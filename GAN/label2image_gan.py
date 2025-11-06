# label2image_gan.py
import os, argparse, random, glob
import numpy as np
from PIL import Image
import cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# -----------------
# Config
# -----------------
CLASS_NAMES = ['Background', 'Forest land', 'Grassland', 'Cropland', 'Settlement', 'Seminatural Grassland']
PALETTE = [
    [250, 62, 119],     # Background
    [168, 232, 84],     # Forest land
    [242, 180, 92],     # Grassland
    [116, 116, 116],    # Cropland
    [255, 214, 33],     # Settlement
    [33, 150, 243],     # Seminatural Grassland
]
C = len(CLASS_NAMES)
COLORS2ID = {tuple(rgb): i for i, rgb in enumerate(PALETTE)}  # RGB -> class id

# Set your synthetic pair resolution here.
# If your GAN was trained at 256, set 256. If you want 512 outputs, set 512.
IMG_SIZE = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------
# Data utils
# -----------------
def remap_mask_ids(msk_np, C, bg_id=0):
    """
    Convert RGB or int mask to ID mask in [0..C-1].
    - RGB masks are mapped using COLORS2ID.
    - Out-of-range ints are clipped to bg_id.
    """
    if msk_np.ndim == 2:  # already IDs
        m = msk_np.astype(np.int64)
        m[(m < 0) | (m >= C)] = bg_id
        return m

    # RGB -> IDs
    h, w, _ = msk_np.shape
    out = np.full((h, w), bg_id, dtype=np.int64)
    flat = msk_np.reshape(-1, 3)
    out_flat = out.reshape(-1)
    for rgb, cid in COLORS2ID.items():
        r, g, b = rgb
        hit = (flat[:, 0] == r) & (flat[:, 1] == g) & (flat[:, 2] == b)
        out_flat[hit] = int(cid)
    return out

# -----------------
# Dataset
# -----------------
class PairDataset(Dataset):
    def __init__(self, root, split="train"):
        self.img_paths = sorted(glob.glob(os.path.join(root, split, "images", "*")))
        self.msk_paths = [p.replace("/images/","/masks/").rsplit(".",1)[0]+".png" for p in self.img_paths]
        self.tf_img = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.tf_msk = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=Image.NEAREST)
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        msk_img = Image.open(self.msk_paths[idx])  # keep RGB/palette

        img_t = self.tf_img(img)  # [3,H,W], 0..1

        # Resize BEFORE mapping; then map RGB->IDs
        msk_np = np.array(self.tf_msk(msk_img))
        msk_ids = remap_mask_ids(msk_np, C=C, bg_id=0).astype(np.int64)  # [H,W]

        onehot = np.eye(C, dtype=np.float32)[msk_ids]          # [H,W,C]
        onehot_t = torch.from_numpy(onehot).permute(2,0,1)     # [C,H,W]

        return img_t, onehot_t, torch.from_numpy(msk_ids)

# -----------------
# Models
# -----------------
def conv_block(in_ch, out_ch, ks=3, s=1, p=1, norm=True):
    layers = [nn.Conv2d(in_ch, out_ch, ks, s, p)]
    if norm: layers += [nn.InstanceNorm2d(out_ch, affine=True)]
    layers += [nn.LeakyReLU(0.2, inplace=True)]
    return nn.Sequential(*layers)

class UNetGen(nn.Module):
    # Conditional UNet: input is semantic onehot [C,H,W]
    def __init__(self, in_ch=C, base=64):
        super().__init__()
        # enc
        self.e1 = conv_block(in_ch, base, norm=False)
        self.e2 = conv_block(base, base*2, s=2)
        self.e3 = conv_block(base*2, base*4, s=2)
        self.e4 = conv_block(base*4, base*8, s=2)
        self.e5 = conv_block(base*8, base*8, s=2)
        # dec
        self.d1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                conv_block(base*8, base*8))
        self.d2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                conv_block(base*8+base*8, base*4))
        self.d3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                conv_block(base*4+base*4, base*2))
        self.d4 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                conv_block(base*2+base*2, base))
        self.outc = nn.Sequential(nn.Conv2d(base+base, 3, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        e1 = self.e1(x)           # [b,64,H,W]
        e2 = self.e2(e1)          # [b,128,H/2,W/2]
        e3 = self.e3(e2)          # [b,256,H/4,W/4]
        e4 = self.e4(e3)          # [b,512,H/8,W/8]
        e5 = self.e5(e4)          # [b,512,H/16,W/16]
        d1 = self.d1(e5)          # [b,512,H/8,W/8]
        d2 = self.d2(torch.cat([d1, e4], 1))
        d3 = self.d3(torch.cat([d2, e3], 1))
        d4 = self.d4(torch.cat([d3, e2], 1))
        out = self.outc(torch.cat([d4, e1], 1))
        return (out+1)/2.0        # 0..1

class PatchDis(nn.Module):
    # Input: concat [RGB(3) + onehot(C)]
    def __init__(self, in_ch=3+C, base=64):
        super().__init__()
        self.c1 = conv_block(in_ch, base, s=2, norm=False)
        self.c2 = conv_block(base, base*2, s=2)
        self.c3 = conv_block(base*2, base*4, s=2)
        self.c4 = conv_block(base*4, base*8, s=1)
        self.out = nn.Conv2d(base*8, 1, 1)
    def forward(self, x):
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        logits = self.out(h4)
        return logits, (h1, h2, h3, h4)  # return features for FM loss

# -----------------
# Losses
# -----------------
def hinge_d(d_real, d_fake):
    return (F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean())

def hinge_g(d_fake):
    return -d_fake.mean()

# -----------------
# Metrics / Aug
# -----------------
def mean_iou(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        p = (pred==cls); t = (target==cls)
        inter = (p & t).sum()
        union = (p | t).sum()
        if union == 0:
            continue
        ious.append((inter / union).item())
    return float(np.mean(ious)) if ious else 0.0

def jitter_mask_np(mask, max_dilate=2, max_erode=2, elastic_alpha=15, elastic_sigma=3):
    m = mask.copy()
    for cls in np.unique(m):
        k = np.random.randint(0, max_dilate+1)
        if k:
            m = np.where(cv2.dilate((m==cls).astype(np.uint8), np.ones((k,k), np.uint8)), cls, m)
        k = np.random.randint(0, max_erode+1)
        if k:
            m = np.where(cv2.erode((m==cls).astype(np.uint8), np.ones((k,k), np.uint8)), cls, m)
    h,w = m.shape
    dx = cv2.GaussianBlur((np.random.rand(h,w)*2-1), (0,0), elastic_sigma)*elastic_alpha
    dy = cv2.GaussianBlur((np.random.rand(h,w)*2-1), (0,0), elastic_sigma)*elastic_alpha
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + dx).astype(np.float32); map_y = (grid_y + dy).astype(np.float32)
    mel = cv2.remap(m.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    return mel.astype(mask.dtype)

# -----------------
# Train
# -----------------
def train(args):
    ds = PairDataset(args.data_root, "train")
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=0, drop_last=True)
    val = PairDataset(args.data_root, "val")
    vl = DataLoader(val, batch_size=args.bs, shuffle=False, num_workers=0)

    G = UNetGen(C).to(DEVICE)
    D = PatchDis(3+C).to(DEVICE)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    for epoch in range(1, args.epochs+1):
        G.train(); D.train()
        for img, onehot, _ in dl:
            img, onehot = img.to(DEVICE), onehot.to(DEVICE)

            # --- Train D ---
            with torch.no_grad():
                fake = G(onehot)
            real_in = torch.cat([img, onehot], 1)
            fake_in = torch.cat([fake, onehot], 1)
            d_real_logits, _ = D(real_in)
            d_fake_logits, _ = D(fake_in)
            loss_d = hinge_d(d_real_logits, d_fake_logits)
            optD.zero_grad(); loss_d.backward(); optD.step()

            # --- Train G ---
            for p in D.parameters():
                p.requires_grad_(False)

            fake = G(onehot)
            fake_in = torch.cat([fake, onehot], 1)
            d_fake_logits, fake_feats = D(fake_in)
            gan = hinge_g(d_fake_logits)

            with torch.no_grad():
                real_in = torch.cat([img, onehot], 1)
                _, real_feats = D(real_in)

            fm = sum(F.l1_loss(f, r) for f, r in zip(fake_feats, real_feats))
            loss_g = gan + 10.0*fm

            optG.zero_grad(); loss_g.backward(); optG.step()
            for p in D.parameters():
                p.requires_grad_(True)

        # quick val sample
        G.eval()
        with torch.no_grad():
            img, onehot, _ = next(iter(vl))
            fake = G(onehot.to(DEVICE)).cpu().clamp(0,1)
            os.makedirs(args.out_dir, exist_ok=True)
            grid = (fake[:4]*255).byte().permute(0,2,3,1).numpy()
            for i,arr in enumerate(grid):
                Image.fromarray(arr).save(os.path.join(args.out_dir, f"epoch{epoch:03d}_{i}.jpg"))
        torch.save(G.state_dict(), os.path.join(args.out_dir, f"G_e{epoch:03d}.pt"))
        torch.save(D.state_dict(), os.path.join(args.out_dir, f"D_e{epoch:03d}.pt"))
        print(f"Epoch {epoch}: saved samples and checkpoints.")

# -----------------
# Generate synthetic pairs (+ optional filter)
# -----------------
def load_segmenter(path_or_none):
    if not path_or_none:
        return None
    # IMPORTANT: Load TorchScript on CPU and keep it on CPU to avoid cuda/cpu mismatches
    m = torch.jit.load(path_or_none, map_location="cpu")
    m.eval()
    return m

@torch.no_grad()
def generate(args):
    # Load Generator (GPU if available)
    G = UNetGen(C).to(DEVICE)
    G.load_state_dict(torch.load(args.gen_ckpt, map_location=DEVICE))
    G.eval()

    # Load available real masks
    any_masks = glob.glob(os.path.join(args.data_root, "train", "masks", "*.png"))
    assert any_masks, "No masks found."
    os.makedirs(args.save_dir, exist_ok=True)

    # Optional segmenter for filtering (TorchScript model on CPU)
    S = load_segmenter(args.segmenter_ckpt)

    kept = 0
    SEG_IN_SIZE = 512  # input size the TS segmenter expects (your export size)

    for i in range(args.num_samples):
        # 1. Pick and prepare a random mask
        m_path = random.choice(any_masks)
        m_rgb = np.array(Image.open(m_path).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
        m = remap_mask_ids(m_rgb, C=C, bg_id=0)
        m = jitter_mask_np(m)

        # 2. Convert mask to one-hot and generate fake image
        onehot = torch.from_numpy(np.eye(C, dtype=np.float32)[m]).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        fake = G(onehot).cpu()[0].clamp(0, 1)  # [3,H,W] on CPU for later CPU segmenter
        img_uint8 = (fake * 255).byte().permute(1, 2, 0).numpy()

        ok = True

        # 3. Self-consistency filter using the segmenter (CPU)
        if S is not None:
            x = fake.unsqueeze(0)  # [1,3,H,W] on CPU, 0..1

            # --- Upsize to 512 for the segmenter if needed ---
            if x.shape[-2] != SEG_IN_SIZE or x.shape[-1] != SEG_IN_SIZE:
                x_seg = F.interpolate(
                    x, size=(SEG_IN_SIZE, SEG_IN_SIZE),
                    mode="bilinear", align_corners=False
                )
            else:
                x_seg = x

            logits = S(x_seg)  # run TorchScript on CPU
            pred = torch.argmax(logits, dim=1)  # [1,Hs,Ws] (CPU)

            # --- Downsize back to GAN image size before comparing ---
            if pred.shape[-2] != IMG_SIZE or pred.shape[-1] != IMG_SIZE:
                pred = F.interpolate(
                    pred.float().unsqueeze(1),
                    size=(IMG_SIZE, IMG_SIZE),
                    mode="nearest"
                ).squeeze(1).long()[0]
            else:
                pred = pred[0]

            # Compute mIoU for filtering
            miou = mean_iou(pred.numpy(), m, C)  # both CPU numpy/int
            ok = miou >= args.tau

        # 4. Save if passed filtering
        if ok:
            base = os.path.join(args.save_dir, f"pair_{i:05d}")
            Image.fromarray(img_uint8).save(base + "_img.jpg")
            Image.fromarray(m.astype(np.uint8)).save(base + "_mask.png")
            kept += 1

    print(f"Generated {args.num_samples} pairs, kept {kept} with tau={args.tau}")

# -----------------
# CLI
# -----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    tr = sub.add_parser("train")
    tr.add_argument("--data_root", required=True)     # expects data/train/{images,masks}, data/val/{images,masks}
    tr.add_argument("--out_dir", default="runs/label2img")
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--bs", type=int, default=8)
    tr.add_argument("--lr", type=float, default=2e-4)

    ge = sub.add_parser("gen")
    ge.add_argument("--data_root", required=True)
    ge.add_argument("--gen_ckpt", required=True)
    ge.add_argument("--save_dir", default="synthetic")
    ge.add_argument("--num_samples", type=int, default=200)
    ge.add_argument("--segmenter_ckpt", default="")   # optional TorchScript model
    ge.add_argument("--tau", type=float, default=0.6) # mIoU threshold

    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "gen":
        generate(args)
    else:
        print("Use 'train' or 'gen'")
