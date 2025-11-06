#!/usr/bin/env python3
import os
import glob
import argparse
import cv2
import numpy as np

# ----- Edit if you want different colors/classes -----
CLASSES = ('Background','Forest land', 'Grassland', 'Cropland', 'Settlement', 'Seminatural Grassland')
PALETTE = [
    [11, 246, 210],   # 0 Background/ignore
    [250, 62, 119],   # 1 Forest land
    [168, 232, 84],   # 2 Grassland
    [242, 180, 92],   # 3 Cropland
    [116, 116, 116],  # 4 Settlement
    [255, 214, 33],   # 5 Seminatural Grassland
]
# -----------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Quick visualizer for image+mask pairs (with contrast stretch).")
    p.add_argument("--image-dir", default="data/Biodiversity_tiff/Train/image", help="Directory with images")
    p.add_argument("--mask-dir",  default="data/Biodiversity_tiff/Train/masks_converted_rgb", help="Directory with masks")
    p.add_argument("--img-suffix", default=".tif", help="Image file suffix (e.g., .tif, .png, .jpg)")
    p.add_argument("--mask-suffix", default=".png", help="Mask file suffix (e.g., .png, .tif)")
    p.add_argument("--alpha", type=float, default=0.5, help="Overlay opacity for masks")
    # Contrast stretch controls
    p.add_argument("--p-lo", type=float, default=2.0, help="Low percentile for contrast stretch")
    p.add_argument("--p-hi", type=float, default=98.0, help="High percentile for contrast stretch")
    p.add_argument("--gamma", type=float, default=1.0, help="Gamma after stretch (1.0 = none)")
    return p.parse_args()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def is_rgb_mask(arr):
    return arr is not None and arr.ndim == 3 and arr.shape[2] == 3



def stretch_uint8(arr, p_lo=2.0, p_hi=98.0, gamma=1.0):
    """
    Percentile stretch to uint8. Works on 2D or 3D arrays (uses first 3 channels).
    - Ignores NaN/inf automatically.
    - Falls back to min-max if percentiles collapse.
    - Optional gamma after normalization.
    """
    a = arr.astype(np.float32)
    def _stretch(ch):
        finite = np.isfinite(ch)
        if not finite.any():
            return np.zeros_like(ch, dtype=np.uint8)
        lo = np.percentile(ch[finite], p_lo)
        hi = np.percentile(ch[finite], p_hi)
        if hi <= lo:
            lo = float(np.nanmin(ch[finite]))
            hi = float(np.nanmax(ch[finite]))
            if hi <= lo:
                return np.zeros_like(ch, dtype=np.uint8)
        x = (ch - lo) / (hi - lo)
        x = np.clip(x, 0, 1)
        if gamma != 1.0:
            x = np.power(x, 1.0 / gamma)
        return (x * 255.0 + 0.5).astype(np.uint8)

    if a.ndim == 2:
        return _stretch(a)
    elif a.ndim == 3:
        # Use first 3 bands for preview; if fewer, tile
        C = a.shape[2]
        if C >= 3:
            chs = [ _stretch(a[..., i]) for i in range(3) ]
        elif C == 2:
            ch0 = _stretch(a[..., 0])
            ch1 = _stretch(a[..., 1])
            chs = [ch0, ch1, ch1]
        else:  # C == 1
            ch0 = _stretch(a[..., 0])
            chs = [ch0, ch0, ch0]
        return np.dstack(chs)
    else:
        # Weird shape: collapse to 2D and tile
        if a.ndim == 1:
            a = a[:, None]
        u = _stretch(a)
        return np.dstack([u, u, u])

def load_image_rgb(path, p_lo=2.0, p_hi=98.0, gamma=1.0):
    """
    Reads a possibly high-bit-depth/float GeoTIFF/PNG and returns
    a contrast-stretched uint8 RGB image suitable for visualization/overlay.
    """
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError(f"Failed to read image: {path}")

    # Force to (H,W,C) with at least 3 channels before stretch (don't do BGR<->RGB yet)
    if im.ndim == 2:
        im3 = np.dstack([im, im, im])
    elif im.ndim == 3:
        # Drop alpha or extra bands; keep first 3 bands
        if im.shape[2] >= 3:
            im3 = im[..., :3]
        else:
            im3 = np.dstack([im[..., 0]] * 3)
    else:
        im3 = np.dstack([im, im, im]) if im.ndim == 1 else im

    # Percentile-stretch to uint8
    vis = stretch_uint8(im3, p_lo=p_lo, p_hi=p_hi, gamma=gamma)

    # OpenCV writes expect BGR; we’ll convert at save time. Keep as RGB here.
    return vis

def load_mask_rgb(path):
    """
    Reads mask. If already RGB, convert BGR->RGB and uint8.
    If single-channel class IDs, colorize with PALETTE.
    """
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    if is_rgb_mask(m):
        m = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
        if m.dtype != np.uint8:
            m = stretch_uint8(m, 2.0, 98.0, 1.0)  # just in case
        return m, "rgb"
    else:
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        if m.dtype != np.uint8:
            m = m.astype(np.uint8)
        return label_to_rgb(m), "indexed"

def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_root   = os.path.join(script_dir, "quickviz")
    out_images = os.path.join(out_root, "images")
    out_masks  = os.path.join(out_root, "rgb")
    out_overlay= os.path.join(out_root, "overlay")
    for d in (out_images, out_masks, out_overlay):
        ensure_dir(d)

    # Build stem -> path maps
    imgs = {}
    for p in glob.glob(os.path.join(args.image_dir, f"*{args.img_suffix}")):
        stem = os.path.splitext(os.path.basename(p))[0]
        imgs[stem] = p

    masks = {}
    for p in glob.glob(os.path.join(args.mask_dir, f"*{args.mask_suffix}")):
        stem = os.path.splitext(os.path.basename(p))[0]
        masks[stem] = p

    common = sorted(set(imgs.keys()) & set(masks.keys()))
    missing_masks = sorted(set(imgs.keys()) - set(masks.keys()))
    missing_imgs  = sorted(set(masks.keys()) - set(imgs.keys()))

    print(f"Found images: {len(imgs)} | masks: {len(masks)} | paired: {len(common)}")
    if missing_masks:
        print(f"Images without masks: {len(missing_masks)} (showing up to 10)\n  " + "\n  ".join(missing_masks[:10]))
    if missing_imgs:
        print(f"Masks without images: {len(missing_imgs)} (showing up to 10)\n  " + "\n  ".join(missing_imgs[:10]))

    count = 0
    for stem in common:
        img_path = imgs[stem]
        msk_path = masks[stem]

        try:
            # Stretch image for visibility
            img = load_image_rgb(img_path, p_lo=args.p_lo, p_hi=args.p_hi, gamma=args.gamma)

            # Prepare mask RGB
            mask_rgb, _ = load_mask_rgb(msk_path)

            # Size check & resize mask if needed (nearest to preserve crisp edges)
            if (mask_rgb.shape[0] != img.shape[0]) or (mask_rgb.shape[1] != img.shape[1]):
                mask_rgb = cv2.resize(mask_rgb, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Blend in float32 then cast back
            img_f  = img.astype(np.float32)
            mask_f = mask_rgb.astype(np.float32)
            overlay = cv2.addWeighted(img_f, 1.0 - args.alpha, mask_f, args.alpha, 0.0)
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            # Save (convert RGB->BGR for cv2.imwrite)
            cv2.imwrite(os.path.join(out_images, f"{stem}.png"),  cv2.cvtColor(img,       cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(out_masks,  f"{stem}.png"),  cv2.cvtColor(mask_rgb,  cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(out_overlay,f"{stem}.png"),  cv2.cvtColor(overlay,   cv2.COLOR_RGB2BGR))

            count += 1
            if count % 50 == 0:
                print(f"Saved {count} visualizations...")
        except Exception as e:
            print(f"[WARN] Skipping {stem}: {e}")

    print(f"Done. Wrote {count} sets into:\n  {out_images}\n  {out_masks}\n  {out_overlay}")

if __name__ == "__main__":
    main()