# Geoseg-GAN Label-to-Image GAN for Synthetic Data Generation

This repository provides a conditional Generative Adversarial Network (GAN) implementation for generating realistic RGB images from semantic segmentation masks.
It is designed to augment limited training datasets by producing new, high-quality image–mask pairs that follow the same class structure and visual distribution as the original data.

The model is based on a U-Net generator and a PatchGAN discriminator, optimized with adversarial and feature-matching losses. It is suitable for tasks such as land-cover synthesis and general semantic-to-realistic image translation.



## 1. Requirements

* Python ≥ 3.8
* PyTorch ≥ 2.0
* torchvision
* Pillow
* OpenCV
* NumPy

Install dependencies using:

```bash
pip install torch torchvision pillow opencv-python numpy
```



## 2. Directory Structure

The code expects the following dataset layout:

```
data/
  train/
    images/    # RGB input images
    masks/     # RGB segmentation masks (same filenames)
  val/
    images/
    masks/
```

Each mask must be a color-coded RGB image following the class palette defined in the script.

Example palette:

| Class                 | RGB Color       |
|  |  |
| Background            | (250, 62, 119)  |
| Forest land           | (168, 232, 84)  |
| Grassland             | (242, 180, 92)  |
| Cropland              | (116, 116, 116) |
| Settlement            | (255, 214, 33)  |
| Seminatural Grassland | (33, 150, 243)  |



## 3. Script Overview

The script `label2image_gan.py` contains the following main components:

* **UNetGen** – the generator network that converts one-hot encoded semantic masks into RGB images.
* **PatchDis** – the discriminator network that distinguishes between real and generated image–mask pairs.
* **Feature Matching Loss** – encourages the generator to produce images whose intermediate discriminator features resemble those of real samples.
* **Hinge Loss** – used for adversarial training to stabilize convergence.
* **Mask Remapping** – automatically maps RGB mask colors to integer class IDs based on the defined palette.



## 4. Training the GAN

To train the model on your dataset:

```bash
python label2image_gan.py train \
  --data_root data \
  --out_dir runs/label2img \
  --epochs 50 \
  --bs 8 \
  --lr 2e-4
```

**Arguments:**

* `--data_root` – root folder containing `train/` and `val/` subfolders.
* `--out_dir` – directory where checkpoints and samples will be saved.
* `--epochs` – total number of training epochs (default: 10).
* `--bs` – batch size (default: 8).
* `--lr` – learning rate (default: 2e-4).

**Output per epoch:**

* Saved sample images in `runs/label2img/epoch###_i.jpg`
* Model checkpoints `G_e###.pt` (generator) and `D_e###.pt` (discriminator)
* A console message such as:

  ```
  Epoch 1: saved samples and checkpoints.
  ```

Early epochs may produce blurry or abstract textures. Visual realism improves gradually; epochs 30–50 typically yield visually coherent outputs.



## 5. Generating Synthetic Image–Mask Pairs

After training, generate new data using:

```bash
python label2image_gan.py gen \
  --data_root data \
  --gen_ckpt runs/label2img/G_e050.pt \
  --save_dir synthetic \
  --num_samples 1000
```

This command:

* Samples random masks from the training set.
* Applies geometric jittering to create novel variations.
* Generates corresponding RGB images using the trained generator.
* Saves results to `synthetic/` as:

  ```
  pair_00000_img.jpg
  pair_00000_mask.png
  ```

### Filtering with a Segmenter (optional)

To retain only high-quality samples, provide a segmentation model checkpoint and enable self-consistency filtering:

```bash
python label2image_gan.py gen \
  --data_root data \
  --gen_ckpt runs/label2img/G_e050.pt \
  --save_dir synthetic_filtered \
  --num_samples 2000 \
  --segmenter_ckpt path/to/segmenter.pt \
  --tau 0.65
```

Only pairs with a mean Intersection-over-Union (mIoU) above the threshold `tau` are kept.
Console output example:

```
Generated 2000 pairs, kept 1578 with tau=0.65
```



## 6. Using Synthetic Data for Augmentation

The generated dataset can be merged with the original dataset to improve segmentation performance, especially in cases of class imbalance or limited data.

Recommended strategy:

* Mix approximately **20–30% synthetic data** per training epoch.
* Optionally oversample rare classes using masks containing specific class IDs.
* Monitor performance on the **real validation set** only.



## 7. Hyperparameter Guidance

| Parameter               | Effect                              | Recommendation                                                   |
| -- | -- | - |
| `--epochs`              | Training duration                   | 30–50 for stable results                                         |
| `--bs`                  | Batch size                          | 8–16 for balance between stability and detail                    |
| `--lr`                  | Learning rate                       | 2e-4 (scale linearly with batch size)                            |
| Feature-matching weight | Controls texture realism            | Default 10.0; can be increased slightly for sharper edges        |
| Jittering parameters    | Defines mask perturbation intensity | Default values usually sufficient; can be adjusted for diversity |

Increasing batch size smooths training and improves stability but may reduce fine detail.
Decreasing it enhances texture sharpness but increases training noise.



## 8. Expected Outputs

After training and generation:

```
runs/label2img/
  epoch001_0.jpg
  epoch050_0.jpg
  G_e050.pt
  D_e050.pt

synthetic/
  pair_00000_img.jpg
  pair_00000_mask.png
  ...
```

`epoch050_0.jpg` should show clear visual correlation between the semantic mask and generated textures.



## 9. Typical Workflow Summary

1. Prepare dataset with RGB masks following the defined palette.
2. Train the label-to-image GAN for 30–50 epochs.
3. Generate several hundred to several thousand synthetic pairs.
4. Optionally filter using a trained segmenter for quality assurance.
5. Mix synthetic and real data to improve your main segmentation model.



## 10. Notes

* The generator can be fine-tuned at higher resolution (e.g., 512×512) after convergence at 256×256.
* The system is intended for data augmentation, not photorealistic rendering.
* Monitor generated sample quality rather than loss values to gauge progress.



This README provides all necessary context for setting up, training, and using the GAN to produce synthetic data for semantic segmentation tasks.
