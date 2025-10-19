#!/usr/bin/env python3
"""
augment_watermark_dataset.py
-----------------------------
Expands your watermark detection dataset via smart augmentations.

Usage (Kaggle / Local):
!python -m src.receipt_ie.models.watermark.augment_watermark_dataset \
  --input_dir /kaggle/input/watermark-dataset \
  --output_dir /kaggle/working/receipt-watermark-augmented \
  --split_ratio 0.8 \
  --aug_per_image 3

Input directory (structure):
    base/
      ├── clean/
      └── watermark/

Output:
    output_root/
      ├── train/{clean,watermarked}
      └── test/{clean,watermarked}
"""

import os
import random
from pathlib import Path
from tqdm import tqdm
import albumentations as A
import cv2
from sklearn.model_selection import train_test_split
import argparse


# ==================== AUGMENTATION PIPELINE ====================
augment = A.Compose([
    A.Rotate(limit=3, p=0.4),
    A.RandomBrightnessContrast(p=0.4),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.ImageCompression(quality_lower=85, quality_upper=100, p=0.4),
    A.CLAHE(p=0.3),
    A.GaussNoise(var_limit=(5, 20), p=0.3),
    A.Affine(scale=(0.98, 1.02), translate_percent=(0.01, 0.02), rotate=(-2, 2), p=0.3)
])

# ==================== HELPERS ====================
def ensure_dirs(base: Path):
    """Create output folders."""
    for split in ["train", "test"]:
        for cls in ["clean", "watermarked"]:
            (base / split / cls).mkdir(parents=True, exist_ok=True)


def augment_and_save(img_path: Path, out_dir: Path, n_aug: int):
    """Apply augmentations and save original + augmented copies."""
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    base_name = img_path.stem

    # Save original
    cv2.imwrite(str(out_dir / f"{base_name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Augmentations
    for i in range(n_aug):
        aug_img = augment(image=img)["image"]
        out_path = out_dir / f"{base_name}_aug{i+1}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))


# ==================== MAIN FUNCTION ====================
def main():
    parser = argparse.ArgumentParser(description="Augment watermark detection dataset.")
    parser.add_argument("--input_dir", required=True, help="Path to base dataset containing clean/ and watermark/ subfolders")
    parser.add_argument("--output_dir", required=True, help="Path to save augmented dataset")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Train/test split ratio")
    parser.add_argument("--aug_per_image", type=int, default=3, help="Number of augmented copies per original image")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    ensure_dirs(output_dir)

    all_classes = [("clean", "clean"), ("watermark", "watermarked")]

    for src_cls, dst_cls in all_classes:
        src_dir = input_dir / src_cls
        if not src_dir.exists():
            print(f"⚠️ Missing {src_dir}")
            continue

        all_imgs = [p for p in src_dir.glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
        train_imgs, test_imgs = train_test_split(all_imgs, train_size=args.split_ratio, random_state=args.seed)

        for split_name, subset in [("train", train_imgs), ("test", test_imgs)]:
            split_dir = output_dir / split_name / dst_cls
            print(f"📸 Processing {src_cls} -> {split_name} ({len(subset)} images)...")
            for img_path in tqdm(subset):
                try:
                    augment_and_save(img_path, split_dir, args.aug_per_image)
                except Exception as e:
                    print(f"❌ Error on {img_path}: {e}")

    print(f" Augmented dataset created at: {output_dir}")


if __name__ == "__main__":
    main()
