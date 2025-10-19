"""
wm_dataset.py
-------------
Dataset class for watermark detection (clean vs watermarked receipts).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class ReceiptWMDataset(Dataset):
    def __init__(self, paths, labels, processor, augment: bool = False):
        """
        paths: list of image paths
        labels: list of 0 (clean) or 1 (watermarked)
        processor: ViTImageProcessor
        augment: whether to apply light augmentations
        """
        self.paths = paths
        self.labels = labels
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        y = int(self.labels[idx])

        with Image.open(p) as im:
            im = im.convert("RGB")

        if self.augment:
            from PIL import ImageEnhance
            if np.random.rand() < 0.30:
                im = im.rotate(np.random.uniform(-2.0, 2.0),
                               resample=Image.BILINEAR,
                               expand=False, fillcolor=(255, 255, 255))
            if np.random.rand() < 0.25:
                im = ImageEnhance.Brightness(im).enhance(np.random.uniform(0.9, 1.1))
            if np.random.rand() < 0.20:
                im = ImageEnhance.Contrast(im).enhance(np.random.uniform(0.9, 1.1))
            if np.random.rand() < 0.20:
                arr = np.array(im).astype(np.float32)
                arr += np.random.normal(0, 3.0, arr.shape)
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                im = Image.fromarray(arr)

        pixel_values = self.processor(images=im, return_tensors="pt")["pixel_values"].squeeze(0)
        return {"pixel_values": pixel_values, "labels": torch.tensor(y, dtype=torch.long)}
