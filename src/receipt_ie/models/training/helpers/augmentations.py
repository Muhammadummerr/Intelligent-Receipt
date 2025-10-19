import random
import numpy as np
import albumentations as A
from PIL import Image
from .data_utils import sanitize_box

augment = A.Compose([
    A.Affine(scale=(0.95, 1.05), translate_percent=(0.02, 0.02), rotate=(-3, 3), p=0.6),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(p=0.3),
], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["texts"], min_visibility=0.2))

def apply_augmentation(image, ocr_data):
    """Apply augmentation while maintaining valid boxes."""
    if random.random() > 0.5:
        texts = [t for t, _ in ocr_data]
        boxes = [sanitize_box(b, *image.size) for _, b in ocr_data]
        try:
            aug = augment(image=np.array(image), bboxes=boxes, texts=texts)
            image_aug = Image.fromarray(aug["image"])
            ocr_aug = list(zip(aug["texts"], [sanitize_box(b, *image.size) for b in aug["bboxes"]]))
            return image_aug, ocr_aug
        except Exception:
            return image, ocr_data
    return image, ocr_data