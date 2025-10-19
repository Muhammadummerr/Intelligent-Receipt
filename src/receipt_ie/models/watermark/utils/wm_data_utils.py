"""
wm_data_utils.py
----------------
Utility functions for loading and scanning watermark dataset splits.
"""

import os
from typing import Tuple

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

def scan_split(split_dir: str) -> Tuple[list, list]:
    """
    Returns (paths, labels) for a split dir that contains:
      split_dir/clean/*.jpg
      split_dir/watermarked/*.jpg
    Label map: clean->0, watermarked->1
    """
    paths, labels = [], []
    for label_name, label_id in [("clean", 0), ("watermarked", 1)]:
        d = os.path.join(split_dir, label_name)
        if not os.path.isdir(d):
            # tolerate missing; just skip
            continue
        for fn in os.listdir(d):
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(d, fn))
                labels.append(label_id)
    return paths, labels