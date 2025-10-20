"""
dataset_infer.py
----------------
Utility dataset class for running LayoutLMv3 inference on OCR-ready receipts.

Responsibilities:
- Loads paired image and OCR box files.
- Converts text and bounding boxes into normalized LayoutLMv3 input format (0–1000 space).
- Preserves reading order and word-token alignment for accurate field extraction.

Used in:
    - run_pipeline.py
    - evaluation scripts for LayoutLMv3 inference
"""

import os
import torch
from typing import List, Optional, Dict
from PIL import Image, UnidentifiedImageError
from transformers import LayoutLMv3Processor
from .boxes import parse_box_file, sort_reading_order
from ..utils.text import split_tokens


IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
BOX_EXTS = [".txt", ".TXT"]


def _find_with_ext(dirpath: str, stem: str, exts: List[str]) -> Optional[str]:
    """Return the first file matching stem + extension (case-insensitive)."""
    if not os.path.isdir(dirpath):
        return None
    for e in exts:
        path = os.path.join(dirpath, stem + e)
        if os.path.isfile(path):
            return path
    return None


class ReceiptInferenceDataset:
    """
    Dataset for preparing LayoutLMv3 inference inputs.

    Args:
        img_dir (str): Directory with receipt images.
        box_dir (str): Directory with OCR box text files.
        processor (LayoutLMv3Processor): Tokenizer + feature extractor.
        max_seq_len (int): Max sequence length (default=512).
        stems_subset (List[str], optional): Specific subset of stems to include.

    Returns:
        Dict[str, torch.Tensor] with:
            - input_ids
            - attention_mask
            - bbox
            - pixel_values
            - word_ids
            - orig_words
            - id
            - line_texts
    """

    def __init__(
        self,
        img_dir: str,
        box_dir: str,
        processor: LayoutLMv3Processor,
        max_seq_len: int = 512,
        stems_subset: Optional[List[str]] = None,
    ):
        self.img_dir = img_dir
        self.box_dir = box_dir
        self.processor = processor
        self.max_seq_len = max_seq_len

        discovered = set()
        for name in os.listdir(img_dir):
            stem, ext = os.path.splitext(name)
            if ext not in IMG_EXTS:
                continue
            if _find_with_ext(box_dir, stem, BOX_EXTS):
                discovered.add(stem)

        all_stems = sorted(discovered)
        self.stems = [s for s in all_stems if not stems_subset or s in stems_subset]

        print(f"📄 Found {len(self.stems)} valid receipts for inference.")

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stem = self.stems[idx]
        img_path = _find_with_ext(self.img_dir, stem, IMG_EXTS)
        box_path = _find_with_ext(self.box_dir, stem, BOX_EXTS)
        if not (img_path and box_path):
            raise FileNotFoundError(f"Missing image or OCR box file for: {stem}")

        try:
            image = Image.open(img_path).convert("RGB")
        except UnidentifiedImageError:
            raise RuntimeError(f"Invalid or corrupted image: {img_path}")

        W, H = image.size
        lines = sort_reading_order(parse_box_file(box_path))
        line_texts = [li.text for li in lines]

        # --- Build token-level inputs ---
        words, boxes = [], []
        for li in lines:
            xmin, ymin, xmax, ymax = li.aabb

            # Clamp to image bounds
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(W - 1, xmax), min(H - 1, ymax)

            # Fix inverted boxes
            if xmax < xmin:
                xmin, xmax = xmax, xmin
            if ymax < ymin:
                ymin, ymax = ymax, ymin

            # Ensure non-zero area
            if xmax == xmin:
                xmax = min(W - 1, xmin + 1)
            if ymax == ymin:
                ymax = min(H - 1, ymin + 1)

            # Scale to [0, 1000]
            sxmin = max(0, min(int(xmin / W * 1000), 1000))
            sxmax = max(0, min(int(xmax / W * 1000), 1000))
            symin = max(0, min(int(ymin / H * 1000), 1000))
            symax = max(0, min(int(ymax / H * 1000), 1000))

            # Tokenize text and duplicate box for each token
            toks = split_tokens(li.text)
            for _ in toks:
                words.append(_)
                boxes.append([sxmin, symin, sxmax, symax])

        # Truncate for model limits
        words, boxes = words[:self.max_seq_len], boxes[:self.max_seq_len]

        enc = self.processor(
            image,
            words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}

        # Word ID alignment
        enc0 = enc.encodings[0]
        wids_attr = getattr(enc0, "word_ids", None)
        wids = wids_attr() if callable(wids_attr) else list(wids_attr) if wids_attr else None
        if wids is None:
            wids = [None] * item["input_ids"].shape[0]

        item["word_ids"] = torch.tensor([-1 if w is None else w for w in wids], dtype=torch.long)
        item["orig_words"] = words
        item["id"] = stem
        item["line_texts"] = line_texts

        return item
