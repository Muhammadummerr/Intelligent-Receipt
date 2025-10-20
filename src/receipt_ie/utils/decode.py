"""
decode.py
----------
Utilities for decoding BIO token labels from LayoutLMv3 predictions
into structured receipt fields (company, date, address, total).

Used by:
 - Training validation and debugging (token → field alignment)
 - Inference pipeline (`run_pipeline.py`)
"""

import re
from typing import List, Dict, Optional

# ------------------------------
# Label definitions (align with training)
# ------------------------------
LABELS = [
    "O",
    "B-COMPANY", "I-COMPANY",
    "B-DATE", "I-DATE",
    "B-ADDRESS", "I-ADDRESS",
    "B-TOTAL", "I-TOTAL",
]

LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

# ------------------------------
# Main decoding utility
# ------------------------------
def group_bio(words: List[str], label_ids: List[int], id2label: Dict[int, str] = None) -> Dict[str, str]:
    """
    Convert word-level BIO ids into field strings.
    Returns dict with keys: company, date, address, total (lowercase).

    Strategy:
      - Merge consecutive B/I- tags into coherent spans
      - Prefer early spans for company/address, last span for total
      - Clean spacing/punctuation
      - Automatically handles sequence length mismatches (LayoutLMv3 512 vs OCR words)
    """
    if id2label is None:
        id2label = ID2LABEL

    # ✅ Safety alignment for mismatched sequence lengths
    if len(label_ids) != len(words):
        min_len = min(len(words), len(label_ids))
        words = words[:min_len]
        label_ids = label_ids[:min_len]

    spans: Dict[str, List[List[str]]] = {}
    cur_field: Optional[str] = None
    buf: List[str] = []

    def _flush(field, buf, spans):
        if field and buf:
            spans.setdefault(field, []).append(buf[:])
        buf.clear()

    # --- Build spans ---
    for w, li in zip(words, label_ids):
        if w == "[PAD]":
            continue  # ignore padding tokens

        lab = id2label.get(int(li), "O")
        if lab == "O":
            _flush(cur_field, buf, spans)
            cur_field = None
            continue

        prefix, _, tag = lab.partition("-")
        if prefix == "B":
            _flush(cur_field, buf, spans)
            cur_field = tag
            buf.append(w)
        elif prefix == "I":
            if cur_field == tag:
                buf.append(w)
            else:
                _flush(cur_field, buf, spans)
                cur_field = tag
                buf.append(w)
        else:
            _flush(cur_field, buf, spans)
            cur_field = None

    _flush(cur_field, buf, spans)

    # --- Pick best span per field ---
    out: Dict[str, str] = {}
    for tag in ("COMPANY", "DATE", "ADDRESS", "TOTAL"):
        cands = spans.get(tag, [])
        if not cands:
            out[tag.lower()] = ""
            continue

        # Prefer early spans for COMPANY/ADDRESS, last for TOTAL
        if tag in ("COMPANY", "ADDRESS"):
            best = cands[0]
        elif tag == "TOTAL":
            best = cands[-1]
        else:  # DATE
            best = max(cands, key=lambda s: len(s))

        # 🌟 Limit company span length to prevent over-spanning
        if tag == "COMPANY" and len(best) > 15:
            best = best[:15]

        clean = " ".join(best).strip(" -:;,._")
        out[tag.lower()] = clean

    return out
