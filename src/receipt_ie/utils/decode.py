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
def group_bio(words: List[str], label_ids: List[int],
              id2label: Dict[int, str] = None) -> Dict[str, str]:
    """
    Convert word-level BIO ids into structured field strings.

    Returns dict with keys: {"company", "date", "address", "total"} (lowercase).

    Strategy:
      - Merge consecutive B/I- tags into coherent spans.
      - Prefer early spans for company/address, longest for date, last for total.
      - Normalize spacing and punctuation.
    """
    if id2label is None:
        id2label = ID2LABEL

    # --- Safety check ---
    assert len(words) == len(label_ids), (
        f"Mismatched sequence lengths: words={len(words)} vs label_ids={len(label_ids)}"
    )

    spans: Dict[str, List[List[str]]] = {}
    current_field: Optional[str] = None
    buffer: List[str] = []

    def _flush(field, buf, spans):
        if field and buf:
            spans.setdefault(field, []).append(buf[:])
        buf.clear()

    # --- Build spans ---
    for w, li in zip(words, label_ids):
        if w == "[PAD]":
            continue  # ignore padding tokens

        label = id2label.get(int(li), "O")
        if label == "O":
            _flush(current_field, buffer, spans)
            current_field = None
            continue

        prefix, _, tag = label.partition("-")
        if prefix == "B":
            _flush(current_field, buffer, spans)
            current_field = tag
            buffer.append(w)
        elif prefix == "I":
            if current_field == tag:
                buffer.append(w)
            else:
                _flush(current_field, buffer, spans)
                current_field = tag
                buffer.append(w)
        else:
            _flush(current_field, buffer, spans)
            current_field = None

    _flush(current_field, buffer, spans)

    # --- Span selection strategies ---
    FIELD_SELECTION = {
        "COMPANY": "first",
        "ADDRESS": "first",
        "DATE": "longest",
        "TOTAL": "last",
    }

    out: Dict[str, str] = {}
    for tag in FIELD_SELECTION.keys():
        cands = spans.get(tag, [])
        if not cands:
            out[tag.lower()] = ""
            continue

        strategy = FIELD_SELECTION[tag]
        if strategy == "first":
            best = cands[0]
        elif strategy == "last":
            best = cands[-1]
        elif strategy == "longest":
            best = max(cands, key=len)
        else:
            best = cands[0]

        # Prevent over-spanning in COMPANY fields
        if tag == "COMPANY" and len(best) > 15:
            best = best[:15]

        clean = re.sub(r"\s{2,}", " ", " ".join(best)).strip(" -:;,_")
        out[tag.lower()] = clean

    return out
