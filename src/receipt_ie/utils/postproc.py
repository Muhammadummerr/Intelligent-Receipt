"""
postproc.py
------------
Text normalization and heuristic post-processing utilities for receipt field extraction.

Handles:
- Normalizing and extracting DATE values (DD/MM/YYYY)
- Cleaning and standardizing TOTAL amounts
- Filtering and refining COMPANY names
- Light address normalization for consistency

Used after model prediction or LLM correction to ensure structured and clean final outputs.
"""

import re
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Change to INFO for debugging

WS = re.compile(r"\s+")
def norm_spaces(s: str) -> str:
    """Normalize whitespace and trim."""
    return WS.sub(" ", (s or "").strip())

# ----------------------
# DATE
# ----------------------
_DATE_TOK = r"(?:\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})"
DATE_LIKE = re.compile(_DATE_TOK, flags=re.IGNORECASE)

def soft_date_norm(s: str) -> str:
    """Normalize dates like 20-04-18 → 20/04/2018."""
    s = norm_spaces(s)
    m = re.match(r"^\s*(\d{1,4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,4})\s*$", s)
    if not m:
        return s

    a, b, c = m.groups()
    if len(a) == 4 and a.startswith("20"): a = a[-2:]
    if len(c) == 4 and c.startswith("20"): c = c[-2:]

    try:
        if len(a) <= 2 and len(c) <= 2:
            return f"{int(a):02d}/{int(b):02d}/{int(c):02d}"
        if len(a) == 4 and len(c) <= 2:
            return f"{int(c):02d}/{int(b):02d}/{a[-2:]}"
    except Exception:
        pass
    return s


def extract_best_date(s: str) -> str:
    """Find and normalize the most plausible date string in text."""
    s_norm = norm_spaces(s)
    hits = DATE_LIKE.findall(s_norm)
    if not hits:
        return ""
    def score(tok: str) -> int:
        t = tok.upper()
        sc = 0
        if re.search(r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b", t):
            sc += 2
        if re.search(r"\d{4}$|\d{2}$", t):
            sc += 1
        if re.search(r"^[A-Z]{3,9}\s+\d{1,2}\s+\d{2}$", t):
            sc -= 1
        return sc
    hits.sort(key=score, reverse=True)
    return soft_date_norm(hits[0])

# ----------------------
# TOTAL
# ----------------------
def soft_total_norm(s: str) -> str:
    """
    Normalize and extract the most likely total amount from text.
    Example: "Total RM 1,234.00" → "1234.00"
    """
    s = norm_spaces(s or "")
    s = re.sub(r"(RM|MYR|\$|USD)\s*", "", s, flags=re.I)
    s = re.sub(r"(\d)\s+(\d)", r"\1\2", s)
    s = re.sub(r"[^\d\.,]", "", s)
    s = re.sub(r"[,\s]", "", s)
    nums = re.findall(r"\d+\.\d{2}", s)
    if not nums:
        return ""
    def _safe_float(x): 
        try: return float(x)
        except: return 0.0
    try:
        best = max(nums, key=_safe_float)
        return f"{_safe_float(best):.2f}"
    except ValueError:
        return ""


def pick_total_from_lines(lines):
    """
    Pick the best total line from OCR text lines, preferring 'TOTAL' or bottom-most values.
    """
    best = ""
    best_val = 0.0
    for i, ln in enumerate(lines):
        if not ln.strip(): continue
        ln = re.sub(r"RM\s*", "", ln, flags=re.I)
        nums = re.findall(r"\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})", ln)
        if not nums: continue
        try:
            val = max(float(re.sub(r"[,\s]", "", n)) for n in nums)
        except ValueError:
            continue
        if re.search(r"\b(total|grand\s*total|amount\s*due|cash|balance\s*due|amt|sales)\b", ln, flags=re.I):
            val += 0.001
        val += i * 1e-5  # positional bias
        if val > best_val:
            best_val, best = val, max(nums, key=lambda x: float(re.sub(r"[,\s]", "", x)))
    return best or ""

# ----------------------
# COMPANY
# ----------------------
def clean_company(s: str) -> str:
    """
    Clean company name from OCR output by removing legal suffixes, noise, and redundant data.
    """
    s = norm_spaces(s)
    s = re.sub(r"^\d{5,}\s*", "", s)
    s = re.sub(r"\b(SDN\s*BHD|SDN|BHD|LTD|LIMITED|ENTERPRISES?|PLT|CO\.?|SB|S\/B)\b.*", "", s, flags=re.I)
    s = re.sub(r"\b(COPY|DUPLICATE|ORIGINAL)\b.*$", "", s, flags=re.I)
    s = re.split(r"\b(TEL|FAX|GST|TAX|INVOICE|RECEIPT|DATE|TIME|NO\.)\b", s)[0]
    s = re.sub(r"[^A-Z0-9\s\.\-&]", "", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

# ----------------------
# ADDRESS
# ----------------------
def soft_addr_norm(s: str) -> str:
    """Normalize address spacing and remove country suffixes."""
    s = norm_spaces(s.upper())
    s = re.sub(r"[.,;:]+", " ", s)
    s = re.sub(r"\b(MALAYSIA|MY)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()

# ---- Compatibility aliases ----
def norm_date(s: str) -> str: return soft_date_norm(s)
def norm_total(s: str) -> str: return soft_total_norm(s)
