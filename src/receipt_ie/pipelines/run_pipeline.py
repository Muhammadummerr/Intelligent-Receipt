"""
run_pipeline.py
----------------
End-to-end intelligent receipt extraction pipeline:
- LayoutLMv3 for field detection
- EasyOCR fallback for missing OCR data
- LLM reasoning for correction and validation
- Post-processing and normalization for final JSON output
"""


import os
import re
import json
import time
import argparse
import torch
from typing import Dict, Any
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import sys
# === Imports from your repo ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.receipt_ie.data.dataset_infer import ReceiptInferenceDataset
from src.receipt_ie.utils.decode import group_bio
from src.receipt_ie.utils.postproc import (
    clean_company,
    extract_best_date,
    pick_total_from_lines,
    norm_total,
    soft_addr_norm,
)
from src.receipt_ie.utils.llm_client import LLMClient
from src.receipt_ie.models.watermark.utils.watermark_filter import check_watermark

# === Optional ===
try:
    import easyocr
    OCR_READER = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
except Exception as e:
    print("⚠️ EasyOCR not available, OCR fallback disabled.")
    OCR_READER = None

os.environ["HF_HOME"] = "/kaggle/temp/hf_home"
os.environ["HF_DATASETS_CACHE"] = "/kaggle/temp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/kaggle/temp/hf_models"

# ===================== SAFE JSON PARSER =====================
def safe_json_loads(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        candidate = re.sub(r",\s*([}\]])", r"\1", match.group(0))
        candidate = candidate.replace("\n", " ").replace("\r", "")
        try:
            return json.loads(candidate)
        except Exception:
            return {}
    return {}

# ===================== PROMPT BUILDER =====================
def build_reasoning_prompt(ocr_text: str, extracted: Dict[str, Any]) -> str:
    """
    Optimized reasoning prompt for robust, deterministic receipt correction.
    - Ensures high precision for structured extraction.
    - Strengthens watermark/tamper detection reasoning.
    - Uses hierarchical instructions + verified few-shot examples.
    """
    return f"""
You are a **Receipt Intelligence Agent**.
Your task is to review OCR text and verify or correct structured receipt data.

---

### OUTPUT REQUIREMENTS
Return **only valid JSON**, with exactly these 5 string keys:
["company", "date", "address", "total", "agent_comment"]

- The output **must be directly parseable by Python json.loads()**.
- Do **not** include Markdown, explanations, or extra text.
- Never guess or hallucinate; if unsure, leave the value as "" (empty string).

---

### OBJECTIVE
You are given two inputs:
1. **OCR_TEXT** — raw text extracted from the receipt.
2. **EXTRACTED_JSON** — initial structured fields from a model.

Your job:
1. Correct or fill in any missing fields using clues in OCR_TEXT.  
2. **Do not modify any field already correct in EXTRACTED_JSON.**  
3. If any sign of watermark, tampering, or heavy occlusion exists → **reject the receipt**.  
   (Return all fields empty with an appropriate agent_comment.)

---

### REJECTION RULES (Watermark / Tampered Receipt)
Reject if **any** of the following are detected in OCR_TEXT:
- Contains words like “sample”, “confidential”, “training”, “void”, “demo”, “watermark”, “practice”, “fake”, or “test”.
- Mentions text overlays such as “NOT FOR SALE”, “COPY”, or “FOR TRAINING USE”.
- Appears to have black bars, blurred text areas, or unrelated overlays covering printed content.

On rejection, return:
{{
  "company": "", "date": "", "address": "", "total": "",
  "agent_comment": "Receipt rejected — detected watermark, tampering, or obstructed text ('<term>')."
}}

---

### FIELD RULES
- **company** → Main merchant name (top 1–3 lines).  
  Usually includes “SDN BHD”, “ENTERPRISE”, or “LTD”.  
  Exclude generic terms like TAX INVOICE, RECEIPT, or TEL.
- **date** → Primary transaction date in format **DD/MM/YYYY**.  
  Normalize variants like 2018/04/20 → 20/04/2018 or 20-04-18 → 20/04/2018.  
  Ignore manufacturing or expiry dates; if multiple dates exist, pick the **latest valid one**.
- **address** → Physical store or branch location.  
  Typically includes “JALAN”, “TAMAN”, “ROAD”, “SELANGOR”, “KUALA LUMPUR”, or postal codes.  
  Merge multi-line addresses and exclude phone numbers or “TEL”.
- **total** → Final payable amount.  
  Usually near “TOTAL”, “CASH”, or “AMOUNT DUE”.  
  Return only the numeric value, without currency symbols.
- **agent_comment** → One concise sentence describing your corrections or reasoning.  
  Never list or mention fields that were already correct.

---

### FEW-SHOT EXAMPLES

#### Example 1 — Normal Correction
OCR_TEXT:
ONE ONE THREE SEAFOOD RESTAURANT SDN BHD  
(1120908-M)  
NO.1, TAMAN SRI DENGKIL, JALAN AIR HITAM  
43800 DENGKIL, SELANGOR.  
DATE: 30-05-2018  
TOTAL (INCLUSIVE OF GST): 87.45  
CASH: 87.45

EXTRACTED_JSON:
{{"company": "", "date": "", "address": "", "total": ""}}

OUTPUT:
{{"company": "ONE ONE THREE SEAFOOD RESTAURANT SDN BHD",
  "date": "30/05/2018",
  "address": "NO.1, TAMAN SRI DENGKIL, JALAN AIR HITAM 43800 DENGKIL, SELANGOR.",
  "total": "87.45",
  "agent_comment": "Extracted all fields directly from labeled OCR lines."}}

---

#### Example 2 — Preserve Correct Fields
OCR_TEXT:
LEMON TREE RESTAURANT  
JTJ FOODS SDN BHD (1179227A)  
NO 3, JALAN PERMAS 10/8, BANDAR BARU PERMAS JAYA,  
81750 MASAI, JOHOR  
INVOICE DATE: 6/1/2018 6:42:02 PM  
TOTAL AMOUNT: 10.30

EXTRACTED_JSON:
{{"company": "", "date": "06/01/2018", "address": "", "total": "10.30"}}

OUTPUT:
{{"company": "LEMON TREE RESTAURANT JTJ FOODS SDN BHD",
  "date": "06/01/2018",
  "address": "NO 3, JALAN PERMAS 10/8, BANDAR BARU PERMAS JAYA, 81750 MASAI, JOHOR",
  "total": "10.30",
  "agent_comment": "Added company and address from OCR context; preserved date and total."}}

---

#### Example 3 — Rejected (Watermark)
OCR_TEXT:
SAMPLE RECEIPT — FOR TRAINING PURPOSES  
tan chay yee  
No. 284, JALAN HARMONI 3/2, TAMAN DESA HARMONI  
81100 JOHOR BAHRU  
TOTAL: 31.00

EXTRACTED_JSON:
{{"company": "", "date": "", "address": "", "total": ""}}

OUTPUT:
{{"company": "", "date": "", "address": "", "total": "",
  "agent_comment": "Receipt rejected — detected watermark text ('SAMPLE / TRAINING')."}}

---

### TASK
Now, analyze the provided OCR_TEXT and EXTRACTED_JSON.
If the receipt is valid, return the corrected JSON.
If it is tampered/watermarked, return an empty JSON with a clear agent_comment reason.
""".strip()




# ===================== NORMALIZATION =====================
def normalize_refined_output(refined: Dict[str, Any]) -> Dict[str, str]:
    out = {
        "company": clean_company(refined.get("company", "")),
        "date": extract_best_date(refined.get("date", "")) or refined.get("date", ""),
        "address": soft_addr_norm(refined.get("address", "")),
        "total": norm_total(refined.get("total", "")),
    }
    comment = refined.get("agent_comment", "")
    out["agent_comment"] = comment.strip() or "Validated using OCR context."
    return out

# ===================== EASYOCR FALLBACK =====================
def run_easyocr(image_path: str) -> str:
    if not OCR_READER:
        return ""
    result = OCR_READER.readtext(image_path, detail=0, paragraph=True)
    return "\n".join(result)

# ===================== MODEL EXTRACTION =====================
def run_layoutlmv3_extraction(image_path: str, model_dir: str, box_dir: str) -> (Dict[str, str], str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = LayoutLMv3Processor.from_pretrained(model_dir, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir).to(device).eval()

    stem = os.path.splitext(os.path.basename(image_path))[0]
    box_path = os.path.join(box_dir, stem + ".txt")

    # --- OCR fallback if no .txt file exists ---
    if not os.path.exists(box_path):
        print(f"⚠️ No OCR file for {stem}, running EasyOCR fallback...")
        ocr_text = run_easyocr(image_path)
        return {"company": "", "date": "", "address": "", "total": ""}, ocr_text

    # --- Use ReceiptInferenceDataset for consistency ---
    ds = ReceiptInferenceDataset(
        img_dir=os.path.dirname(image_path),
        box_dir=box_dir,
        processor=processor,
        max_seq_len=512,
        stems_subset=[stem],
    )
    item = ds[0]
    with torch.no_grad():
        outputs = model(
            input_ids=item["input_ids"].unsqueeze(0).to(device),
            bbox=item["bbox"].unsqueeze(0).to(device),
            pixel_values=item["pixel_values"].unsqueeze(0).to(device),
            attention_mask=item["attention_mask"].unsqueeze(0).to(device),
        )
    pred_ids = outputs.logits.argmax(-1).squeeze(0).cpu().tolist()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    extracted = group_bio(item["orig_words"], pred_ids, id2label=id2label)

    # --- Postprocess ---
    if extracted.get("company"):
        extracted["company"] = clean_company(extracted["company"])
    if not extracted.get("date"):
        extracted["date"] = extract_best_date(" ".join(item["line_texts"]))
    if not extracted.get("total"):
        extracted["total"] = pick_total_from_lines(item["line_texts"])
    else:
        extracted["total"] = norm_total(extracted["total"])

    ocr_text = "\n".join(item["line_texts"])
    return extracted, ocr_text

# ===================== PIPELINE CORE =====================
def run_pipeline_single(image_path: str, model_dir: str, box_dir: str,
                        llm_provider="groq", llm_model="openai/gpt-oss-120b") -> Dict[str, Any]:
    print(f"🔍 Processing: {os.path.basename(image_path)}")

    print("🛡️ Checking for watermarks or tampering...")
    is_marked, reason = check_watermark(image_path)
    if is_marked:
        result = {
            "company": "",
            "date": "",
            "address": "",
            "total": "",
            "agent_comment": f"Receipt rejected automatically → {reason}"
        }
        print(f"🚫 Rejected {os.path.basename(image_path)} — {reason}")
        return result

    # Step 1: LayoutLMv3 extraction + OCR fallback
    try:
        extracted, ocr_text = run_layoutlmv3_extraction(image_path, model_dir, box_dir)
    except Exception as e:
        print(f"⚠️ LayoutLMv3 extraction failed: {e}")
        extracted, ocr_text = {"company": "", "date": "", "address": "", "total": ""}, run_easyocr(image_path)

    print(f"🧾 Extracted: {extracted}")

    # Step 2: LLM reasoning correction
    llm = LLMClient(provider=llm_provider, model=llm_model, temperature=0.0)
    prompt = build_reasoning_prompt(ocr_text, extracted)

    refined = None
    for attempt in range(3):
        try:
            resp = llm.generate(prompt)
            refined = safe_json_loads(resp)
            if isinstance(refined, dict):
                break
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")
            time.sleep(2)

    if not refined:
        refined = {**extracted, "agent_comment": "LLM fallback — invalid output."}

    final_output = normalize_refined_output(refined)
    print("✅ Final corrected output ready.")
    return final_output

# ===================== CLI =====================
def main():
    parser = argparse.ArgumentParser(description="End-to-end intelligent receipt pipeline.")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--box_dir", required=True)
    parser.add_argument("--model_dir", default="muhammadummerrr/layoutlmv3-receipt-epochs-20")
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    args = parser.parse_args()

    result = run_pipeline_single(args.image_path, args.model_dir, args.box_dir, args.provider, args.model)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Saved final result to {args.out_path}")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
