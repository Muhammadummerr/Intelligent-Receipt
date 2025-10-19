"""
train_layoutlmv3.py
-------------------
Fine-tunes LayoutLMv3 for receipt key information extraction (KIE).

Features:
- Albumentations-based image + box augmentations
- Smart token-to-entity alignment
- Early stopping & best checkpoint saving
- Compatible with Kaggle and Hugging Face format

Author: Muhammad Umer
License: MIT
Usage:
    python src/receipt_ie/models/training/train_layoutlmv3.py
"""




from src.receipt_ie.models.training.helpers.data_utils import (
    load_ocr_file, load_entities_txt, normalize_box, sanitize_box
)
from src.receipt_ie.models.training.helpers.augmentations import apply_augmentation
from src.receipt_ie.models.training.helpers.entity_utils import text_match

import os, torch
import argparse
from datasets import Dataset
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from PIL import Image
from tqdm import tqdm

# ==========================
# PATHS
# ==========================
parser = argparse.ArgumentParser(description="Fine-tune LayoutLMv3 for receipt information extraction.")
parser.add_argument("--data_root", type=str, default="/kaggle/input/receipt-dataset",
                    help="Path to dataset root directory containing train/test folders.")
parser.add_argument("--output_dir", type=str, default="/kaggle/temp/outputs_layoutlmv3",
                    help="Where to save model checkpoints and logs.")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

data_root = args.data_root
output_dir = args.output_dir

# ==========================
# LABELS
# ==========================
labels = [
    "O",
    "B-COMPANY", "I-COMPANY",
    "B-DATE", "I-DATE",
    "B-ADDRESS", "I-ADDRESS",
    "B-TOTAL", "I-TOTAL"
]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

# ==========================
# PROCESSOR
# ==========================
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)



def prepare_example(img_path, ent_path, ocr_path):
    """
    Convert one receipt (image + OCR + entity labels)
    into LayoutLMv3-compatible token features with bounding boxes.
    """
    image = Image.open(img_path).convert("RGB")
    W, H = image.size
    ocr_data = load_ocr_file(ocr_path)
    entities = load_entities_txt(ent_path)

    image, ocr_data = apply_augmentation(image, ocr_data)
    W, H = image.size

    words, boxes, label_ids = [], [], []

    for text, box in ocr_data:
        box = sanitize_box(box, W, H)
        box_norm = normalize_box(box, W, H)
        label = "O"
        for key, val in entities.items():
            if text_match(text, val):
                label = f"B-{key.upper()}" if label == "O" else f"I-{key.upper()}"
        words.append(text)
        boxes.append(box_norm)
        label_ids.append(label2id.get(label, 0))

    encoding = processor(
        image,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    label_ids = label_ids[:512] + [0] * max(0, 512 - len(label_ids))
    encoding = {k: v.squeeze(0) for k, v in encoding.items()}
    encoding["labels"] = torch.tensor(label_ids, dtype=torch.long)
    return encoding

# ==========================
# DATASET BUILDING
# ==========================
samples = []
for split in ["train", "test"]:
    img_dir = os.path.join(data_root, split, "img")
    ent_dir = os.path.join(data_root, split, "entities")
    ocr_dir = os.path.join(data_root, split, "box")
    for fname in tqdm(os.listdir(img_dir), desc=f"scan-{split}"):
        stem, _ = os.path.splitext(fname)
        img_path = os.path.join(img_dir, fname)
        ent_path = os.path.join(ent_dir, stem + ".txt")
        ocr_path = os.path.join(ocr_dir, stem + ".txt")
        if not (os.path.exists(img_path) and os.path.exists(ent_path) and os.path.exists(ocr_path)):
            continue
        samples.append(prepare_example(img_path, ent_path, ocr_path))

print(f"✅ Loaded {len(samples)} samples total.")
dataset = Dataset.from_list(samples)
split_ds = dataset.train_test_split(test_size=0.2, seed=42)
train_ds, val_ds = split_ds["train"], split_ds["test"]

# ==========================
# MODEL & TRAINING
# ==========================
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    num_train_epochs=20,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    warmup_ratio=0.1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=50,
    fp16=True,
    remove_unused_columns=False,
    report_to=[],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=processor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# ==========================
# TRAIN
# ==========================
trainer.train()
trainer.save_model(os.path.join(output_dir, "final_model"))
processor.save_pretrained(os.path.join(output_dir, "final_model"))
print("✅ LayoutLMv3 fine-tuning complete (smart entity labeling enabled).")
