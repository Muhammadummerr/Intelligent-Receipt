# 🧾 Intelligent Receipt — End-to-End AI for Receipt Understanding  

**Intelligent Receipt** is a unified AI system that automates receipt understanding using:  
- 🧠 **LayoutLMv3** for document field extraction  
- 🖼️ **ViT** for watermark/tampering detection  
- 🤖 **LLM reasoning (Groq / OpenAI / Mistral)** for validation and correction  

It can automatically detect fake receipts, extract structured information, and produce clean, human-readable JSON outputs.  

---

## 🔍 Key Features  

- 🧩 **Hybrid Pipeline:** Combines computer vision, document AI, and reasoning in one workflow  
- 🧾 **Field Extraction:** Extracts `company`, `date`, `address`, and `total` using LayoutLMv3  
- 🕵️ **Watermark Detection:** ViT-based classifier detects tampered or fake receipts  
- 🧠 **LLM Reasoning:** Uses Groq / GPT / Mistral to refine and validate extracted fields  
- ☁️ **Hugging Face Integration:** Upload and reuse models seamlessly  
- ⚙️ **Modular Architecture:** Independent training, inference, and reasoning components  

---

## 🧩 System Overview  

```text
Receipt Image
   │
   ├──► ViT Watermark Detector
   │        └── Rejects tampered/fake receipts
   │
   ├──► LayoutLMv3 Extractor
   │        └── Extracts text regions and entity tags
   │
   ├──► LLM Reasoning Agent
   │        └── Validates and corrects structured fields
   │
   └──► Final JSON Output

---

## 📦 Datasets Used
Receipt-dataset → (Provided by the Unikrew Solution)
link: https://unikrew-my.sharepoint.com/:u:/p/uzair_mughal/Eco5MROMYQlHm2mGgAD2PCABeFhdtuEvBBb2g-lGXtTPew?e=LGImL7

Watermark-dataset → Custom-created by blurring or overlaying different regions of real receipts.
link: https://drive.google.com/drive/folders/1JVLB1lK9vLtUr_VkuxvYq7zauB1MB1RT?usp=drive_link
Note:
For real-world use, watermark types can be extended to include multiple tampering and forgery styles and increasing the dataset size.

---

## 🧰 Tech Stack
| Category      | Tools / Frameworks                     |
| ------------- | -------------------------------------- |
| Document AI   | LayoutLMv3 (Hugging Face Transformers) |
| Vision        | ViT Base (Image Classification)        |
| LLM Reasoning | Groq / OpenAI GPT / Mistral            |
| OCR           | EasyOCR                                |
| Hosting       | Hugging Face Hub                       |
| Training      | PyTorch + Accelerate                   |
| Environment   | Kaggle / Local GPU                     |

---


## ⚙️ Setup and Installation
Clone the repository and install dependencies:

```
    git clone https://github.com/muhammadummerr/Intelligent-Receipt.git
    cd Intelligent-Receipt
    pip install -r requirements.txt
    export PYTHONPATH=./src
```

If using Kaggle, simply add:

```
    %env PYTHONPATH=./src
``` 

## 🧠 Training the Models
# 🧾 LayoutLMv3 — Field Extraction

```
    python -m src.receipt_ie.models.training.train_layoutlmv3

```
After training, the model will be saved at:

/kaggle/temp/outputs_layoutlmv3/final_model/

## 🕵️ ViT — Watermark Detection
# 1️⃣ Augment Dataset

```
    python -m src.receipt_ie.models.watermark.augment_watermark_dataset \
    --input_dir /kaggle/input/wm-dataset/watermark-dataset \
    --output_dir /kaggle/working/receipt-watermark-augmented \
    --split_ratio 0.8 --aug_per_image 3
```

# 2️⃣ Train ViT Classifier

```
    python -m src.receipt_ie.models.watermark.train_vit_watermark_classifier \
    --data_root /kaggle/working/receipt-watermark-augmented \
    --out_dir /kaggle/working/wm_vit_output \
    --epochs 10 --batch 32 --lr 5e-5
```

# ☁️ Upload Models to Hugging Face

```
    from huggingface_hub import HfApi, upload_folder
    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HUGGINGFACE_TOKEN")

    api = HfApi()
    repo_id = "muhammadummerrr/layoutlmv3-receipt-epochs-20"
    api.create_repo(repo_id=repo_id, private=False, token=hf_token, exist_ok=True)

    upload_folder(
        folder_path="/kaggle/temp/outputs_layoutlmv3/final_model",
        repo_id=repo_id,
        token=hf_token,
)
```

✅ Uploaded models:

LayoutLMv3 Receipt Model

ViT Watermark Detector

## Running the Full Inference Pipeline

Once both models are trained and uploaded, run:

```
    python -m src.receipt_ie.pipelines.run_pipeline \
    --image_path /kaggle/input/receipt-dataset/test/img/X51005587261.jpg \
    --box_dir /kaggle/input/receipt-dataset/test/box \
    --out_path /kaggle/working/X51005230605_result.json \
    --provider groq \
    --model llama-3.3-70b-versatile
```

Example output:

{
  "company": "CHAY SEAFOOD RESTAURANT",
  "date": "15/01/2019",
  "address": "JALAN HARMONI 3/2, TAMAN DESA HARMONI, 81100 JOHOR BAHRU",
  "total": "45.80",
  "agent_comment": "Validated and normalized fields using OCR and LLM reasoning."
}

🧩 Project Structure
   
+---configs
+---data
ª   +---processed
ª   +---raw
ª   +---samples
+---models
ª   +---pretrained
+---notebooks
ª       intelligent-receipt-pipeline.ipynb    ← main training & inference notebook
ª       
+---src
ª   +---receipt_ie
ª       +---data
ª       ª       boxes.py
ª       ª       dataset_infer.py
ª       ª       text.py
ª       ª       __init__.py
ª       ª       
ª       +---infer
ª       +---models
ª       ª   +---training
ª       ª   ª   ª   train_layoutlmv3.py
ª       ª   ª   ª   __init__.py
ª       ª   ª   ª   
ª       ª   ª   +---helpers
ª       ª   ª           augmentations.py
ª       ª   ª           data_utils.py
ª       ª   ª           entity_utils.py
ª       ª   ª           
ª       ª   +---watermark
ª       ª       ª   augment_watermark_dataset.py
ª       ª       ª   train_vit_watermark_classifier.py
ª       ª       ª   __init__.py
ª       ª       ª   
ª       ª       +---utils
ª       ª               watermark_filter.py
ª       ª               wm_dataset.py
ª       ª               wm_data_utils.py
ª       ª               __init__.py
ª       ª               
ª       +---pipelines
ª       ª       run_pipeline.py
ª       ª       __init.py
ª       ª       
ª       +---utils
ª               decode.py
ª               llm_client.py
ª               postproc.py
ª               __init__.py
ª               
+---tests


## 📈 Results:

| Field   | Extracted               | Ground Truth | Notes                           |
| ------- | ----------------------- | ------------ | ------------------------------- |
| Company | CHAY SEAFOOD RESTAURANT | ✅            | Reconstructed                  |
| Date    | 15/01/2019              | ✅            | Normalized                     |
| Address | JALAN HARMONI, JOHOR    | ✅            | Reconstructed                  |
| Total   | 45.80                   | ✅            | Verified using reasoning agent |



## ▶️ Quick Start (Notebook)

To run the full training and reasoning pipeline:

Open notebooks/intelligent_receipt_pipeline.ipynb

Execute all cells sequentially

Adjust dataset paths (for receipt-dataset and watermark-dataset) as per your environment


## 🎥 Demo  

<video src="assets/demo.mp4" width="640" controls></video>

> 🕒 Duration: 42 seconds  
> 📍 Shows: LayoutLMv3 + ViT + LLM reasoning end-to-end inference

---
## 👤 Author

Muhammad Umer Farooq
🎓 Bachelor’s in Computer Science — Namal University
📚 Research Focus: Computer Vision, Document AI, and LLM Reasoning
