"""
llm_client.py
--------------
Unified interface for calling different Large Language Models (LLMs):
- OpenAI GPT models (gpt-3.5, gpt-4, etc.)
- Hugging Face Hub (Llama, Mistral, Falcon, etc.)
- Groq API (free, ultra-fast inference)


Usage:
    from src.receipt_ie.utils.llm_client import LLMClient
    llm = LLMClient(provider="groq", model="llama3-70b-8192")
    response = llm.generate("Summarize this receipt text...")
"""

import os
import time
import logging
from typing import Literal, Optional

# ----------------------------
# Environment & Secrets Handling (Kaggle + Local safe)
# ----------------------------
try:
    from kaggle_secrets import UserSecretsClient
except ImportError:
    UserSecretsClient = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

IN_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

if IN_KAGGLE and UserSecretsClient:
    user_secrets = UserSecretsClient()
    for key in ["OPENAI_API_KEY", "HUGGINGFACEHUB_API_TOKEN", "GROQ_API_KEY"]:
        try:
            os.environ[key] = user_secrets.get_secret(key) or os.environ.get(key, "")
        except Exception:
            # Skip silently if secret label does not exist
            os.environ[key] = os.environ.get(key, "")
elif load_dotenv:
    load_dotenv()  # Fallback for local .env

# ----------------------------
# Optional Provider Imports
# ----------------------------
try:
    import openai
except ImportError:
    openai = None

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

try:
    from groq import Groq
except ImportError:
    Groq = None

# ----------------------------
# Logging Setup
# ----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)


# ----------------------------
# Core LLM Client Class
# ----------------------------
class LLMClient:
    """Flexible multi-provider LLM wrapper for OpenAI, HuggingFace, and Groq APIs."""

    def __init__(
        self,
        provider: Literal["openai", "huggingface", "groq"] = "groq",
        model: Optional[str] = "openai/gpt-oss-120b",
        temperature: float = 0.2,
        max_tokens: int = 512,
        retry: int = 3,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry = retry

        # --- Provider Initialization ---
        if provider == "openai":
            if openai is None:
                raise ImportError("OpenAI package not installed. Run `pip install openai`.")
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.client = openai
            self.model = model or "gpt-4-turbo"

        elif provider == "huggingface":
            if InferenceClient is None:
                raise ImportError("huggingface_hub not installed. Run `pip install huggingface_hub`.")
            hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            if not hf_token:
                raise EnvironmentError("Missing HUGGINGFACEHUB_API_TOKEN.")
            self.client = InferenceClient(model or "mistralai/Mistral-7B-Instruct-v0.2", token=hf_token)

        elif provider == "groq":
            if Groq is None:
                raise ImportError("groq package not installed. Run `pip install groq`.")
            groq_key = os.getenv("GROQ_API_KEY")
            if not groq_key:
                raise EnvironmentError("Missing GROQ_API_KEY.")
            self.client = Groq(api_key=groq_key)
            self.model = model or "llama3-70b-8192"

        else:
            raise ValueError(f"Unsupported provider: {provider}")

        logger.info(f"LLM initialized — provider={self.provider}, model={self.model}")

    # ------------------------------------------------------------------ #
    def generate(self, prompt: str) -> str:
        """Generate text using the configured provider, with retry and exponential backoff."""
        for attempt in range(self.retry):
            try:
                if self.provider == "openai":
                    return self._generate_openai(prompt)
                elif self.provider == "huggingface":
                    return self._generate_hf(prompt)
                elif self.provider == "groq":
                    return self._generate_groq(prompt)
            except Exception as e:
                delay = 2 ** attempt
                logger.warning(f"⚠️ Attempt {attempt+1}/{self.retry} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
        raise RuntimeError("❌ All LLM generation attempts failed after retries.")

    # ------------------------------------------------------------------ #
    def _generate_openai(self, prompt: str) -> str:
        """Generate text using OpenAI ChatCompletion API."""
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        msg = response.choices[0].message.content.strip()
        logger.debug(f"[OpenAI] {msg[:200]}...")
        return msg

    # ------------------------------------------------------------------ #
    def _generate_hf(self, prompt: str) -> str:
        """Generate text using Hugging Face Inference API."""
        output = self.client.text_generation(
            prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=True,
        )
        msg = output[0]["generated_text"] if isinstance(output, list) else str(output).strip()
        logger.debug(f"[HuggingFace] {msg[:200]}...")
        return msg

    # ------------------------------------------------------------------ #
    def _generate_groq(self, prompt: str) -> str:
        """Generate text using Groq’s ChatCompletion API."""
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        msg = chat_completion.choices[0].message.content.strip()
        logger.debug(f"[Groq] Output length={len(msg)} chars")
        return msg

    # ------------------------------------------------------------------ #
    def __repr__(self):
        return f"<LLMClient provider={self.provider} model={self.model}>"
