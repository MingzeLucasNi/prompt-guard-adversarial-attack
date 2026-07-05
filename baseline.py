"""
Baseline evaluation: how many known prompt-injection texts does Prompt Guard
correctly flag as INJECTION/JAILBREAK (i.e. NOT benign) *before* any attack.

Also dumps the correctly-detected texts to detected_injections.json, which
run_attack.py samples from as the attack seed pool (we only try to evade
prompts the model already catches -- attacking prompts it already misses
would be meaningless).
"""
import os

# Must be set before `transformers` is imported. With tensorflow pip-installed
# in this venv (needed for TextFooler/CLARE's USE constraint), transformers'
# TF-availability probe during from_pretrained() hangs for 2+ minutes on this
# machine. We only need the PyTorch backend here, so hard-disable the TF path.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_CACHE_DIR = os.path.join(PROJECT_DIR, ".cache")
os.environ.setdefault("HF_HOME", os.path.join(LOCAL_CACHE_DIR, "huggingface"))
os.environ.setdefault(
    "HF_DATASETS_CACHE", os.path.join(LOCAL_CACHE_DIR, "huggingface", "datasets")
)
os.environ.setdefault(
    "SENTENCE_TRANSFORMERS_HOME",
    os.path.join(LOCAL_CACHE_DIR, "sentence-transformers"),
)
os.environ.setdefault("NLTK_DATA", os.path.join(LOCAL_CACHE_DIR, "nltk"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(LOCAL_CACHE_DIR, "matplotlib"))

for cache_dir in (
    os.environ["HF_HOME"],
    os.environ["HF_DATASETS_CACHE"],
    os.environ["SENTENCE_TRANSFORMERS_HOME"],
    os.environ["NLTK_DATA"],
    os.environ["MPLCONFIGDIR"],
):
    os.makedirs(cache_dir, exist_ok=True)

import collections
import json

import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "Niansuh/Prompt-Guard-86M"  # ungated mirror of meta-llama/Prompt-Guard-86M (same weights, verified by matching md5)
OUT_FILE = "detected_injections.json"


def predict(wrapper_model, tok, texts, device, batch_size=16):
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(
                batch, return_tensors="pt", truncation=True, padding=True, max_length=512
            ).to(device)
            logits = wrapper_model(**enc).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
    return preds


def main():
    # Real CUDA GPUs (e.g. a server) are used when available. MPS (Apple GPU)
    # is deliberately skipped even when present: it recompiles its Metal graph
    # on every new (variable) sequence length, which made this take 10+
    # minutes for 263 short texts on a Mac -- CPU was actually faster there.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()
    print("id2label:", model.config.id2label)

    ds = load_dataset("deepset/prompt-injections")
    full = concatenate_datasets([ds["train"], ds["test"]])
    injections = [ex["text"] for ex in full if ex["label"] == 1]
    print(f"Total ground-truth injection texts in dataset: {len(injections)}")

    preds = predict(model, tok, injections, device)
    id2label = model.config.id2label
    labels = [id2label[p] for p in preds]

    print("Prediction distribution over ground-truth injection texts:", collections.Counter(labels))

    detected = [t for t, l in zip(injections, labels) if l != "BENIGN"]
    rate = len(detected) / len(injections)
    print(f"\nBASELINE detection rate (before any attack): {len(detected)}/{len(injections)} = {rate:.2%}")

    with open(OUT_FILE, "w") as f:
        json.dump(detected, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(detected)} correctly-detected injection texts to {OUT_FILE}")


if __name__ == "__main__":
    main()
