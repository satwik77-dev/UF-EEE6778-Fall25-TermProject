import os
import time
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.special import softmax

import faiss
from sentence_transformers import SentenceTransformer

from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from captum.attr import LayerIntegratedGradients

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = BASE / "data" / "processed"
MODEL_DIR = BASE / "models" / "classifier" / "roberta_finetuned_v2"
FAISS_DIR = DATA_DIR / "faiss_index"

FAISS_INDEX_FILE = FAISS_DIR / "claimverify_faiss_index.bin"
FAISS_META_FILE = FAISS_DIR / "claimverify_faiss_metadata.csv"

TEMP_FILE = MODEL_DIR / "temperature_scaling.pt"
LABEL_MAP_FILE = MODEL_DIR / "label_mapping.pkl"

# ---------------------------------------------------------------------
# DEVICE SETUP
# ---------------------------------------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print("Device:", device)

# ---------------------------------------------------------------------
# LOAD METADATA + FAISS
# ---------------------------------------------------------------------
print("📂 Loading metadata...")
metadata = pd.read_csv(FAISS_META_FILE)

print("📂 Loading FAISS index...")
index = faiss.read_index(str(FAISS_INDEX_FILE))
print(f"   Loaded FAISS with {index.ntotal} vectors.")

# ---------------------------------------------------------------------
# LOAD SENTENCE TRANSFORMER FOR RETRIEVAL
# ---------------------------------------------------------------------
print("⚙️ Loading SentenceTransformer model...")
retrieval_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("   Retrieval model loaded.")

# ---------------------------------------------------------------------
# LOAD CLASSIFIER + TOKENIZER
# ---------------------------------------------------------------------
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)
model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

# ---------------------------------------------------------------------
# LABEL MAP
# ---------------------------------------------------------------------
import pickle
with open(LABEL_MAP_FILE, "rb") as f:
    label_map = pickle.load(f)

id2label = {v: k for k, v in label_map.items()}

# ---------------------------------------------------------------------
# LOAD TEMPERATURE PARAMETER
# ---------------------------------------------------------------------
temp = torch.load(TEMP_FILE)["temperature"]
print("Temperature loaded:", temp)

# ---------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------
import re
import string

def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.replace("’", "'")

    text = re.sub(r"\s+", " ", text).strip()
    text = text.translate(str.maketrans("", "", string.punctuation))

    return text

# ---------------------------------------------------------------------
# RETRIEVAL
# ---------------------------------------------------------------------
def retrieve_claims(query, top_k=5):
    start = time.time()

    q_norm = normalize_text(query)
    emb = retrieval_model.encode([q_norm], normalize_embeddings=True)

    scores, idxs = index.search(emb, top_k)
    sim = scores[0]
    ids = idxs[0]

    rows = []
    for rank, (i, s) in enumerate(zip(ids, sim), start=1):
        row = metadata.iloc[i].copy()
        row["rank"] = rank
        row["similarity"] = float(s)
        rows.append(row)

    return rows, (time.time() - start) * 1000

# ---------------------------------------------------------------------
# CLASSIFICATION (CALIBRATED)
# ---------------------------------------------------------------------
def classify_claim(text):
    start = time.time()

    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits[0]
        logits = logits / temp
        probs = softmax(logits.cpu().numpy())

    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))

    return id2label[pred_idx], confidence, (time.time() - start) * 1000

# ---------------------------------------------------------------------
# GOOGLE FALLBACK (STUB)
# ---------------------------------------------------------------------
def google_fallback_stub(query):
    """
    This is a mock version used only for Deliverable 3.
    """
    return [
        {
            "rank": 1,
            "claim_id": "GOOG001",
            "claim_text": "Google Search mock evidence for: " + query,
            "similarity": 0.42,
            "verdict_mapped": "Uncertain",
            "summary": "Mock Google summary",
            "url": "https://google.com/search?q=" + query.replace(" ", "+"),
            "dataset_source": "GoogleAPI"
        }
    ]

# ---------------------------------------------------------------------
# EXPLAINABILITY (INTEGRATED GRADIENTS)
# ---------------------------------------------------------------------
embedding_layer = model.roberta.embeddings.word_embeddings
lig = LayerIntegratedGradients(lambda ids, mask: model(input_ids=ids, attention_mask=mask).logits,
                               embedding_layer)

def explain_with_ig(text, target_idx):
    start = time.time()

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length"
    ).to(device)

    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]

    baseline = torch.zeros_like(input_ids).to(device)

    # target_idx MUST be int not numpy.int64
    target_idx = int(target_idx)

    atts, _ = lig.attribute(
        input_ids,
        baselines=baseline,
        additional_forward_args=(attn,),
        target=target_idx,
        n_steps=50,
        return_convergence_delta=True
    )

    token_atts = atts.sum(dim=-1).squeeze().detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().cpu())

    # normalize
    max_abs = np.max(np.abs(token_atts)) + 1e-9
    token_atts = token_atts / max_abs

    result = [
        {"token": t, "score": float(s)}
        for t, s in zip(tokens, token_atts)
        if t not in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]
    ]

    return result, (time.time() - start) * 1000

# ---------------------------------------------------------------------
# MAIN INFERENCE PIPELINE
# ---------------------------------------------------------------------
def claimverify_infer(user_claim):

    overall_start = time.time()

    # --------------------------
    # 1) Preprocess & Retrieve
    # --------------------------
    retrieval_results, retr_t = retrieve_claims(user_claim)

    # --------------------------
    # Similarity Threshold Check
    # --------------------------
    top_sim = retrieval_results[0]["similarity"]

    if top_sim < 0.70:
        evidence = google_fallback_stub(user_claim)
        source = "Hybrid Fallback (Google Stub)"
    else:
        evidence = retrieval_results
        source = "OfflineDB"

    # --------------------------
    # 2) Classification
    # --------------------------
    pred_label, confidence, cls_t = classify_claim(user_claim)

    # Uncertainty threshold
    if confidence < 0.55:
        pred_label = "Uncertain"

    # --------------------------
    # 3) Explainability
    # --------------------------
    target_idx = list(label_map.values())[
        list(label_map.keys()).index(pred_label)
    ]

    explanation, exp_t = explain_with_ig(user_claim, int(target_idx))

    total_t = (time.time() - overall_start) * 1000

    return {
        "verdict": pred_label,
        "confidence": confidence,
        "evidence": evidence,
        "explanation": explanation,
        "source": source,
        "runtime_ms": {
            "retrieval": retr_t,
            "classification": cls_t,
            "explainability": exp_t,
            "total": total_t
        }
    }


# END OF FILE
