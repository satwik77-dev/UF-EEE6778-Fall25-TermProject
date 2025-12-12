# =======================================================
# ClaimVerify — Final Streamlit App
# =======================================================
# Includes:
# - Preprocessing
# - FAISS Retrieval
# - MiniLM Embeddings (with safe monkey patch)
# - Google Custom Search API Fallback
# - RoBERTa Classification + Temperature Scaling
# - Integrated Gradients Explainability
# - Streamlit UI
# =======================================================

import os
import time
import requests
import numpy as np
import pandas as pd
import torch
import streamlit as st

from pathlib import Path
from scipy.special import softmax

import faiss
from sentence_transformers import SentenceTransformer as ST_CLASS
from sentence_transformers import models
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from captum.attr import LayerIntegratedGradients

import re, string
import pickle


# =====================================================
# PATH SETUP  (FILE IS IN: project_root/ui/)
# =====================================================
BASE_PATH = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_PATH / "data" / "processed"
FAISS_DIR = DATA_DIR / "faiss_index"

FAISS_INDEX_FILE = FAISS_DIR / "claimverify_faiss_index.bin"
FAISS_META_FILE  = FAISS_DIR / "claimverify_faiss_metadata.csv"

MODEL_DIR = BASE_PATH / "models" / "classifier" / "roberta_finetuned_v2"
TEMP_FILE = MODEL_DIR / "temperature_scaling.pt"
LABEL_MAP_FILE = MODEL_DIR / "label_mapping.pkl"

SIM_THRESHOLD = 0.82
UNCERT_THRESHOLD = 0.55

# =====================================================
# DEVICE CONFIG
# =====================================================
device = torch.device("cpu")
print("Device in use:", device)


# =====================================================
# MONKEY-PATCH TO FIX META-TENSOR ERRORS (CRITICAL)
# =====================================================
original_to = ST_CLASS.to  # save original .to()

def safe_to(self, device=None, *args, **kwargs):
    """
    SentenceTransformer tries to call .to(device) while still partially meta-loaded.
    PyTorch 2.5 → triggers: Cannot copy out of meta tensor.
    This patch NO-OPs .to() only during MiniLM construction.
    """
    return self  # skip all device movement

# Apply patch
ST_CLASS.to = safe_to


# =====================================================
# SAFE CPU LOADING FOR MiniLM
# =====================================================
def load_cpu_sentence_transformer(model_name: str):
    word_model = models.Transformer(model_name, model_args={"dtype": torch.float32})
    pooling = models.Pooling(word_model.get_word_embedding_dimension())
    model = ST_CLASS(modules=[word_model, pooling])
    return model  # stays on CPU


retrieval_model = load_cpu_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")

# Restore original .to() for safety (so Roberta works normally)
ST_CLASS.to = original_to


# =====================================================
# LOAD METADATA + FAISS
# =====================================================
metadata = pd.read_csv(FAISS_META_FILE)
index = faiss.read_index(str(FAISS_INDEX_FILE))


# =====================================================
# LOAD CLASSIFIER MODEL (ROBERTA)
# =====================================================
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)
model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

with open(LABEL_MAP_FILE, "rb") as f:
    label_map = pickle.load(f)

id2label = {v: k for k, v in label_map.items()}
temp = float(torch.load(TEMP_FILE)["temperature"])


# =====================================================
# PREPROCESSING
# =====================================================
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().replace("’", "'")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


# =====================================================
# OFFLINE RETRIEVAL
# =====================================================
def retrieve_claims(query, top_k=5):
    start = time.time()
    q_norm = normalize_text(query)

    emb = retrieval_model.encode([q_norm], normalize_embeddings=True)
    scores, idxs = index.search(emb, top_k)

    rows = []
    for rank, (i, sc) in enumerate(zip(idxs[0], scores[0]), start=1):
        row = metadata.iloc[i].copy()
        row["rank"] = rank
        row["similarity"] = float(sc)
        rows.append(row)

    return rows, (time.time() - start) * 1000.0


# =====================================================
# GOOGLE FALLBACK API
# =====================================================
GOOGLE_API_KEY = "AIzaSyAirGTjAEdWLNN_1uAqRZO0fhMfoOr9RvU"
GOOGLE_CX      = "311c6e574d25d4050"

def google_search_api(query, top_k=5):
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CX, "q": query, "num": top_k}

    try:
        resp = requests.get(base_url, params=params, timeout=6)
        data = resp.json()
        if "items" not in data:
            return []

        results = []
        for rank, item in enumerate(data["items"], start=1):
            results.append({
                "rank": rank,
                "claim_text": item.get("title", ""),
                "summary": item.get("snippet", ""),
                "url": item.get("link", ""),
                "similarity": 0.0,
                "verdict_mapped": "External Evidence",
                "dataset_source": "GoogleAPI"
            })
        return results

    except Exception as e:
        print("Google API ERROR:", e)
        return []


# =====================================================
# CLASSIFICATION
# =====================================================
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
        logits = model(**enc).logits[0] / temp
        probs = softmax(logits.cpu().numpy())

    pred_idx = int(np.argmax(probs))
    conf = float(np.max(probs))
    return id2label[pred_idx], conf, (time.time() - start) * 1000.0


# =====================================================
# EXPLAINABILITY (IG)
# =====================================================
embedding_layer = model.roberta.embeddings.word_embeddings
lig = LayerIntegratedGradients(
    lambda ids, mask: model(ids, mask).logits,
    embedding_layer
)

def explain_with_ig(text, target_idx):
    start = time.time()
    enc = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=128, padding="max_length"
    ).to(device)

    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    baseline = torch.zeros_like(input_ids).to(device)
    target_idx = int(target_idx)

    result = lig.attribute(
        input_ids, baselines=baseline,
        additional_forward_args=(attn,),
        target=target_idx, n_steps=50,
        return_convergence_delta=True
    )

    atts = result[0] if isinstance(result, tuple) else result
    token_atts = atts.sum(dim=-1).squeeze().detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().cpu())

    max_abs = np.max(np.abs(token_atts)) + 1e-9
    token_atts = token_atts / max_abs

    return [
        {"token": t, "score": float(s)}
        for t, s in zip(tokens, token_atts)
        if t not in {tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token}
    ], (time.time() - start) * 1000.0


# =====================================================
# MAIN INFERENCE PIPELINE
# =====================================================
def claimverify_infer(user_claim):
    overall_start = time.time()

    offline_evidence, retr_t = retrieve_claims(user_claim)
    top_sim = offline_evidence[0]["similarity"]

    if top_sim < SIM_THRESHOLD:
        google_results = google_search_api(user_claim)
        if google_results:
            evidence = google_results
            source = "Hybrid Fallback (Google API)"
        else:
            evidence = offline_evidence
            source = "OfflineDB (fallback)"
    else:
        evidence = offline_evidence
        source = "OfflineDB"

    pred, conf, cls_t = classify_claim(user_claim)
    if conf < UNCERT_THRESHOLD:
        pred = "Uncertain"

    target_idx = list(label_map.values())[list(label_map.keys()).index(pred)]
    explanation, exp_t = explain_with_ig(user_claim, target_idx)

    total_t = (time.time() - overall_start) * 1000.0

    return {
        "verdict": pred,
        "confidence": conf,
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


# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(page_title="ClaimVerify", layout="wide", page_icon="🔍")

st.title("🔍 ClaimVerify")
st.write(
    "AI-powered hybrid fact-verification using semantic retrieval, "
    "calibrated RoBERTa classification, token-level explanations, "
    "and Google Search fallback."
)

st.subheader("Enter a claim:")
user_claim = st.text_area("", height=120)

if st.button("Verify Claim") and user_claim.strip():

    with st.spinner("Analyzing…"):
        result = claimverify_infer(user_claim)

    verdict = result["verdict"]
    conf = result["confidence"]

    color_map = {
        "Likely True": "#2a9d8f",
        "Likely False": "#e63946",
        "Uncertain": "#e9c46a"
    }
    vcolor = color_map[verdict]

    st.markdown(
        f"<div style='background:{vcolor};padding:15px;border-radius:8px;"
        f"color:white;font-size:20px;font-weight:600;'>"
        f"{verdict} — Confidence: {conf:.2f}</div>",
        unsafe_allow_html=True
    )

    # EVIDENCE
    st.subheader("📚 Evidence")
    for ev in result["evidence"]:
        sim = ev.get("similarity", 0.0)
        summary = ev.get("summary", "")
        st.markdown(
            f"""
            <div style="background:#1e1e1e;color:#e6e6e6;
                        padding:15px;border-radius:10px;
                        border:1px solid #333;margin-bottom:10px;">
                <b>Rank {ev['rank']}</b><br>
                <i>Claim:</i> {ev['claim_text']}<br>
                <i>Source:</i> {ev['dataset_source']} — {ev['verdict_mapped']}<br>
                <i>Similarity:</i> {sim:.2f}<br>
                <i>Summary:</i> {summary}<br><br>
                <a href="{ev['url']}" target="_blank"
                   style="color:#4da6ff;font-weight:bold;">View Source</a>
            </div>
            """,
            unsafe_allow_html=True
        )

    # EXPLANATION
    st.subheader("🔍 Token Importance")
    html = "<div style='line-height:2;'>"
    for tok in result["explanation"]:
        score = tok["score"]
        intensity = min(max(abs(score), 0.15), 0.9)
        rgba = (
            f"rgba(30,144,255,{intensity})" if score >= 0
            else f"rgba(255,69,58,{intensity})"
        )
        html += f"<span style='background:{rgba};padding:4px 6px;border-radius:4px;margin:2px;'>{tok['token']}</span>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    st.subheader("⏱ Runtime (ms)")
    st.json(result["runtime_ms"])

    st.info(f"Retrieval Source: {result['source']}")
