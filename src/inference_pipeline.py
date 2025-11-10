# ClaimVerify Unified Inference Pipeline
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import softmax
from captum.attr import LayerIntegratedGradients
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sentence_transformers import SentenceTransformer
import faiss
import pickle

def claimverify_infers(user_claim: str):
    """
    Dummy fallback inference function for demo purposes.
    Always returns a simulated response without model execution.
    """

    # Example simulated result
    sample_explanation = [
        {"token": "COVID-19", "score": 0.9},
        {"token": "vaccines", "score": -0.5},
        {"token": "cause", "score": 0.8},
        {"token": "infertility", "score": -0.3},
    ]

    sample_evidence = [
        {
            "claim_text": "COVID-19 vaccines cause infertility.",
            "similarity": 0.78,
            "verdict": "Likely False",
            "url": "https://www.politifact.com/factchecks/2021/may/01/fact-check/",
            "dataset_source": "PolitiFact"
        },
        {
            "claim_text": "COVID-19 vaccines do not affect fertility according to studies.",
            "similarity": 0.74,
            "verdict": "Likely True",
            "url": "https://www.snopes.com/fact-check/vaccine-fertility/",
            "dataset_source": "Snopes"
        }
    ]

    return {
        "verdict": "Likely False",
        "confidence": 0.86,
        "explanation": sample_explanation,
        "evidence": sample_evidence,
        "source": "Demo Mode"
    }


# Configuration
BASE_PATH = Path("/Users/satwik/Documents/GitHub/UF-EEE6778-Fall25-TermProject")
FAISS_INDEX_PATH = BASE_PATH / "data/processed/faiss_index/claimverify_faiss_index.bin"
FAISS_METADATA_PATH = BASE_PATH / "data/processed/faiss_index/claimverify_faiss_metadata.csv"
MODEL_DIR = BASE_PATH / "models/classifier/roberta_finetuned"


# Load Retrieval Components
print("📥 Loading FAISS index and metadata...")
index = faiss.read_index(str(FAISS_INDEX_PATH))
metadata = pd.read_csv(FAISS_METADATA_PATH)
print(f" Loaded FAISS index with {index.ntotal} vectors.")

# Load SentenceTransformer (same one used for embeddings)
retrieval_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Load Classifier + Calibration
print("📥 Loading RoBERTa classifier and tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained(str(MODEL_DIR))
model = RobertaForSequenceClassification.from_pretrained(str(MODEL_DIR))

# Load label map
with open(MODEL_DIR / "label_mapping.pkl", "rb") as f:
    label_map = pickle.load(f)
id2label = {v: k for k, v in label_map.items()}

# Load temperature
temp_file = MODEL_DIR / "temperature_scaling.pt"
temperature_data = torch.load(temp_file, map_location="cpu")
temperature = temperature_data["temperature"]
print(f"✅ Model, tokenizer, and calibration loaded (T={temperature:.4f}).")


# Explainability Setup (Integrated Gradients)
embedding_layer = model.roberta.embeddings.word_embeddings
lig = LayerIntegratedGradients(lambda ids, mask: model(ids, mask).logits, embedding_layer)

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
model.to(device)
model.eval()


# Utility: Evidence Retrieval
def retrieve_evidence(user_claim: str, top_k: int = 5, threshold: float = 0.70):
    """
    Retrieve top similar claims using FAISS.
    Returns a list of evidence dictionaries.
    """
    query_vec = retrieval_model.encode([user_claim], normalize_embeddings=True)
    scores, indices = index.search(query_vec, top_k)

    evidence = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        row = metadata.iloc[idx]
        evidence.append({
            "rank": rank + 1,
            "claim_text": row["claim_text"],
            "verdict": row["verdict_mapped"],
            "url": row["url"],
            "dataset_source": row["dataset_source"],
            "similarity": float(score)
        })

    # Determine source flag
    source = "OfflineDB" if scores[0][0] >= threshold else "OnlineFallback"
    return evidence, source


# Utility: Classification + Calibration
def classify_claim(user_claim: str):
    """
    Classify the claim using fine-tuned RoBERTa and apply temperature scaling.
    Returns label, confidence, and logits.
    """
    enc = tokenizer(user_claim, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logits = logits / temperature  # Apply calibration
        probs = softmax(logits.cpu().numpy(), axis=1)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = id2label[pred_idx]
    confidence = float(np.max(probs))

    return pred_label, confidence, logits, input_ids, attention_mask


#Explainability
def generate_explanation(input_ids, attention_mask, target_idx, n_steps: int = 50):
    """
    Compute token-level attributions using Integrated Gradients.
    Works across CPU / CUDA / MPS without Captum target-type errors.
    """
    # --- 🔧 Normalize target ---
    if isinstance(target_idx, np.generic):  # numpy scalar (e.g., np.int64)
        target_idx = int(target_idx)
    elif isinstance(target_idx, np.ndarray):
        target_idx = int(target_idx.item())
    elif isinstance(target_idx, torch.Tensor):
        target_idx = int(target_idx.detach().cpu().item())
    elif not isinstance(target_idx, int):
        raise ValueError(f"Invalid target_idx type: {type(target_idx)}")

    # --- Ensure it's a plain Python int ---
    target_idx = int(target_idx)

    # Move tensors to correct device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Prepare baseline (padding)
    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id).to(device)

    # --- 🧠 Integrated Gradients ---
    attributions, _ = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        target=target_idx,       # plain Python int
        n_steps=n_steps,
        return_convergence_delta=True
    )

    # Normalize importance
    token_importance = attributions.sum(dim=-1).squeeze(0)
    token_importance = (
        token_importance / (torch.max(torch.abs(token_importance)) + 1e-10)
    ).detach().cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    special_tokens = {tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token}

    explanation = [
        {"token": tok, "score": float(score)}
        for tok, score in zip(tokens, token_importance)
        if tok not in special_tokens
    ]
    return explanation





#  Unified Inference Function
def claimverify_infer(user_claim: str, top_k: int = 5):
    """
    Complete pipeline:
    1. Retrieve similar evidence from FAISS
    2. Classify claim
    3. Apply temperature scaling
    4. Generate token-level explanations
    5. Return structured JSON-ready dict
    """
    # Step 1: Evidence retrieval
    evidence, source = retrieve_evidence(user_claim, top_k=top_k)

    # Step 2: Classification
    verdict, confidence, logits, input_ids, attention_mask = classify_claim(user_claim)
    target_idx = [k for k, v in id2label.items() if v == verdict][0]

    # Step 3: Explainability
    explanation = generate_explanation(input_ids, attention_mask, target_idx)

    # Final structured output
    response = {
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "evidence": evidence,
        "source": source,
        "explanation": explanation
    }

    return response