# ClaimVerify: Offline Retrieval Function
# Author: Sai Satwik Yarapothini

import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer


# Load FAISS and metadata (initialize once)

class ClaimRetrievalEngine:
    def __init__(self, base_path=None):
        # Define paths
        if base_path is None:
            base_path = Path("/Users/satwik/Documents/GitHub/UF-EEE6778-Fall25-TermProject")
        self.base_path = Path(base_path)

        self.index_path = self.base_path / "data/processed/faiss_index/claimverify_faiss_index.bin"
        self.meta_path = self.base_path / "data/processed/faiss_index/claimverify_faiss_metadata.csv"

        # Load FAISS index
        print("📂 Loading FAISS index and metadata...")
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = pd.read_csv(self.meta_path)
        print(f"✅ Loaded FAISS index with {self.index.ntotal} vectors.")
        print(f" Metadata records: {len(self.metadata)}")

        # Load embedding model
        print("⚙️ Loading SentenceTransformer model...")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Model loaded successfully.")

    # ------------------------------------------------------------
    # Core retrieval function
    # ------------------------------------------------------------
    def retrieve_similar_claims(self, query_text, top_k=5):
        """
        Given a claim (query_text), retrieve top_k similar verified claims
        from the FAISS index.
        Returns a pandas DataFrame with claim info + similarity scores.
        """

        if not query_text or not isinstance(query_text, str):
            raise ValueError("Query text must be a non-empty string.")

        # Embed and normalize
        query_vec = self.model.encode([query_text], normalize_embeddings=True)

        # Search FAISS
        scores, indices = self.index.search(query_vec, top_k)

        # Build result DataFrame
        results = []
        for rank, idx in enumerate(indices[0]):
            row = self.metadata.iloc[idx]
            results.append({
                "rank": rank + 1,
                "claim_id": row["claim_id"],
                "claim_text": row["claim_text"],
                "similarity": float(scores[0][rank]),
                "verdict_mapped": row["verdict_mapped"],
                "summary": row.get("summary", None),
                "url": row.get("url", None),
                "dataset_source": row.get("dataset_source", None)
            })

        return pd.DataFrame(results).sort_values(by="similarity", ascending=False)
