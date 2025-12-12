import time
import faiss
import pandas as pd
from typing import List, Tuple
from src.preprocessing import normalize_text

class FAISSRetriever:
    """
    Offline semantic retrieval using MiniLM embeddings + FAISS.
    """

    def __init__(self, index_path: str, metadata_path: str, embedding_model):
        self.index = faiss.read_index(index_path)
        self.metadata = pd.read_csv(metadata_path)
        self.embedding_model = embedding_model

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> Tuple[List[dict], float]:
        start = time.time()

        query_norm = normalize_text(query)
        embedding = self.embedding_model.encode(
            [query_norm],
            normalize_embeddings=True
        )

        scores, idxs = self.index.search(embedding, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(idxs[0], scores[0]), start=1):
            row = self.metadata.iloc[idx].to_dict()
            row["rank"] = rank
            row["similarity"] = float(score)
            results.append(row)

        latency = (time.time() - start) * 1000.0
        return results, latency
