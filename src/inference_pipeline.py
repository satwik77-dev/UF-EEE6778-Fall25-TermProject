import time
import requests

class ClaimVerifyPipeline:
    """
    End-to-end hybrid inference pipeline.
    """

    def __init__(
        self,
        retriever,
        classifier,
        explainer,
        similarity_threshold: float = 0.82,
        uncertainty_threshold: float = 0.55,
        google_api_key: str = None,
        google_cx: str = None
    ):
        self.retriever = retriever
        self.classifier = classifier
        self.explainer = explainer
        self.sim_thresh = similarity_threshold
        self.uncert_thresh = uncertainty_threshold
        self.google_api_key = google_api_key
        self.google_cx = google_cx

    def google_fallback(self, query: str, top_k: int = 5):
        if not self.google_api_key or not self.google_cx:
            return []

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.google_api_key,
            "cx": self.google_cx,
            "q": query,
            "num": top_k
        }

        resp = requests.get(url, params=params, timeout=6)
        data = resp.json()

        results = []
        for rank, item in enumerate(data.get("items", []), start=1):
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

    def run(self, claim: str):
        overall_start = time.time()

        evidence, retr_t = self.retriever.retrieve(claim)
        top_sim = evidence[0]["similarity"]

        if top_sim < self.sim_thresh:
            google_results = self.google_fallback(claim)
            if google_results:
                evidence = google_results
                source = "Hybrid Fallback (Google API)"
            else:
                source = "Offline Database (fallback)"
        else:
            source = "Offline Database"

        label, confidence = self.classifier.predict(claim)
        if confidence < self.uncert_thresh:
            label = "Uncertain"

        target_idx = self.classifier.label_map[label]
        explanation, exp_t = self.explainer.explain(claim, target_idx)

        total_t = (time.time() - overall_start) * 1000.0

        return {
            "verdict": label,
            "confidence": confidence,
            "evidence": evidence,
            "explanation": explanation,
            "source": source,
            "runtime_ms": {
                "retrieval": retr_t,
                "explainability": exp_t,
                "total": total_t
            }
        }
