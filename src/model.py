import torch
import pickle
import numpy as np
from scipy.special import softmax
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

class ClaimClassifier:
    """
    Calibrated RoBERTa-based claim classifier.
    """

    def __init__(self, model_dir: str, temperature_path: str, label_map_path: str):
        self.device = torch.device("cpu")

        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)
        self.model = RobertaForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        with open(label_map_path, "rb") as f:
            self.label_map = pickle.load(f)

        self.id2label = {v: k for k, v in self.label_map.items()}
        self.temperature = float(torch.load(temperature_path)["temperature"])

    def predict(self, text: str):
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits[0] / self.temperature
            probs = softmax(logits.cpu().numpy())

        pred_idx = int(np.argmax(probs))
        confidence = float(np.max(probs))
        label = self.id2label[pred_idx]

        return label, confidence
