import time
import numpy as np
import torch
from captum.attr import LayerIntegratedGradients

class IGExplainer:
    """
    Token-level explanation using Integrated Gradients.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_layer = model.roberta.embeddings.word_embeddings

        self.ig = LayerIntegratedGradients(
            lambda ids, mask: self.model(ids, mask).logits,
            self.embedding_layer
        )

    def explain(self, text: str, target_idx: int):
        start = time.time()

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]
        baseline = torch.zeros_like(input_ids)

        attributions, _ = self.ig.attribute(
            input_ids,
            baselines=baseline,
            additional_forward_args=(attn_mask,),
            target=target_idx,
            n_steps=50,
            return_convergence_delta=True
        )

        token_atts = attributions.sum(dim=-1).squeeze().detach().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze())

        token_atts /= (np.max(np.abs(token_atts)) + 1e-9)

        explanation = [
            {"token": t, "score": float(s)}
            for t, s in zip(tokens, token_atts)
            if t not in {
                self.tokenizer.pad_token,
                self.tokenizer.cls_token,
                self.tokenizer.sep_token
            }
        ]

        latency = (time.time() - start) * 1000.0
        return explanation, latency
