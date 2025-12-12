import torch
from sentence_transformers import SentenceTransformer as ST_CLASS
from sentence_transformers import models

def load_minilm_cpu(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Safely load MiniLM on CPU to avoid meta-tensor issues in PyTorch 2.x.
    """
    word_model = models.Transformer(
        model_name,
        model_args={"dtype": torch.float32}
    )
    pooling = models.Pooling(word_model.get_word_embedding_dimension())
    model = ST_CLASS(modules=[word_model, pooling])
    return model
