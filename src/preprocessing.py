import re
import string

def normalize_text(text: str) -> str:
    """
    Normalize input text for both retrieval and classification.
    Ensures consistency across the pipeline.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower().replace("’", "'")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text
