import re
import string

def clean_text(text: str) -> str:
    """Limpeza básica de texto para normalização."""
    text = text.replace("\n", " ").replace("\r", "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str) -> list[str]:
    """Tokenização simples para o BM25."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()