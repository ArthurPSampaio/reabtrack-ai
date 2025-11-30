import os
import json
import faiss
from typing import List, Dict, Any, Tuple
from app.core.models import get_models  # <-- Importação modular

# Salva os dados na raiz do projeto/data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def _paths_for(paciente_id: str) -> Tuple[str, str]:
    return (
        os.path.join(DATA_DIR, f"faiss_{paciente_id}.index"),
        os.path.join(DATA_DIR, f"faiss_{paciente_id}.meta.json")
    )

def upsert_docs(paciente_id: str, docs: List[Dict[str, Any]]) -> int:
    if not docs: return 0
    idx_path, meta_path = _paths_for(paciente_id)
    model = get_models().embedder
    
    if os.path.exists(idx_path) and os.path.exists(meta_path):
        index = faiss.read_index(idx_path)
        with open(meta_path, "r", encoding="utf-8") as f: meta = json.load(f)
    else:
        dim = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)
        meta = []

    texts = [d["text"] for d in docs]
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    
    index.add(embs)
    meta.extend(docs)

    faiss.write_index(index, idx_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    
    return len(docs)

def search_faiss(paciente_id: str, query: str, k: int = 10) -> List[Dict[str, Any]]:
    idx_path, meta_path = _paths_for(paciente_id)
    if not os.path.exists(idx_path): return []

    index = faiss.read_index(idx_path)
    with open(meta_path, "r", encoding="utf-8") as f: meta = json.load(f)
    
    model = get_models().embedder
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    
    k = min(k, index.ntotal)
    _, indices = index.search(q_emb, k)

    return [meta[i] for i in indices[0] if 0 <= i < len(meta)]

def reset_index(paciente_id: str):
    p1, p2 = _paths_for(paciente_id)
    if os.path.exists(p1): os.remove(p1)
    if os.path.exists(p2): os.remove(p2)
    return True