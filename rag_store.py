# rag_store.py
import os
import json
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss

# Vamos reutilizar o mesmo modelo de embeddings já carregado em summarizer.py
from summarizer import emb_model  # <-- único ponto de acoplamento

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def _paths_for(paciente_id: str) -> Tuple[str, str]:
    idx_path = os.path.join(DATA_DIR, f"faiss_{paciente_id}.index")
    meta_path = os.path.join(DATA_DIR, f"faiss_{paciente_id}.meta.json")
    return idx_path, meta_path

def _new_index() -> faiss.IndexFlatIP:
    dim = emb_model.get_sentence_embedding_dimension()
    return faiss.IndexFlatIP(dim)

def _load_index(paciente_id: str) -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]:
    idx_path, meta_path = _paths_for(paciente_id)
    if os.path.exists(idx_path) and os.path.exists(meta_path):
        index = faiss.read_index(idx_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return index, meta
    return _new_index(), []

def _save_index(paciente_id: str, index: faiss.IndexFlatIP, meta: List[Dict[str, Any]]):
    idx_path, meta_path = _paths_for(paciente_id)
    faiss.write_index(index, idx_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

def upsert_docs(paciente_id: str, docs: List[Dict[str, Any]]) -> int:
    """
    docs: [{"id": "abc", "text": "conteúdo textual", "meta": {...}}, ...]
    Estratégia simples: sempre append (sem delete). Suficiente para TCC.
    """
    if not docs:
        return 0
    index, meta = _load_index(paciente_id)

    texts = [d["text"] for d in docs]
    # Embeddings normalizados => produto interno vira cosseno
    embs = emb_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    index.add(embs)
    meta.extend(docs)

    _save_index(paciente_id, index, meta)
    return len(docs)

def search_topk(paciente_id: str, query: str, k: int = 6) -> List[Dict[str, Any]]:
    index, meta = _load_index(paciente_id)
    if index.ntotal == 0:
        return []

    q = emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    # busca os top-k (ou até o total)
    k = min(k, index.ntotal)
    D, I = index.search(q, k)

    out = []
    for idx in I[0]:
        if 0 <= idx < len(meta):
            out.append(meta[idx])
    return out

def reset_index(paciente_id: str) -> List[str]:
    idx_path, meta_path = _paths_for(paciente_id)
    removed = []
    for p in (idx_path, meta_path):
        if os.path.exists(p):
            os.remove(p)
            removed.append(p)
    return removed

def stats(paciente_id: str) -> Dict[str, Any]:
    index, meta = _load_index(paciente_id)
    return {"vectors": int(index.ntotal), "docs": len(meta)}
