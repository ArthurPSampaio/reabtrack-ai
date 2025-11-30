import os
import json
import faiss
import pickle
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from app.core.models import get_models
from app.services.text_processing import tokenize  

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def _paths_for(paciente_id: str) -> Tuple[str, str, str]:
    return (
        os.path.join(DATA_DIR, f"faiss_{paciente_id}.index"),
        os.path.join(DATA_DIR, f"meta_{paciente_id}.json"),
        os.path.join(DATA_DIR, f"bm25_{paciente_id}.pkl") 
    )

def upsert_docs(paciente_id: str, docs: List[Dict[str, Any]]) -> int:
    if not docs: return 0
    idx_path, meta_path, bm25_path = _paths_for(paciente_id)
    model = get_models().embedder
    
    meta = []
    if os.path.exists(idx_path) and os.path.exists(meta_path):
        index = faiss.read_index(idx_path)
        with open(meta_path, "r", encoding="utf-8") as f: meta = json.load(f)
    else:
        dim = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)
    
    new_texts = [d["text"] for d in docs]
    embs = model.encode(new_texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    index.add(embs)
    meta.extend(docs)
    
    all_texts = [d["text"] for d in meta]
    tokenized_corpus = [tokenize(doc) for doc in all_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    faiss.write_index(index, idx_path)
    with open(meta_path, "w", encoding="utf-8") as f: json.dump(meta, f, ensure_ascii=False)
    with open(bm25_path, "wb") as f: pickle.dump(bm25, f)
    
    return len(docs)

def search_hybrid(paciente_id: str, query: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Realiza busca híbrida (Vetorial + Lexical) com fusão de ranqueamento (RRF).
    """
    idx_path, meta_path, bm25_path = _paths_for(paciente_id)
    if not os.path.exists(idx_path): return []

    index = faiss.read_index(idx_path)
    with open(meta_path, "r", encoding="utf-8") as f: meta = json.load(f)
    if not os.path.exists(bm25_path): return [] 
    with open(bm25_path, "rb") as f: bm25 = pickle.load(f)

    model = get_models().embedder
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    k_search = min(k * 2, len(meta)) 
    _, faiss_indices = index.search(q_emb, k_search)
    faiss_hits = [meta[i] for i in faiss_indices[0] if 0 <= i < len(meta)]

    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_n = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k_search]
    bm25_hits = [meta[i] for i in top_n]

    rrf_score = {}
    const_k = 60

    def add_scores(hits, weight=1.0):
        for rank, doc in enumerate(hits):
            doc_id = doc.get("id") 
            if not doc_id: continue 
            if doc_id not in rrf_score:
                rrf_score[doc_id] = {"score": 0, "doc": doc}
            rrf_score[doc_id]["score"] += weight * (1 / (const_k + rank + 1))

    add_scores(faiss_hits)
    add_scores(bm25_hits)

    sorted_docs = sorted(rrf_score.values(), key=lambda x: x["score"], reverse=True)
    
    return [item["doc"] for item in sorted_docs[:k]]

def reset_index(paciente_id: str):
    paths = _paths_for(paciente_id)
    for p in paths:
        if os.path.exists(p): os.remove(p)
    return True