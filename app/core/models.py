import os
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class ModelManager:
    _instance = None
    
    def __init__(self):
        # 1. Embedding (Bi-Encoder) - Rápido para busca no FAISS
        emb_model_name = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        print(f"[CORE] Carregando Embedder: {emb_model_name}...")
        self.embedder = SentenceTransformer(emb_model_name)
        
        # 2. Re-ranking (Cross-Encoder) - Preciso para filtrar relevância
        rerank_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        print(f"[CORE] Carregando Re-ranker: {rerank_model_name}...")
        self.reranker = CrossEncoder(rerank_model_name)
        
        # 3. Generativo (LLM) - O "Escritor"
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.gemini = genai.GenerativeModel(
                model_name='models/gemini-2.0-flash', # Modelo estável e rápido
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                ]
            )
            print("[CORE] Gemini Conectado.")
        else:
            self.gemini = None
            print("[CORE] ERRO: GEMINI_API_KEY ausente.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

def get_models():
    return ModelManager.get_instance()