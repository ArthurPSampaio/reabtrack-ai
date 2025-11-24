# summarizer.py
import os
import re
import time
from typing import List, Dict

from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# =========================================================
# Config via .env
# =========================================================
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print(">> Carregando modelos de IA (embeddings + geração)...")
print(f">> EMB_MODEL = {EMB_MODEL}")

# --- Carregando o Modelo de Embedding (para o FAISS) ---
emb_model = SentenceTransformer(EMB_MODEL)

# --- Carregando o Modelo de Geração (Gemini) ---
try:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY não encontrada no .env")
        
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Configurações de segurança (para evitar bloqueios)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    # --- A CORREÇÃO ESTÁ AQUI ---
    # Usando o nome exato que o SEU script 'check_models.py' encontrou.
    gen_model = genai.GenerativeModel(
        model_name='models/gemini-flash-latest',
        safety_settings=safety_settings
    )
    print(">> Modelo de Geração (Gemini) carregado.")

except Exception as e:
    print(f"ERRO FATAL AO CARREGAR O GEMINI: {e}")
    print("Verifique se a GEMINI_API_KEY está correta no seu .env")
    gen_model = None # Vai falhar na geração

# =========================================================
# Funções de Limpeza (mantidas)
# =========================================================

_WS = re.compile(r"\s+")

def _clean_line(s: str, max_chars: int = 240) -> str:
    """Higieniza texto de entrada (remove nulos, colapsa espaços, troca bullets)
    e limita o tamanho de cada sentença.
    """
    if not s:
        return ""
    s = s.replace("\x00", " ")
    s = _WS.sub(" ", s).strip()
    s = s.replace("•", "-")
    return s[:max_chars]


# =========================================================
# Função de Geração (Reescrita para o Gemini)
# =========================================================

def gerar_resumo(contexto: List[str], indicadores: Dict, instrucoes_extra: str | None = None) -> str:
    """Gera um resumo usando a API do Gemini."""
    
    if gen_model is None:
        return "ERRO: O modelo de geração Gemini não foi carregado. Verifique a API Key."

    # --- 1. Formatar os dados como "Prosa" (o melhor formato que encontramos) ---
    clean_ctx = [_clean_line(str(t)) for t in contexto if _clean_line(str(t))]
    
    observacoes = []
    
    # Adicionar os indicadores como prosa (frases completas)
    ind_bits = []
    if "totalSessoes" in indicadores:
        ind_bits.append(f"{indicadores['totalSessoes']} sessões no total")
    if "mediaDor" in indicadores:
        ind_bits.append(f"dor média de {indicadores['mediaDor']}/10")
    if "taxaConclusao" in indicadores:
        try:
            pct = int(round(100 * float(indicadores["taxaConclusao"])))
        except Exception:
            pct = indicadores["taxaConclusao"]
        ind_bits.append(f"adesão de {pct}%")
    if ind_bits:
        observacoes.append("Indicadores gerais: " + ", ".join(ind_bits) + ".")

    # Adicionar o contexto (as notas reais)
    observacoes.extend(clean_ctx)

    # Formatar como uma lista numerada
    fonte_formatada = "\n".join(f"{i+1}. {obs}" for i, obs in enumerate(observacoes))

    # --- 2. Criar o Prompt (o Gemini entende isso perfeitamente) ---
    prompt = (
        "Você é um assistente de fisioterapia. "
        "Gere um breve resumo clínico profissional em português com base nas seguintes observações. "
        "O resumo deve focar na evolução do paciente (comparando as sessões), sintomas atuais, e adesão. "
        "Seja factual e use um tom clínico.\n\n"
        "OBSERVAÇÕES:\n"
        f"{fonte_formatada}\n\n"
        "RESUMO CLÍNICO:"
    )

    # ===== Logs de depuração =====
    start = time.strftime("%H:%M:%S")
    print(f"\n[IA][{start}] --- Nova geração (Gemini) ---")
    print(f"[IA] Prompt (primeiros 500 chars): {prompt[:500]!r}")

    # --- 3. Chamar a API do Gemini ---
    try:
        response = gen_model.generate_content(prompt)
        texto = response.text
        
    except Exception as e:
        print(f"[IA] ERRO na chamada do Gemini: {e}")
        texto = f"Falha ao gerar resumo: {e}"

    # ===== Logs da saída final =====
    print(f"[IA] Output (limpo): {texto[:400]!r}")
    print(f"[IA] --- Fim da geração ---\n")

    return texto