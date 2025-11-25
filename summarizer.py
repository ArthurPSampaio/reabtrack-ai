# summarizer.py
import os
import re
import time
from typing import List, Dict

from sentence_transformers import SentenceTransformer
import google.generativeai as genai

EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print(">> Carregando modelos de IA (embeddings + geração)...")
print(f">> EMB_MODEL = {EMB_MODEL}")

emb_model = SentenceTransformer(EMB_MODEL)

try:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY não encontrada no .env")
        
    genai.configure(api_key=GEMINI_API_KEY)
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    gen_model = genai.GenerativeModel(
        model_name='models/gemini-flash-latest',
        safety_settings=safety_settings
    )
    print(">> Modelo de Geração (Gemini) carregado.")

except Exception as e:
    print(f"ERRO FATAL AO CARREGAR O GEMINI: {e}")
    print("Verifique se a GEMINI_API_KEY está correta no seu .env")
    gen_model = None 


_WS = re.compile(r"\s+")

def _clean_line(s: str, max_chars: int = 240) -> str:
    if not s:
        return ""
    s = s.replace("\x00", " ")
    s = _WS.sub(" ", s).strip()
    s = s.replace("•", "-")
    return s[:max_chars]


def gerar_resumo(contexto: List[str], indicadores: Dict, instrucoes_extra: str | None = None) -> str:
    """Gera um relatório clínico estruturado usando a API do Gemini."""
    
    if gen_model is None:
        return "ERRO: O modelo de geração Gemini não foi carregado."

    clean_ctx = [_clean_line(str(t)) for t in contexto if _clean_line(str(t))]
    
    observacoes = []
    
    ind_bits = []
    if "totalSessoes" in indicadores:
        ind_bits.append(f"Total de Sessões: {indicadores['totalSessoes']}")
    if "mediaDor" in indicadores:
        ind_bits.append(f"Média de Dor: {indicadores['mediaDor']}/10")
    if "taxaConclusao" in indicadores:
        try:
            pct = int(round(100 * float(indicadores["taxaConclusao"])))
        except:
            pct = indicadores["taxaConclusao"]
        ind_bits.append(f"Taxa de Adesão aos Exercícios: {pct}%")
    
    if ind_bits:
        observacoes.append("DADOS QUANTITATIVOS: " + " | ".join(ind_bits))

    observacoes.extend(clean_ctx)

    fonte_formatada = "\n".join(f"- {obs}" for obs in observacoes)

    prompt = (
        "Atue como um sistema inteligente de análise de prontuários. "
        "Gere um **Relatório de Evolução Fisioterapêutica** com base nos dados brutos abaixo. "
        "O documento deve ter um tom profissional, técnico e limpo, semelhante a um laudo ou exame, "
        "evitando linguagem coloquial, saudações de chat ou emojis.\n\n"
        "DADOS DO PRONTUÁRIO:\n"
        f"{fonte_formatada}\n\n"
        "ESTRUTURA DO RELATÓRIO (Use Markdown Limpo):\n"
        "### 1. Análise da Evolução\n"
        "(Parágrafo técnico comparativo descrevendo a progressão do quadro álgico (dor), tolerância ao esforço e funcionalidade desde o início até o momento atual.)\n\n"
        "### 2. Observações Relevantes\n"
        "(Lista com bullet points destacando adesão ao tratamento, ganho de amplitude, força ou quaisquer intercorrências relatadas.)\n\n"
        "### 3. Conduta e Recomendações\n"
        "(Diretrizes objetivas para a continuidade do tratamento e orientações ao paciente.)"
    )

    start = time.strftime("%H:%M:%S")
    print(f"\n[IA][{start}] --- Nova geração (Gemini - Laudo) ---")
    
    try:
        response = gen_model.generate_content(prompt)
        texto = response.text
        
    except Exception as e:
        print(f"[IA] ERRO na chamada do Gemini: {e}")
        texto = f"Não foi possível gerar o relatório. Erro técnico: {e}"

    print(f"[IA] Output (limpo): {texto[:400]!r}")
    print(f"[IA] --- Fim da geração ---\n")

    return texto