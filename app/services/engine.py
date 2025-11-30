import time
from typing import List, Dict
from app.services.rag import search_faiss
from app.core.models import get_models

def advanced_retrieve(paciente_id: str, query: str) -> List[str]:
    start = time.time()
    
    # 1. Recuperação Densa (Rápida, traz candidatos)
    raw_docs = search_faiss(paciente_id, query, k=15)
    if not raw_docs: return []
    
    # 2. Re-ranking (Inteligente, filtra o melhor contexto)
    # O Cross-Encoder lê a pergunta E o documento juntos
    pairs = [[query, d["text"]] for d in raw_docs]
    reranker = get_models().reranker
    scores = reranker.predict(pairs)
    
    # Ordena por relevância real
    ranked = sorted(zip(scores, raw_docs), key=lambda x: x[0], reverse=True)
    
    # Pega o Top 5 de alta qualidade
    final_docs = [doc["text"] for score, doc in ranked[:5]]
    
    print(f"[ENGINE] Pipeline executado em {time.time()-start:.2f}s. Contexto refinado: {len(final_docs)} docs.")
    return final_docs

def generate_clinical_report(paciente_id: str, indicadores: Dict) -> str:
    # Query estratégica para buscar sintomas e evolução
    query = "Evolução da dor, capacidade funcional e adesão ao tratamento"
    
    contexto = advanced_retrieve(paciente_id, query)
    
    stats_str = "\n".join([f"- {k}: {v}" for k, v in indicadores.items()])
    historico_str = "\n".join([f"- {c}" for c in contexto])
    
    prompt = (
        "Atue como um Fisioterapeuta Sênior Especialista. Analise os dados abaixo e redija um **Laudo de Evolução Clínica**.\n"
        "O texto deve ser extremamente profissional, técnico e estruturado, adequado para prontuário médico.\n\n"
        f"### INDICADORES QUANTITATIVOS:\n{stats_str}\n\n"
        f"### HISTÓRICO CLÍNICO (Contexto RAG):\n{historico_str}\n\n"
        "### ESTRUTURA DO LAUDO (Markdown):\n"
        "**1. SÍNTESE DA EVOLUÇÃO**\n"
        "(Descreva tecnicamente a progressão do quadro álgico e funcional.)\n\n"
        "**2. ANÁLISE DE INDICADORES**\n"
        "(Interprete a adesão e a resposta aos exercícios.)\n\n"
        "**3. CONDUTA TERAPÊUTICA**\n"
        "(Sugira o plano de tratamento futuro.)"
    )
    
    try:
        llm = get_models().gemini
        if not llm: return "Serviço de IA indisponível."
        return llm.generate_content(prompt).text
    except Exception as e:
        return f"Erro na geração do laudo: {e}"