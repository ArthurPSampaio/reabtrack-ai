import time
from typing import Dict, List
from app.services.rag import search_hybrid
from app.core.models import get_models

def expand_query(query: str) -> str:
    return f"{query} sintomas dores evolução tratamento fisioterapia"

def advanced_retrieve(paciente_id: str, query: str) -> List[str]:
    start = time.time()
    
    expanded_query = expand_query(query)
    
    raw_docs = search_hybrid(paciente_id, expanded_query, k=15)
    if not raw_docs: return []
    
    pairs = [[query, d["text"]] for d in raw_docs]
    reranker = get_models().reranker
    scores = reranker.predict(pairs)
    
    ranked = sorted(zip(scores, raw_docs), key=lambda x: x[0], reverse=True)
    final_docs = [doc["text"] for score, doc in ranked[:5]]
    
    print(f"[ENGINE] Pipeline Híbrido em {time.time()-start:.2f}s. {len(raw_docs)} -> {len(final_docs)} docs.")
    return final_docs

def generate_clinical_report(paciente_id: str, indicadores: Dict) -> str:
    query = "Evolução da dor, capacidade funcional e adesão ao tratamento"
    contexto = advanced_retrieve(paciente_id, query)
    
    stats_str = "\n".join([f"- {k}: {v}" for k, v in indicadores.items()])
    historico_str = "\n".join([f"- {c}" for c in contexto])
    
    prompt = (
        "Atue como um Fisioterapeuta Sênior Especialista. Escreva o **CORPO TEXTUAL** de um Laudo de Evolução Clínica.\n"
        "Seu objetivo é fornecer uma análise aprofundada e técnica para compor o prontuário do paciente.\n\n"
        
        "⚠️ ORDEM DE EXECUÇÃO RÍGIDA (SIGA ESTRITAMENTE):\n"
        "1. Gere a **Tabela de Indicadores** (MarkDown) no topo absoluto.\n"
        "2. Pule uma linha.\n"
        "3. Escreva os **3 Tópicos de Texto** na sequência.\n\n"

        "🚫 REGRAS DE EXCLUSÃO:\n"
        "- NÃO gere cabeçalhos, rodapés, datas ou assinaturas.\n"
        "- NÃO invente nomes de clínicas.\n\n"
        
        f"### DADOS QUANTITATIVOS:\n{stats_str}\n\n"
        f"### HISTÓRICO CLÍNICO (RAG):\n{historico_str}\n\n"
        
        "### FORMATO DE SAÍDA OBRIGATÓRIO (MARKDOWN):\n"
        
        "| Indicador | Resultado |\n"
        "| :--- | :--- |\n"
        "| Total de Sessões | (valor) |\n"
        "| Média de Dor (EVA) | (valor)/10 |\n"
        "| Adesão ao Plano | (valor)% |\n\n"
        
        "### 1. Análise Detalhada da Evolução\n"
        "(Escreva 2 parágrafos técnicos detalhados. Compare o estado inicial com o atual. Cite a evolução específica da dor, amplitude de movimento (ADM) e força muscular. Use terminologia culta, ex: 'algique', 'cinesiofobia', 'ganho funcional'.)\n\n"
        
        "### 2. Considerações Clínicas\n"
        "(Destaque qualitativo sobre a resposta do paciente ao tratamento. Mencione se houve intercorrências, como o paciente reagiu à progressão de carga e seu nível de cooperação.)\n\n"
        
        "### 3. Planejamento Terapêutico\n"
        "(Defina as diretrizes para o próximo ciclo. Sugira manutenção ou alteração de conduta, progressão de exercícios e metas de curto prazo.)"
    )
    
    try:
        llm = get_models().gemini
        if not llm: return "Serviço de IA indisponível."
        return llm.generate_content(prompt).text
    except Exception as e:
        return f"Erro na geração do laudo: {e}"