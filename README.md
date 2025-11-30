# 🧠 ReabTrack AI Core

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google%20gemini&logoColor=white)

> **Inteligência Clínica Avançada com RAG Híbrido.**

Este é o microsserviço de Inteligência Artificial do projeto **ReabTrack**. Ele é responsável por analisar o histórico não estruturado dos pacientes e gerar laudos de evolução clínica de alta precisão.

## 🔬 Engenharia e Inovação

Diferente de sistemas básicos de chat, este núcleo implementa uma arquitetura de **RAG (Retrieval-Augmented Generation) Híbrido em Dois Estágios**:

1.  **Recuperação Híbrida (Hybrid Search):** Combina busca semântica (**FAISS** / `all-MiniLM-L6-v2`) com busca lexical (**BM25**) para capturar tanto o contexto quanto termos médicos exatos.
2.  **Re-ranking (Cross-Encoder):** Um modelo especialista (`ms-marco-MiniLM`) reavalia os documentos recuperados para filtrar alucinações e garantir relevância máxima.
3.  **Geração (LLM):** Utiliza o modelo **Google Gemini 2.0 Flash** para raciocínio clínico e redação do laudo estruturado.

## 📂 Estrutura do Projeto

* `app/services/rag.py`: Motor de busca vetorial e lexical.
* `app/services/engine.py`: Pipeline de inteligência e Prompt Engineering blindado.
* `app/core/models.py`: Gerenciamento de memória e modelos (Singleton).

## 🚀 Como Rodar

1.  **Ambiente Virtual:**
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Linux/Mac:
    source .venv/bin/activate
    ```

2.  **Instalação:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Nota: Na primeira execução, o sistema baixará automaticamente os modelos de Embedding e Cross-Encoder).*

3.  **Configuração:**
    Crie o arquivo `.env`:
    ```env
    GEMINI_API_KEY=sua_chave_do_google_ai_studio
    EMB_MODEL=sentence-transformers/all-MiniLM-L6-v2
    ```

4.  **Execução:**
    ```bash
    uvicorn app.main:app --reload --port 8000
    ```

---
Desenvolvido por **Arthur Sampaio** | TCC 2025