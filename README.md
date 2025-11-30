# 🧠 ReabTrack AI Core

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google%20gemini&logoColor=white)

> **Inteligência Clínica Avançada com RAG Híbrido.**

Este é o "cérebro" do projeto ReabTrack. Um microsserviço Python de alta performance responsável por analisar históricos clínicos não estruturados e gerar laudos de evolução técnica utilizando o estado da arte em IA Generativa.

## 🔬 Engenharia e Inovação

Diferente de sistemas que apenas "resumem" textos, este núcleo implementa uma arquitetura de **Recuperação Híbrida em Dois Estágios (Two-Stage Hybrid RAG)** para garantir precisão médica e zero alucinação:

1.  **Busca Híbrida (Hybrid Search):** Combina a busca semântica (**FAISS**) com a busca lexical (**BM25**), capturando tanto o contexto ("dor no joelho") quanto termos técnicos exatos ("LCA", "Dipirona").
2.  **Fusão e Re-ranking:** Utiliza o algoritmo **RRF (Reciprocal Rank Fusion)** seguido de um modelo **Cross-Encoder** (`ms-marco-MiniLM`) para reordenar os resultados e selecionar apenas as evidências clínicas mais relevantes.
3.  **Geração Clínica:** Utiliza o modelo **Google Gemini 2.0 Flash** com Engenharia de Prompt avançada para redigir laudos estruturados em formato Markdown.

## 📂 Estrutura do Projeto

A arquitetura segue o padrão de Clean Architecture simplificada:
* `app/services/rag.py`: Motor de busca vetorial e lexical.
* `app/services/engine.py`: Pipeline de inteligência e orquestração.
* `app/core/models.py`: Singleton para gerenciamento eficiente de memória dos modelos de ML.

## 🚀 Como Rodar

1.  **Prepare o ambiente:**
    ```bash
    git clone [https://github.com/ArthurPSampaio/reabtrack-ai.git](https://github.com/ArthurPSampaio/reabtrack-ai.git)
    cd reabtrack-ai
    python -m venv .venv
    # Ative o venv (Windows: .venv\Scripts\activate | Mac/Linux: source .venv/bin/activate)
    ```

2.  **Instale os pacotes:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Nota: O download dos modelos de ML ocorrerá automaticamente na primeira execução).*

3.  **Configure:**
    Crie o arquivo `.env`:
    ```env
    GEMINI_API_KEY=sua_chave_do_google_ai_studio
    EMB_MODEL=sentence-transformers/all-MiniLM-L6-v2
    ```

4.  **Execute o servidor:**
    ```bash
    uvicorn app.main:app --reload --port 8000
    ```

## 🔗 Parte do Ecossistema ReabTrack

* **Consumido por:** [reabtrack-backend](https://github.com/ArthurPSampaio/reabtrack-backend)

---
Desenvolvido por **Arthur Sampaio** | TCC 2025