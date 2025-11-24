import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Header, HTTPException
from schemas import (
    SummarizeInput, SummarizeOutput,
    IngestInput, IngestOutput,
    SummarizeByPacienteInput
)
from summarizer import gerar_resumo
from rag_store import upsert_docs, search_topk, reset_index, stats

API_KEY = os.getenv("API_KEY", "")

app = FastAPI(title="ReabTrack AI (RAG)")

def auth(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

@app.get("/health")
def health(x_api_key: str | None = Header(default=None)):
    auth(x_api_key)
    return {"status": "ok"}

@app.post("/ingest", response_model=IngestOutput)
def ingest(body: IngestInput, x_api_key: str | None = Header(default=None)):
    auth(x_api_key)
    try:
        added = upsert_docs(body.pacienteId, [d.model_dump() for d in body.docs])
        return {"pacienteId": body.pacienteId, "added": added}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"falha na ingestão: {e}")

@app.get("/stats/{pacienteId}")
def get_stats(pacienteId: str, x_api_key: str | None = Header(default=None)):
    auth(x_api_key)
    return stats(pacienteId)

@app.delete("/reset_index/{pacienteId}")
def reset(pacienteId: str, x_api_key: str | None = Header(default=None)):
    auth(x_api_key)
    removed = reset_index(pacienteId)
    return {"removed": removed}

@app.post("/summarize/by-paciente", response_model=SummarizeOutput)
def summarize_by_paciente(body: SummarizeByPacienteInput, x_api_key: str | None = Header(default=None)):
    auth(x_api_key)

    docs = search_topk(body.pacienteId, body.query, k=6)
    if not docs:
        raise HTTPException(status_code=400, detail="sem contexto no índice FAISS para esse paciente")

    contexto = [d["text"] for d in docs]
    texto = gerar_resumo(contexto, body.indicadores, instrucoes_extra=None)
    return SummarizeOutput(texto=texto)

@app.post("/summarize", response_model=SummarizeOutput)
def summarize(body: SummarizeInput, x_api_key: str | None = Header(default=None)):
    auth(x_api_key)
    if not body.contexto:
        raise HTTPException(status_code=400, detail="contexto vazio")
    texto = gerar_resumo(body.contexto, body.indicadores, body.instrucoes)
    return SummarizeOutput(texto=texto)
