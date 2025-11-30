from fastapi import FastAPI, HTTPException
from app.schemas import IngestInput, IngestOutput, SummarizeByPacienteInput, SummarizeOutput
from app.services.rag import upsert_docs, reset_index
from app.services.engine import generate_clinical_report

app = FastAPI(title="ReabTrack AI Core", version="2.0 (Two-Stage RAG)")

@app.get("/")
def root():
    return {"system": "ReabTrack AI", "status": "online", "architecture": "Modular"}

@app.post("/ingest", response_model=IngestOutput)
def ingest_handler(body: IngestInput):
    try:
        n = upsert_docs(body.pacienteId, [d.model_dump() for d in body.docs])
        return {"pacienteId": body.pacienteId, "added": n}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/summarize/by-paciente", response_model=SummarizeOutput)
def summarize_handler(body: SummarizeByPacienteInput):
    try:
        txt = generate_clinical_report(body.pacienteId, body.indicadores)
        return {"texto": txt}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.delete("/reset/{pacienteId}")
def reset_handler(pacienteId: str):
    reset_index(pacienteId)
    return {"status": "deleted"}