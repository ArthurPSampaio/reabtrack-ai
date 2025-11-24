# schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class SummarizeInput(BaseModel):
    # Para já: passaremos “contexto” diretamente (lista de trechos). Na próxima etapa, isso virá do FAISS.
    contexto: List[str] = Field(default_factory=list)
    indicadores: Dict[str, float | int | str] = Field(default_factory=dict)
    instrucoes: Optional[str] = None  # opcional: tom/estilo extra

class SummarizeOutput(BaseModel):
    texto: str

# schemas.py (adicione)
from typing import Any

class IngestDoc(BaseModel):
    id: str
    text: str
    meta: Dict[str, Any] = Field(default_factory=dict)

class IngestInput(BaseModel):
    pacienteId: str
    docs: List[IngestDoc]

class IngestOutput(BaseModel):
    pacienteId: str
    added: int

class SummarizeByPacienteInput(BaseModel):
    pacienteId: str
    query: str = "Resumo clínico do paciente enfatizando progresso, sinais de alerta e próxima sessão."
    indicadores: Dict[str, float | int | str] = Field(default_factory=dict)
