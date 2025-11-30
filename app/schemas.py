from pydantic import BaseModel, Field
from typing import List, Dict, Any

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
    indicadores: Dict[str, Any] = Field(default_factory=dict)

class SummarizeOutput(BaseModel):
    texto: str