from langgraph.graph import StateGraph
from typing  import TypedDict
from langchain_core.documents import Document

# patient info
class PatientInfo(TypedDict):
    symptoms: dict
    severity: dict
    history: dict
    
# creating the State for our Workflow
class DiagnosticState(TypedDict):
    image_path: str
    prediction: str
    confidence: float
    grad_cam_output: str
    patient: PatientInfo
    retrieval_query: str
    retrieved_docs: list[Document]
    web_docs: list[dict]
    sufficient_info: bool
    coverage_score: float
    missing_info: list[str]
    final_report: str

