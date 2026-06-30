from src.rag.retriever import get_retriever
from src.graph.state import DiagnosticState

# 3. ------- RAG node --------
def rag_node(state: DiagnosticState):
   query = f"""
    The AI model predicted **{state["prediction"]}** with a confidence of **{state["confidence"]:.2f}%**.

    Patient Information

    Symptoms:
    {state["patient"]["symptoms"]}

    Severity:
    {state["patient"]["severity"]}

    Medical History:
    {state["patient"]["history"]}

    Grad-CAM Findings:
    {state["grad_cam_output"]}

    Based on the above clinical and radiological information, retrieve the most relevant tuberculosis medical literature covering:

    • Disease overview
    • Interpretation of similar chest X-ray findings
    • Clinical symptoms associated with this presentation
    • Diagnostic criteria
    • Differential diagnosis
    • Recommended confirmatory laboratory investigations
    • Treatment recommendations
    • WHO or standard clinical guidelines
    • Prognosis and recommended follow-up
    """
   retriever = get_retriever()
   docs = retriever.invoke(query)
   return {
       'retrieved_docs':docs
   }
   
    