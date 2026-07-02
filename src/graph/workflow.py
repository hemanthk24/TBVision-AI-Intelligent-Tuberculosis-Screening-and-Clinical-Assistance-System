from langgraph.graph import StateGraph, START, END
from src.graph.state import DiagnosticState
from src.graph.prediction_node import prediction_node
from src.graph.grad_cam_node import grad_cam_node
from src.graph.rag_node import rag_node
from src.graph.router_node import router_node
from src.graph.routing_and_web_node import route_after_router, web_search_node
from src.graph.final_report_node import final_report_node

# --------------------------
#  Graph Workflow Definition 
# --------------------------
builder = StateGraph(DiagnosticState)

# ------- Nodes -------
builder.add_node("prediction", prediction_node)
builder.add_node("gradcam", grad_cam_node)
builder.add_node("rag", rag_node)
builder.add_node("router", router_node)
builder.add_node("web_search", web_search_node)
builder.add_node("final_report", final_report_node)

# ------ Edges -------
builder.add_edge(START, "prediction")
builder.add_edge("prediction", "gradcam")
builder.add_edge("gradcam", "rag")
builder.add_edge("rag", "router")
builder.add_conditional_edges(
    "router",
    route_after_router,
    {
        "final_report": "final_report",
        "web_search": "web_search"
    }
)
builder.add_edge("web_search", "final_report")
builder.add_edge("final_report", END)


# ------ compile the graph -------
graph = builder.compile()

