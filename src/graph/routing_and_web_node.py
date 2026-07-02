import os
from src.graph.state import DiagnosticState
from langchain_tavily import TavilySearch


os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# --- initialize TavilySearch ---
tavily_search = TavilySearch(max_results=3)

# ---- routing conditional edge ----
def route_after_router(state: DiagnosticState):

    if state["sufficient_info"]:
        return "final_report"

    return "web_search"

# ---- web search node ----
def web_search_node(state: DiagnosticState):

    query = (
        f"Tuberculosis. "
        f"Prediction: {state['prediction']}. "
        f"Confidence: {state['confidence']}. "
        f"Missing topics: {', '.join(state['missing_info'])}. "
        f"Provide the latest WHO guidelines, treatment recommendations, "
        f"diagnostic criteria, and clinical evidence."
    )

    results = tavily_search.invoke(query)

    print("Web Search Results:", results)
    print(type(results))

    return {
        "web_docs": results["results"]
    }
