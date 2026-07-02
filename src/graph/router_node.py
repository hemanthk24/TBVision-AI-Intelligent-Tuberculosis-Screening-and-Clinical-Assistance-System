from src.graph.state import DiagnosticState
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# ---------- Structured Output ----------
class RetrievalEvaluation(BaseModel):
    sufficient: bool = Field(
        ...,
        description="Whether the retrieved information is sufficient to generate the final diagnostic report."
    )

    coverage_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="Coverage score between 1 and 10."
    )

    missing_topics: list[str] = Field(
        default_factory=list,
        description="Important medical topics missing from the retrieved literature."
    )


# ---------- LLM ----------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

structured_llm = llm.with_structured_output(RetrievalEvaluation)


# ---------- Prompt ----------
ROUTER_PROMPT = ChatPromptTemplate.from_messages(
[
(
"system",
"""
You are an expert medical retrieval evaluator.

Your task is NOT to answer the medical question.

Evaluate whether the retrieved tuberculosis medical literature is sufficient
to generate a complete and reliable diagnostic report.

Evaluate whether the retrieved information covers:

• Disease overview
• Chest X-ray findings
• Clinical symptoms
• Diagnostic criteria
• Differential diagnosis
• Laboratory investigations
• Treatment recommendations
• WHO or other standard clinical guidelines
• Prognosis and follow-up

Scoring Rules:

- Give a coverage score from 1 to 10.
- If the score is 7 or above, mark the information as sufficient.
- If important medical topics are missing, include them in missing_topics.
- Return only the structured output.
"""
),

(
"human",
"""
Original Retrieval Query

{query}

--------------------------------------------

Retrieved Medical Literature

{retrieved_context}
"""
)
]
)


# ---------- Chain ----------
router_chain = ROUTER_PROMPT | structured_llm


# ---------- Router Node ----------
def router_node(state: DiagnosticState):

    retrieved_context = "\n\n".join(
        doc.page_content
        for doc in state["retrieved_docs"]
    )

    response = router_chain.invoke({
        "query": state["retrieval_query"],
        "retrieved_context": retrieved_context
    })

    return {
        "sufficient_info": response.sufficient,
        "coverage_score": response.coverage_score,
        "missing_info": response.missing_topics
    }