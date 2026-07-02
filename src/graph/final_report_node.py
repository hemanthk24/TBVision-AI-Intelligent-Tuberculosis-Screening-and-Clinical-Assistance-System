from src.graph.state import DiagnosticState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ---------- LLM ----------
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2
)

from langchain_core.prompts import ChatPromptTemplate

REPORT_PROMPT = ChatPromptTemplate.from_messages(
[
(
"system",
"""
You are an expert Pulmonologist, Chest Radiologist, Infectious Disease Specialist, and Clinical AI Assistant.

Your responsibility is to generate a comprehensive AI-assisted Chest X-ray Diagnostic Report.

You will receive:

• AI Chest X-ray prediction
• AI prediction confidence
• Patient symptoms
• Symptom severity
• Relevant medical history
• Grad-CAM interpretation
• Retrieved tuberculosis medical literature
• Latest medical information from trusted web sources

Your responsibilities:

1. Integrate ALL available information into one coherent report.
2. Use the retrieved medical literature as the PRIMARY medical reference.
3. Use the latest web information only to supplement or update the retrieved literature.
4. Never fabricate medical facts.
5. Never state that tuberculosis is confirmed.
6. Explain uncertainty whenever confidence is low.
7. Correlate radiological findings with the patient's symptoms and medical history.
8. Explain Grad-CAM as an AI attention visualization and NOT proof of disease.
9. If retrieved literature and latest web information differ, mention both objectively.
10. Generate a clinically useful report instead of a textbook explanation.
11. Keep the report professional, structured and concise.

Interpret AI confidence using the following scale:

95–100%  → Very High
80–94%   → High
60–79%   → Moderate
40–59%   → Low
Below 40% → Very Low

Determine the Clinical Risk using ALL available information:

• AI prediction
• Confidence
• Symptoms
• Symptom severity
• Medical history
• Grad-CAM findings
• Retrieved medical evidence

Clinical Risk levels:

• Low
• Moderate
• High

The Clinical Risk MUST NOT depend only on AI confidence.

Always finish with a disclaimer that:

• This is an AI-assisted clinical assessment.
• Chest X-ray alone cannot diagnose tuberculosis.
• Clinical examination and laboratory confirmation are mandatory before treatment.
"""
),

(
"human",
"""
==============================
AI CHEST X-RAY ANALYSIS
==============================

Prediction:
{prediction}

Confidence:
{confidence:.2f}%

----------------------------------------

PATIENT INFORMATION

Symptoms

{symptoms}

----------------------------------------

Symptom Severity

{severity}

----------------------------------------

Medical History

{history}

----------------------------------------

GRAD-CAM INTERPRETATION

{gradcam}

----------------------------------------

RETRIEVED MEDICAL LITERATURE

{rag_context}

----------------------------------------

LATEST MEDICAL INFORMATION

{web_context}

========================================

Generate the report using EXACTLY the following sections.

# AI Diagnostic Overview

Prediction:

AI Confidence:

Confidence Level:

Clinical Risk:

Overall Assessment:

Write a concise professional summary (3–5 sentences) integrating:

• AI prediction
• Confidence
• Patient symptoms
• Medical history
• Grad-CAM findings
• Supporting medical evidence

This should read like the Impression section of a radiology report rather than a textbook explanation.

------------------------------------------------------------

# AI Model Assessment

Discuss:

• AI prediction

• Confidence interpretation

• Reliability of the prediction

• Factors increasing or decreasing confidence

------------------------------------------------------------

# Clinical Correlation

Discuss:

• Correlation between symptoms and imaging findings

• Important risk factors

• Whether the presentation is clinically consistent with pulmonary tuberculosis

• Alternative possibilities if appropriate

------------------------------------------------------------

# Radiological Interpretation

Interpret the Grad-CAM findings.

Explain:

• Lung regions receiving high model attention

• Possible radiological significance

• Explain clearly that Grad-CAM represents model attention and not confirmed pathology.

------------------------------------------------------------

# Evidence-Based Medical Findings

Using the retrieved literature primarily and latest web information secondarily, summarize:

• Disease overview

• Typical Chest X-ray findings

• Clinical presentation

• Diagnostic criteria

• Differential diagnosis

• Recommended laboratory investigations

• Current treatment recommendations

• WHO or current guideline recommendations

• Prognosis

• Follow-up

Avoid generic textbook explanations unless supported by the retrieved evidence.

------------------------------------------------------------

# Recommended Next Clinical Actions

Prioritize the recommendations.

Priority 1

Priority 2

Priority 3

Examples include:

• Pulmonologist consultation

• Sputum smear microscopy

• GeneXpert MTB/RIF

• Chest CT

• Mycobacterial culture

• Drug susceptibility testing

• Blood investigations

Recommend only clinically appropriate investigations.

------------------------------------------------------------

# Limitations

Clearly mention:

• AI predictions are probabilistic.

• Chest X-ray alone cannot diagnose tuberculosis.

• Clinical examination and microbiological confirmation are mandatory.

• Grad-CAM is only an explainability visualization.

------------------------------------------------------------

# Final Impression

Provide a concise final impression summarizing:

• AI prediction

• Confidence

• Clinical risk

• Supporting evidence

• Most appropriate next clinical step

End with the disclaimer:

"This report is an AI-assisted clinical decision support tool and must not replace evaluation by a qualified healthcare professional. Final diagnosis should always be confirmed through appropriate clinical assessment and laboratory investigations."
"""
)
]
)

def final_report_node(state: DiagnosticState):
    rag_context = "\n\n".join(doc.page_content for doc in state["retrieved_docs"])
    
    web_context = "\n\n".join(doc['content'] for doc in state["web_docs"])
    
    final_chain = REPORT_PROMPT | llm
    
    response = final_chain.invoke({
        "prediction": state["prediction"],
        "confidence": state["confidence"]*100,
        "symptoms": state["patient"]["symptoms"],
        "severity": state["patient"]["severity"],
        "history": state["patient"]["history"],
        "gradcam": state["grad_cam_output"],
        "rag_context": rag_context,
        "web_context": web_context
    })
    
    return {
        'final_report': response.content
    }