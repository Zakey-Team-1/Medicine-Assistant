"""LangGraph agent for the Medicine Assistant."""

from typing import TypedDict
import json
import re

from langchain_core.messages import BaseMessage, HumanMessage

from llm import get_llm
from rag import RAGComponent


# System Prompt for Medicine Assistant
SYSTEM_PROMPT = """You are an expert medical AI assistant specializing in diabetes management across all types (Type 1, Type 2, gestational, and related metabolic conditions). You support clinicians by analyzing patient data and providing evidence-based medication recommendations.

## Core Responsibilities

### 1. Evidence-Based Recommendations
- Base ALL recommendations on the RAG-retrieved context: {context}
- Cite specific guidelines, protocols, and evidence from retrieved documents
- Include document references (title, page number, or section) for each recommendation
- If RAG context is insufficient, explicitly state this limitation and provide general evidence-based guidance while noting the need for verification

### 2. Comprehensive Patient Analysis
Evaluate the complete clinical picture:

**Metabolic Control:**
- HbA1c trends and current value
- Blood glucose patterns and immediate readings
- Time in target range (if available)

**Patient Profile:**
- Age, gender, weight, BMI
- Diabetes type and duration
- Comorbidities (cardiovascular, renal, hepatic)

**Organ Function:**
- Renal function (eGFR/creatinine) - critical for medication safety
- Hepatic function markers
- Cardiovascular status (BP, lipids, cardiac history)

**Current Treatment:**
- All diabetes medications with doses and duration
- Recent adjustments and patient response
- Adherence patterns and barriers

**Clinical Context:**
- Current symptoms and concerns
- Hypoglycemia history and risk factors
- Patient preferences and lifestyle factors

### 3. Personalized Recommendations
Provide tailored guidance considering:
- Appropriate medication classes based on patient profile
- Specific dosing adjusted for weight, renal function, and age
- Titration schedules with clear steps and timelines
- Contraindications and drug interactions specific to this patient
- Special population considerations (elderly, pregnancy, pediatric, renal impairment)

### 4. Structured Clinical Output
Generate TWO comprehensive reports:

**A. PHYSICIAN REPORT**
Include detailed clinical reasoning:
- Current Status Assessment: Interpretation of all labs and clinical data
- Medication Efficacy Evaluation: Analysis of current regimen's effectiveness
- Recommendations with Rationale: Why each change is suggested, citing evidence
- Safety Considerations: Contraindications, interactions, monitoring needs
- Monitoring Plan: Specific tests, frequency, and target values
- Red Flags: Any urgent concerns requiring immediate attention
- Evidence Citations: Reference RAG documents for each major recommendation

**B. PATIENT REPORT**
Provide clear, accessible information:
- Current Health Summary: Simple explanation of diabetes control status
- Medication Instructions: What to take, when, how, and with what
- Important Precautions: Side effects to watch for, when to call doctor
- Lifestyle Guidance: Diet, exercise, and monitoring recommendations
- Follow-up Plan: When to return, what tests are needed
- Questions to Ask: Encourage patient engagement with their care team

### 5. Safety Framework
- Flag hypoglycemia risk factors prominently
- Adjust recommendations for renal/hepatic impairment
- Identify potential drug-drug interactions
- Note pregnancy/breastfeeding contraindications
- Highlight need for dose reduction in elderly patients

## Critical Reminders
⚠️ **CLINICAL JUDGMENT SUPERSEDES AI RECOMMENDATIONS**
- This tool assists but does NOT replace clinical decision-making
- Final treatment decisions must be made by qualified healthcare professionals
- All recommendations should be verified against local protocols and guidelines
- In case of uncertainty or incomplete data, recommend clinician review

## Response Format
When context is available: Provide evidence-based recommendations with citations
When context is limited: State limitations clearly, offer general guidance, encourage guideline verification
Always: Prioritize patient safety and acknowledge uncertainty when appropriate

DO NOT include any welcome messages or follow-up statements. Focus solely on delivering the reports as specified.
The report should be concise, structured, and easy to navigate for busy clinicians and patients.
DO NOT include any disclaimers about being an AI model or similar statements.
DO NOT reference the RAG process in the final reports.
DO NOT say things like Of course. Here is a patient-friendly report... etc
"""



class AgentState(TypedDict):
    """Enhanced state schema for the medicine assistant agent."""
    messages: list[BaseMessage]
    context: str
    patient_info: dict  # Changed from str to dict for structured data
    recommendations: str
    physician_report: str  # Separate storage for physician report
    patient_report: str    # Separate storage for patient report
    error_message: str  # Error handling


class MedicineAssistantAgent:
    """Enhanced LangGraph-based agent for medicine assistance with improved workflow."""

    def __init__(self, rag_component: RAGComponent | None = None):
        """
        Initialize the Medicine Assistant Agent.

        Args:
            rag_component: RAGComponent for document retrieval.
                         If None, a new instance will be created.
        """
        self.rag = rag_component or RAGComponent()
        self.llm = get_llm()
        # Simplified flow: no StateGraph. We'll call methods sequentially in invoke/ainvoke.

    # Note: The prior implementation used a StateGraph to orchestrate steps. To
    # simplify maintenance and readability we've removed the graph and call
    # components sequentially in `invoke` / `ainvoke`.

    def _extract_patient_data(self, state: dict) -> dict:
        """Extract and structure patient data from the query."""
        messages = state.get("messages", [])
        patient_info = state.get("patient_info", {})
        # If patient_info already provided (from Flask route), use it
        if patient_info and isinstance(patient_info, dict) and len(patient_info) > 0:
            return {"patient_info": patient_info}

        # If there's a human message, store the raw message text directly
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, HumanMessage):
                return {"patient_info": {"raw": last_message.content}}

        # Fallback: empty patient_info
        return {"patient_info": {}}

    def _retrieve_context(self, state: dict) -> dict:
        """Retrieve relevant context from RAG with enhanced query."""
        messages = state.get("messages", [])
        patient_info = state.get("patient_info", {})
        
        # Build enhanced query for retrieval
        query_parts = []
        
        # Add original query
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query_parts.append(msg.content)
                break
        
        # Add patient-specific context for better retrieval
        if patient_info.get("diabetes_type"):
            query_parts.append(f"diabetes type: {patient_info['diabetes_type']}")
        
        if patient_info.get("egfr"):
            egfr_val = patient_info["egfr"]
            try:
                egfr = float(egfr_val)
                if egfr < 30:
                    query_parts.append("severe renal impairment medication adjustments")
                elif egfr < 60:
                    query_parts.append("moderate renal impairment dosing")
            except (ValueError, TypeError):
                pass
        
        if patient_info.get("age"):
            try:
                age = int(patient_info["age"])
                if age >= 65:
                    query_parts.append("elderly diabetes management")
            except (ValueError, TypeError):
                pass
        
        if patient_info.get("latest_hba1c"):
            try:
                hba1c = float(patient_info["latest_hba1c"])
                if hba1c > 9:
                    query_parts.append("poor glycemic control intensification")
            except (ValueError, TypeError):
                pass
        
        enhanced_query = " ".join(query_parts)
        
        # Retrieve with multiple passes for comprehensive coverage
        all_docs = []
        
        # Main retrieval
        docs = self.rag.retrieve(enhanced_query, k=5)
        all_docs.extend(docs)
        
        # Retrieve medication-specific guidance if current meds are known
        current_meds = patient_info.get("current_medications") or patient_info.get("current_meds")
        if current_meds:
            # Handle both string and list formats
            if isinstance(current_meds, str):
                # Simple extraction of medication names
                med_names = [word for word in current_meds.split() 
                           if len(word) > 3 and word[0].isupper()]
            else:
                med_names = [med.get("name") if isinstance(med, dict) else str(med) 
                           for med in current_meds]
            
            for med_name in med_names[:3]:  # Limit to 3 to avoid too many queries
                if med_name:
                    med_docs = self.rag.retrieve(f"{med_name} dosing monitoring", k=2)
                    all_docs.extend(med_docs)
        
        # Deduplicate and format context
        seen_content = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_docs.append(doc)
        
        # Format context with source attribution
        context_parts = []
        for i, doc in enumerate(unique_docs[:10]):  # Limit to top 10
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            context_parts.append(f"[Source {i+1}: {source}, Page {page}]\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        return {"context": context}

    def _generate_physician_report(self, state: dict) -> dict:
        """Generate detailed physician report with clinical reasoning."""
        context = state.get("context", "")
        patient_info = state.get("patient_info", {})
        physician_prompt = f"""
{SYSTEM_PROMPT.format(context=context)}

Generate a PHYSICIAN REPORT ONLY for the following patient case.

{self._format_patient_data(patient_info)}

Provide a comprehensive physician report with:
1. **Clinical Assessment** - Interpret all labs and current status
2. **Current Regimen Evaluation** - Efficacy and safety analysis
3. **Recommendations with Rationale** - Specific changes with evidence citations
4. **Safety Considerations** - Contraindications, interactions, monitoring
5. **Monitoring Plan** - Tests, frequency, targets
6. **Immediate Concerns** - Any red flags or urgent actions needed

Use clear headings and be concise, don't make it too long. This is for the treating physician who understands this topic don't be too extensive.
"""
        
        response = self.llm.invoke([HumanMessage(content=physician_prompt)])
        
        return {"physician_report": response.content}

    def _generate_patient_report(self, state: dict) -> dict:
        """Generate patient-friendly report with clear instructions."""
        context = state.get("context", "")
        patient_info = state.get("patient_info", {})
        physician_report = state.get("physician_report", "")
        
        patient_prompt = f"""
Based on the physician report below, create a PATIENT-FRIENDLY REPORT.

Physician Analysis:
{physician_report}

Patient Information:
{self._format_patient_data(patient_info)}

Generate a clear, compassionate patient report with:
1. **Your Diabetes Status** - Simple explanation of control (avoid medical jargon)
2. **Your Medications** - Clear instructions: what, when, how, with/without food
3. **Important Warnings** - What symptoms need immediate medical attention
4. **Daily Management Tips** - Practical advice for blood sugar, diet, lifestyle
5. **Your Follow-Up Plan** - When to return, what tests you'll need
6. **Questions to Ask Your Doctor** - Important topics to discuss

Use simple language. Be encouraging and supportive. Avoid medical terminology or explain it clearly.
"""
        
        response = self.llm.invoke([HumanMessage(content=patient_prompt)])
        
        return {"patient_report": response.content}

    # Clarification handling removed — callers should supply sufficient data.

    def _format_patient_data(self, patient_info: dict) -> str:
        """Format patient data for LLM prompts."""
        if not patient_info:
            return "No patient data available."
        
        formatted = "**PATIENT DATA:**\n\n"
        
        sections = {
            "Demographics": ["name", "patient_id", "age", "gender", "weight", "height", "bmi"],
            "Diabetes Profile": ["diabetes_type", "duration_years"],
            "Glycemic Control": ["latest_hba1c", "blood_glucose"],
            "Cardiovascular": ["blood_pressure", "lipid_panel"],
            "Renal Function": ["egfr", "creatinine"],
            "Current Treatment": ["current_medications", "current_meds", "treatment_adjustments"],
            "Clinical Notes": ["symptoms_notes", "recent_symptoms", "comorbidities", "allergies"]
        }
        
        for section, fields in sections.items():
            section_data = {k: v for k, v in patient_info.items() if k in fields and v}
            if section_data:
                formatted += f"\n**{section}:**\n"
                for key, value in section_data.items():
                    formatted += f"- {key.replace('_', ' ').title()}: {json.dumps(value) if isinstance(value, (dict, list)) else value}\n"
        
        return formatted

    def invoke(self, message: str, patient_info: dict | str | None = None) -> dict:
        """
        Process a user message and return both reports.

        Args:
            message: The user's input message (can include structured patient data).
            patient_info: Optional patient information (dict preferred, str accepted for backwards compatibility).

        Returns:
            Dictionary with `physician_report`, `patient_report`, and `patient_info`.
        """
        # Handle backwards compatibility - convert string to dict if needed
        if isinstance(patient_info, str):
            try:
                patient_info = json.loads(patient_info)
            except (json.JSONDecodeError, TypeError):
                # If it's not valid JSON, let extraction handle it
                patient_info = {}
        
        # Prepare initial state
        messages: list[BaseMessage] = [HumanMessage(content=message)]
        state: dict = {
            "messages": messages,
            "context": "",
            "patient_info": patient_info or {},
            "recommendations": "",
            "physician_report": "",
            "patient_report": "",
            "error_message": "",
        }

        # 1) Extract patient data if not already provided
        extract_out = self._extract_patient_data(state)
        state["patient_info"] = extract_out.get("patient_info", state["patient_info"])

        # 2) (Validation step removed) Proceeding directly after extraction.

        # 3) Retrieve context and identify safety concerns
        ctx = self._retrieve_context(state)
        state["context"] = ctx.get("context", "")

        # Safety-alert identification removed; nothing to set here.

        # 4) Generate reports sequentially
        phys = self._generate_physician_report(state)
        state["physician_report"] = phys.get("physician_report", "")

        pat = self._generate_patient_report(state)
        state["patient_report"] = pat.get("patient_report", "")

        return {
            "physician_report": state["physician_report"],
            "patient_report": state["patient_report"],
            "patient_info": state.get("patient_info", {}),
        }

    async def ainvoke(self, message: str, patient_info: dict | str | None = None) -> dict:
        """Async version of invoke."""
        # Handle backwards compatibility
        if isinstance(patient_info, str):
            try:
                patient_info = json.loads(patient_info)
            except (json.JSONDecodeError, TypeError):
                patient_info = {}
        # For simplicity, call the synchronous flow in an executor-like manner.
        # If a fully async LLM/RAG is used, this method should be adapted.
        return self.invoke(message, patient_info)


# Example usage
if __name__ == "__main__":
    agent = MedicineAssistantAgent()
    
    # Example with structured query
    query = """
    Patient: John Doe (ID: P12345)
    Age: 58, Male, Weight: 92kg
    Type 2 Diabetes, 8 years duration
    
    Latest Labs:
    - HbA1c: 8.2%
    - Blood Glucose: 165 mg/dL
    - BP: 145/88 mmHg
    - eGFR: 58 mL/min
    - LDL: 145, HDL: 38, TG: 210
    
    Current Meds: Metformin 1000mg BID, Glimepiride 4mg daily
    Recent Changes: Glimepiride increased from 2mg to 4mg 3 months ago
    Notes: Occasional morning hyperglycemia, mild peripheral neuropathy
    """
    
    result = agent.invoke(query)
    
    print("\n=== PATIENT INFO (raw) ===")
    print(result.get("patient_info", {}))

    print("\n=== PHYSICIAN REPORT ===")
    print(result["physician_report"])
    
    print("\n=== PATIENT REPORT ===")
    print(result["patient_report"])