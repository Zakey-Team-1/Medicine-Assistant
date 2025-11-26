"""LangGraph agent for the Medicine Assistant."""

from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph

from llm import get_llm
from rag import RAGComponent
from state import AgentState

# System prompt for the medicine assistant
SYSTEM_PROMPT = """You are a specialist medical assistant AI focused on diabetes care (type 1, type 2,
and related metabolic disorders). Your role is to help clinicians select appropriate
antidiabetic medications, insulin regimens, dosing adjustments, monitoring plans, and
education for patients with diabetes.

Your responsibilities:
1. Always use the RAG-retrieved context (`{context}`) when formulating recommendations —
    incorporate relevant passages, guidelines, or local protocols found in the retrieved
    documents into your answer.
2. Analyze patient information (age, weight, renal/hepatic function, comorbidities, current
    medications, pregnancy status) and tailor medication and dosing suggestions accordingly.
3. Recommend dosing ranges, titration steps, monitoring schedules (glucose, A1c, renal
    function), and when to intensify or de-escalate therapy.
4. Highlight contraindications, drug interactions, hypoglycemia risk, and special
    populations (pregnancy, pediatrics, elderly, renal impairment).
5. Cite supporting evidence from the retrieved context: for each clinical recommendation,
    include a brief citation (document title or filename and a short locator such as page
    number or paragraph) when available.

IMPORTANT: Always remind the clinician that final decisions rest with a qualified
healthcare professional. This tool assists clinical decision-making and does not replace
clinical judgment or institutional protocols.

If the RAG context is empty or does not provide direct guidance, state that explicitly,
provide evidence-based general guidance (with common-dose ranges and monitoring), and
encourage verification against authoritative guidelines. Prioritize patient safety and note
uncertainty when appropriate."""


class MedicineAssistantAgent:
    """LangGraph-based agent for medicine assistance."""

    def __init__(self, rag_component: RAGComponent | None = None):
        """
        Initialize the Medicine Assistant Agent.

        Args:
            rag_component: RAGComponent for document retrieval.
                         If None, a new instance will be created.
        """
        self.rag = rag_component or RAGComponent()
        self.llm = get_llm()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the graph with our state schema
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("retrieve", self._retrieve_context)
        workflow.add_node("analyze", self._analyze_query)
        workflow.add_node("respond", self._generate_response)

        # Set entry point
        workflow.set_entry_point("retrieve")

        # Add edges
        workflow.add_edge("retrieve", "analyze")
        workflow.add_conditional_edges(
            "analyze",
            self._should_continue,
            {
                "respond": "respond",
                "end": END,
            },
        )
        workflow.add_edge("respond", END)

        return workflow.compile()

    def _retrieve_context(self, state: AgentState) -> dict:
        """Retrieve relevant context from the RAG system."""
        # Get the last human message for retrieval
        messages = state.get("messages", [])
        query = ""

        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break

        if not query:
            return {"context": ""}

        # Retrieve relevant documents
        docs = self.rag.retrieve(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        return {"context": context}

    def _analyze_query(self, state: AgentState) -> dict:
        """Analyze the user's query to determine the type of response needed."""
        messages = state.get("messages", [])
        patient_info = state.get("patient_info", "")

        # Extract patient info if provided in the message
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, HumanMessage):
                content = last_message.content.lower()
                # Simple extraction of patient info (could be enhanced)
                if "patient" in content or "age" in content or "weight" in content:
                    patient_info = last_message.content

        return {"patient_info": patient_info}

    def _should_continue(self, state: AgentState) -> Literal["respond", "end"]:
        """Determine if we should continue to generate a response."""
        messages = state.get("messages", [])
        if not messages:
            return "end"
        return "respond"

    def _generate_response(self, state: AgentState) -> dict:
        """Generate a response using the LLM with retrieved context."""
        context = state.get("context", "No specific context available.")
        messages = state.get("messages", [])

        # Create the prompt with context
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        # Generate response
        chain = prompt | self.llm
        response = chain.invoke({"messages": messages})

        return {"messages": [response]}

    def invoke(self, message: str, patient_info: str = "") -> str:
        """
        Process a user message and return the agent's response.

        Args:
            message: The user's input message.
            patient_info: Optional patient information.

        Returns:
            The agent's response as a string.
        """
        initial_state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "context": "",
            "patient_info": patient_info,
            "recommendations": "",
        }

        result = self.graph.invoke(initial_state)

        # Extract the AI response
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage):
                return msg.content or "No response generated."

        return "I apologize, but I couldn't generate a response. Please try again."

    async def ainvoke(self, message: str, patient_info: str = "") -> str:
        """
        Asynchronously process a user message and return the agent's response.

        Args:
            message: The user's input message.
            patient_info: Optional patient information.

        Returns:
            The agent's response as a string.
        """
        initial_state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "context": "",
            "patient_info": patient_info,
            "recommendations": "",
        }

        result = await self.graph.ainvoke(initial_state)

        # Extract the AI response
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage):
                return msg.content or "No response generated."

        return "I apologize, but I couldn't generate a response. Please try again."

    def chat(self, messages: list[BaseMessage]) -> AgentState:
        """
        Process a conversation and return the full state.

        Args:
            messages: List of conversation messages.

        Returns:
            The full agent state after processing.
        """
        initial_state: AgentState = {
            "messages": messages,
            "context": "",
            "patient_info": "",
            "recommendations": "",
        }

        return self.graph.invoke(initial_state)
