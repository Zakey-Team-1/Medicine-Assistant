"""LangGraph agent for the Medicine Assistant."""

from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph

from medicine_assistant.llm import get_llm
from medicine_assistant.rag import RAGComponent
from medicine_assistant.state import AgentState

# System prompt for the medicine assistant
SYSTEM_PROMPT = """You are a knowledgeable medical assistant AI designed to help doctors
select appropriate medications and dosages for their patients. You have access to a
comprehensive database of medicines, their indications, contraindications, and dosing guidelines.

Your responsibilities:
1. Analyze patient information and symptoms
2. Suggest appropriate medications based on the available context
3. Recommend proper dosages based on patient characteristics (age, weight, conditions)
4. Highlight potential drug interactions and contraindications
5. Provide clear, professional medical reports

IMPORTANT: Always remind the doctor that final decisions should be made by qualified
healthcare professionals. This tool is meant to assist, not replace, clinical judgment.

Use the following context from the medical database to inform your recommendations:
{context}

If the context doesn't contain relevant information, acknowledge this and provide
general guidance based on your training, while emphasizing the need for verification
with authoritative sources."""


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
                return msg.content

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
                return msg.content

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
