"""Tests for the state module."""

from langchain_core.messages import AIMessage, HumanMessage

from medicine_assistant.state import AgentState


class TestAgentState:
    """Test cases for the AgentState schema."""

    def test_state_creation(self):
        """Test that AgentState can be created with required fields."""
        state: AgentState = {
            "messages": [HumanMessage(content="Hello")],
            "context": "Test context",
            "patient_info": "Patient info",
            "recommendations": "",
        }

        assert len(state["messages"]) == 1
        assert state["context"] == "Test context"
        assert state["patient_info"] == "Patient info"
        assert state["recommendations"] == ""

    def test_state_with_multiple_messages(self):
        """Test state with multiple messages."""
        state: AgentState = {
            "messages": [
                HumanMessage(content="Question 1"),
                AIMessage(content="Answer 1"),
                HumanMessage(content="Question 2"),
            ],
            "context": "",
            "patient_info": "",
            "recommendations": "",
        }

        assert len(state["messages"]) == 3
        assert isinstance(state["messages"][0], HumanMessage)
        assert isinstance(state["messages"][1], AIMessage)
