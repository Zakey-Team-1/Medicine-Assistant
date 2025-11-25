"""OpenRouter LLM integration using LangChain."""

from langchain_openai import ChatOpenAI

from medicine_assistant.config import settings


def get_llm() -> ChatOpenAI:
    """
    Create and return a ChatOpenAI instance configured for OpenRouter.

    Returns:
        ChatOpenAI: A LangChain chat model configured to use OpenRouter API.
    """
    settings.validate()

    return ChatOpenAI(
        model=settings.MODEL_NAME,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base=settings.OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/ibrhr/Medicine-Assistant",
            "X-Title": "Medicine Assistant",
        },
        temperature=0.7,
    )
