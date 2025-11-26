"""OpenRouter LLM integration using LangChain."""

from langchain_openai import ChatOpenAI

from config import settings


def get_llm() -> ChatOpenAI:
    """
    Create and return a ChatOpenAI instance configured for OpenRouter.

    Returns:
        ChatOpenAI: A LangChain chat model configured to use OpenRouter API.
    """
    settings.validate()

    return ChatOpenAI(
        model=settings.MODEL_NAME,
        api_key=lambda: settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/ibrhr/Medicine-Assistant",
            "X-Title": "Medicine Assistant",
        },
        temperature=0.7,
    )
