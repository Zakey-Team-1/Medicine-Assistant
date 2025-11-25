"""Configuration settings for the Medicine Assistant."""

import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # OpenRouter API configuration
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # Model configuration - using a capable model from OpenRouter
    MODEL_NAME: str = os.getenv("MODEL_NAME", "openai/gpt-5-mini")

    # RAG configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

    # Vector store configuration
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "medicine_docs")

    @classmethod
    def validate(cls) -> None:
        """Validate that required settings are configured."""
        if not cls.OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required. "
                "Please set it in your .env file or environment."
            )


settings = Settings()
