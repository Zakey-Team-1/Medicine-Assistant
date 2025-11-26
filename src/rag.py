"""RAG (Retrieval Augmented Generation) components for the Medicine Assistant."""

from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from config import settings


class RAGComponent:
    """Handles retrieval from Chroma vector store for RAG."""

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the RAG component for retrieval only.

        Args:
            persist_directory: Directory where the vector store is persisted.
                             Defaults to settings.CHROMA_PERSIST_DIRECTORY.
        """
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
        # Using OpenRouter (OpenAI-compatible) embeddings via LangChain
        # Model: `openai/text-embedding-3-small`
        settings.validate()
        self.embeddings = OpenAIEmbeddings(
            model="openai/text-embedding-3-small",
            api_key=lambda: settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
        )
        self._vector_store: Optional[Chroma] = None

    @property
    def vector_store(self) -> Chroma:
        """Get or create the vector store."""
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=settings.COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
        return self._vector_store

    def get_retriever(self) -> VectorStoreRetriever:
        """
        Get a retriever for the vector store.

        Returns:
            VectorStoreRetriever configured with the current settings.
        """
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.TOP_K_RESULTS},
        )

    def retrieve(self, query: str) -> list[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The search query.

        Returns:
            List of relevant documents.
        """
        return self.vector_store.similarity_search(
            query,
            k=settings.TOP_K_RESULTS,
        )
