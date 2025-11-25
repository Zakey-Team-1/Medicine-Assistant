"""RAG (Retrieval Augmented Generation) components for the Medicine Assistant."""

from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings


class RAGComponent:
    """Handles document loading, embedding, and retrieval for RAG."""

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the RAG component.

        Args:
            persist_directory: Directory to persist the vector store.
                             Defaults to settings.CHROMA_PERSIST_DIRECTORY.
        """
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
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

    def load_documents(self, documents_path: str) -> list[Document]:
        """
        Load documents from a directory or file.

        Args:
            documents_path: Path to the documents directory or file.

        Returns:
            List of loaded documents.
        """
        path = Path(documents_path)

        if path.is_file():
            if path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path))
            else:
                loader = TextLoader(str(path))
            return loader.load()

        if path.is_dir():
            documents = []
            # Load text files
            txt_loader = DirectoryLoader(
                str(path),
                glob="**/*.txt",
                loader_cls=TextLoader,
                silent_errors=True,
            )
            documents.extend(txt_loader.load())

            # Load PDF files
            pdf_loader = DirectoryLoader(
                str(path),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                silent_errors=True,
            )
            documents.extend(pdf_loader.load())

            return documents

        return []

    def add_documents(self, documents: list[Document]) -> None:
        """
        Split and add documents to the vector store.

        Args:
            documents: List of documents to add.
        """
        if not documents:
            return

        splits = self.text_splitter.split_documents(documents)
        self.vector_store.add_documents(splits)

    def ingest_documents(self, documents_path: str) -> int:
        """
        Load and ingest documents from a path into the vector store.

        Args:
            documents_path: Path to the documents directory or file.

        Returns:
            Number of document chunks added.
        """
        documents = self.load_documents(documents_path)
        if not documents:
            return 0

        splits = self.text_splitter.split_documents(documents)
        self.vector_store.add_documents(splits)
        return len(splits)

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

    def clear(self) -> None:
        """Clear all documents from the vector store."""
        if self._vector_store is not None:
            try:
                # Delete the existing collection to clear all data
                self._vector_store.delete_collection()
            except Exception:
                pass  # Collection may not exist
            self._vector_store = None

        # Reinitialize empty vector store
        self._vector_store = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
