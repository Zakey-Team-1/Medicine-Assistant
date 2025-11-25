"""Tests for the RAG module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from medicine_assistant.rag import RAGComponent


class TestRAGComponent:
    """Test cases for the RAGComponent class."""

    @pytest.fixture
    def mock_embeddings(self):
        """Create a mock embeddings model."""
        mock = MagicMock()
        mock.embed_documents.return_value = [[0.1] * 384]  # Mock 384-dim vectors
        mock.embed_query.return_value = [0.1] * 384
        return mock

    @pytest.fixture
    def rag_component(self, mock_embeddings):
        """Create a RAG component with mocked embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "medicine_assistant.rag.HuggingFaceEmbeddings",
                return_value=mock_embeddings,
            ):
                yield RAGComponent(persist_directory=tmpdir)

    @pytest.fixture
    def sample_documents_dir(self):
        """Create a temporary directory with sample documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a sample text file
            sample_file = Path(tmpdir) / "medicine_info.txt"
            sample_file.write_text(
                "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID). "
                "It is used for treating pain, fever, and inflammation. "
                "Common dosage for adults is 200-400mg every 4-6 hours."
            )
            yield tmpdir

    def test_initialization(self, rag_component):
        """Test that RAGComponent initializes correctly."""
        assert rag_component.text_splitter is not None
        assert rag_component.embeddings is not None

    def test_load_documents_from_file(self, rag_component, sample_documents_dir):
        """Test loading documents from a single file."""
        file_path = Path(sample_documents_dir) / "medicine_info.txt"
        docs = rag_component.load_documents(str(file_path))

        assert len(docs) > 0
        assert "Ibuprofen" in docs[0].page_content

    def test_load_documents_from_directory(self, rag_component, sample_documents_dir):
        """Test loading documents from a directory."""
        docs = rag_component.load_documents(sample_documents_dir)

        assert len(docs) > 0

    def test_load_documents_nonexistent(self, rag_component):
        """Test loading documents from nonexistent path."""
        docs = rag_component.load_documents("/nonexistent/path")
        assert docs == []

    @pytest.mark.skip(reason="Requires vector store with real embeddings")
    def test_ingest_documents(self, rag_component, sample_documents_dir):
        """Test ingesting documents into the vector store."""
        count = rag_component.ingest_documents(sample_documents_dir)
        assert count > 0

    @pytest.mark.skip(reason="Requires vector store with real embeddings")
    def test_retrieve_after_ingest(self, rag_component, sample_documents_dir):
        """Test retrieval after document ingestion."""
        rag_component.ingest_documents(sample_documents_dir)
        results = rag_component.retrieve("What is Ibuprofen?")

        assert len(results) > 0
