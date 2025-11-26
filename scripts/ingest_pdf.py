#!/usr/bin/env python
"""Script to ingest PDF files into Chroma vector store using OpenRouter embeddings."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "medicine_docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


def validate_config() -> None:
    """Validate that required configuration is set."""
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is required. "
            "Please set it in your .env file or environment."
        )


def get_embeddings() -> OpenAIEmbeddings:
    """Create and return OpenAI embeddings configured for OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set")
    
    return OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=lambda: OPENROUTER_API_KEY,  # type: ignore
        base_url=OPENROUTER_BASE_URL,
    )


def get_vector_store(embeddings: OpenAIEmbeddings) -> Chroma:
    """Get or create the Chroma vector store."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
    )


def ingest_pdf(pdf_path: str, persist_directory: str | None = None) -> None:
    """
    Ingest a single PDF file into the vector store.

    Args:
        pdf_path: Path to the PDF file to ingest.
        persist_directory: Optional custom Chroma persist directory.

    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
        ValueError: If the file is not a PDF.
    """
    pdf_file = Path(pdf_path)

    # Validate file exists
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Validate it's a PDF
    if pdf_file.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")

    print(f"📄 Loading PDF: {pdf_file.name}")

    # Setup embeddings and vector store
    embeddings = get_embeddings()
    db_path = persist_directory or CHROMA_PERSIST_DIRECTORY
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=db_path,
    )

    # Load PDF documents
    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()
    print(f"  ✓ Loaded {len(documents)} pages")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    splits = text_splitter.split_documents(documents)
    print(f"  ✓ Split into {len(splits)} chunks")

    # Add to vector store
    vector_store.add_documents(splits)
    print(f"✅ Successfully ingested {pdf_file.name}")
    print(f"📊 Created {len(splits)} document chunks")
    print(f"💾 Persisted to: {db_path}")


def ingest_directory(
    directory_path: str, persist_directory: str | None = None
) -> None:
    """
    Ingest all PDF files from a directory into the vector store.

    Args:
        directory_path: Path to the directory containing PDF files.
        persist_directory: Optional custom Chroma persist directory.

    Raises:
        FileNotFoundError: If the directory doesn't exist.
        ValueError: If the directory contains no PDF files.
    """
    directory = Path(directory_path)

    # Validate directory exists
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    # Find all PDF files
    pdf_files = list(directory.glob("**/*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {directory_path}")

    print(f"📁 Found {len(pdf_files)} PDF file(s)")

    # Setup embeddings and vector store
    embeddings = get_embeddings()
    db_path = persist_directory or CHROMA_PERSIST_DIRECTORY
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=db_path,
    )

    # Setup text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    total_chunks = 0

    for pdf_file in pdf_files:
        print(f"\n📄 Processing: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()
        print(f"  ✓ Loaded {len(documents)} pages")

        splits = text_splitter.split_documents(documents)
        vector_store.add_documents(splits)
        total_chunks += len(splits)
        print(f"  ✅ Created {len(splits)} chunks")

    print(f"\n📊 Total chunks created: {total_chunks}")
    print(f"💾 Persisted to: {db_path}")


def main() -> None:
    """Main entry point for the ingest script."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF files into Chroma vector store using OpenRouter embeddings"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a PDF file or directory containing PDF files",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help=f"Custom Chroma database path (default: {CHROMA_PERSIST_DIRECTORY})",
    )

    args = parser.parse_args()

    try:
        validate_config()
        path = Path(args.path)

        if path.is_file():
            ingest_pdf(str(path), persist_directory=args.db_path)
        elif path.is_dir():
            ingest_directory(str(path), persist_directory=args.db_path)
        else:
            print(f"❌ Error: Path does not exist: {args.path}", file=sys.stderr)
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
