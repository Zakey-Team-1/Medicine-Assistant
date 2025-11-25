"""Main entry point for the Medicine Assistant."""

import argparse
import sys

from agent import MedicineAssistantAgent
from .config import settings
from rag import RAGComponent


def create_agent() -> MedicineAssistantAgent:
    """Create and return a configured Medicine Assistant agent."""
    rag = RAGComponent()
    return MedicineAssistantAgent(rag_component=rag)


def run_interactive(agent: MedicineAssistantAgent) -> None:
    """Run the agent in interactive mode."""
    print("=" * 60)
    print("Medicine Assistant - Interactive Mode")
    print("=" * 60)
    print("\nI'm your medical assistant. I can help you:")
    print("- Find appropriate medications for patient conditions")
    print("- Recommend dosages based on patient information")
    print("- Identify potential drug interactions")
    print("\nType 'quit' or 'exit' to end the session.")
    print("Type 'ingest <path>' to add documents to the knowledge base.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nDoctor: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                print("\nThank you for using Medicine Assistant. Goodbye!")
                break

            # Handle document ingestion command
            if user_input.lower().startswith("ingest "):
                path = user_input[7:].strip()
                try:
                    count = agent.rag.ingest_documents(path)
                    print(f"\n✓ Successfully ingested {count} document chunks from: {path}")
                except Exception as e:
                    print(f"\n✗ Error ingesting documents: {e}")
                continue

            # Get agent response
            response = agent.invoke(user_input)
            print(f"\nAssistant: {response}")

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or type 'quit' to exit.")


def main() -> None:
    """Main entry point for the Medicine Assistant CLI."""
    parser = argparse.ArgumentParser(
        description="Medicine Assistant - An AI agent to help doctors with medication decisions"
    )
    parser.add_argument(
        "--ingest",
        type=str,
        help="Path to documents to ingest into the knowledge base",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to process (non-interactive mode)",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit",
    )

    args = parser.parse_args()

    # Validate configuration
    if args.validate_config:
        try:
            settings.validate()
            print("✓ Configuration is valid")
            sys.exit(0)
        except ValueError as e:
            print(f"✗ Configuration error: {e}")
            sys.exit(1)

    # Validate API key before proceeding
    try:
        settings.validate()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Create the agent
    agent = create_agent()

    # Ingest documents if specified
    if args.ingest:
        try:
            count = agent.rag.ingest_documents(args.ingest)
            print(f"✓ Successfully ingested {count} document chunks from: {args.ingest}")
        except Exception as e:
            print(f"✗ Error ingesting documents: {e}")
            sys.exit(1)

    # Process single query or run interactive mode
    if args.query:
        response = agent.invoke(args.query)
        print(response)
    else:
        run_interactive(agent)


if __name__ == "__main__":
    main()
