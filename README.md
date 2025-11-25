# Medicine-Assistant

Medicine Assistant is an AI Agent that assists doctors in picking the correct medicine and dosage for patients, while producing comprehensive reports about medications. Built using **LangGraph** for agent orchestration and **OpenRouter** for LLM access.

## Features

- 🤖 **LangGraph Agent**: Structured workflow for processing medical queries
- 📚 **RAG (Retrieval Augmented Generation)**: Ingests and retrieves from medical documents
- 🔗 **OpenRouter Integration**: Access to various LLM models via OpenRouter API
- 💊 **Medicine Recommendations**: Suggests appropriate medications based on patient conditions
- 📋 **Dosage Guidelines**: Provides dosage recommendations based on patient characteristics
- ⚠️ **Interaction Warnings**: Highlights potential drug interactions and contraindications

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Medicine Assistant Agent                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Retrieve   │ -> │   Analyze    │ -> │   Respond    │  │
│  │   Context    │    │    Query     │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                                       │           │
│         v                                       v           │
│  ┌──────────────┐                      ┌──────────────┐    │
│  │  RAG System  │                      │   OpenRouter │    │
│  │  (ChromaDB)  │                      │     LLM      │    │
│  └──────────────┘                      └──────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10 or higher
- An OpenRouter API key ([Get one here](https://openrouter.ai/keys))

### Install from source

```bash
# Clone the repository
git clone https://github.com/ibrhr/Medicine-Assistant.git
cd Medicine-Assistant

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# For development
pip install -e ".[dev]"
```

## Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your OpenRouter API key:
   ```env
   OPENROUTER_API_KEY=your_api_key_here
   ```

### Available Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key (required) | - |
| `MODEL_NAME` | LLM model to use | `openai/gpt-4o-mini` |
| `CHUNK_SIZE` | Document chunk size for RAG | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `TOP_K_RESULTS` | Number of documents to retrieve | `5` |
| `CHROMA_PERSIST_DIRECTORY` | Vector store location | `./chroma_db` |
| `COLLECTION_NAME` | ChromaDB collection name | `medicine_docs` |

## Usage

### Interactive Mode

```bash
# Start the interactive assistant
medicine-assistant

# Or run directly
python -m medicine_assistant.main
```

### Command Line Options

```bash
# Ingest documents into the knowledge base
medicine-assistant --ingest /path/to/medical/documents

# Process a single query
medicine-assistant --query "What medication should I consider for a patient with hypertension?"

# Validate configuration
medicine-assistant --validate-config
```

### Python API

```python
from medicine_assistant.agent import MedicineAssistantAgent
from medicine_assistant.rag import RAGComponent

# Create the agent
rag = RAGComponent()
agent = MedicineAssistantAgent(rag_component=rag)

# Ingest medical documents (optional)
rag.ingest_documents("/path/to/medical/documents")

# Get a recommendation
response = agent.invoke(
    "Patient is a 45-year-old male with type 2 diabetes and hypertension. "
    "What medication would you recommend for blood pressure control?"
)
print(response)
```

### Async Support

```python
import asyncio
from medicine_assistant.agent import MedicineAssistantAgent

async def main():
    agent = MedicineAssistantAgent()
    response = await agent.ainvoke(
        "What are the common side effects of metformin?"
    )
    print(response)

asyncio.run(main())
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=medicine_assistant
```

### Linting

```bash
# Run ruff linter
ruff check src tests

# Auto-fix issues
ruff check --fix src tests
```

## Project Structure

```
Medicine-Assistant/
├── src/
│   └── medicine_assistant/
│       ├── __init__.py      # Package initialization
│       ├── agent.py         # LangGraph agent implementation
│       ├── config.py        # Configuration settings
│       ├── llm.py           # OpenRouter LLM integration
│       ├── main.py          # CLI entry point
│       ├── rag.py           # RAG components
│       └── state.py         # Agent state definitions
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_rag.py
│   └── test_state.py
├── .env.example             # Example environment configuration
├── pyproject.toml           # Project configuration
├── LICENSE
└── README.md
```

## Supported Models

The agent uses OpenRouter to access various LLM models. Some recommended models:

- `openai/gpt-4o-mini` (default) - Good balance of cost and performance
- `openai/gpt-4o` - Higher capability for complex medical queries
- `anthropic/claude-3-sonnet` - Strong reasoning capabilities
- `google/gemini-pro` - Alternative option

See [OpenRouter Models](https://openrouter.ai/models) for the full list.

## ⚠️ Important Disclaimer

This tool is designed to **assist** healthcare professionals, not replace clinical judgment. All recommendations should be verified by qualified medical professionals before making treatment decisions. The AI may make errors and should never be used as the sole basis for medical decisions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
