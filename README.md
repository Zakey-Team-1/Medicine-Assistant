# Medicine Assistant

Concise, developer-focused README for the Medicine Assistant project.

## What
An AI agent that helps clinicians pick medicines and dosages using a LangGraph agent and a RAG system backed by ChromaDB and OpenRouter LLMs.

## Quick Start (recommended: conda)
1. Create and activate the conda env:

```bash
conda create -n medicine-assistant python=3.10 -y
conda activate medicine-assistant
```

2. Install project and dev dependencies:

```bash
pip install -e .
pip install -e ".[dev]"
```

3. Set all values in .env.example:

### Configuration
- Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`.
- Key settings are in `src/config.py` (model name, chunk size, top-k).

4. Run the project!

```bash
python src/web/app.py
```

## Project layout (actual)

```
Medicine-Assistant/
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── config.py
│   ├── llm.py
│   ├── main.py
│   ├── rag.py
│   └── state.py
├── tests/
├── pyproject.toml
├── README.md
└── LICENSE
```

## License
MIT
