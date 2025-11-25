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

3. Set OpenRouter API key (required for integration tests and live LLM calls):

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key"
```

4. Run tests:

```bash
# from repository root
python -m pytest -q
```

## Configuration
- Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`.
- Key settings are in `src/config.py` (model name, chunk size, top-k).

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

## Notes
- Tests import from the `src` package (run tests with `python -m pytest` or `PYTHONPATH=src pytest`).
- Integration tests that call OpenRouter are skipped unless `OPENROUTER_API_KEY` is set.

## License
MIT
