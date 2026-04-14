# Celeste + Idun RAG agent

A RAG pipeline that answers questions about internal HR documents (French PDFs). Uses [Celeste AI](https://docs.withceleste.com) for generation and embeddings, [LangGraph](https://langchain-ai.github.io/langgraph/) for orchestration, and [Idun Agent Platform](https://cloud.idunplatform.com) to serve and manage the agent.

## How it works

```
rewrite_query → retrieve → generate
```

1. Rewrites the user's question into a vector-search-friendly query via Celeste (`gemini-3-flash-preview`)
2. Searches an in-memory vector store built from the PDFs using batch Celeste embeddings (`gemini-embedding-2-preview`)
3. Generates a grounded answer from the retrieved document chunks

## Project structure

```
main.py              # Graph nodes and wiring
state.py             # InputState, GraphState, OutputState
celeste_adapter.py   # CelesteEmbeddings adapter + call_celeste wrapper
docs/                # HR PDF documents (NexaTech)
docs/blog/           # Blog post
tests/               # Lightweight adapter tests
```

## Setup

```bash
uv sync
```

Set your API keys in `.env`:

```
GOOGLE_API_KEY=...
```

The Celeste calls use model-based provider inference, so no provider-specific SDK setup is needed beyond the API key.

## Run

```bash
idun agent serve --source=manager
```

The agent fetches its config from [Idun Cloud](https://cloud.idunplatform.com) and starts on `http://localhost:8800`.

## Links

- [Celeste AI docs](https://docs.withceleste.com)
- [Celeste AI PyPI](https://pypi.org/project/celeste-ai/)
- [Idun Agent Platform](https://github.com/Idun-Group/idun-agent-platform)
- [Idun Cloud](https://cloud.idunplatform.com)
- [LangGraph docs](https://langchain-ai.github.io/langgraph/)
