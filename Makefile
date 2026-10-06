# Local development targets for the LangGraph ICD-10 coding agent.

VENV := .venv

# Embedding backend.
RAG_EMBED_BASE_URL ?= http://localhost:11434/v1
RAG_EMBED_MODEL ?= mxbai-embed-large

.PHONY: install test

install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"

test:
	$(VENV)/bin/python -m pytest
