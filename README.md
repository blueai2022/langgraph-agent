# LangGraph Agent

The agent calls RAG-based retrieval as a *tool*, so it does not make a mistake due to poor
"memory" — a non-existent ICD-10 code.

The agent loop and tool execution are provided by LangGraph (`StateGraph` + `ToolNode` +
`ChatOpenAI`); the RAG layer is exposed as a LangChain `BaseRetriever`.

## Requirements

- Python 3.12+
- An OpenAI-compatible endpoint with tool-calling support (chat) and an OpenAI-compatible
  embeddings endpoint.

## Running it

**Local, via Ollama** (no API key; models must support tools/embeddings):

```bash
ollama pull llama3.1:8b        # tool-calling capable model
ollama pull mxbai-embed-large  # embedding model
```
