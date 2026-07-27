# Open Reasoning Engine

A fully open-source Reasoning Engine that you can run on your own machine or server:

```text
FastAPI -> LangGraph reasoning workflow -> LlamaIndex RAG -> Qdrant / Milvus
                                   \-> Ollama (Qwen / Llama)
```

The default stack is **LangGraph + LlamaIndex + Qdrant + Ollama/Qwen**. Documents, vectors, and
model inference stay on your infrastructure, with no dependency on a closed model API.

## More than basic RAG

Each query passes through the following workflow:

1. Analyze the question and generate a standalone retrieval query.
2. Retrieve evidence from the vector database.
3. Generate an answer grounded in the evidence with `[1]`-style citations.
4. Validate that the evidence supports the answer.
5. If validation fails, rewrite the retrieval query and retry once.
6. Return the answer, sources, relevance scores, and observable reasoning-step summaries.

## Quick start

Requirements: Docker Desktop or Docker Engine and at least 8 GB of memory. The default 7B model
can run on CPU-only systems, although inference will be slower.

```bash
cp .env.example .env
docker compose up -d --build
```

The first startup downloads the Qwen model and the English embedding model, which can take a few
minutes. Once the services are ready, open:

- API documentation: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/health>
- Qdrant dashboard: <http://localhost:6333/dashboard>

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
docker compose up -d --build
```

## Ingest documents

The API accepts `.txt`, `.md`, `.pdf`, `.docx`, and `.csv` files. The default per-file
limit is 25 MB.

```bash
curl -X POST http://localhost:8000/v1/documents \
  -F "files=@data/example.md"
```

You can also ingest a directory with the CLI from a local Python environment:

```bash
pip install -e .
reasoning-engine ingest ./data
```

## Run a reasoning query

```bash
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the reasoning workflow used by this system?"}'
```

Example response:

```json
{
  "answer": "The system analyzes the question and retrieves evidence before answering. [1]",
  "rewritten_question": "Complete Open Reasoning Engine workflow",
  "citations": [
    {
      "index": 1,
      "source": "example.md",
      "text": "The reasoning workflow has five stages...",
      "score": 0.82,
      "metadata": {}
    }
  ],
  "reasoning_steps": [
    "Analyzed the question and created a retrieval plan",
    "Retrieved evidence from the vector knowledge base",
    "Generated a cited answer from the evidence",
    "Validated the answer against the evidence",
    "Produced the final answer"
  ],
  "retries": 0
}
```

CLI query:

```bash
reasoning-engine ask "Which vector database does this system use by default?"
```

## Change the model

Edit `.env`:

```dotenv
OLLAMA_MODEL=qwen3:8b
```

Then pull the model and restart the API:

```bash
docker compose run --rm model-pull
docker compose restart api
```

You can also use Llama, a DeepSeek-R1 distilled model, or another model available through Ollama.
The model name must match a tag installed in your local Ollama instance.

## Switch to Milvus

Milvus is better suited to large datasets and horizontally scalable deployments. The Compose
override automatically installs the optional Milvus dependency in the API image:

```bash
pip install -e ".[milvus]"
docker compose -f docker-compose.yml -f docker-compose.milvus.yml up -d
```

Qdrant remains the recommended default for everyday single-node use.

## Local development

Python 3.11-3.13:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cp .env.example .env
pytest
ruff check .
uvicorn reasoning_engine.api:app --reload
```

When running the API directly on the host, update the service URLs in `.env`:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
QDRANT_URL=http://localhost:6333
```

## Main configuration

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Generation and validation model |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local English embedding model |
| `VECTOR_STORE` | `qdrant` | `qdrant` or `milvus` |
| `SIMILARITY_TOP_K` | `5` | Number of evidence chunks per retrieval |
| `MAX_RETRIES` | `1` | Maximum retries after failed validation |
| `CHUNK_SIZE` | `700` | Document chunk size |
| `CHUNK_OVERLAP` | `100` | Overlap between adjacent chunks |

## Production deployment notes

- Put the API behind a reverse proxy with authentication, TLS, rate limiting, and request-size
  limits.
- Do not expose Qdrant, Milvus, or Ollama ports to the public internet. The default Compose
  configuration binds them to the local host only.
- Snapshot and back up vector database volumes, and pin image versions that you have validated.
- Monitor model latency, unsupported-answer rates, retrieval hit rates, and retry counts.
- When changing the embedding model or vector dimensions, create a new collection and ingest the
  documents again to avoid dimension conflicts.

## Project structure

```text
src/reasoning_engine/
|-- api.py          # FastAPI interface
|-- engine.py       # LangGraph reasoning and correction workflow
|-- retrieval.py    # LlamaIndex + Qdrant/Milvus
|-- llm.py          # Ollama model adapter
|-- config.py       # Environment configuration
`-- cli.py          # Ingestion and query CLI
```

## License

MIT
