from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile

from .config import get_settings
from .schemas import HealthResponse, IngestResponse, QueryRequest, QueryResponse
from .service import get_service

settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Fully self-hosted LangGraph + LlamaIndex reasoning and RAG API.",
)

ALLOWED_SUFFIXES = {".txt", ".md", ".pdf", ".docx", ".csv"}
MAX_FILE_BYTES = 25 * 1024 * 1024


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok", model=settings.ollama_model, vector_store=settings.vector_store
    )


@app.post("/v1/query", response_model=QueryResponse, tags=["reasoning"])
def query(request: QueryRequest) -> QueryResponse:
    try:
        return get_service().engine.query(
            request.question,
            top_k=request.top_k,
            max_retries=request.max_retries,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Reasoning service unavailable: {exc}"
        ) from exc


@app.post("/v1/documents", response_model=IngestResponse, tags=["knowledge"])
def ingest_documents(files: Annotated[list[UploadFile], File()]) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    saved: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(prefix="reasoning-engine-") as temp_dir:
            root = Path(temp_dir)
            for upload in files:
                safe_name = Path(upload.filename or "document.txt").name
                suffix = Path(safe_name).suffix.lower()
                if suffix not in ALLOWED_SUFFIXES:
                    raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix}")
                target = root / safe_name
                with target.open("wb") as output:
                    shutil.copyfileobj(upload.file, output)
                if target.stat().st_size > MAX_FILE_BYTES:
                    raise HTTPException(status_code=413, detail=f"File too large: {safe_name}")
                saved.append(target)
            count = get_service().knowledge_base.ingest_paths(saved)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ingestion failed: {exc}") from exc
    finally:
        for upload in files:
            upload.file.close()

    return IngestResponse(
        files=len(saved), documents=count, collection=settings.collection_name
    )
