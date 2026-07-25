from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AppSettings


@dataclass(slots=True)
class RetrievedContext:
    text: str
    source: str
    score: float | None
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata,
        }


class Retriever:
    def retrieve(self, query: str, top_k: int) -> list[RetrievedContext]:
        raise NotImplementedError


class LlamaIndexKnowledgeBase(Retriever):
    """LlamaIndex ingestion/retrieval backed by Qdrant or Milvus."""

    def __init__(self, settings: AppSettings) -> None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        self.settings = settings
        self.embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model)
        self.vector_store = self._build_vector_store()
        self._index = self._load_index()

    def _build_vector_store(self) -> Any:
        if self.settings.vector_store == "milvus":
            try:
                from llama_index.vector_stores.milvus import MilvusVectorStore
            except ImportError as exc:
                raise RuntimeError(
                    "Milvus support is not installed. Run: pip install '.[milvus]'"
                ) from exc
            return MilvusVectorStore(
                uri=self.settings.milvus_uri,
                token=self.settings.milvus_token or "",
                collection_name=self.settings.collection_name,
                dim=self.settings.embedding_dimension,
                overwrite=False,
            )

        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            timeout=30,
        )
        return QdrantVectorStore(
            client=client,
            collection_name=self.settings.collection_name,
        )

    def _load_index(self) -> Any:
        from llama_index.core import VectorStoreIndex

        return VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model,
        )

    def ingest_paths(self, paths: Iterable[Path]) -> int:
        from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
        from llama_index.core.node_parser import SentenceSplitter

        file_paths = [str(path) for path in paths]
        if not file_paths:
            return 0
        documents = SimpleDirectoryReader(input_files=file_paths).load_data()
        splitter = SentenceSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        storage = StorageContext.from_defaults(vector_store=self.vector_store)
        self._index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage,
            embed_model=self.embed_model,
            transformations=[splitter],
            show_progress=True,
        )
        return len(documents)

    def ingest_directory(self, directory: Path) -> int:
        supported = {".txt", ".md", ".pdf", ".docx", ".csv"}
        paths = [
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in supported
        ]
        return self.ingest_paths(paths)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedContext]:
        retriever = self._index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        contexts: list[RetrievedContext] = []
        for item in nodes:
            score = float(item.score) if item.score is not None else None
            if score is not None and score < self.settings.min_relevance_score:
                continue
            metadata = dict(item.node.metadata or {})
            source = str(
                metadata.get("file_name")
                or metadata.get("filename")
                or metadata.get("source")
                or item.node.node_id
            )
            contexts.append(
                RetrievedContext(
                    text=item.node.get_content(),
                    source=source,
                    score=score,
                    metadata=metadata,
                )
            )
        return contexts
