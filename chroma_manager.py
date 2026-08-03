"""ChromaDB long-term memory manager."""
from __future__ import annotations

import time
import uuid
from typing import Any

import chromadb

from config import config
from models.embeddings import Embedder
from utils.logger import get_logger

logger = get_logger(__name__)


class ChromaManager:
    """Persist and retrieve metadata and conversation summaries.

    Note:
        Raw datasets are never stored — only textual summaries,
        schemas, column descriptions, and conversation summaries.
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        """Initialize the persistent client and collection.

        Args:
            embedder: Embedding provider (defaults to a new Embedder).
        """
        self.embedder = embedder or Embedder()
        self.client = chromadb.PersistentClient(path=config.chroma_db_path)
        self.collection = self.client.get_or_create_collection(
            name="analyst_memory"
        )

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a memory entry.

        Args:
            text: Content to store and embed.
            metadata: Optional metadata (e.g. ``{"type": "schema"}``).
        """
        if not text.strip():
            return
        try:
            embedding = self.embedder.embed([text])[0]
            self.collection.add(
                ids=[str(uuid.uuid4())],
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata or {"type": "note", "ts": time.time()}],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Chroma add failed: %s", exc)

    def query(self, text: str, n_results: int = 4) -> list[str]:
        """Retrieve the most relevant stored documents.

        Args:
            text: Query text.
            n_results: Number of documents to return.

        Returns:
            List of matching document strings (possibly empty).
        """
        try:
            if self.collection.count() == 0:
                return []
            embedding = self.embedder.embed([text])[0]
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=min(n_results, self.collection.count()),
            )
            docs = results.get("documents") or [[]]
            return docs[0]
        except Exception as exc:  # noqa: BLE001
            logger.error("Chroma query failed: %s", exc)
            return []

    def reset(self) -> None:
        """Delete and recreate the memory collection."""
        try:
            self.client.delete_collection("analyst_memory")
        except Exception:  # noqa: BLE001
            pass
        self.collection = self.client.get_or_create_collection(
            name="analyst_memory"
        )
