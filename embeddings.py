"""Embedding wrapper around the Hugging Face Inference API."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from huggingface_hub import InferenceClient

from config import config
from utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """Generate sentence embeddings via the HF feature-extraction API."""

    def __init__(
        self,
        model: str | None = None,
        token: str | None = None,
    ) -> None:
        """Initialize the embedding client.

        Args:
            model: Embedding model repo id (defaults to config).
            token: HF token (defaults to config).
        """
        self.model = model or config.embedding_model
        self._client = InferenceClient(token=token or config.hf_token)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors. On failure returns zero vectors.
        """
        vectors: list[list[float]] = []
        for text in texts:
            try:
                raw = self._client.feature_extraction(
                    text, model=self.model
                )
                arr = np.asarray(raw, dtype=float)
                # Mean-pool if the model returns token-level embeddings.
                if arr.ndim == 2:
                    arr = arr.mean(axis=0)
                vectors.append(arr.astype(float).tolist())
            except Exception as exc:  # noqa: BLE001
                logger.error("Embedding failed: %s", exc)
                vectors.append([0.0] * 768)
        return vectors
