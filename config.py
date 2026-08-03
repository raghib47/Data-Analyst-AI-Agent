"""Central configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    """Immutable application configuration.

    Attributes:
        hf_token: Hugging Face API token.
        chat_model: Chat/instruct model repo id.
        embedding_model: Embedding model repo id.
        chroma_db_path: Directory for the persistent ChromaDB store.
        data_dir: Directory for uploaded / sample data.
        log_dir: Directory for log files.
    """

    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    chat_model: str = field(
        default_factory=lambda: os.getenv(
            "CHAT_MODEL", "meta-llama/Llama-3.1-8B-Instruct"
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "google/embeddinggemma-300m"
        )
    )
    chroma_db_path: str = field(
        default_factory=lambda: os.getenv("CHROMA_DB_PATH", "./chroma_db")
    )
    data_dir: str = "./data"
    log_dir: str = "./logs"

    def validate(self) -> list[str]:
        """Return a list of human-readable configuration problems."""
        problems: list[str] = []
        if not self.hf_token:
            problems.append("HF_TOKEN is not set. Add it to your .env file.")
        return problems


config = Config()
