"""Retrieval-augmented context from ChromaDB."""
from __future__ import annotations

from database.chroma_manager import ChromaManager


class RAGTool:
    """Fetch and store long-term memory for the agent."""

    def __init__(self, chroma: ChromaManager) -> None:
        """Initialize with a Chroma manager.

        Args:
            chroma: Long-term memory manager.
        """
        self.chroma = chroma

    def retrieve(self, question: str, n_results: int = 4) -> str:
        """Retrieve relevant memory as a single context block.

        Args:
            question: The user's question.
            n_results: Max documents to retrieve.

        Returns:
            Newline-joined context string (empty if nothing found).
        """
        docs = self.chroma.query(question, n_results=n_results)
        return "\n".join(f"- {d}" for d in docs)

    def store_dataset_metadata(self, name: str, summary_text: str) -> None:
        """Persist a dataset summary."""
        self.chroma.add(summary_text, {"type": "dataset_summary", "name": name})

    def store_schema(self, schema_text: str) -> None:
        """Persist a database schema description."""
        self.chroma.add(schema_text, {"type": "schema"})

    def store_conversation(self, summary: str) -> None:
        """Persist a conversation/analysis summary."""
        self.chroma.add(summary, {"type": "conversation"})
