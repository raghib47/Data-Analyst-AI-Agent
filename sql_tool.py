"""Natural-language-to-SQL generation and execution."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from database.sqlite_manager import SQLiteManager
from models.llm import ChatLLM
from utils.logger import get_logger

logger = get_logger(__name__)

_SQL_SYSTEM = (
    "You are an expert SQLite analyst. Given a schema and a question, "
    "return ONLY a single valid SQLite SELECT query. No explanation, "
    "no markdown fences. Never modify data."
)


@dataclass
class SQLResult:
    """Result of an NL-to-SQL run.

    Attributes:
        sql: The generated SQL.
        dataframe: Query result, if successful.
        error: Error message, if any.
    """

    sql: str
    dataframe: pd.DataFrame | None = None
    error: str = ""


class SQLTool:
    """Generate and execute SQL from natural language."""

    def __init__(self, manager: SQLiteManager, llm: ChatLLM | None = None) -> None:
        """Initialize the tool.

        Args:
            manager: Connected SQLite manager.
            llm: Chat model (defaults to a new ChatLLM).
        """
        self.manager = manager
        self.llm = llm or ChatLLM()

    @staticmethod
    def _clean_sql(text: str) -> str:
        """Strip fences/prose and keep the SELECT statement."""
        match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL)
        sql = (match.group(1) if match else text).strip()
        # Keep only the first statement.
        return sql.split(";")[0].strip() + ";"

    def generate_sql(self, question: str) -> str:
        """Generate a SQL query for a natural-language question."""
        schema = self.manager.schema_text()
        prompt = f"Schema:\n{schema}\n\nQuestion: {question}\n\nSQL:"
        raw = self.llm.chat(_SQL_SYSTEM, prompt, temperature=0.0)
        return self._clean_sql(raw)

    def run(self, question: str) -> SQLResult:
        """Generate and execute SQL for a question.

        Args:
            question: Natural-language question.

        Returns:
            A :class:`SQLResult`.
        """
        sql = self.generate_sql(question)
        if not sql.lower().lstrip().startswith("select"):
            return SQLResult(sql=sql, error="Only SELECT queries are allowed.")
        try:
            df = self.manager.run_query(sql)
            return SQLResult(sql=sql, dataframe=df)
        except Exception as exc:  # noqa: BLE001
            logger.error("SQL execution failed: %s", exc)
            return SQLResult(sql=sql, error=f"{type(exc).__name__}: {exc}")
