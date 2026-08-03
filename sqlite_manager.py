"""SQLite connection and inspection helper."""
from __future__ import annotations

import sqlite3
from typing import Any

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


class SQLiteManager:
    """Manage a connection to a SQLite database file."""

    def __init__(self, db_path: str) -> None:
        """Open a connection to the database.

        Args:
            db_path: Path to the SQLite ``.db``/``.sqlite`` file.

        Raises:
            sqlite3.Error: If the connection cannot be opened.
        """
        self.db_path = db_path
        try:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
        except sqlite3.Error as exc:
            logger.error("SQLite connection failed: %s", exc)
            raise

    def list_tables(self) -> list[str]:
        """Return all user table names in the database."""
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        rows = self.conn.execute(query).fetchall()
        return [r[0] for r in rows]

    def get_schema(self, table: str) -> pd.DataFrame:
        """Return the schema of a table.

        Args:
            table: Table name.

        Returns:
            DataFrame with column metadata.
        """
        return pd.read_sql_query(f"PRAGMA table_info({table});", self.conn)

    def schema_text(self) -> str:
        """Return a natural-language description of the whole schema."""
        parts: list[str] = []
        for table in self.list_tables():
            schema = self.get_schema(table)
            cols = ", ".join(
                f"{row['name']} ({row['type']})"
                for _, row in schema.iterrows()
            )
            parts.append(f"Table {table}: {cols}")
        return "\n".join(parts)

    def preview(self, table: str, limit: int = 100) -> pd.DataFrame:
        """Return the first ``limit`` rows of a table."""
        return pd.read_sql_query(
            f"SELECT * FROM {table} LIMIT {int(limit)};", self.conn
        )

    def run_query(self, sql: str) -> pd.DataFrame:
        """Execute a read query and return the result.

        Args:
            sql: SQL statement.

        Returns:
            Query result as a DataFrame.

        Raises:
            Exception: Propagates any SQL/execution error.
        """
        logger.info("Executing SQL: %s", sql)
        return pd.read_sql_query(sql, self.conn)

    def load_table(self, table: str) -> pd.DataFrame:
        """Load an entire table into a DataFrame."""
        return pd.read_sql_query(f"SELECT * FROM {table};", self.conn)

    def close(self) -> None:
        """Close the underlying connection."""
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass
