"""Load CSV and Excel files into DataFrames."""
from __future__ import annotations

from typing import IO

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


class DatasetLoader:
    """Load tabular datasets from uploaded files."""

    @staticmethod
    def load_csv(source: str | IO[bytes]) -> pd.DataFrame:
        """Load a CSV file.

        Args:
            source: File path or file-like object.

        Returns:
            Parsed DataFrame.
        """
        logger.info("Loading CSV dataset")
        return pd.read_csv(source)

    @staticmethod
    def load_excel(source: str | IO[bytes]) -> pd.DataFrame:
        """Load the first sheet of an Excel workbook.

        Args:
            source: File path or file-like object.

        Returns:
            Parsed DataFrame.
        """
        logger.info("Loading Excel dataset")
        return pd.read_excel(source, engine="openpyxl")

    @classmethod
    def load(cls, source: str | IO[bytes], filename: str) -> pd.DataFrame:
        """Dispatch to the correct loader based on file extension.

        Args:
            source: File path or file-like object.
            filename: Original filename (used to detect the type).

        Returns:
            Parsed DataFrame.

        Raises:
            ValueError: If the extension is unsupported.
        """
        lower = filename.lower()
        if lower.endswith(".csv"):
            return cls.load_csv(source)
        if lower.endswith((".xlsx", ".xls")):
            return cls.load_excel(source)
        raise ValueError(f"Unsupported file type: {filename}")
