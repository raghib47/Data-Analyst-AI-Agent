"""Generic helper utilities."""
from __future__ import annotations

import io
import time
from contextlib import contextmanager
from typing import Any, Iterator

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


@contextmanager
def timed(label: str) -> Iterator[None]:
    """Context manager that logs the execution time of a block.

    Args:
        label: Description used in the log line.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("Execution time [%s]: %.3fs", label, elapsed)


def dataframe_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Produce a compact structural summary of a DataFrame.

    Args:
        df: The DataFrame to summarize.

    Returns:
        Dictionary describing shape, dtypes, missing values, etc.
    """
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns.astype(str)),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {
            c: int(v) for c, v in df.isna().sum().items() if v > 0
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
    }


def summary_to_text(name: str, summary: dict[str, Any]) -> str:
    """Convert a DataFrame summary into a natural-language description.

    Args:
        name: Dataset/table name.
        summary: Output of :func:`dataframe_summary`.

    Returns:
        Human-readable summary string suitable for embedding.
    """
    missing = summary.get("missing_values", {})
    missing_desc = (
        ", ".join(f"{k} ({v})" for k, v in missing.items())
        if missing
        else "none"
    )
    cols = ", ".join(
        f"{c}:{t}" for c, t in summary.get("dtypes", {}).items()
    )
    return (
        f"Dataset '{name}' has {summary['rows']} rows and "
        f"{summary['columns']} columns. Columns and types: {cols}. "
        f"Missing values: {missing_desc}. "
        f"Duplicate rows: {summary['duplicate_rows']}."
    )


def df_to_markdown(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Render a DataFrame to markdown, truncating large frames.

    Args:
        df: DataFrame to render.
        max_rows: Maximum number of rows to include.

    Returns:
        Markdown table string.
    """
    truncated = df.head(max_rows)
    try:
        return truncated.to_markdown(index=False)
    except Exception:  # pragma: no cover - fallback if tabulate missing
        buf = io.StringIO()
        truncated.to_string(buf, index=False)
        return buf.getvalue()
