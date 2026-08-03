"""Deterministic pandas analysis operations."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class PandasTool:
    """Reusable, LLM-free pandas analytics helpers."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Bind the tool to a DataFrame.

        Args:
            df: DataFrame to operate on.
        """
        self.df = df

    def info(self) -> dict[str, Any]:
        """Return a structural overview of the DataFrame."""
        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": {c: str(t) for c, t in self.df.dtypes.items()},
            "memory_bytes": int(self.df.memory_usage(deep=True).sum()),
        }

    def describe(self) -> pd.DataFrame:
        """Return descriptive statistics for all columns."""
        return self.df.describe(include="all").transpose()

    def missing_values(self) -> pd.DataFrame:
        """Return per-column missing value counts and percentages."""
        counts = self.df.isna().sum()
        pct = (counts / len(self.df) * 100).round(2) if len(self.df) else counts
        return pd.DataFrame({"missing": counts, "percent": pct})

    def duplicates(self) -> pd.DataFrame:
        """Return all duplicated rows."""
        return self.df[self.df.duplicated(keep=False)]

    def correlation(self) -> pd.DataFrame:
        """Return the numeric correlation matrix."""
        return self.df.select_dtypes(include=np.number).corr()

    def group_by(
        self, by: str, agg_col: str, func: str = "mean"
    ) -> pd.DataFrame:
        """Group by a column and aggregate another.

        Args:
            by: Grouping column.
            agg_col: Column to aggregate.
            func: Aggregation function name.

        Returns:
            Aggregated DataFrame sorted descending by the result.
        """
        result = (
            self.df.groupby(by)[agg_col].agg(func).reset_index()
        )
        return result.sort_values(agg_col, ascending=False)

    def outliers(self, column: str) -> pd.DataFrame:
        """Detect outliers in a numeric column using the IQR method.

        Args:
            column: Numeric column name.

        Returns:
            Rows considered outliers.
        """
        series = self.df[column]
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return self.df[(series < low) | (series > high)]

    def top_n(self, column: str, n: int = 10) -> pd.DataFrame:
        """Return the top ``n`` rows sorted by a column descending."""
        return self.df.sort_values(column, ascending=False).head(n)
