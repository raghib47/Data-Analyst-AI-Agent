"""Plotly chart builders."""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px


class VisualizationTool:
    """Create common Plotly charts from a DataFrame."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Bind the tool to a DataFrame.

        Args:
            df: Source DataFrame.
        """
        self.df = df

    def line(self, x: str, y: str) -> Any:
        """Return a line chart."""
        return px.line(self.df, x=x, y=y, title=f"{y} over {x}")

    def bar(self, x: str, y: str) -> Any:
        """Return a bar chart."""
        return px.bar(self.df, x=x, y=y, title=f"{y} by {x}")

    def scatter(self, x: str, y: str) -> Any:
        """Return a scatter plot."""
        return px.scatter(self.df, x=x, y=y, title=f"{y} vs {x}")

    def pie(self, names: str, values: str) -> Any:
        """Return a pie chart."""
        return px.pie(self.df, names=names, values=values)

    def histogram(self, column: str) -> Any:
        """Return a histogram of a column."""
        return px.histogram(self.df, x=column, title=f"Distribution of {column}")

    def box(self, column: str) -> Any:
        """Return a box plot of a column."""
        return px.box(self.df, y=column, title=f"Box plot of {column}")

    def heatmap(self) -> Any:
        """Return a correlation heatmap for numeric columns."""
        corr = self.df.select_dtypes("number").corr()
        return px.imshow(
            corr, text_auto=".2f", title="Correlation Heatmap"
        )
