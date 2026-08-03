"""Basic unit tests that run without any external API calls."""
from __future__ import annotations

import pandas as pd

from tools.pandas_tool import PandasTool
from tools.visualization_tool import VisualizationTool
from utils.helpers import dataframe_summary, summary_to_text
from utils.safe_executor import safe_execute


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["A", "B", "A", "B", "A"],
            "revenue": [10, 20, 30, 40, 1000],
            "year": [2021, 2021, 2022, 2022, 2023],
        }
    )


def test_dataframe_summary() -> None:
    summary = dataframe_summary(_sample_df())
    assert summary["rows"] == 5
    assert summary["columns"] == 3
    assert "revenue" in summary["column_names"]


def test_summary_to_text() -> None:
    text = summary_to_text("sales", dataframe_summary(_sample_df()))
    assert "sales" in text and "5 rows" in text


def test_pandas_group_by() -> None:
    result = PandasTool(_sample_df()).group_by("region", "revenue", "sum")
    assert set(result["region"]) == {"A", "B"}


def test_pandas_outliers() -> None:
    outliers = PandasTool(_sample_df()).outliers("revenue")
    assert (outliers["revenue"] == 1000).any()


def test_visualization_histogram() -> None:
    fig = VisualizationTool(_sample_df()).histogram("revenue")
    assert fig is not None


def test_safe_execute_success() -> None:
    result = safe_execute("result = df['revenue'].sum()", _sample_df())
    assert result.success is True
    assert result.result == 1100


def test_safe_execute_blocks_import() -> None:
    result = safe_execute("import os\nresult = 1", _sample_df())
    assert result.success is False
    assert "Disallowed" in result.error
