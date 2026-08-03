"""LLM-driven Python code generation and safe execution."""
from __future__ import annotations

import pandas as pd

from models.llm import ChatLLM
from utils.safe_executor import ExecutionResult, safe_execute

_PY_SYSTEM = (
    "You are a Python data analyst. Write concise pandas/numpy/plotly code "
    "to answer the user's question about a DataFrame named `df`. "
    "Available names: df, pd, np, px. "
    "Assign the final tabular/scalar answer to a variable named `result`. "
    "If a chart is appropriate, build a Plotly figure and assign it to `fig`. "
    "Do NOT import anything, read files, or access the network. "
    "Return ONLY code inside a single ```python``` block."
)


class PythonTool:
    """Generate and safely execute Python analysis code."""

    def __init__(self, llm: ChatLLM | None = None) -> None:
        """Initialize the tool.

        Args:
            llm: Chat model (defaults to a new ChatLLM).
        """
        self.llm = llm or ChatLLM()

    def generate_code(self, question: str, df: pd.DataFrame) -> str:
        """Generate Python code for a question given a DataFrame."""
        columns = ", ".join(f"{c} ({t})" for c, t in df.dtypes.items())
        prompt = (
            f"DataFrame columns: {columns}\n"
            f"Rows: {len(df)}\n\n"
            f"Question: {question}"
        )
        return self.llm.chat(_PY_SYSTEM, prompt, temperature=0.1)

    def run(self, question: str, df: pd.DataFrame) -> tuple[str, ExecutionResult]:
        """Generate and execute code for a question.

        Args:
            question: Natural-language question.
            df: Active DataFrame.

        Returns:
            Tuple of (generated code, execution result).
        """
        code = self.generate_code(question, df)
        result = safe_execute(code, df)
        return code, result
