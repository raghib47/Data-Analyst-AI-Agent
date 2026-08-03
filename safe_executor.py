"""Sandboxed execution of LLM-generated Python code."""
from __future__ import annotations

import contextlib
import io
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px

from utils.logger import get_logger

logger = get_logger(__name__)

# Substrings that are never allowed in generated code.
_FORBIDDEN = (
    "import os",
    "import sys",
    "import subprocess",
    "import shutil",
    "open(",
    "__import__",
    "eval(",
    "exec(",
    "globals(",
    "locals(",
    "getattr(",
    "setattr(",
    "os.",
    "sys.",
    "subprocess",
    "socket",
    "requests",
    "pickle",
)


@dataclass
class ExecutionResult:
    """Result of executing generated code.

    Attributes:
        success: Whether execution completed without error.
        stdout: Captured standard output.
        error: Error message if execution failed.
        result: Value bound to ``result`` in the namespace, if any.
        figure: Plotly figure bound to ``fig``, if any.
    """

    success: bool
    stdout: str = ""
    error: str = ""
    result: Any = None
    figure: Any = None


def _strip_code_fences(code: str) -> str:
    """Remove markdown code fences from an LLM response."""
    match = re.search(r"```(?:python)?\s*(.*?)```", code, re.DOTALL)
    return (match.group(1) if match else code).strip()


def _is_safe(code: str) -> tuple[bool, str]:
    """Check code against the forbidden-substring blocklist."""
    lowered = code.lower()
    for token in _FORBIDDEN:
        if token in lowered:
            return False, f"Disallowed operation detected: '{token}'"
    return True, ""


def safe_execute(code: str, df: pd.DataFrame | None) -> ExecutionResult:
    """Execute generated Python code in a restricted namespace.

    The namespace exposes ``df``, ``pd``, ``np`` and ``px``. Any figure
    assigned to ``fig`` and any value assigned to ``result`` is captured.

    Args:
        code: Python source (may include markdown fences).
        df: Active DataFrame, or ``None``.

    Returns:
        An :class:`ExecutionResult`.
    """
    code = _strip_code_fences(code)
    safe, reason = _is_safe(code)
    if not safe:
        logger.warning("Blocked unsafe code: %s", reason)
        return ExecutionResult(success=False, error=reason)

    namespace: dict[str, Any] = {
        "df": df.copy() if df is not None else None,
        "pd": pd,
        "np": np,
        "px": px,
        "result": None,
        "fig": None,
    }

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, {"__builtins__": __builtins__}, namespace)  # noqa: S102
        return ExecutionResult(
            success=True,
            stdout=stdout.getvalue(),
            result=namespace.get("result"),
            figure=namespace.get("fig"),
        )
    except Exception as exc:  # noqa: BLE001 - surface all errors to user
        logger.error("Code execution failed: %s", exc)
        return ExecutionResult(
            success=False,
            stdout=stdout.getvalue(),
            error=f"{type(exc).__name__}: {exc}",
        )
