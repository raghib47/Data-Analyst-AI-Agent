"""High-level agent orchestrating tools, LLM, and memory."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from database.chroma_manager import ChromaManager
from database.sqlite_manager import SQLiteManager
from models.llm import ChatLLM
from prompts.analyst_prompt import build_explanation_prompt
from prompts.system_prompt import (
    EXPLANATION_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
)
from tools.pandas_tool import PandasTool
from tools.python_tool import PythonTool
from tools.rag_tool import RAGTool
from tools.sql_tool import SQLTool
from tools.visualization_tool import VisualizationTool
from utils.helpers import df_to_markdown, timed
from utils.logger import get_logger

logger = get_logger(__name__)

_VALID_TOOLS = {"pandas", "sql", "python", "visualization"}


@dataclass
class AgentResponse:
    """Structured response returned to the UI.

    Attributes:
        answer: Natural-language explanation.
        tool: Tool selected by the router.
        dataframe: Optional tabular result.
        figure: Optional Plotly figure.
        code: Optional generated code/SQL.
        error: Optional error message.
    """

    answer: str = ""
    tool: str = ""
    dataframe: pd.DataFrame | None = None
    figure: Any = None
    code: str = ""
    error: str = ""


@dataclass
class AgentContext:
    """Mutable runtime dependencies for the agent.

    Attributes:
        df: Active DataFrame, if any.
        sqlite: Connected SQLite manager, if any.
    """

    df: pd.DataFrame | None = None
    sqlite: SQLiteManager | None = None


class DataAnalystAgent:
    """Routes questions to tools and produces explanations."""

    def __init__(self, chroma: ChromaManager | None = None) -> None:
        """Initialize the agent and its shared dependencies.

        Args:
            chroma: Long-term memory manager (created if omitted).
        """
        self.llm = ChatLLM()
        self.chroma = chroma or ChromaManager()
        self.rag = RAGTool(self.chroma)
        self.python_tool = PythonTool(self.llm)

    def route(self, question: str, ctx: AgentContext) -> str:
        """Decide which tool should handle the question.

        Args:
            question: User question.
            ctx: Runtime context.

        Returns:
            One of ``pandas``, ``sql``, ``python``, ``visualization``.
        """
        if ctx.df is None and ctx.sqlite is not None:
            # Default to SQL when only a database is available.
            hint = "sql"
        else:
            hint = "pandas"
        raw = self.llm.chat(
            ROUTER_SYSTEM_PROMPT, question, temperature=0.0, max_tokens=10
        )
        tool = raw.strip().lower().split()[0] if raw.strip() else hint
        return tool if tool in _VALID_TOOLS else hint

    def _explain(
        self, question: str, tool: str, output: str, context: str
    ) -> str:
        """Generate a business-friendly explanation of a result."""
        prompt = build_explanation_prompt(question, tool, output, context)
        return self.llm.chat(EXPLANATION_SYSTEM_PROMPT, prompt)

    def answer(self, question: str, ctx: AgentContext) -> AgentResponse:
        """Answer a user question end-to-end.

        Args:
            question: Natural-language question.
            ctx: Runtime context (DataFrame and/or SQLite).

        Returns:
            An :class:`AgentResponse`.
        """
        with timed("agent.answer"):
            logger.info("User question: %s", question)
            context = self.rag.retrieve(question)
            tool = self.route(question, ctx)
            logger.info("Routed to tool: %s", tool)

            if tool == "sql":
                resp = self._handle_sql(question, ctx, context)
            elif tool == "visualization":
                resp = self._handle_python(question, ctx, context, "visualization")
            elif tool == "python":
                resp = self._handle_python(question, ctx, context, "python")
            else:
                resp = self._handle_python(question, ctx, context, "pandas")

            self._remember(question, resp)
            return resp

    def _handle_sql(
        self, question: str, ctx: AgentContext, context: str
    ) -> AgentResponse:
        """Handle database questions via the SQL tool."""
        if ctx.sqlite is None:
            return AgentResponse(
                tool="sql", error="No SQLite database connected."
            )
        sql_tool = SQLTool(ctx.sqlite, self.llm)
        result = sql_tool.run(question)
        if result.error:
            return AgentResponse(tool="sql", code=result.sql, error=result.error)
        output = df_to_markdown(result.dataframe)
        answer = self._explain(question, "sql", output, context)
        return AgentResponse(
            answer=answer,
            tool="sql",
            dataframe=result.dataframe,
            code=result.sql,
        )

    def _handle_python(
        self, question: str, ctx: AgentContext, context: str, tool: str
    ) -> AgentResponse:
        """Handle pandas/python/visualization via code execution."""
        if ctx.df is None:
            return AgentResponse(
                tool=tool, error="No dataset loaded. Upload or select one."
            )
        code, exec_result = self.python_tool.run(question, ctx.df)
        if not exec_result.success:
            return AgentResponse(tool=tool, code=code, error=exec_result.error)

        output_parts: list[str] = []
        result_df: pd.DataFrame | None = None
        if isinstance(exec_result.result, pd.DataFrame):
            result_df = exec_result.result
            output_parts.append(df_to_markdown(result_df))
        elif exec_result.result is not None:
            output_parts.append(str(exec_result.result))
        if exec_result.stdout:
            output_parts.append(exec_result.stdout)

        output = "\n".join(output_parts) or "(figure generated)"
        answer = self._explain(question, tool, output, context)
        return AgentResponse(
            answer=answer,
            tool=tool,
            dataframe=result_df,
            figure=exec_result.figure,
            code=code,
        )

    def _remember(self, question: str, resp: AgentResponse) -> None:
        """Store a short conversation summary in long-term memory."""
        snippet = resp.answer[:400] if resp.answer else resp.error[:200]
        summary = f"Q: {question}\nTool: {resp.tool}\nA: {snippet}"
        self.rag.store_conversation(summary)
