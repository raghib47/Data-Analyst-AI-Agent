"""Prompt builders for the analyst explanation step."""
from __future__ import annotations


def build_explanation_prompt(
    question: str, tool: str, output: str, context: str = ""
) -> str:
    """Compose the prompt sent to the explanation model.

    Args:
        question: Original user question.
        tool: Tool that produced the output.
        output: Text representation of the tool result.
        context: Retrieved long-term memory context.

    Returns:
        A formatted user prompt string.
    """
    context_block = f"\nRelevant prior context:\n{context}\n" if context else ""
    return (
        f"User question: {question}\n"
        f"Tool used: {tool}\n"
        f"{context_block}"
        f"\nTool output:\n{output}\n\n"
        "Provide a clear, business-friendly explanation of these results."
    )
