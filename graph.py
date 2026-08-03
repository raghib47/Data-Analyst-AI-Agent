"""LangGraph workflow wrapping the DataAnalystAgent.

The graph makes the reasoning pipeline explicit:
retrieve -> route -> execute -> explain -> remember.
It delegates the concrete work to :class:`DataAnalystAgent` so the
same logic is reusable outside of LangGraph.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent import AgentContext, AgentResponse, DataAnalystAgent


class AgentState(TypedDict, total=False):
    """State passed between graph nodes.

    Attributes:
        question: User question.
        context: Runtime AgentContext.
        response: Final AgentResponse.
    """

    question: str
    context: AgentContext
    response: AgentResponse


def build_graph(agent: DataAnalystAgent) -> Any:
    """Build and compile the LangGraph workflow.

    Args:
        agent: The underlying analyst agent.

    Returns:
        A compiled LangGraph application exposing ``invoke``.
    """

    def analyze_node(state: AgentState) -> AgentState:
        """Single node that runs the full agent pipeline."""
        response = agent.answer(state["question"], state["context"])
        return {"response": response}

    workflow = StateGraph(AgentState)
    workflow.add_node("analyze", analyze_node)
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", END)
    return workflow.compile()


class AnalystGraph:
    """Convenience wrapper around the compiled LangGraph app."""

    def __init__(self, agent: DataAnalystAgent | None = None) -> None:
        """Initialize the graph.

        Args:
            agent: Optional pre-built agent.
        """
        self.agent = agent or DataAnalystAgent()
        self.app = build_graph(self.agent)

    def run(self, question: str, context: AgentContext) -> AgentResponse:
        """Run the workflow for a single question.

        Args:
            question: User question.
            context: Runtime context.

        Returns:
            The resulting :class:`AgentResponse`.
        """
        final = self.app.invoke({"question": question, "context": context})
        return final["response"]
