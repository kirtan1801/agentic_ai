
import logging
from typing import Literal

from app.agent.state import AgentState

logger = logging.getLogger(__name__)


def router_after_decision(
    state: AgentState,
) -> Literal["direct", "rag", "web_search", "combined"]:
    decision = state["decision"]

    if decision is None:
        raise ValueError("Agent decision is missing")

    logger.info(
        "Routing decision: requires_rag=%s requires_tool=%s tool_name=%s",
        decision.requires_rag,
        decision.requires_tool,
        decision.tool_name,
    )

    if decision.requires_tool and decision.tool_name != "web_search":
        raise ValueError(f"Unsupported tool: {decision.tool_name}")

    if decision.requires_rag and decision.requires_tool:
        logger.info("Routing to: combined")
        return "combined"

    if decision.requires_tool:
        logger.info("Routing to: web_search")
        return "web_search"

    if decision.requires_rag:
        logger.info("Routing to: rag")
        return "rag"

    logger.info("Routing to: direct")
    return "direct"