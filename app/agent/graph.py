
from langgraph.graph import END, START, StateGraph

from app.agent.node import (
    decide_node,
    direct_response_node,
    web_search_node,
    generate_response_node,
    rag_node,
    retrieve_node,
    generate_combined_response_node,
    combined_start_node
)
from app.agent.routing import router_after_decision
from app.agent.state import AgentState

builder = StateGraph(AgentState)

builder.add_node("decide", decide_node)
builder.add_node("direct", direct_response_node)
builder.add_node("web_search", web_search_node)
builder.add_node("rag", rag_node)
builder.add_node("combined_start", combined_start_node)
builder.add_node("combined_web_search", web_search_node)
builder.add_node("combined_rag", retrieve_node)
builder.add_node("generate_combined_response", generate_combined_response_node)
builder.add_node("generate_response", generate_response_node)

builder.add_edge(START, "decide")

builder.add_conditional_edges(
    "decide",
    router_after_decision,
    {
        "direct": "direct",
        "web_search": "web_search",
        "rag": "rag",
        "combined": "combined_start",
    },
)

# Existing workflow
builder.add_edge("direct", END)
builder.add_edge("rag", END)
builder.add_edge("web_search", "generate_response")
builder.add_edge("generate_response", END)

# Combined workflow: execute both retrieval nodes in parallel
builder.add_edge("combined_start", "combined_web_search")
builder.add_edge("combined_start", "combined_rag")

# Wait for BOTH retrieval nodes before generating the answer
builder.add_edge(
    ["combined_web_search", "combined_rag"],
    "generate_combined_response",
)
builder.add_edge("generate_combined_response", END)

graph = builder.compile()