"""Run a deep agent once to check it works and shows up in LangSmith.

Usage: uv run python -m app.script.try_deep_agent
"""
from deepagents import create_deep_agent

from app.ai.llm import chat_model

agent = create_deep_agent(model=chat_model)
result = agent.invoke({"messages": [{"role": "user", "content": "What is LangSmith?"}]})
print(result["messages"][-1].content)
