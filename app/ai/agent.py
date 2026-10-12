from app.ai.llm import router_llm
from app.schema.ai import AgentDecision

agent_llm = router_llm.with_structured_output(AgentDecision)
