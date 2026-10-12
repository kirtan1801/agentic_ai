import logging

from app.agent.state import AgentState
from app.agent.graph import graph

logger = logging.getLogger(__name__)

class AgentService:
    async def decide(self, message: str) -> dict:
        """Run the agent graph. Returns the response plus the context it was grounded on."""
        logger.info("Agent request received")

        state: AgentState = {
            'message': message,
            'decision': None,
            'response': None,
            'tool_result': None,
            'rag_result': None
        }

        logger.info("Invoking agent graph")

        result = await graph.ainvoke(state)

        logger.info("Agent graph completed")

        response = result['response']

        if response is None: 
            logger.error("Agent response missing after graph execution")
            raise ValueError("Agent response is missing.")

        context = "\n\n".join(
            part for part in (result["rag_result"], result["tool_result"]) if part
        )

        return {"response": response, "context": context}
