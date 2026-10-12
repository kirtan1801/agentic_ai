from app.ai.llm import llm

class ChatService:

    async def chat(self, message: str) -> str:
        response = await llm.ainvoke(message)

        return response.content