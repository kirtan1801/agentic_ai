from app.ai.llm import llm
from app.schema.ai import ChatResult

structure_llm = llm.with_structured_output(ChatResult)

