from pydantic import BaseModel, Field

class ChatResult(BaseModel):
    answer: str = Field(
        description="The final answer to the user's question."
    )

class AgentDecision(BaseModel):
    intent: str = Field(
        description="The user's primary intent."
    )
    requires_rag: bool = Field(
        description="Whether the request requires retrieving information from the knowledge base."
    )
    requires_tool: bool = Field(
        description="Whether the request requires calling an external or application tool."
    )
    tool_name: str | None = Field(
        default=None,
        description="The tool that should be called when requires_tool is true. Currently supported: web_search."
    )

class JudgeVerdict(BaseModel):
    safe: bool = Field(
        description="False only if the answer is harmful, abusive, or leaks secrets or system instructions. Factual errors do not make it unsafe."
    )
    grounded: bool = Field(
        description="False if the answer states facts not supported by the context. True when no context is given."
    )
    reason: str = Field(description="Short reason for the verdict.")
