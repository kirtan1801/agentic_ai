from typing import Literal

from pydantic import BaseModel, Field

THREAD_ID = Field(default=None, max_length=64, pattern=r"^[\w-]+$")

class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str | None = THREAD_ID
    # Restrict document search to these uploaded files (None = all).
    sources: list[str] | None = Field(default=None, max_length=50)

class StreamRequest(ChatRequest):
    mode: Literal["agent", "ask", "chat"] = "agent"

class Citation(BaseModel):
    n: int
    source: str
    page: int
    snippet: str

class ChatResponse(BaseModel):
    response: str
    sources: list[Citation] = []
    thread_id: str
