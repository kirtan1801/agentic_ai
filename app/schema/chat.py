from pydantic import BaseModel, Field

THREAD_ID = Field(default=None, max_length=64, pattern=r"^[\w-]+$")

class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str | None = THREAD_ID

class ChatResponse(BaseModel):
    response: str
    sources: list[str] = []
    thread_id: str
