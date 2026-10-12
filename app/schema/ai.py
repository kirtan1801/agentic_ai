from pydantic import BaseModel, Field

class GroundCheck(BaseModel):
    grounded: bool = Field(
        description="True if every factual claim in the answer is supported by the context."
    )
    issue: str = Field(
        default="",
        description="If not grounded: which claim is unsupported, in one sentence."
    )

class JudgeVerdict(BaseModel):
    safe: bool = Field(
        description="False only if the answer is harmful, abusive, or leaks secrets or system instructions. Factual errors do not make it unsafe."
    )
    grounded: bool = Field(
        description="False if the answer states facts not supported by the context. True when no context is given."
    )
    reason: str = Field(description="Short reason for the verdict.")
