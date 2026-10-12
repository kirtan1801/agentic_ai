"""Layer 2: NeMo Guardrails self-check rails, used as a pass/fail check only."""
from functools import cache
from pathlib import Path

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import RailStatus, RailType

from app.ai.llm import router_llm

CONFIG_DIR = Path(__file__).parent / "nemo"


@cache
def _rails() -> LLMRails:
    return LLMRails(RailsConfig.from_path(str(CONFIG_DIR)), llm=router_llm)


async def nemo_blocks_input(message: str) -> bool:
    result = await _rails().check_async([{"role": "user", "content": message}])
    return result.status == RailStatus.BLOCKED


async def nemo_blocks_output(message: str, answer: str) -> bool:
    result = await _rails().check_async(
        [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ],
        rail_types=[RailType.OUTPUT],
    )
    return result.status == RailStatus.BLOCKED
