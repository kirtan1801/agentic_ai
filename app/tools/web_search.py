import asyncio
import logging

from langchain_core.tools import tool
from app.config import settings
from langchain_tavily import TavilySearch

logger = logging.getLogger(__name__)

# Web results are the biggest prompt bloat: 5 results x ~700 chars keeps it near 4k
# while leaving room for several distinct sources.
MAX_RESULTS = 5
RESULT_CHARS = 700

tavily_search = TavilySearch(
    max_results=MAX_RESULTS,
    topic="general",
    tavily_api_key=settings.tavily_api_key
)

def _format(result: dict) -> str:
    # Title + URL + trimmed snippet per result, instead of the raw dict cut mid-result.
    rows = result.get("results") or []
    if not rows:
        return "No web results found. Try a broader query."
    return "\n\n".join(
        f"{row.get('title', '')}\n{row.get('url', '')}\n{str(row.get('content', ''))[:RESULT_CHARS]}" for row in rows
    )


@tool
async def web_search(query: str) -> str:
    """Search the web for current or public information. Phrase the query about the topic itself
    (e.g. "backend scaling best practices"), not about an uploaded document's name, to get varied sources."""
    try:
        result = await asyncio.wait_for(tavily_search.ainvoke({'query': query}), settings.tool_timeout)
    except Exception:
        # Tell the model plainly, so it answers instead of retrying the search.
        logger.warning("Web search failed", exc_info=True)
        return "Web search is unavailable right now. Answer from the documents, or say you couldn't look it up."

    return f"<untrusted>\n{_format(result)}\n</untrusted>"
