from langchain_core.tools import tool
from app.config import settings
from langchain_tavily import TavilySearch

MAX_RESULT_CHARS = 4000

tavily_search = TavilySearch(
    max_results=3,
    topic="general",
    tavily_api_key=settings.tavily_api_key
)

@tool
async def web_search(query: str) -> str:
    """
    Search the web for the current information...
    """
    result = await tavily_search.ainvoke({
        'query': query
    })

    # Web results are the biggest prompt bloat; cap them.
    return f"<untrusted>\n{str(result)[:MAX_RESULT_CHARS]}\n</untrusted>"
