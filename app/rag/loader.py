from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".md", ".txt", ".html"}


def load_document(
    content: bytes,
    filename: str,
) -> Document:
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported file type. Upload .md, .txt, or .html files."
        )

    text = content.decode("utf-8")

    if extension == ".html":
        soup = BeautifulSoup(text, "html.parser")

        for element in soup(["script", "style", "noscript"]):
            element.decompose()

        text = soup.get_text(separator="\n", strip=True)

    text = text.strip()

    if not text:
        raise ValueError("The uploaded document is empty.")

    return Document(
        page_content=text,
        metadata={
            "source": Path(filename).name,
            "file_type": extension.lstrip("."),
        },
    )
