from io import BytesIO
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".md", ".txt", ".html", ".pdf"}


def load_document(
    content: bytes,
    filename: str,
) -> list[Document]:
    """Return one Document per PDF page, or a single Document for text formats."""
    extension = Path(filename).suffix.lower()
    source = Path(filename).name

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported file type. Upload .pdf, .md, .txt, or .html files."
        )

    metadata = {"source": source, "file_type": extension.lstrip(".")}

    if extension == ".pdf":
        try:
            reader = PdfReader(BytesIO(content))
            pages = [
                Document(page_content=text.strip(), metadata={**metadata, "page": number})
                for number, page in enumerate(reader.pages, start=1)
                if (text := page.extract_text() or "").strip()
            ]
        except Exception as exc:
            raise ValueError("Could not read the PDF file.") from exc

        if not pages:
            # ponytail: no OCR; scanned PDFs need e.g. pytesseract.
            raise ValueError("No extractable text in this PDF (scanned PDFs are not supported).")
        return pages

    text = content.decode("utf-8")

    if extension == ".html":
        soup = BeautifulSoup(text, "html.parser")

        for element in soup(["script", "style", "noscript"]):
            element.decompose()

        text = soup.get_text(separator="\n", strip=True)

    text = text.strip()

    if not text:
        raise ValueError("The uploaded document is empty.")

    return [Document(page_content=text, metadata={**metadata, "page": 1})]
