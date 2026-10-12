from bs4 import BeautifulSoup
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".md", ".txt", ".html"}

def load_doc(content: bytes, filename: str) -> Document:
    if "." not in filename:
        raise ValueError("File must have extension")
    
    extension = f".{filename.rsplit('.', maxsplit=1)[-1].lower()}"

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Invalid file extension.")
    
    text = content.decode("utf-8")

    if extension == '.html':
        soup = BeautifulSoup(text, "html.parser")

        for element in soup(["script", "style", "noscript"]):
            element.decompose()

        text = soup.get_text(separator="\n",strip=True)

    if not text.strip():
        raise ValueError("Document is empty.")
    
    return Document(
        page_content=text, 
        metadata= {
            "filename": filename,
           "format": extension.lstrip("."),
        }
    )