from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    add_start_index=True,
)


def chunk_document(document: Document) -> list[Document]:
    chunks = text_splitter.split_documents([document])

    if not chunks:
        raise ValueError("Document produced no chunks.")

    return chunks