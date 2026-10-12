from fastapi import APIRouter, File, HTTPException, Path, UploadFile

from app.rag.loader import load_document
from app.rag.chunker import chunk_documents
from app.rag.ingestion import delete_document, list_documents, store_chunk
from app.config import settings

router = APIRouter()


@router.post("/docs")
async def upload_document(
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    try:
        content = await file.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {settings.max_upload_bytes // 1024} KB).",
            )

        documents = load_document(content, file.filename)

        chunks = chunk_documents(documents)
        chunk_ids = await store_chunk(chunks)

        return {
            "message": "Document uploaded and indexed successfully.",
            "filename": documents[0].metadata["source"],
            "pages": len(documents),
            "chunk_count": len(chunks),
            "stored_chunk_count": len(chunk_ids),
        }

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="The document must use UTF-8 encoding.",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    finally:
        await file.close()


@router.get("/docs")
async def get_documents() -> dict:
    return {"documents": await list_documents()}


@router.delete("/docs/{source}")
async def remove_document(source: str = Path(min_length=1, max_length=255)) -> dict:
    deleted = await delete_document(source)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"source": source, "deleted_chunks": deleted}
