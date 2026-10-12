from fastapi import APIRouter, File, HTTPException, UploadFile

from app.rag.loader import load_document
from app.rag.chunker import chunk_document
from app.rag.ingestion import store_chunk
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

        document = load_document(content, file.filename)

        chunks = chunk_document(document=document)
        chunk_ids = await store_chunk(chunks)

        return {
            "message": "Document uploaded and indexed successfully.",
            "filename": document.metadata["source"],
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
