
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.pdf_service import (
    extract_pdf_pages,
    PDFProcessingError
)

from app.services.chunking_service import chunk_text
from app.services.embedding_service import generate_embeddings
from app.services.database_service import save_document


router = APIRouter(
    prefix="/documents",
    tags=["Documents"]
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    # Check the file extension.
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    try:
        # Read up to 10 MB + 1 byte to detect oversized files.
        file_bytes = await file.read(MAX_FILE_SIZE + 1)

        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File is too large. Maximum size is 10 MB."
            )

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        # Step 1: Extract text from PDF.
        extracted_pages = extract_pdf_pages(file_bytes)

        # Step 2: Divide extracted text into chunks.
        chunks = chunk_text(
            pages=extracted_pages,
            chunk_size=200,
            overlap=40
        )

        # Step 3: Generate embeddings for each chunk.
        chunk_texts = [chunk["text"] for chunk in chunks]

        embeddings = generate_embeddings(chunk_texts)

        # Step 4: Save document, chunks and embeddings.
        database_result = save_document(
            filename=file.filename,
            file_size=len(file_bytes),
            total_pages=len(extracted_pages),
            chunks=chunks,
            embeddings=embeddings
        )

        # Step 5: Return processing and database results.
        return {
            "message": "Document uploaded and saved successfully!",
            "filename": file.filename,
            "total_pages": len(extracted_pages),
            "total_chunks": len(chunks),
            "embedding_count": len(embeddings),
            "embedding_dimensions": len(embeddings[0]) if embeddings else 0,
            "database": database_result,
            "chunks": chunks
        }

    except PDFProcessingError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    finally:
        await file.close()
