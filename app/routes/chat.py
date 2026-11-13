
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.retrieval_service import search_similar_chunks
from app.services.llm_service import generate_answer


router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)

# Minimum vector similarity for a chunk to be considered relevant.
MIN_SIMILARITY = 0.30


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=20)


@router.post("/ask")
def ask_question(request: ChatRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        # Step 1: Retrieve chunks using hybrid search.
        retrieved_chunks = search_similar_chunks(
            question=question,
            top_k=request.top_k
        )

        # Step 2: Keep relevant chunks.
        relevant_chunks = [
            chunk for chunk in retrieved_chunks
            if (
                chunk["similarity"] >= MIN_SIMILARITY
                or chunk["keyword_rank"] is not None
            )
        ]

        print(f"Retrieved chunks: {len(retrieved_chunks)}")
        print(f"Relevant chunks: {len(relevant_chunks)}")

        # Step 3: Generate answer using relevant context.
        result = generate_answer(
            question=question,
            retrieved_chunks=relevant_chunks
        )

        # Step 4: Return answer and sources.
        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_chunks": len(relevant_chunks)
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        print(f"Chat error: {error}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question."
        )
