
from pgvector import Vector
from pgvector.psycopg import register_vector

from app.database import get_connection


def save_document(
    filename: str,
    file_size: int,
    total_pages: int,
    chunks: list[dict],
    embeddings: list[list[float]]
) -> dict:
    """
    Save document information, text chunks and embeddings
    into PostgreSQL.
    """

    if len(chunks) != len(embeddings):
        raise ValueError(
            "The number of chunks and embeddings must match."
        )

    with get_connection() as connection:

        # Register pgvector support for this connection.
        register_vector(connection)

        # Step 1: Save document metadata.
        document = connection.execute(
            """
            INSERT INTO documents (
                filename,
                file_size,
                total_pages
            )
            VALUES (%s, %s, %s)
            RETURNING id;
            """,
            (filename, file_size, total_pages)
        ).fetchone()

        document_id = document[0]

        # Step 2: Save chunks and their embeddings.
        for chunk, embedding in zip(chunks, embeddings, strict=True):

            saved_chunk = connection.execute(
                """
                INSERT INTO document_chunks (
                    document_id,
                    page_number,
                    chunk_index,
                    content,
                    word_count
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    document_id,
                    chunk["page_number"],
                    chunk["chunk_id"],
                    chunk["text"],
                    chunk["word_count"]
                )
            ).fetchone()

            chunk_id = saved_chunk[0]

            # Step 3: Save the corresponding vector.
            connection.execute(
                """
                INSERT INTO document_embeddings (
                    chunk_id,
                    embedding
                )
                VALUES (%s, %s);
                """,
                (chunk_id, Vector(embedding))
            )

    # Reaching here means the transaction committed successfully.
    return {
        "document_id": document_id,
        "total_chunks_saved": len(chunks),
        "total_embeddings_saved": len(embeddings)
    }
