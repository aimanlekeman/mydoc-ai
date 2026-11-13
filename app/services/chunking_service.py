
def chunk_text(
    pages: list[dict],
    chunk_size: int = 200,
    overlap: int = 40
) -> list[dict]:
    """
    Divide extracted PDF text into smaller chunks.

    Each chunk retains its original PDF page number.
    """

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be zero or greater, and smaller than chunk size."
        )

    chunks = []
    chunk_counter = 1

    # Process each PDF page separately to preserve page references.
    for page in pages:
        page_number = page["page_number"]
        text = page.get("text", "").strip()

        if not text:
            continue

        words = text.split()
        step = chunk_size - overlap

        for start in range(0, len(words), step):
            end = start + chunk_size
            chunk_words = words[start:end]

            if not chunk_words:
                break

            chunks.append({
                "chunk_id": chunk_counter,
                "page_number": page_number,
                "text": " ".join(chunk_words),
                "word_count": len(chunk_words)
            })

            chunk_counter += 1

            if end >= len(words):
                break

    return chunks
