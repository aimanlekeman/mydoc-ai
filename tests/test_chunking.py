
from app.services.chunking_service import chunk_text


# Generate sample text containing 450 words.
sample_text = " ".join(
    [f"word{i}" for i in range(1, 451)]
)

sample_pages = [
    {
        "page_number": 1,
        "text": sample_text
    }
]

chunks = chunk_text(
    pages=sample_pages,
    chunk_size=200,
    overlap=40
)

# Display results.
print(f"Total chunks: {len(chunks)}")

for chunk in chunks:
    print("\n-------------------------")
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Page: {chunk['page_number']}")
    print(f"Word count: {chunk['word_count']}")
    print(f"First words: {chunk['text'][:60]}")
    print(f"Last words: {chunk['text'][-60:]}")

# Basic checks.
assert len(chunks) == 3
assert chunks[0]["word_count"] == 200
assert chunks[1]["word_count"] == 200
assert chunks[2]["word_count"] == 130
assert chunks[0]["page_number"] == 1

print("\nAll chunking tests passed!")
