
from app.services.retrieval_service import search_similar_chunks

question = (
    "What are the seven Personal Data Protection "
    "Principles under the PDPA 2010?"
)

results = search_similar_chunks(
    question=question,
    top_k=10
)

print("\n========== Q01 RETRIEVAL RESULTS ==========\n")

for i, chunk in enumerate(results, start=1):

    print(f"Rank: {i}")
    print(f"Page: {chunk['page_number']}")
    print(f"Chunk: {chunk['chunk_index']}")
    print(f"Similarity: {chunk['similarity']:.4f}")
    print(f"Keyword rank: {chunk['keyword_rank']}")
    print(f"Phrase rank: {chunk['phrase_rank']}")
    print(f"Adjacent: {chunk['is_adjacent']}")
    print("-" * 60)
    print(chunk["content"][:700])
    print("\n")
