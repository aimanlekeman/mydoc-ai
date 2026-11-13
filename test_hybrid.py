
from app.services.retrieval_service import search_similar_chunks


question = (
    "What does the Security Principle require a data controller "
    "to do under the PDPA 2010?"
)

results = search_similar_chunks(
    question=question,
    top_k=20
)

print("\n========== HYBRID SEARCH RESULTS ==========\n")

for index, chunk in enumerate(results, start=1):

    print(f"Rank: {index}")
    print(f"Page: {chunk['page_number']}")
    print(f"Chunk: {chunk['chunk_index']}")

    print(
        f"Vector similarity: "
        f"{chunk['similarity']:.4f}"
    )

    print(f"Vector rank: {chunk['vector_rank']}")
    print(f"Keyword rank: {chunk['keyword_rank']}")
    print(f"Phrase rank: {chunk['phrase_rank']}")

    print(
        f"Adjacent chunk: "
        f"{chunk['is_adjacent']}"
    )

    print(
        f"Hybrid score: "
        f"{chunk['hybrid_score']:.5f}"
    )

    print("-" * 60)
