
from app.services.retrieval_service import search_similar_chunks

question = (
    "What information must a data controller include in a written "
    "notice to a data subject under Section 7 of the PDPA 2010?"
)

print("\n" + "=" * 80)
print("MYDOC AI - Q10 RETRIEVAL DIAGNOSTIC")
print("=" * 80)
print(f"\nQuestion: {question}\n")

results = search_similar_chunks(
    question=question,
    top_k=10
)

print(f"Total chunks retrieved: {len(results)}")

for i, chunk in enumerate(results, start=1):
    print("\n" + "-" * 80)
    print(f"Rank:          {i}")
    print(f"Page:          {chunk.get('page_number')}")
    print(f"Chunk:         {chunk.get('chunk_index')}")
    print(f"Similarity:    {chunk.get('similarity')}")
    print(f"Keyword rank:  {chunk.get('keyword_rank')}")
    print(f"Phrase rank:   {chunk.get('phrase_rank')}")
    print(f"Adjacent:      {chunk.get('is_adjacent')}")

    print("\nCONTENT:")
    print(chunk.get("content", "")[:1200])

print("\n" + "=" * 80)
print("END OF Q10 DIAGNOSTIC")
print("=" * 80)
