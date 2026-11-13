
from app.services.retrieval_service import search_similar_chunks


def test_retrieval():

    question = "What are the advantages of Shallow Neural Networks?"

    print("\nQuestion:")
    print(question)

    print("\nSearching for relevant document chunks...")

    results = search_similar_chunks(
        question=question,
        top_k=3
    )

    print("\nRetrieved Results:")

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")
        print(f"Filename: {result['filename']}")
        print(f"Page: {result['page_number']}")
        print(f"Chunk: {result['chunk_index']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Content: {result['content']}")


if __name__ == "__main__":
    test_retrieval()
