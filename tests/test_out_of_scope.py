
from app.services.retrieval_service import search_similar_chunks


def test_out_of_scope():

    question = "Who won the FIFA World Cup in 2022?"

    print("\nQuestion:")
    print(question)

    print("\nSearching PostgreSQL for relevant chunks...")

    results = search_similar_chunks(
        question=question,
        top_k=3
    )

    print("\nRetrieved Results:")

    if not results:
        print("No document chunks found.")

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")
        print(f"Filename: {result['filename']}")
        print(f"Page: {result['page_number']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Content: {result['content'][:150]}...")


if __name__ == "__main__":
    test_out_of_scope()
