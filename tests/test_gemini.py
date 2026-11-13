
from app.services.llm_service import generate_answer


def test_gemini():

    question = "What are the advantages of Shallow Neural Networks?"

    # Temporary sample context for testing Gemini.
    sample_chunks = [
        {
            "filename": "Lab3_snn_di230069.pdf",
            "page_number": 2,
            "chunk_index": 2,
            "content": (
                "Shallow Neural Networks offer several advantages. "
                "They require less computational power, have a lower "
                "risk of overfitting, and are easier to implement."
            ),
            "similarity": 0.85
        }
    ]

    print("\nQuestion:")
    print(question)

    print("\nSending question and context to Gemini...")

    result = generate_answer(
        question=question,
        retrieved_chunks=sample_chunks
    )

    print("\nGemini's Answer:")
    print(result["answer"])

    print("\nSources:")

    for source in result["sources"]:
        print(
            f"- {source['filename']}, "
            f"Page {source['page_number']}"
        )


if __name__ == "__main__":
    test_gemini()
