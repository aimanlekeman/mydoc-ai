
import os
import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


# Load environment variables from the project root.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")


MODEL_NAME = "gemini-3.5-flash-lite"

# Retry configuration.
MAX_RETRIES = 3
RETRY_DELAYS = [2, 4]

RETRYABLE_ERROR_CODES = {
    429,
    500,
    502,
    503,
    504
}

_client = None


def get_gemini_client():
    """
    Create and return the Gemini API client.
    """

    global _client

    if _client is None:

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. Check your .env file."
            )

        _client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=60000)
        )

    return _client


def generate_content_with_retry(client, prompt):
    """
    Send a request to Gemini with automatic retries
    for temporary API errors.
    """

    for attempt in range(MAX_RETRIES):

        try:

            print(
                f"Sending request to Gemini "
                f"(attempt {attempt + 1}/{MAX_RETRIES})..."
            )

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            return response

        except Exception as error:

            error_code = getattr(error, "code", None)
            error_message = str(error).upper()

            is_retryable = (
                error_code in RETRYABLE_ERROR_CODES
                or any(
                    code in error_message
                    for code in [
                        "429",
                        "500",
                        "502",
                        "503",
                        "504",
                        "UNAVAILABLE",
                        "RESOURCE_EXHAUSTED",
                        "INTERNAL",
                        "DEADLINE_EXCEEDED"
                    ]
                )
            )

            if not is_retryable:
                raise

            if attempt == MAX_RETRIES - 1:

                print(
                    "Gemini request failed after "
                    f"{MAX_RETRIES} attempts."
                )

                raise

            delay = RETRY_DELAYS[attempt]

            print(
                f"Gemini temporarily unavailable. "
                f"Retrying in {delay} seconds..."
            )

            time.sleep(delay)

    raise RuntimeError("Gemini request failed unexpectedly.")


def remove_duplicate_chunks(chunks):
    """
    Remove identical chunks while preserving their original order.
    """

    unique_chunks = []
    seen_chunks = set()

    for chunk in chunks:

        normalized_content = re.sub(
            r"\s+",
            " ",
            chunk["content"]
        ).strip()

        chunk_key = (
            chunk["filename"],
            chunk["page_number"],
            normalized_content
        )

        if chunk_key not in seen_chunks:

            unique_chunks.append(chunk)
            seen_chunks.add(chunk_key)

    return unique_chunks


def generate_answer(
    question: str,
    retrieved_chunks: list[dict]
) -> dict:
    """
    Generate an answer using retrieved document chunks
    and return only the sources cited by Gemini.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not retrieved_chunks:
        return {
            "answer": (
                "I couldn't find relevant information in the "
                "uploaded documents to answer your question."
            ),
            "sources": []
        }

    # Step 1: Remove duplicate chunks.
    unique_chunks = remove_duplicate_chunks(retrieved_chunks)

    print(
        f"Context chunks: {len(unique_chunks)} "
        f"(removed {len(retrieved_chunks) - len(unique_chunks)} duplicates)"
    )

    # Step 2: Build document context.
    context_parts = []

    for index, chunk in enumerate(unique_chunks, start=1):

        context_parts.append(
            f"""
[Source {index}]
Filename: {chunk["filename"]}
Page: {chunk["page_number"]}
Content:
{chunk["content"]}
"""
        )

    context = "\n".join(context_parts)

    # Step 3: Build the Gemini prompt.
    prompt = f"""
You are MyDoc AI, a document question-answering assistant.

Your task is to provide accurate, complete and well-supported
answers using ONLY the supplied document context.

IMPORTANT ANSWERING RULES:

1. Use only information explicitly supported by the supplied
   document context. Never invent facts or use outside knowledge.

2. Read ALL supplied source chunks before answering.

3. Information may continue across multiple chunks and pages.
   Combine relevant information from different sources when
   necessary to answer the complete question.

4. If the question refers to a particular section, article,
   clause or subsection, examine every relevant part of that
   provision available in the supplied context.

5. Pay particular attention to continuation paragraphs and
   subsequent subsections. Do not stop after finding the
   first relevant requirement.

6. If a question asks for all requirements, responsibilities,
   rights, conditions or procedures, include every relevant
   point that can be supported by the supplied context.

7. Preserve the distinction between the main requirements
   and additional requirements such as timing, exceptions,
   language, notification and procedures.

8. Before finalizing your answer, check whether another
   supplied source contains relevant information that has
   not yet been included.

9. Do not silently omit relevant information.

10. If the context provides only part of the answer, explain
    what is available and clearly state what is missing.

11. Never claim that information is absent merely because
    the filename, document title or year differs from the
    wording used in the user's question.

12. Every factual claim must be supported by the supplied
    document context.

13. Identify the source numbers that directly support
    each part of your answer.

14. Only cite sources that genuinely support the answer.

15. Source numbers must match the supplied context exactly.

ANSWER COMPLETENESS CHECK:

Before returning the answer, internally verify:

- Have all relevant source chunks been examined?
- Are there additional relevant subsections?
- Are there continuation paragraphs on later pages?
- Have all relevant conditions and exceptions been included?
- Have timing and procedural requirements been included
  where they are available in the context?
- Are the cited sources sufficient to support the answer?

Return ONLY valid JSON in this format:

{{
    "answer": "Your complete answer with filename and page citations.",
    "source_numbers": [1, 2, 3]
}}

If the supplied context contains no relevant information,
explain that the uploaded documents do not provide enough
information and return an empty source_numbers list.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    # Step 4: Send the prompt to Gemini with automatic retries.
    client = get_gemini_client()

    response = generate_content_with_retry(
        client=client,
        prompt=prompt
    )

    response_text = response.text

    if not response_text:
        raise RuntimeError("Gemini returned an empty response.")

    # Step 5: Parse Gemini's JSON response.
    try:

        cleaned_response = response_text.strip()

        cleaned_response = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            cleaned_response,
            flags=re.IGNORECASE
        ).strip()

        result = json.loads(cleaned_response)

    except json.JSONDecodeError as error:

        raise RuntimeError(
            f"Gemini returned invalid JSON: {response_text}"
        ) from error

    answer = result.get("answer", "")
    source_numbers = result.get("source_numbers", [])

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Gemini returned an empty answer.")

    if not isinstance(source_numbers, list):
        raise RuntimeError("Gemini returned invalid source numbers.")

    # Step 6: Map source numbers to actual retrieved chunks.
    selected_sources = []
    seen_sources = set()

    for number in source_numbers:

        if (
            isinstance(number, bool)
            or not isinstance(number, int)
            or number < 1
            or number > len(unique_chunks)
        ):
            continue

        chunk = unique_chunks[number - 1]

        source_key = (
            chunk["filename"],
            chunk["page_number"]
        )

        if source_key not in seen_sources:

            selected_sources.append({
                "filename": chunk["filename"],
                "page_number": chunk["page_number"]
            })

            seen_sources.add(source_key)

    return {
        "answer": answer.strip(),
        "sources": selected_sources
    }
