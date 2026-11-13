
import re

from pgvector import Vector
from pgvector.psycopg import register_vector

from app.database import get_connection
from app.services.embedding_service import generate_embeddings


RRF_K = 60
CANDIDATE_LIMIT = 50
NEIGHBOUR_EXPANSION_LIMIT = 3
SECTION_CONTEXT_LIMIT = 3

STOP_WORDS = {
    "what", "which", "when", "where", "who", "why", "how",
    "does", "do", "did", "is", "are", "was", "were",
    "the", "a", "an", "and", "or", "of", "to", "in",
    "on", "for", "by", "with", "under", "from", "about",
    "can", "could", "would", "should", "must", "shall",
    "this", "that", "these", "those", "it", "its"
}

PDPA_PRINCIPLES = {
    "General Principle": 6,
    "Notice and Choice Principle": 7,
    "Disclosure Principle": 8,
    "Security Principle": 9,
    "Retention Principle": 10,
    "Data Integrity Principle": 11,
    "Access Principle": 12
}


def build_keyword_query(question: str) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", question.lower())

    keywords = []
    seen = set()

    for word in words:
        if word in STOP_WORDS:
            continue

        if len(word) <= 2 and not word.isdigit():
            continue

        if word not in seen:
            keywords.append(word)
            seen.add(word)

    return " OR ".join(keywords) if keywords else question


def build_phrase_queries(question: str) -> list[str]:
    phrases = []

    quoted_phrases = re.findall(r'"([^"]+)"', question)
    phrases.extend(quoted_phrases)

    principle_phrases = re.findall(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Principle)\b",
        question
    )
    phrases.extend(principle_phrases)

    section_phrases = re.findall(
        r"\bSection\s+\d+(?:\(\d+\))?\b",
        question,
        flags=re.IGNORECASE
    )
    phrases.extend(section_phrases)

    unique_phrases = []
    seen = set()

    for phrase in phrases:
        phrase = phrase.strip()

        if phrase.lower() not in seen:
            unique_phrases.append(phrase)
            seen.add(phrase.lower())

    return unique_phrases[:5]


def is_principle_list_question(question: str) -> bool:
    """
    Detect questions asking for the names or complete list
    of the Personal Data Protection Principles.
    """

    has_principles = bool(
        re.search(r"\bprinciples?\b", question, re.IGNORECASE)
    )

    has_list_indicator = bool(
        re.search(
            r"\b(seven|7|list|names?|all)\b",
            question,
            re.IGNORECASE
        )
    )

    return has_principles and has_list_indicator


def extract_section_numbers(question: str) -> list[int]:
    """
    Detect explicit PDPA section numbers and principle names.

    Also detects Section 5 when a question asks for the
    list of Personal Data Protection Principles.
    """

    detected_sections = []

    # Extract multiple numbers from wording such as:
    # "sections 6, 7, 8, 9, 10, 11 and 12"

    explicit_sections = re.finditer(
        r"\bsections?\s+([0-9][0-9,\s]*(?:and\s*[0-9]+)?)",
        question,
        flags=re.IGNORECASE
    )

    for match in explicit_sections:

        numbers = re.findall(r"\d+", match.group(1))

        for number in numbers:

            section_number = int(number)

            if section_number not in detected_sections:
                detected_sections.append(section_number)

    # Detect named PDPA principles.

    principle_keywords = {
        "general": 6,
        "notice and choice": 7,
        "disclosure": 8,
        "security": 9,
        "retention": 10,
        "data integrity": 11,
        "access": 12
    }

    for keyword, section_number in principle_keywords.items():

        pattern = r"\b" + re.escape(keyword) + r"\b"

        if re.search(pattern, question, flags=re.IGNORECASE):

            if section_number not in detected_sections:
                detected_sections.append(section_number)

    # Section 5 contains the official list of all seven principles.

    if is_principle_list_question(question):

        if 5 not in detected_sections:
            detected_sections.append(5)

    return detected_sections


def make_chunk_key(chunk: dict):
    return (
        chunk["document_id"],
        chunk["page_number"],
        chunk["chunk_index"]
    )


def add_or_update_chunk(results: dict, chunk: dict):

    key = make_chunk_key(chunk)

    if key not in results:
        results[key] = chunk
        return

    existing = results[key]

    existing["hybrid_score"] = max(
        existing["hybrid_score"],
        chunk["hybrid_score"]
    )

    existing["similarity"] = max(
        existing["similarity"],
        chunk["similarity"]
    )

    for rank_field in [
        "keyword_rank",
        "phrase_rank",
        "vector_rank"
    ]:

        new_rank = chunk.get(rank_field)

        if new_rank is not None:

            old_rank = existing.get(rank_field)

            if old_rank is None:
                existing[rank_field] = new_rank
            else:
                existing[rank_field] = min(old_rank, new_rank)

    if chunk.get("is_adjacent"):
        existing["is_adjacent"] = True


def search_similar_chunks(
    question: str,
    top_k: int = 3
) -> list[dict]:

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    candidate_limit = min(
        max(top_k * 5, CANDIDATE_LIMIT),
        100
    )

    question_embedding = generate_embeddings([question])[0]

    keyword_query = build_keyword_query(question)
    phrase_queries = build_phrase_queries(question)
    section_numbers = extract_section_numbers(question)

    print(f"Keyword query: {keyword_query}")
    print(f"Phrase queries: {phrase_queries}")
    print(f"Detected sections: {section_numbers}")

    combined_results = {}

    with get_connection() as connection:

        register_vector(connection)

        # SEARCH 1: VECTOR SEARCH

        vector_results = connection.execute(
            """
            SELECT
                d.id,
                d.filename,
                c.page_number,
                c.chunk_index,
                c.content,
                1 - (e.embedding <=> %s) AS similarity

            FROM document_embeddings e

            JOIN document_chunks c
                ON e.chunk_id = c.id

            JOIN documents d
                ON c.document_id = d.id

            ORDER BY e.embedding <=> %s

            LIMIT %s;
            """,
            (
                Vector(question_embedding),
                Vector(question_embedding),
                candidate_limit
            )
        ).fetchall()

        for rank, row in enumerate(vector_results, start=1):

            chunk = {
                "document_id": row[0],
                "filename": row[1],
                "page_number": row[2],
                "chunk_index": row[3],
                "content": row[4],
                "similarity": float(row[5]),
                "vector_rank": rank,
                "keyword_rank": None,
                "phrase_rank": None,
                "hybrid_score": 1 / (RRF_K + rank),
                "is_adjacent": False
            }

            add_or_update_chunk(combined_results, chunk)

        # SEARCH 2: KEYWORD SEARCH

        keyword_results = connection.execute(
            """
            SELECT
                d.id,
                d.filename,
                c.page_number,
                c.chunk_index,
                c.content,
                ts_rank_cd(
                    to_tsvector('simple', c.content),
                    websearch_to_tsquery('simple', %s)
                ) AS keyword_score,
                1 - (e.embedding <=> %s) AS similarity

            FROM document_chunks c

            JOIN documents d
                ON c.document_id = d.id

            JOIN document_embeddings e
                ON e.chunk_id = c.id

            WHERE to_tsvector('simple', c.content)
                @@ websearch_to_tsquery('simple', %s)

            ORDER BY keyword_score DESC

            LIMIT %s;
            """,
            (
                keyword_query,
                Vector(question_embedding),
                keyword_query,
                candidate_limit
            )
        ).fetchall()

        for rank, row in enumerate(keyword_results, start=1):

            chunk = {
                "document_id": row[0],
                "filename": row[1],
                "page_number": row[2],
                "chunk_index": row[3],
                "content": row[4],
                "similarity": float(row[6]),
                "vector_rank": None,
                "keyword_rank": rank,
                "phrase_rank": None,
                "hybrid_score": 1 / (RRF_K + rank),
                "is_adjacent": False
            }

            add_or_update_chunk(combined_results, chunk)

        # SEARCH 3: PHRASE SEARCH

        for phrase in phrase_queries:

            phrase_results = connection.execute(
                """
                SELECT
                    d.id,
                    d.filename,
                    c.page_number,
                    c.chunk_index,
                    c.content,
                    ts_rank_cd(
                        to_tsvector('simple', c.content),
                        phraseto_tsquery('simple', %s)
                    ) AS phrase_score,
                    1 - (e.embedding <=> %s) AS similarity

                FROM document_chunks c

                JOIN documents d
                    ON c.document_id = d.id

                JOIN document_embeddings e
                    ON e.chunk_id = c.id

                WHERE to_tsvector('simple', c.content)
                    @@ phraseto_tsquery('simple', %s)

                ORDER BY phrase_score DESC

                LIMIT %s;
                """,
                (
                    phrase,
                    Vector(question_embedding),
                    phrase,
                    candidate_limit
                )
            ).fetchall()

            for rank, row in enumerate(phrase_results, start=1):

                chunk = {
                    "document_id": row[0],
                    "filename": row[1],
                    "page_number": row[2],
                    "chunk_index": row[3],
                    "content": row[4],
                    "similarity": float(row[6]),
                    "vector_rank": None,
                    "keyword_rank": None,
                    "phrase_rank": rank,
                    "hybrid_score": 1 / (RRF_K + rank),
                    "is_adjacent": False
                }

                add_or_update_chunk(combined_results, chunk)

        # SEARCH 4: SECTION-AWARE SEARCH

        for section_number in section_numbers:

            print(
                f"Searching for the actual heading of "
                f"Section {section_number}..."
            )

            if section_number == 5:

                # Section 5 is the official list of seven principles.

                heading_pattern = (
                    r"Personal Data Protection Principles"
                    r"[[:space:]]+5\."
                )

            else:

                principle_name = next(
                    (
                        name
                        for name, number in PDPA_PRINCIPLES.items()
                        if number == section_number
                    ),
                    None
                )

                if principle_name:

                    heading_pattern = (
                        rf"(^|[[:space:]])"
                        rf"{re.escape(principle_name)}"
                        rf"[[:space:]]+{section_number}\."
                    )

                else:

                    heading_pattern = (
                        rf"(^|[[:space:]])"
                        rf"[A-Z][a-z]+"
                        rf"([[:space:]]+[A-Z][a-z]+){{0,5}}"
                        rf"[[:space:]]+{section_number}\."
                    )

            section_anchors = connection.execute(
                """
                SELECT
                    d.id,
                    d.filename,
                    c.page_number,
                    c.chunk_index,
                    c.content,
                    1 - (e.embedding <=> %s) AS similarity

                FROM document_chunks c

                JOIN documents d
                    ON c.document_id = d.id

                JOIN document_embeddings e
                    ON e.chunk_id = c.id

                WHERE c.content ~* %s

                ORDER BY c.page_number, c.chunk_index

                LIMIT 10;
                """,
                (
                    Vector(question_embedding),
                    heading_pattern
                )
            ).fetchall()

            print(
                f"Section {section_number} heading matches: "
                f"{len(section_anchors)}"
            )

            for anchor in section_anchors:

                document_id = anchor[0]
                anchor_index = anchor[3]

                section_chunks = connection.execute(
                    """
                    SELECT
                        d.id,
                        d.filename,
                        c.page_number,
                        c.chunk_index,
                        c.content,
                        1 - (e.embedding <=> %s) AS similarity

                    FROM document_chunks c

                    JOIN documents d
                        ON c.document_id = d.id

                    JOIN document_embeddings e
                        ON e.chunk_id = c.id

                    WHERE c.document_id = %s
                      AND c.chunk_index BETWEEN %s AND %s

                    ORDER BY c.chunk_index;
                    """,
                    (
                        Vector(question_embedding),
                        document_id,
                        anchor_index,
                        anchor_index + SECTION_CONTEXT_LIMIT
                    )
                ).fetchall()

                for row in section_chunks:

                    distance = row[3] - anchor_index

                    section_score = 1.0 - (distance * 0.05)

                    chunk = {
                        "document_id": row[0],
                        "filename": row[1],
                        "page_number": row[2],
                        "chunk_index": row[3],
                        "content": row[4],
                        "similarity": float(row[5]),
                        "vector_rank": None,
                        "keyword_rank": None,
                        "phrase_rank": None,
                        "hybrid_score": section_score,
                        "is_adjacent": distance > 0
                    }

                    add_or_update_chunk(
                        combined_results,
                        chunk
                    )

        # INITIAL RANKING

        ranked_results = sorted(
            combined_results.values(),
            key=lambda chunk: chunk["hybrid_score"],
            reverse=True
        )

        # SEARCH 5: NEIGHBOUR EXPANSION

        selected_results = ranked_results[
            :min(
                NEIGHBOUR_EXPANSION_LIMIT,
                len(ranked_results)
            )
        ]

        for parent in selected_results:

            neighbour_indices = [
                parent["chunk_index"] - 1,
                parent["chunk_index"] + 1
            ]

            neighbour_rows = connection.execute(
                """
                SELECT
                    d.id,
                    d.filename,
                    c.page_number,
                    c.chunk_index,
                    c.content,
                    1 - (e.embedding <=> %s) AS similarity

                FROM document_chunks c

                JOIN documents d
                    ON c.document_id = d.id

                JOIN document_embeddings e
                    ON e.chunk_id = c.id

                WHERE c.document_id = %s
                  AND c.chunk_index IN (%s, %s);
                """,
                (
                    Vector(question_embedding),
                    parent["document_id"],
                    neighbour_indices[0],
                    neighbour_indices[1]
                )
            ).fetchall()

            for row in neighbour_rows:

                chunk = {
                    "document_id": row[0],
                    "filename": row[1],
                    "page_number": row[2],
                    "chunk_index": row[3],
                    "content": row[4],
                    "similarity": float(row[5]),
                    "vector_rank": None,
                    "keyword_rank": None,
                    "phrase_rank": None,
                    "hybrid_score": parent["hybrid_score"] * 0.95,
                    "is_adjacent": True
                }

                add_or_update_chunk(
                    combined_results,
                    chunk
                )

        # FINAL RANKING AND CONTENT-BASED DEDUPLICATION

        final_results = sorted(
            combined_results.values(),
            key=lambda chunk: chunk["hybrid_score"],
            reverse=True
        )

        unique_results = []
        seen_chunks = set()

        for chunk in final_results:

            normalized_content = re.sub(
                r"\s+",
                " ",
                chunk["content"]
            ).strip()

            key = (
                chunk["filename"],
                chunk["page_number"],
                chunk["chunk_index"],
                normalized_content
            )

            if key in seen_chunks:
                continue

            seen_chunks.add(key)
            unique_results.append(chunk)

            if len(unique_results) >= top_k:
                break

        print(
            f"Final unique chunks returned: "
            f"{len(unique_results)}"
        )

        return unique_results
