
from app.services.embedding_service import generate_embeddings


sentences = [
    "Organisations must protect personal data.",
    "Companies should safeguard people's personal information."
]

embeddings = generate_embeddings(sentences)

print("Number of embeddings:", len(embeddings))

for index, embedding in enumerate(embeddings, start=1):
    print(f"\nEmbedding {index}")
    print("Dimensions:", len(embedding))
    print("First 5 values:", embedding[:5])

# Basic validation.
assert len(embeddings) == 2
assert len(embeddings[0]) == 384
assert len(embeddings[1]) == 384

print("\nAll embedding tests passed!")
