
from sentence_transformers import util

from app.services.embedding_service import generate_embeddings


sentences = [
    "Organisations must protect personal data.",
    "Companies should safeguard people's personal information.",
    "The weather is sunny and warm today."
]

# Generate embeddings.
embeddings = generate_embeddings(sentences)

# Calculate cosine similarity.
similarity_related = util.cos_sim(
    embeddings[0],
    embeddings[1]
).item()

similarity_unrelated = util.cos_sim(
    embeddings[0],
    embeddings[2]
).item()

print("Related sentence similarity:", round(similarity_related, 4))
print("Unrelated sentence similarity:", round(similarity_unrelated, 4))

# Check whether the related sentences are more similar.
assert similarity_related > similarity_unrelated

print("\nAll semantic similarity tests passed!")
