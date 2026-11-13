
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Load the model only once.
_model = None


def get_embedding_model():
    global _model

    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer(MODEL_NAME)
        print("Embedding model loaded successfully!")

    return _model


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text passages into numerical vectors.
    """

    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("Text passages cannot be empty.")

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.tolist()
