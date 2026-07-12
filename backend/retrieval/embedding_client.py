from backend.config import settings
from backend.retrieval.azure_openai_client import get_azure_openai_client


def generate_query_embedding(text: str) -> list[float]:
    if not text or not text.strip():
        raise ValueError("Cannot generate embedding for empty text.")

    if not settings.azure_openai_embedding_deployment:
        raise ValueError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT is not configured.")

    client = get_azure_openai_client()

    response = client.embeddings.create(
        model=settings.azure_openai_embedding_deployment,
        input=text.strip(),
        dimensions=settings.vector_dimensions,
    )

    embedding = response.data[0].embedding

    if len(embedding) != settings.vector_dimensions:
        raise ValueError(
            f"Embedding dimension mismatch. Expected {settings.vector_dimensions}, got {len(embedding)}."
        )

    return embedding
