from vertexai.language_models import TextEmbeddingModel
from typing import List
from src.askdata import logger  # Import your logger
import vertexai

def create_embeddings(texts: List[str], model_name: str) -> List[List[float]]:
    """Creates embeddings using a Vertex AI TextEmbeddingModel."""
    try:
        model = TextEmbeddingModel.from_pretrained(model_name)
        embeddings = model.get_embeddings(texts)
        return [embedding.values for embedding in embeddings]
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise