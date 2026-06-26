from sentence_transformers import SentenceTransformer
from functools import lru_cache
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache()
def get_embedding_model() -> SentenceTransformer:
    settings = get_settings()
    logger.info(f"Loading embedding model: {settings.embed_model}")
    model = SentenceTransformer(settings.embed_model)
    logger.info("Embedding model loaded successfully")
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    model = get_embedding_model()
    embedding = model.encode([query], convert_to_numpy=True)
    return embedding[0].tolist()
