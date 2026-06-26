from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    groq_api_key: str
    api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    embed_model: str = "all-MiniLM-L6-v2"
    faiss_index_path: str = "./data/faiss_index/index.faiss"
    faiss_meta_path: str = "./data/faiss_index/metadata.json"
    upload_dir: str = "./uploads"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 4
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()
