from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


# Ezek csak alapértelmezett (default) értékek.
# Induláskor a BaseSettings automatikusan megkeresi a .env fájlt, és ha talál benne
# egyező nevű változót (kis/nagybetű mindegy), akkor azzal felülírja az itt lévőt.
# A .env.example-t a program nem nézi, az csak dokumentáció a többi fejlesztőnek, hogy setupolja fel a saját .env-jét
class Settings(BaseSettings):
    app_name: str = "Enterprise Knowledge Assistant"
    log_level: str = "INFO"

    llm_provider: Literal["gemini", "ollama"] = "gemini"
    embedding_provider: Literal["gemini", "ollama"] = "gemini"

    gemini_api_key: str = ""
    gemini_llm_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"

    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.2"
    ollama_embedding_model: str = "nomic-embed-text"

    data_dir: str = "data"
    chroma_persist_dir: str = "chroma_db"
    chroma_collection_name: str = "company_knowledge_base"
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_top_k: int = 3
    min_score: float = 0.25
    llm_timeout_seconds: float = 30.0

    # Itt adjuk meg, melyik fájlból olvasson, és hogy ha a .env-ben van valami plusz
    # felesleges sor, ami itt nincs definiálva, azt simán hagyja figyelmen kívül.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

# Az @lru_cache miatt csak az első híváskor olvassa be a .env fájlt a lemezről,
# utána már ugyanazt a kész példányt adja vissza a memóriából.
@lru_cache
def get_settings() -> Settings:
    return Settings()