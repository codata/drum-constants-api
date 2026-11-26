from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from pathlib import Path

class Settings(BaseSettings):
    API_TITLE: str = "Drum API"
    API_DESCRIPTION: str = "Open source API package for fundamental physical constants"
    CORS_ORIGINS: List[str] = ["*"]
    DATA_DIR: Path = Path(__file__).resolve().parent / "data"

    # SPARQL result cache (LRU). Sizes are in result rows / triples, an
    # approximate indicator of memory use (about 1 kB per triple for this
    # dataset), not a hard memory limit. SPARQL_CACHE_MAX_ENTRIES=0 disables it.
    SPARQL_CACHE_MAX_ENTRIES: int = Field(512, ge=0)
    SPARQL_CACHE_MAX_TOTAL_ROWS: int = Field(100_000, ge=0)
    SPARQL_CACHE_MAX_ENTRY_ROWS: int = Field(10_000, ge=0)

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
