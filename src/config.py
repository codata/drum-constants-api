import re

from pydantic import Field, field_validator
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

    # Limits for client-supplied queries on /sparql (see docs/SPARQL_LIMITS.md).
    # All limits apply per worker process.
    SPARQL_TIMEOUT_SECONDS: float = Field(10.0, gt=0, le=600)
    SPARQL_MAX_QUERY_MEMORY_MB: int = Field(512, ge=64)
    SPARQL_MAX_CONCURRENT_QUERIES: int = Field(4, ge=1, le=64)
    SPARQL_MAX_WAITING_QUERIES: int = Field(4, ge=0)
    SPARQL_MAX_WAIT_SECONDS: float = Field(2.0, ge=0, le=60)
    SPARQL_MAX_RESULT_ROWS: int = Field(100_000, ge=1)
    # The largest full-dataset response today is ~17 MB (SELECT * as XML).
    SPARQL_MAX_RESPONSE_BYTES: int = Field(32 * 1024 * 1024, ge=1024)
    # Run queries in-process, without time or memory limits, where query
    # isolation is unavailable (anything but Linux). Local development only.
    SPARQL_ALLOW_UNISOLATED: bool = False

    # Subpath hosting behind a reverse proxy (see docs/SUBPATH_HOSTING.md).
    # TRUST_X_FORWARDED_PREFIX=False ignores the header; use that with a fixed
    # ROOT_PATH. When it is used, the proxy must *set* (overwrite) the header,
    # and FORWARDED_PREFIX_ALLOWED should list the expected prefix(es), e.g.
    # ["/drum/constants"]; empty accepts any well-formed absolute path.
    TRUST_X_FORWARDED_PREFIX: bool = True
    FORWARDED_PREFIX_ALLOWED: List[str] = []

    @field_validator("FORWARDED_PREFIX_ALLOWED")
    @classmethod
    def _check_prefixes(cls, values: List[str]) -> List[str]:
        cleaned = []
        for v in values:
            v = v[:-1] if v.endswith("/") else v
            if not re.fullmatch(r"(/[A-Za-z0-9._~-]+)+", v) or "/./" in v + "/" or "/../" in v + "/":
                raise ValueError(f"not a plain absolute path: {v!r}")
            cleaned.append(v)
        return cleaned

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
