from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from pathlib import Path

class Settings(BaseSettings):
    API_TITLE: str = "Drum API"
    API_DESCRIPTION: str = "Open source API package for fundamental physical constants"
    CORS_ORIGINS: List[str] = ["*"]
    DATA_DIR: Path = Path(__file__).resolve().parent / "data"

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
