"""Application settings for the FruitDetector service."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment."""

    model_path: str = Field(
        default="models/fruitdetector.pt",
        description="Path to the YOLO model weights file.",
    )
    allowed_classes: tuple[str, ...] = ("apple", "banana", "tomato")
    default_imgsz: int = Field(default=320, gt=0)
    fallback_imgsz: int = Field(default=416, gt=0)
    confidence_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum confidence required to emit a detection. "
            "Default 0.0 to return every box the model predicts."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="FRUIT_",
        env_file=".env",
        protected_namespaces=("settings_",),
    )


@lru_cache
def get_settings() -> Settings:
    """Memoized settings instance."""

    return Settings()


