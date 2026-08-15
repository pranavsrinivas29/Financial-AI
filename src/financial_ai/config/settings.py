import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parents[3]


class Settings:
    APP_ENV = os.getenv("APP_ENV", "development")

    DATA_DIR = BASE_DIR / os.getenv(
        "DATA_DIR",
        "data",
    )

    RAW_DATA_DIR = BASE_DIR / os.getenv(
        "RAW_DATA_DIR",
        "data/raw",
    )

    PROCESSED_DATA_DIR = BASE_DIR / os.getenv(
        "PROCESSED_DATA_DIR",
        "data/processed",
    )

    FEATURE_DATA_DIR = BASE_DIR / os.getenv(
        "FEATURE_DATA_DIR",
        "data/features",
    )

    DEFAULT_TICKER = os.getenv(
        "DEFAULT_TICKER",
        "AAPL",
    )

    DEFAULT_START_DATE = os.getenv(
        "DEFAULT_START_DATE",
        "2016-01-01",
    )


settings = Settings()