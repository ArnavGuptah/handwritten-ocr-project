from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Handwritten OCR API"
    app_version: str = "1.0.0"
    tesseract_cmd: str = "C:/Program Files/Tesseract-OCR/tesseract.exe"
    ocr_engine: str = "tesseract"
    ocr_language: str = "eng"
    upload_dir: Path = Path("uploads")
    output_dir: Path = Path("outputs")

    class Config:
        env_file = ".env"


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)