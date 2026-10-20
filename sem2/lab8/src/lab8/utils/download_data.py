"""Загрузка Seattle Weather через официальный Kaggle API."""

from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "ananthr1/weather-prediction"
DATA_DIR = PROJECT_ROOT / "data" / "raw"
CSV_PATH = DATA_DIR / "seattle-weather.csv"


def download_and_extract() -> Path:
    """Скачать датасет, если CSV ещё отсутствует, и вернуть путь к нему."""
    if CSV_PATH.is_file() and CSV_PATH.stat().st_size > 0:
        print(f"Датасет уже скачан: {CSV_PATH}")
        return CSV_PATH

    # Kaggle может проверять авторизацию уже при импорте пакета.
    # Поэтому сначала читаем .env, а при наличии CSV вообще не импортируем API.
    load_dotenv(PROJECT_ROOT / ".env")
    from kaggle.api.kaggle_api_extended import KaggleApi

    print("Авторизация в Kaggle...")
    api = KaggleApi()
    api.authenticate()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Скачивание {DATASET_NAME}...")
    api.dataset_download_files(
        dataset=DATASET_NAME,
        path=str(DATA_DIR),
        unzip=True,
        quiet=False,
    )

    if not CSV_PATH.is_file() or CSV_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"После загрузки не найден непустой CSV: {CSV_PATH}")

    print(f"Датасет готов: {CSV_PATH}")
    return CSV_PATH


if __name__ == "__main__":
    download_and_extract()
