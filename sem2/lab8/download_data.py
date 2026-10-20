from pathlib import Path
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi


PROJECT_ROOT = Path.cwd().resolve()
LAB_ROOT = Path( PROJECT_ROOT / "sem2" / "lab8" )
DATASET_NAME = "ananthr1/weather-prediction"

DATA_DIR = LAB_ROOT / "data" / "raw"
CSV_PATH = DATA_DIR / "seattle-weather.csv"


def download_and_extract() -> Path:

    if CSV_PATH.is_file() and CSV_PATH.stat().st_size > 0:
        print(f"Датасет уже скачан: {CSV_PATH}")
        return CSV_PATH

    # Kaggle может проверять авторизацию уже при импорте пакета.
    # Поэтому сначала читаем .env, а при наличии CSV вообще не импортируем API.
    load_dotenv(LAB_ROOT / ".env")

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
