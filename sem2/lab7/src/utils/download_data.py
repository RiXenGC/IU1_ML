import os
from pathlib import Path
import zipfile

from dotenv import load_dotenv

from sem1.lab4.config import Config

_ = load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402


target_dir = Config().target_dir
dataset_name = Config().dataset_name


def download_and_extract() -> None:
    target_path = Path(target_dir)
    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        print(f" Набор данных уже скачан и находится в: {os.path.abspath(target_dir)}")
        return

    print("Авторизация в kaggle")
    api = KaggleApi()
    api.authenticate()

    os.makedirs(target_dir, exist_ok=True)

    print("Скачивание датасета...")
    api.competition_download_files(dataset_name, target_dir, quiet=False)

    while True:
        zip_files = list(target_path.rglob("*.zip"))

        if not zip_files:
            break

        for zip_file in zip_files:
            print(f" Распаковка: {zip_file.relative_to(target_path)}")
            try:
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(zip_file.parent)
                zip_file.unlink()
            except zipfile.BadZipFile:
                print(
                    f"Ошибка: Файл {zip_file.name} поврежден или не является ZIP-архивом."
                )
                zip_file.unlink()


if __name__ == "__main__":
    download_and_extract()
