import os
import zipfile

from dotenv import load_dotenv

from sem1.lab3.config import Config

_ = load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402


target_dir = Config().target_dir
dataset_name = Config().dataset_name
zip_path = Config().zip_path


def download_and_extract() -> None:

    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        print(f" Набор данных уже скачан и находится в: {os.path.abspath(target_dir)}")
        return

    print("Авторизация в kaggle")
    api = KaggleApi()
    api.authenticate()

    os.makedirs(target_dir, exist_ok=True)

    print("Скачивание датасета...")
    api.dataset_download_files(dataset_name, target_dir, quiet=False)  # pyright: ignore[reportUnknownMemberType]

    if os.path.exists(zip_path):
        print("Распаковка")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)

    os.remove(zip_path)


if __name__ == "__main__":
    download_and_extract()
