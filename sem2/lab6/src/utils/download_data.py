import os
import tarfile
import zipfile

from config import Config

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402


def download_data() -> None:

    if os.path.exists(Config.target_dir) and len(os.listdir(Config.target_dir)) > 0:
        print(f" Набор данных уже скачан и находится в: {os.path.abspath(Config.target_dir)}")
        return

    os.makedirs(Config.target_dir, exist_ok=True)
    os.makedirs(Config.processed_dir, exist_ok=True)

    # Полный набор:
    # files_to_download = [
    #     "sample.zarr",      # Демо-набор для исследования
    #     "validate.zarr",    # Валидационный набор (большой, можно пропустить)
    #     "aerial_map",       # Карты для визуализации
    #     "semantic_map",     # Семантические карты
    #     "multi_mode_sample_submission.csv",
    #     "single_mode_sample_submission.csv"
    # ]
    files_to_download = [
        "sample.zarr.zip",      
        # "aerial_map.tar.xz",      
        # "semantic_map.tar.xz",     
    ]
    
    # Почему-то не работает kagglehub ???
    api = KaggleApi()
    api.authenticate()

    for file in files_to_download:
        print(f"Скачиваем {file}...")
        api.competition_download_file(Config.competition_name, file, Config.target_dir)

    # Распаковка: zarr в data/raw/scenes/, карты в data/raw/
    print("Распаковываем...")
    scenes_dir = os.path.join(Config.target_dir, "scenes")
    os.makedirs(scenes_dir, exist_ok=True)

    with zipfile.ZipFile(os.path.join(Config.target_dir, "sample.zarr.zip")) as zf:
        zf.extractall(scenes_dir)

    for map_archive in ["aerial_map.tar.xz", "semantic_map.tar.xz"]:
        with tarfile.open(os.path.join(Config.target_dir, map_archive), "r:xz") as tf:
            tf.extractall(Config.target_dir)

    # Архивы больше не нужны
    for archive in files_to_download:
        os.remove(os.path.join(Config.target_dir, archive))

    print(f"Готово. Данные в: {os.path.abspath(Config.target_dir)}")
    

if __name__ == "__main__":
    download_data()
