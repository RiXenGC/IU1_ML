import os
import subprocess
import zipfile
import kagglehub
from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

from lab6.config import Config


def download_and_extract() -> None:

    if os.path.exists(Config.target_dir) and len(os.listdir(Config.target_dir)) > 0:
        print(f" Набор данных уже скачан и находится в: {os.path.abspath(Config.target_dir)}")
        return

    os.makedirs(Config.target_dir, exist_ok=True)
    os.makedirs(Config.processed_dir, exist_ok=True)

    # Датасет
    # files_to_download = [
    #     "sample.zarr",      # Демо-набор для исследования
    #     "validate.zarr",    # Валидационный набор (большой, можно пропустить)
    #     "aerial_map",       # Карты для визуализации
    #     "semantic_map",     # Семантические карты
    #     "multi_mode_sample_submission.csv",
    #     "single_mode_sample_submission.csv"
    # ]
    
    path = kagglehub.competition_download('lyft-motion-prediction-autonomous-vehicles')
    
    # files_to_download = ["sample.zarr"]
        
    # cmd = [
    #     "kaggle", "competitions", "download",
    #     Config.competition_name,
    #     "-p", Config.target_dir,
    #     "--force"   # Перезаписывает существующие файлы
    # ]
    
    # # Добавляем флаг -f для установки только указанных в files_to_download файлов
    # for file in files_to_download:
    #     print(f"Скачиваем {file}...")
        
    #     cmd.extend(["-f", file])
    
    #     try:
    #         subprocess.run(cmd, check=True)
    #         print("Демо-данные успешно скачаны")
    #         print(f"Файлы сохранены в: {Config.target_dir}")
            
    #     except subprocess.CalledProcessError as e:
    #         print(f"Ошибка при скачивании: {e}")
    #         # print("\nАльтернативный способ - скачать все файлы:")
    #         #           path = kagglehub.competition_download(
    #         #           Config.competition_name,
    #         #           path=Config.target_dir,
    #         #           force_download=True
            #       )


if __name__ == "__main__":
    download_and_extract()
