import os


class Config:
    dataset_name: str = "salikhussaini49/prediction-of-sepsis"
    target_dir: str = "lab5/data/raw"
    zip_path: str = os.path.join(target_dir, "prediction-of-sepsis.zip")
    processed_dir: str = "lab5/data/processed"
