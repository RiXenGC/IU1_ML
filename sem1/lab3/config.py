import os

class Config:
    dataset_name: str = "plameneduardo/sarscov2-ctscan-dataset"
    target_dir: str = "lab3/data/raw"
    zip_path: str = os.path.join(target_dir, "sarscov2-ctscan-dataset.zip")
