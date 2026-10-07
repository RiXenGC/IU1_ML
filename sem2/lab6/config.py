class Config:
    competition_name: str = "lyft-motion-prediction-autonomous-vehicles"
    
    # Dirs
    target_dir: str = "data/raw"
    # sample_zarr: str = os.path.join(target_dir, "sample.zarr")
    # validate_zarr: str = os.path.join(target_dir, "validate.zarr")
    processed_dir: str = "data/processed"
    
    # Maps
    # aerial_map: str = os.path.join(target_dir, "aerial_map")
    # semantic_map: str = os.path.join(target_dir, "semantic_map")
