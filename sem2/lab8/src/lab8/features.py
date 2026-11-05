import numpy as np
import pandas as pd


def window_features(window, mode="dynamics"):

    features = window[["precipitation", "temp_max", "temp_min", "wind"]].astype(float).copy()
    features["precipitation"] = np.log1p(features["precipitation"])
    year_days = np.where(window.index.is_leap_year, 366, 365)
    phase = 2 * np.pi * (window.index.dayofyear - 1) / year_days
    features["year_sin"] = np.sin(phase)
    features["year_cos"] = np.cos(phase)

    if mode in ("dynamics", "extended"):
        for column in ["precipitation", "wind"]:
            smooth = window[column].rolling(3, min_periods=1, center=False).mean()
            features[f"{column}_mean3"] = smooth
            features[f"{column}_change3"] = smooth.diff().fillna(0.0)

    if mode == "extended":
        smooth_temperature = window["temp_max"].rolling(3, min_periods=1).mean()
        features["temp_max_mean3"] = smooth_temperature
        features["temp_max_change3"] = smooth_temperature.diff().fillna(0.0)
        features["temperature_range"] = window["temp_max"] - window["temp_min"]
        features["wind_std3"] = window["wind"].rolling(3, min_periods=1).std(ddof=0)
        
    if mode not in ("base", "dynamics", "extended"):
        raise ValueError("Неизвестный набор признаков")
    
    return features


def prepare_windows(df, mode="dynamics", lookback=30):

    windows = [window_features(df.iloc[i-lookback:i], mode) for i in range(lookback, len(df))]
    names = list(windows[0].columns)
    X_raw = np.stack([w.to_numpy(dtype=float) for w in windows])
    dates = df.index[lookback:]
    train_mask = dates < pd.Timestamp("2014-01-01")

    # Каждый элемент обучающих окон имеет одинаковый вес при стандартизации.
    training_values = X_raw[train_mask].reshape(-1, len(names))
    means = training_values.mean(axis=0)
    stds = training_values.std(axis=0)
    stds[stds == 0] = 1.0
    train_target = df.loc[df.index < "2014-01-01", "temp_max"]
    y_mean = float(train_target.mean())
    y_std = float(train_target.std(ddof=0))
    X = ((X_raw - means) / stds).astype(np.float32)
    y = ((df["temp_max"].iloc[lookback:].to_numpy() - y_mean) / y_std).astype(np.float32)
    masks = {
        "train": train_mask,
        "val": (dates >= "2014-01-01") & (dates < "2015-01-01"),
        "test": dates >= "2015-01-01",
    }
    arrays = {}
    for name, mask in masks.items():
        arrays[f"X_{name}"] = X[mask]
        arrays[f"y_{name}"] = y[mask]
        arrays[f"dates_{name}"] = dates[mask].strftime("%Y-%m-%d").to_numpy(dtype="U10")
    meta = {
        "target": "temp_max", "lookback": lookback, "horizon": 1,
        "features": names, "feature_mode": mode, "feature_version": "window_causal_v1",
        "smoothing_days": 3, "scaler_fit": "training_windows_only",
        "x_mean": dict(zip(names, means.tolist())), "x_std": dict(zip(names, stds.tolist())),
        "y_mean": y_mean, "y_std": y_std,
        "train_end_exclusive": "2014-01-01", "val_end_exclusive": "2015-01-01",
    }
    
    return arrays, meta
