from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

BUNDLE_DIR = Path(__file__).resolve().parent

try:
    from .intensity_data import ensure_hourly_mean
except ImportError:
    from intensity_data import ensure_hourly_mean


DEFAULT_OPENMETEO_TEMPERATURE_PATH = BUNDLE_DIR / "openmeteo_temperature_history_extended_model.csv"
DEFAULT_ENWEX_TEMPERATURE_PATH = BUNDLE_DIR / "enwex_GER_temp_v25_combined.csv"
SUPPORTED_TEMPERATURE_SOURCES = ("openmeteo", "enwex")


def download_openmeteo_temperature_history_extended_model(
    *,
    path: str | Path = DEFAULT_OPENMETEO_TEMPERATURE_PATH,
    latitude: float = 52.52,
    longitude: float = 13.41,
    start_date: str = "2020-08-10",
    end_date: str = "2025-08-20",
    timeout: float = 120.0,
) -> pd.DataFrame:
    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": "temperature_2m",
        "timezone": "UTC",
    }
    url = "https://archive-api.open-meteo.com/v1/era5?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=float(timeout)) as response:
        payload = json.load(response)
    hourly = payload["hourly"]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(hourly["time"], utc=True),
            "temperature_2m": pd.to_numeric(hourly["temperature_2m"], errors="coerce"),
        }
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return frame


def load_openmeteo_temperature_history_extended_model(
    path: str | Path = DEFAULT_OPENMETEO_TEMPERATURE_PATH,
    *,
    auto_download: bool = False,
    latitude: float = 52.52,
    longitude: float = 13.41,
    start_date: str = "2020-08-10",
    end_date: str = "2025-08-20",
) -> pd.Series:
    csv_path = Path(path)
    if not csv_path.exists():
        if not auto_download:
            raise FileNotFoundError(
                f"Historical Open-Meteo temperature cache not found at {csv_path}. "
                "Either create the cache first or call with auto_download=True."
            )
        download_openmeteo_temperature_history_extended_model(
            path=csv_path,
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
        )

    frame = pd.read_csv(csv_path)
    if "date" not in frame.columns or "temperature_2m" not in frame.columns:
        raise KeyError(
            f"Expected columns 'date' and 'temperature_2m' in {csv_path}, "
            f"found {sorted(frame.columns)}."
        )
    index = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    series = pd.Series(
        pd.to_numeric(frame["temperature_2m"], errors="coerce").to_numpy(dtype=float),
        index=index,
        name="temperature_history",
    )
    return ensure_hourly_mean(series, ffill_limit=1)


def load_enwex_temperature_extended_model(
    path: str | Path = DEFAULT_ENWEX_TEMPERATURE_PATH,
) -> pd.Series:
    df = pd.read_csv(path, sep=";", decimal=",")
    temp_col = next((c for c in df.columns if "temp" in c.lower()), None)
    if temp_col is None:
        raise KeyError(f"Could not identify a temperature column in {path}.")
    if "DateTimeUTC" in df.columns:
        idx = pd.to_datetime(df["DateTimeUTC"], format="%d.%m.%Y %H:%M", errors="coerce")
    elif "DateTime" in df.columns:
        idx = pd.to_datetime(df["DateTime"], format="%d.%m.%Y %H:%M", errors="coerce")
    else:
        raise KeyError(f"Expected either 'DateTimeUTC' or 'DateTime' in {path}.")
    series = pd.Series(
        pd.to_numeric(df[temp_col], errors="coerce").to_numpy(dtype=float),
        index=idx,
        name="temperature_enwex",
    )
    return ensure_hourly_mean(series, ffill_limit=3)


def temperature_summary_extended_model(
    series: pd.Series,
    *,
    source: str = "openmeteo",
    latitude: Optional[float] = 52.52,
    longitude: Optional[float] = 13.41,
) -> pd.DataFrame:
    s = pd.Series(series, copy=True).astype(float).dropna()
    idx = pd.DatetimeIndex(s.index)
    source_name = str(source).lower()
    row = {
        "source": source_name,
        "start": idx.min(),
        "end": idx.max(),
        "n_obs": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "min": float(s.min()),
        "max": float(s.max()),
    }
    row["latitude"] = float(latitude) if latitude is not None and source_name == "openmeteo" else np.nan
    row["longitude"] = float(longitude) if longitude is not None and source_name == "openmeteo" else np.nan
    return pd.DataFrame([row])


def openmeteo_temperature_summary_extended_model(
    series: pd.Series,
    *,
    latitude: float = 52.52,
    longitude: float = 13.41,
) -> pd.DataFrame:
    return temperature_summary_extended_model(
        series,
        source="openmeteo",
        latitude=latitude,
        longitude=longitude,
    )


__all__ = [
    "DEFAULT_ENWEX_TEMPERATURE_PATH",
    "DEFAULT_OPENMETEO_TEMPERATURE_PATH",
    "SUPPORTED_TEMPERATURE_SOURCES",
    "download_openmeteo_temperature_history_extended_model",
    "load_enwex_temperature_extended_model",
    "load_openmeteo_temperature_history_extended_model",
    "openmeteo_temperature_summary_extended_model",
    "temperature_summary_extended_model",
]
