from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_SPOT_PATH = Path("Vb-Academy PPAs Application/DayAheadPrices_2021_2025.csv")
DEFAULT_SOLAR_PATH = Path("enwex_GER_solar_v25_combined.csv")


def ensure_hourly_mean(series: pd.Series, *, ffill_limit: int = 0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna()
    s.index = pd.DatetimeIndex(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    s = s.sort_index()
    if s.index.duplicated().any():
        s = s[~s.index.duplicated(keep="first")]
    s = s.asfreq("1h")
    if ffill_limit > 0:
        s = s.ffill(limit=ffill_limit)
    return s.dropna()


def load_day_ahead_prices(path: str | Path = DEFAULT_SPOT_PATH) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    series = pd.to_numeric(df.iloc[:, 0], errors="coerce").rename("spot_price")
    return ensure_hourly_mean(series, ffill_limit=0)


def load_solar_capacity_factor(path: str | Path = DEFAULT_SOLAR_PATH) -> pd.Series:
    df = pd.read_csv(path, sep=";", decimal=",")
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%d.%m.%Y %H:%M", errors="coerce")
    solar_col = next(col for col in df.columns if "solar" in col.lower())
    series = pd.Series(df[solar_col].to_numpy(dtype=float), index=df["DateTime"], name="solar_cf")
    series = ensure_hourly_mean(series, ffill_limit=3)
    if float(series.max()) > 1.5:
        series = series / 100.0
    return series.clip(0.0, 1.0)


@dataclass
class JointSolarMarketData:
    spot: pd.Series
    solar_cf: pd.Series
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "spot": self.spot,
                "solar_cf": self.solar_cf,
            }
        )


def load_joint_market_data_solar(
    *,
    spot_path: str | Path = DEFAULT_SPOT_PATH,
    solar_path: str | Path = DEFAULT_SOLAR_PATH,
    start: Optional[str | pd.Timestamp] = None,
    end: Optional[str | pd.Timestamp] = None,
) -> JointSolarMarketData:
    spot = load_day_ahead_prices(spot_path)
    solar = load_solar_capacity_factor(solar_path)
    idx = spot.index.intersection(solar.index)
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    spot = spot.loc[idx].rename("spot")
    solar = solar.loc[idx].rename("solar_cf")
    return JointSolarMarketData(
        spot=spot,
        solar_cf=solar,
        start=idx.min(),
        end=idx.max(),
    )


__all__ = [
    "DEFAULT_SOLAR_PATH",
    "DEFAULT_SPOT_PATH",
    "JointSolarMarketData",
    "ensure_hourly_mean",
    "load_day_ahead_prices",
    "load_joint_market_data_solar",
    "load_solar_capacity_factor",
]
