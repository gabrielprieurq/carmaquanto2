from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

BUNDLE_DIR = Path(__file__).resolve().parent
DEFAULT_SPOT_PATH = BUNDLE_DIR / "DayAheadPrices_2021_2025.csv"
DEFAULT_WIND_PATH = BUNDLE_DIR / "enwex_GER_wind_v25_combined.csv"


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


def load_wind_capacity_factor(path: str | Path = DEFAULT_WIND_PATH) -> pd.Series:
    df = pd.read_csv(path, sep=";", decimal=",")
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%d.%m.%Y %H:%M", errors="coerce")
    wind_col = next(col for col in df.columns if "wind" in col.lower())
    series = pd.Series(df[wind_col].to_numpy(dtype=float), index=df["DateTime"], name="wind_cf")
    series = ensure_hourly_mean(series, ffill_limit=3)
    if float(series.max()) > 1.5:
        series = series / 100.0
    return series.clip(0.0, 1.0)


def clip_capacity_factor(x: np.ndarray | pd.Series, eps: float = 1e-4) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), eps, 1.0 - eps)


def logit_capacity_factor(x: np.ndarray | pd.Series, eps: float = 1e-4) -> np.ndarray:
    z = clip_capacity_factor(x, eps=eps)
    return np.log(z / (1.0 - z))


@dataclass
class JointMarketData:
    spot: pd.Series
    wind_cf: pd.Series
    wind_logit: pd.Series
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "spot": self.spot,
                "wind_cf": self.wind_cf,
                "wind_logit": self.wind_logit,
            }
        )


def load_joint_market_data(
    *,
    spot_path: str | Path = DEFAULT_SPOT_PATH,
    wind_path: str | Path = DEFAULT_WIND_PATH,
    start: Optional[str | pd.Timestamp] = None,
    end: Optional[str | pd.Timestamp] = None,
) -> JointMarketData:
    spot = load_day_ahead_prices(spot_path)
    wind = load_wind_capacity_factor(wind_path)
    idx = spot.index.intersection(wind.index)
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    spot = spot.loc[idx].rename("spot")
    wind = wind.loc[idx].rename("wind_cf")
    wind_logit = pd.Series(logit_capacity_factor(wind), index=idx, name="wind_logit")
    return JointMarketData(
        spot=spot,
        wind_cf=wind,
        wind_logit=wind_logit,
        start=idx.min(),
        end=idx.max(),
    )
