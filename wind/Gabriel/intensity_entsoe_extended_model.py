from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import re
import warnings

import pandas as pd
import requests
try:
    from entsoe import EntsoePandasClient
except ImportError:  # Cached CSV loading does not require entsoe-py.
    EntsoePandasClient = None

try:
    from .intensity_data import ensure_hourly_mean
except ImportError:
    from intensity_data import ensure_hourly_mean


BUNDLE_DIR = Path(__file__).resolve().parent
DEFAULT_ENTSOE_CACHE_PATH = BUNDLE_DIR / "entsoe_de_lu_load_extended_model.csv"
DEFAULT_ENTSOE_AREA = "DE_LU"
DEFAULT_DATAHANDLING_PATH = BUNDLE_DIR / "DataHandlingEntsoe.py"


@dataclass
class EntsoeLoadBundleExtendedModel:
    actual_load: pd.Series
    load_forecast: pd.Series
    renewable_forecast: pd.Series
    residual_load_forecast: pd.Series
    source: str
    country_code: str = DEFAULT_ENTSOE_AREA


def _to_utc_naive_index(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx.tz_localize(None)


def _regularize_hourly(series: pd.Series, *, name: str, ffill_limit: int = 1) -> pd.Series:
    s = pd.Series(pd.to_numeric(series, errors="coerce").to_numpy(dtype=float), index=_to_utc_naive_index(series.index), name=name)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return ensure_hourly_mean(s, ffill_limit=ffill_limit)


def _extract_api_key_from_datahandling(path: str | Path = DEFAULT_DATAHANDLING_PATH) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    text = p.read_text(errors="ignore")
    m = re.search(r"api_key\s*=\s*['\"]([^'\"]+)['\"]", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"Entsoe_key\s*=\s*['\"]([^'\"]+)['\"]", text)
    if m:
        return m.group(1).strip()
    return None


def _chunk_boundaries(start: pd.Timestamp, end: pd.Timestamp, *, days: int = 360) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    step = pd.Timedelta(days=days)
    while cur < end:
        nxt = min(cur + step, end)
        out.append((cur, nxt))
        cur = nxt
    return out


def _normalize_interval_endpoint(ts: str | pd.Timestamp) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        out = out.tz_localize("UTC")
    else:
        out = out.tz_convert("UTC")
    return out.tz_localize(None)


def _required_hourly_index(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    s = _normalize_interval_endpoint(start)
    e = _normalize_interval_endpoint(end)
    return pd.date_range(s, e, freq="1h")


def _bundle_covers_interval(
    bundle: EntsoeLoadBundleExtendedModel,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> bool:
    required = _required_hourly_index(start, end)
    if len(required) == 0:
        return False
    idx = bundle.actual_load.index
    if len(idx) == 0 or idx.min() > required.min() or idx.max() < required.max():
        return False
    missing = required.difference(idx)
    return len(missing) == 0


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No ENTSO-E frames to concatenate.")
    df = pd.concat(frames, axis=0)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _fetch_entsoe_frame(
    client: EntsoePandasClient,
    *,
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    query: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for s, e in _chunk_boundaries(start, end):
        fn = getattr(client, query)
        if query == "query_wind_and_solar_forecast":
            df = fn(country_code, start=s, end=e, psr_type=None, process_type="A01")
        elif query == "query_load_forecast":
            df = fn(country_code, start=s, end=e, process_type="A01")
        else:
            df = fn(country_code, start=s, end=e)
        if isinstance(df, pd.Series):
            df = df.to_frame()
        frames.append(df)
    return _concat_frames(frames)


def _renewable_forecast_from_frame(df: pd.DataFrame) -> pd.Series:
    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [" ".join(str(x) for x in col if str(x) != "").strip() for col in data.columns]
    lower = {c: str(c).lower() for c in data.columns}
    keep = [c for c, lc in lower.items() if ("wind" in lc or "solar" in lc or "photovoltaic" in lc)]
    if not keep:
        raise KeyError("Could not identify wind/solar forecast columns in ENTSO-E forecast frame.")
    series = data[keep].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    return _regularize_hourly(series.rename("renewable_forecast"), name="renewable_forecast", ffill_limit=1)


def load_entsoe_de_lu_bundle_from_csv(
    path: str | Path = DEFAULT_ENTSOE_CACHE_PATH,
) -> EntsoeLoadBundleExtendedModel:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ENTSO-E cache file not found: {p}")
    df = pd.read_csv(p)
    if "timestamp_utc" not in df.columns:
        raise KeyError(f"Expected 'timestamp_utc' column in {p}.")
    idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)).tz_convert("UTC").tz_localize(None)
    required = ["actual_load", "load_forecast", "renewable_forecast", "residual_load_forecast"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required ENTSO-E cache columns {missing} in {p}.")
    actual = _regularize_hourly(pd.Series(df["actual_load"].to_numpy(dtype=float), index=idx), name="de_load_actual")
    forecast = _regularize_hourly(pd.Series(df["load_forecast"].to_numpy(dtype=float), index=idx), name="de_load_forecast")
    renewable = _regularize_hourly(pd.Series(df["renewable_forecast"].to_numpy(dtype=float), index=idx), name="renewable_forecast")
    residual = _regularize_hourly(pd.Series(df["residual_load_forecast"].to_numpy(dtype=float), index=idx), name="residual_load_forecast")
    return EntsoeLoadBundleExtendedModel(
        actual_load=actual,
        load_forecast=forecast,
        renewable_forecast=renewable,
        residual_load_forecast=residual,
        source=f"csv:{p}",
    )


def download_entsoe_de_lu_bundle(
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    api_key: Optional[str] = None,
    cache_path: str | Path = DEFAULT_ENTSOE_CACHE_PATH,
    country_code: str = DEFAULT_ENTSOE_AREA,
    datahandling_path: str | Path = DEFAULT_DATAHANDLING_PATH,
) -> EntsoeLoadBundleExtendedModel:
    if EntsoePandasClient is None:
        raise ImportError(
            "entsoe-py is required only when auto-downloading ENTSO-E data. "
            "Use the bundled entsoe_de_lu_load_extended_model.csv cache or install entsoe-py."
        )
    token = api_key or os.environ.get("ENTSOE_API_KEY") or _extract_api_key_from_datahandling(datahandling_path)
    if not token:
        raise ValueError("No ENTSO-E API key available. Pass api_key, set ENTSOE_API_KEY, or keep a key in DataHandlingEntsoe.py.")
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    if s.tzinfo is None:
        s = s.tz_localize("Europe/Brussels")
    if e.tzinfo is None:
        e = e.tz_localize("Europe/Brussels")
    client = EntsoePandasClient(api_key=token, retry_count=5, retry_delay=5, timeout=60)
    actual_df = _fetch_entsoe_frame(client, country_code=country_code, start=s, end=e, query="query_load")
    forecast_df = _fetch_entsoe_frame(client, country_code=country_code, start=s, end=e, query="query_load_forecast")
    renew_df = _fetch_entsoe_frame(client, country_code=country_code, start=s, end=e, query="query_wind_and_solar_forecast")

    actual = _regularize_hourly(actual_df["Actual Load"].rename("de_load_actual"), name="de_load_actual")
    forecast = _regularize_hourly(forecast_df["Forecasted Load"].rename("de_load_forecast"), name="de_load_forecast")
    renewable = _renewable_forecast_from_frame(renew_df)

    idx = actual.index.intersection(forecast.index).intersection(renewable.index)
    actual = actual.loc[idx]
    forecast = forecast.loc[idx]
    renewable = renewable.loc[idx]
    residual = (forecast - renewable).rename("residual_load_forecast")

    out = pd.DataFrame(
        {
            "timestamp_utc": idx,
            "actual_load": actual.to_numpy(dtype=float),
            "load_forecast": forecast.to_numpy(dtype=float),
            "renewable_forecast": renewable.to_numpy(dtype=float),
            "residual_load_forecast": residual.to_numpy(dtype=float),
        }
    )
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    return EntsoeLoadBundleExtendedModel(
        actual_load=actual,
        load_forecast=forecast,
        renewable_forecast=renewable,
        residual_load_forecast=residual,
        source=f"api:{country_code}",
        country_code=country_code,
    )


def load_entsoe_de_lu_bundle(
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cache_path: str | Path = DEFAULT_ENTSOE_CACHE_PATH,
    api_key: Optional[str] = None,
    auto_download: bool = False,
    country_code: str = DEFAULT_ENTSOE_AREA,
    datahandling_path: str | Path = DEFAULT_DATAHANDLING_PATH,
) -> EntsoeLoadBundleExtendedModel:
    p = Path(cache_path)
    token = api_key or os.environ.get("ENTSOE_API_KEY") or _extract_api_key_from_datahandling(datahandling_path)
    if auto_download:
        try:
            bundle = download_entsoe_de_lu_bundle(
                start=start,
                end=end,
                api_key=token,
                cache_path=p,
                country_code=country_code,
                datahandling_path=datahandling_path,
            )
        except (requests.exceptions.RequestException, TimeoutError, OSError) as exc:
            if not p.exists():
                raise
            warnings.warn(
                "ENTSO-E download failed; falling back to cached DE/LU load data and using the "
                "available cache overlap only. "
                f"Original error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            bundle = load_entsoe_de_lu_bundle_from_csv(p)
    elif p.exists():
        bundle = load_entsoe_de_lu_bundle_from_csv(p)
        if not _bundle_covers_interval(bundle, start=start, end=end):
            warnings.warn(
                "Cached ENTSO-E data does not fully cover the requested interval and "
                "`auto_download=False`, so the loader is using the available cache overlap only.",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        raise FileNotFoundError(
            f"ENTSO-E cache file {p} not found and auto_download=False. "
            "Provide a downloaded file or enable the ENTSO-E download path."
        )

    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    if s.tzinfo is not None:
        s = s.tz_convert("UTC").tz_localize(None)
    if e.tzinfo is not None:
        e = e.tz_convert("UTC").tz_localize(None)
    idx = bundle.actual_load.index
    idx = idx[(idx >= s) & (idx <= e)]
    if len(idx) == 0:
        raise ValueError("The selected ENTSO-E DE/LU sample is empty.")
    return EntsoeLoadBundleExtendedModel(
        actual_load=bundle.actual_load.loc[idx],
        load_forecast=bundle.load_forecast.loc[idx],
        renewable_forecast=bundle.renewable_forecast.loc[idx],
        residual_load_forecast=bundle.residual_load_forecast.loc[idx],
        source=bundle.source,
        country_code=bundle.country_code,
    )


__all__ = [
    "DEFAULT_DATAHANDLING_PATH",
    "DEFAULT_ENTSOE_AREA",
    "DEFAULT_ENTSOE_CACHE_PATH",
    "EntsoeLoadBundleExtendedModel",
    "download_entsoe_de_lu_bundle",
    "load_entsoe_de_lu_bundle",
    "load_entsoe_de_lu_bundle_from_csv",
]
