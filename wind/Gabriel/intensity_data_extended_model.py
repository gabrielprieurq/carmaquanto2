from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

BUNDLE_DIR = Path(__file__).resolve().parent

try:
    from .intensity_data import (
        DEFAULT_SPOT_PATH,
        DEFAULT_WIND_PATH,
        ensure_hourly_mean,
        load_day_ahead_prices,
        load_wind_capacity_factor,
        logit_capacity_factor,
    )
    from .intensity_temperature_extended_model import (
        DEFAULT_ENWEX_TEMPERATURE_PATH as ENWEX_TEMPERATURE_PATH_DEFAULT,
        DEFAULT_OPENMETEO_TEMPERATURE_PATH,
        load_enwex_temperature_extended_model,
        load_openmeteo_temperature_history_extended_model,
    )
    from .intensity_entsoe_extended_model import (
        DEFAULT_DATAHANDLING_PATH,
        DEFAULT_ENTSOE_CACHE_PATH,
        load_entsoe_de_lu_bundle,
    )
except ImportError:
    from intensity_data import (
        DEFAULT_SPOT_PATH,
        DEFAULT_WIND_PATH,
        ensure_hourly_mean,
        load_day_ahead_prices,
        load_wind_capacity_factor,
        logit_capacity_factor,
    )
    from intensity_temperature_extended_model import (
        DEFAULT_ENWEX_TEMPERATURE_PATH as ENWEX_TEMPERATURE_PATH_DEFAULT,
        DEFAULT_OPENMETEO_TEMPERATURE_PATH,
        load_enwex_temperature_extended_model,
        load_openmeteo_temperature_history_extended_model,
    )
    from intensity_entsoe_extended_model import (
        DEFAULT_DATAHANDLING_PATH,
        DEFAULT_ENTSOE_CACHE_PATH,
        load_entsoe_de_lu_bundle,
    )


DEFAULT_LOAD_PATH = BUNDLE_DIR / "Important_Data_all_included.csv"
DEFAULT_REAL_LOAD_PATH = DEFAULT_ENTSOE_CACHE_PATH
DEFAULT_ENWEX_TEMPERATURE_PATH = ENWEX_TEMPERATURE_PATH_DEFAULT
DEFAULT_TEMPERATURE_PATH = DEFAULT_ENWEX_TEMPERATURE_PATH
SUPPORTED_TEMPERATURE_SOURCES = ("openmeteo", "enwex")
LEGACY_SOLAR_VOLUME_PATH = BUNDLE_DIR / "Solar Power Germany ERA5 SSRD 2020 B_SAMPLE.csv"
DEFAULT_SOLAR_PATH = BUNDLE_DIR / "enwex_GER_solar_v25_combined.csv"


def _parse_utc_index(values: pd.Series | np.ndarray) -> pd.DatetimeIndex:
    idx = pd.to_datetime(values, errors="coerce", utc=True)
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


def load_de_load_forecast_proxy(
    path: str | Path = DEFAULT_LOAD_PATH,
    *,
    column: str = "query_load_and_forecast_DE_LU",
) -> pd.Series:
    df = pd.read_csv(path)
    if column not in df.columns:
        raise KeyError(f"Column {column!r} not found in {path}.")
    if "Unnamed: 0" not in df.columns:
        raise KeyError(f"Expected timestamp column 'Unnamed: 0' in {path}.")
    idx = _parse_utc_index(df["Unnamed: 0"])
    series = pd.Series(pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float), index=idx, name="de_load_forecast_proxy")
    return ensure_hourly_mean(series, ffill_limit=1)


def load_temperature_forecast(
    path: str | Path = DEFAULT_ENWEX_TEMPERATURE_PATH,
) -> pd.Series:
    return load_enwex_temperature_extended_model(path).rename("temperature_forecast")


def load_temperature_history(
    path: str | Path = DEFAULT_OPENMETEO_TEMPERATURE_PATH,
    *,
    auto_download: bool = False,
) -> pd.Series:
    return load_openmeteo_temperature_history_extended_model(
        path=path,
        auto_download=auto_download,
    ).rename("temperature_history")


def load_solar_volume(
    path: str | Path = LEGACY_SOLAR_VOLUME_PATH,
) -> pd.Series:
    df = pd.read_csv(path)
    required = {"DATE", "HOUR", "HOURLY_SOLARPOWER"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required solar columns {sorted(missing)} in {path}.")
    date = pd.to_datetime(df["DATE"], format="%Y-%m-%d", errors="coerce")
    hour = pd.to_numeric(df["HOUR"], errors="coerce").fillna(0.0).astype(int)
    idx = date + pd.to_timedelta(hour, unit="h")
    series = pd.Series(pd.to_numeric(df["HOURLY_SOLARPOWER"], errors="coerce").to_numpy(dtype=float), index=idx, name="solar_volume")
    return ensure_hourly_mean(series, ffill_limit=1).clip(lower=0.0)


def load_solar_capacity_factor(
    path: str | Path = DEFAULT_SOLAR_PATH,
) -> pd.Series:
    df = pd.read_csv(path, sep=";", decimal=",")
    if "DateTime" not in df.columns:
        raise KeyError(f"Expected 'DateTime' in {path}.")
    solar_col = next((col for col in df.columns if "solar" in col.lower()), None)
    if solar_col is None:
        raise KeyError(f"Could not identify a solar-capacity-factor column in {path}.")
    idx = pd.to_datetime(df["DateTime"], format="%d.%m.%Y %H:%M", errors="coerce")
    series = pd.Series(pd.to_numeric(df[solar_col], errors="coerce").to_numpy(dtype=float), index=idx, name="solar_cf")
    series = ensure_hourly_mean(series, ffill_limit=3)
    if float(series.max()) > 1.5:
        series = series / 100.0
    return series.clip(0.0, 1.0)


@dataclass
class ExtendedJointMarketData:
    spot: pd.Series
    wind_cf: pd.Series
    wind_logit: pd.Series
    de_load: pd.Series
    de_load_actual: pd.Series
    de_load_forecast: pd.Series
    residual_load_forecast: pd.Series
    renewable_forecast: pd.Series
    log_de_load: pd.Series
    temperature: pd.Series
    temperature_source: str
    solar_cf: pd.Series
    load_source: str
    use_real_entsoe_load: bool
    start: pd.Timestamp
    end: pd.Timestamp
    full_index: pd.DatetimeIndex

    @property
    def solar_volume(self) -> pd.Series:
        """Backward-compatible alias used by older notebook cells."""
        return self.solar_cf

    @property
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "spot": self.spot,
                "wind_cf": self.wind_cf,
                "wind_logit": self.wind_logit,
                "de_load": self.de_load,
                "de_load_actual": self.de_load_actual,
                "de_load_forecast": self.de_load_forecast,
                "residual_load_forecast": self.residual_load_forecast,
                "renewable_forecast": self.renewable_forecast,
                "log_de_load": self.log_de_load,
                "temperature": self.temperature,
                "temperature_source": self.temperature_source,
                "solar_cf": self.solar_cf,
                "solar_volume": self.solar_cf,
            }
        )


def _resolve_temperature_source(
    *,
    use_historical_temperature: bool,
    temperature_source: Optional[str],
) -> str:
    if temperature_source is None:
        return "openmeteo" if bool(use_historical_temperature) else "enwex"
    source = str(temperature_source).strip().lower()
    if source not in SUPPORTED_TEMPERATURE_SOURCES:
        raise ValueError(f"Unsupported temperature_source {temperature_source!r}. Expected one of {SUPPORTED_TEMPERATURE_SOURCES}.")
    return source


def load_extended_market_data(
    *,
    spot_path: str | Path = DEFAULT_SPOT_PATH,
    wind_path: str | Path = DEFAULT_WIND_PATH,
    load_path: str | Path = DEFAULT_LOAD_PATH,
    real_load_path: str | Path = DEFAULT_REAL_LOAD_PATH,
    temperature_path: str | Path = DEFAULT_TEMPERATURE_PATH,
    solar_path: str | Path = DEFAULT_SOLAR_PATH,
    use_real_entsoe_load: bool = False,
    auto_download_real_load: bool = False,
    entsoe_api_key: Optional[str] = None,
    datahandling_path: str | Path = DEFAULT_DATAHANDLING_PATH,
    use_historical_temperature: bool = True,
    temperature_source: Optional[str] = None,
    auto_download_temperature: bool = False,
    start: Optional[str | pd.Timestamp] = None,
    end: Optional[str | pd.Timestamp] = None,
) -> ExtendedJointMarketData:
    spot = load_day_ahead_prices(spot_path)
    wind = load_wind_capacity_factor(wind_path)
    resolved_temperature_source = _resolve_temperature_source(
        use_historical_temperature=use_historical_temperature,
        temperature_source=temperature_source,
    )
    if resolved_temperature_source == "openmeteo":
        temperature = load_temperature_history(
            temperature_path if Path(temperature_path) != DEFAULT_ENWEX_TEMPERATURE_PATH else DEFAULT_OPENMETEO_TEMPERATURE_PATH,
            auto_download=auto_download_temperature,
        )
    else:
        temperature = load_temperature_forecast(temperature_path)
    solar = load_solar_capacity_factor(solar_path)

    non_load_start = max(spot.index.min(), wind.index.min(), temperature.index.min(), solar.index.min())
    non_load_end = min(spot.index.max(), wind.index.max(), temperature.index.max(), solar.index.max())
    if start is not None:
        non_load_start = max(non_load_start, pd.Timestamp(start))
    if end is not None:
        non_load_end = min(non_load_end, pd.Timestamp(end))
    if non_load_end < non_load_start:
        raise ValueError("The non-load series do not have a non-empty common interval.")

    if use_real_entsoe_load:
        bundle = load_entsoe_de_lu_bundle(
            start=non_load_start,
            end=non_load_end,
            cache_path=real_load_path,
            api_key=entsoe_api_key,
            auto_download=auto_download_real_load,
            datahandling_path=datahandling_path,
        )
        de_load_actual = bundle.actual_load.rename("de_load_actual")
        de_load_forecast = bundle.load_forecast.rename("de_load_forecast")
        residual_load_forecast = bundle.residual_load_forecast.rename("residual_load_forecast")
        renewable_forecast = bundle.renewable_forecast.rename("renewable_forecast")
        de_load = de_load_forecast.rename("de_load")
        load_source = bundle.source
    else:
        de_load = load_de_load_forecast_proxy(load_path)
        de_load_actual = de_load.rename("de_load_actual")
        de_load_forecast = de_load.rename("de_load_forecast")
        residual_load_forecast = pd.Series(np.nan, index=de_load.index, name="residual_load_forecast")
        renewable_forecast = pd.Series(np.nan, index=de_load.index, name="renewable_forecast")
        load_source = f"proxy:{load_path}"

    idx = spot.index
    idx = idx.intersection(wind.index)
    idx = idx.intersection(de_load.index)
    idx = idx.intersection(de_load_actual.index)
    idx = idx.intersection(de_load_forecast.index)
    idx = idx.intersection(residual_load_forecast.index)
    idx = idx.intersection(renewable_forecast.index)
    idx = idx.intersection(temperature.index)
    idx = idx.intersection(solar.index)
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    if len(idx) == 0:
        raise ValueError("The selected extended-data sample is empty.")

    spot = spot.loc[idx].rename("spot")
    wind = wind.loc[idx].rename("wind_cf")
    wind_logit = pd.Series(logit_capacity_factor(wind), index=idx, name="wind_logit")
    de_load = de_load.loc[idx].rename("de_load")
    de_load_actual = de_load_actual.loc[idx].rename("de_load_actual")
    de_load_forecast = de_load_forecast.loc[idx].rename("de_load_forecast")
    residual_load_forecast = residual_load_forecast.loc[idx].rename("residual_load_forecast")
    renewable_forecast = renewable_forecast.loc[idx].rename("renewable_forecast")
    log_de_load = pd.Series(np.log(np.clip(de_load.to_numpy(dtype=float), 1e-6, None)), index=idx, name="log_de_load")
    temperature = temperature.loc[idx].rename("temperature")
    solar = solar.loc[idx].rename("solar_cf")

    return ExtendedJointMarketData(
        spot=spot,
        wind_cf=wind,
        wind_logit=wind_logit,
        de_load=de_load,
        de_load_actual=de_load_actual,
        de_load_forecast=de_load_forecast,
        residual_load_forecast=residual_load_forecast,
        renewable_forecast=renewable_forecast,
        log_de_load=log_de_load,
        temperature=temperature,
        temperature_source=resolved_temperature_source,
        solar_cf=solar,
        load_source=load_source,
        use_real_entsoe_load=bool(use_real_entsoe_load),
        start=idx.min(),
        end=idx.max(),
        full_index=idx,
    )


def extended_sample_summary_table(data: ExtendedJointMarketData) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "start": data.start,
                "end": data.end,
                "n_hours": int(len(data.full_index)),
                "sample_years": float((data.end - data.start).total_seconds() / (365.25 * 24.0 * 3600.0)),
                "spot_min": float(data.spot.min()),
                "spot_max": float(data.spot.max()),
                "wind_min": float(data.wind_cf.min()),
                "wind_max": float(data.wind_cf.max()),
                "de_load_min": float(data.de_load.min()),
                "de_load_max": float(data.de_load.max()),
                "de_load_actual_min": float(data.de_load_actual.min()),
                "de_load_actual_max": float(data.de_load_actual.max()),
                "residual_load_forecast_min": float(data.residual_load_forecast.min()) if data.residual_load_forecast.notna().any() else np.nan,
                "residual_load_forecast_max": float(data.residual_load_forecast.max()) if data.residual_load_forecast.notna().any() else np.nan,
                "load_source": data.load_source,
                "use_real_entsoe_load": bool(data.use_real_entsoe_load),
                "temperature_min": float(data.temperature.min()),
                "temperature_max": float(data.temperature.max()),
                "solar_min": float(data.solar_cf.min()),
                "solar_max": float(data.solar_cf.max()),
            }
        ]
    )


__all__ = [
    "DEFAULT_LOAD_PATH",
    "DEFAULT_REAL_LOAD_PATH",
    "DEFAULT_ENWEX_TEMPERATURE_PATH",
    "LEGACY_SOLAR_VOLUME_PATH",
    "DEFAULT_SOLAR_PATH",
    "DEFAULT_TEMPERATURE_PATH",
    "SUPPORTED_TEMPERATURE_SOURCES",
    "ExtendedJointMarketData",
    "extended_sample_summary_table",
    "load_de_load_forecast_proxy",
    "load_extended_market_data",
    "load_solar_capacity_factor",
    "load_solar_volume",
    "load_temperature_history",
    "load_temperature_forecast",
]
