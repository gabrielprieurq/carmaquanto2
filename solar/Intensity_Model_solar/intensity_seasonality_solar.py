from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


OMEGA_YEAR = 2.0 * np.pi / (365.25 * 24.0)
OMEGA_WEEK = 2.0 * np.pi / (7.0 * 24.0)
OMEGA_DAY = 2.0 * np.pi / 24.0
SOLAR_EPS = 1e-4


def hours_from_origin(index: pd.DatetimeIndex, origin: Optional[pd.Timestamp] = None) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    ref = idx[0] if origin is None else pd.Timestamp(origin)
    return ((idx - ref).total_seconds() / 3600.0).to_numpy(dtype=float)


def harmonic_design(hours: np.ndarray) -> pd.DataFrame:
    h = np.asarray(hours, dtype=float)
    return pd.DataFrame(
        {
            "const": np.ones_like(h),
            "t": h,
            "t2": h**2,
            "year_cos": np.cos(OMEGA_YEAR * h),
            "year_sin": np.sin(OMEGA_YEAR * h),
            "week_cos": np.cos(OMEGA_WEEK * h),
            "week_sin": np.sin(OMEGA_WEEK * h),
            "day_cos": np.cos(OMEGA_DAY * h),
            "day_sin": np.sin(OMEGA_DAY * h),
        }
    )


@dataclass
class HarmonicSeasonalityParameters:
    intercept: float
    trend: float
    quad: float
    year_cos: float
    year_sin: float
    week_cos: float
    week_sin: float
    day_cos: float
    day_sin: float


@dataclass
class PaperSeasonalityParameters:
    c0: float
    c1: float
    c2: float
    amp_year: float
    amp_week: float
    amp_day: float
    tau_year: float
    tau_week: float
    tau_day: float


@dataclass
class SolarTransformParameters:
    alpha: float
    beta: float
    epsilon: float
    clear_sky_floor: float
    clear_sky_quantile: float
    clear_sky_smoothing_days: int


@dataclass
class SeasonalityFit:
    name: str
    parameterization: str
    origin: pd.Timestamp
    series: pd.Series
    fit_mask: pd.Series
    fitted: pd.Series
    residual: pd.Series
    ols_result: object
    harmonic_params: Optional[HarmonicSeasonalityParameters]
    paper_params: Optional[PaperSeasonalityParameters]
    notebook_fourier_order: int = 0
    notebook_exog_names: tuple[str, ...] = ()
    inverse_kind: str = "identity"
    physical_series: Optional[pd.Series] = None
    risk_driver: Optional[pd.Series] = None
    normalized_driver: Optional[pd.Series] = None
    clear_sky_proxy: Optional[pd.Series] = None
    clear_sky_climatology: Optional[np.ndarray] = None
    transform_params: Optional[SolarTransformParameters] = None
    solar_interaction_order: int = 0

    @property
    def rmse(self) -> float:
        err = self.series.loc[self.fit_mask] - self.fitted.loc[self.fit_mask]
        return float(np.sqrt(np.mean(np.square(err.to_numpy(dtype=float)))))

    @property
    def physical_rmse(self) -> float:
        if self.physical_series is None or self.inverse_kind == "identity":
            return self.rmse
        physical_fit = pd.Series(
            latent_to_physical(self, self.fitted.index, self.fitted.to_numpy(dtype=float)),
            index=self.fitted.index,
        )
        mask = self.fit_mask.reindex(physical_fit.index).fillna(False).astype(bool)
        err = self.physical_series.loc[mask] - physical_fit.loc[mask]
        return float(np.sqrt(np.mean(np.square(err.to_numpy(dtype=float)))))


def _harmonic_to_paper(params: HarmonicSeasonalityParameters) -> PaperSeasonalityParameters:
    def amp_phase(a: float, b: float) -> tuple[float, float]:
        amp = float(np.hypot(a, b))
        tau = float(np.arctan2(-b, a))
        return amp, tau

    amp_year, tau_year = amp_phase(params.year_cos, params.year_sin)
    amp_week, tau_week = amp_phase(params.week_cos, params.week_sin)
    amp_day, tau_day = amp_phase(params.day_cos, params.day_sin)
    return PaperSeasonalityParameters(
        c0=params.intercept,
        c1=params.trend,
        c2=params.quad,
        amp_year=amp_year,
        amp_week=amp_week,
        amp_day=amp_day,
        tau_year=tau_year,
        tau_week=tau_week,
        tau_day=tau_day,
    )


def evaluate_harmonic(params: HarmonicSeasonalityParameters, hours: np.ndarray) -> np.ndarray:
    X = harmonic_design(hours)
    beta = np.array(
        [
            params.intercept,
            params.trend,
            params.quad,
            params.year_cos,
            params.year_sin,
            params.week_cos,
            params.week_sin,
            params.day_cos,
            params.day_sin,
        ],
        dtype=float,
    )
    return X.to_numpy(dtype=float) @ beta


def evaluate_paper_phase(params: PaperSeasonalityParameters, hours: np.ndarray) -> np.ndarray:
    h = np.asarray(hours, dtype=float)
    return (
        params.c0
        + params.c1 * h
        + params.c2 * h**2
        + params.amp_year * np.cos(params.tau_year + OMEGA_YEAR * h)
        + params.amp_week * np.cos(params.tau_week + OMEGA_WEEK * h)
        + params.amp_day * np.cos(params.tau_day + OMEGA_DAY * h)
    )


def notebook_calendar_design(index: pd.DatetimeIndex, *, fourier_order: int = 2) -> pd.DataFrame:
    idx = pd.DatetimeIndex(index)
    df = pd.DataFrame(index=idx)
    df["hour"] = idx.hour.astype(str)
    df["weekday"] = idx.weekday.astype(str)
    df["month"] = idx.month.astype(str)
    df["doy"] = idx.dayofyear.astype(float)
    for k in range(1, int(fourier_order) + 1):
        angle = 2.0 * np.pi * float(k) * df["doy"] / 365.25
        df[f"sin{k}"] = np.sin(angle)
        df[f"cos{k}"] = np.cos(angle)
    X_dummies = pd.get_dummies(df[["hour", "weekday", "month"]], drop_first=True, dtype=float)
    fourier_cols = [f"sin{k}" for k in range(1, int(fourier_order) + 1)] + [
        f"cos{k}" for k in range(1, int(fourier_order) + 1)
    ]
    X = pd.concat([X_dummies, df[fourier_cols]], axis=1)
    X = sm.add_constant(X, has_constant="add")
    return X


def solar_calendar_design(
    index: pd.DatetimeIndex,
    *,
    fourier_order: int = 3,
    interaction_order: int = 2,
) -> pd.DataFrame:
    idx = pd.DatetimeIndex(index)
    df = pd.DataFrame(index=idx)
    df["hour"] = idx.hour.astype(str)
    df["month"] = idx.month.astype(str)
    df["doy"] = idx.dayofyear.astype(float)

    hour_dummies = pd.get_dummies(df["hour"], prefix="hour", drop_first=True, dtype=float)
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True, dtype=float)
    columns: dict[str, np.ndarray | float] = {"const": 1.0}
    for col in hour_dummies.columns:
        columns[col] = hour_dummies[col].to_numpy(dtype=float)
    for col in month_dummies.columns:
        columns[col] = month_dummies[col].to_numpy(dtype=float)

    hour_cols = tuple(hour_dummies.columns.tolist())
    for k in range(1, int(fourier_order) + 1):
        angle = 2.0 * np.pi * float(k) * df["doy"] / 365.25
        sin_k = np.sin(angle)
        cos_k = np.cos(angle)
        columns[f"sin{k}"] = sin_k.to_numpy(dtype=float)
        columns[f"cos{k}"] = cos_k.to_numpy(dtype=float)
        if k <= int(interaction_order):
            for col in hour_cols:
                base = hour_dummies[col].to_numpy(dtype=float)
                columns[f"{col}:sin{k}"] = base * sin_k.to_numpy(dtype=float)
                columns[f"{col}:cos{k}"] = base * cos_k.to_numpy(dtype=float)
    return pd.DataFrame(columns, index=idx)


def _circular_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if int(window) <= 1:
        return arr.copy()
    win = int(window)
    if win % 2 == 0:
        win += 1
    half = win // 2
    padded = np.concatenate([arr[-half:], arr, arr[:half]])
    smooth = pd.Series(padded).rolling(window=win, center=True, min_periods=1).mean().to_numpy(dtype=float)
    return smooth[half : half + len(arr)]


def _fill_clear_sky_gaps(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    doy_index = np.arange(1, 367, dtype=int)
    month_by_doy = pd.Series(pd.date_range("2020-01-01", "2020-12-31", freq="D").month, index=doy_index)
    for hour in out.columns:
        col = out[hour].copy()
        if col.isna().all():
            out[hour] = 0.0
            continue
        monthly_means = col.groupby(month_by_doy).transform("mean")
        col = col.fillna(monthly_means)
        col = col.fillna(col.mean())
        out[hour] = col.fillna(0.0)
    return out


def estimate_solar_clear_sky_climatology(
    solar_cf: pd.Series,
    *,
    quantile: float = 0.98,
    smoothing_days: int = 21,
) -> np.ndarray:
    s = pd.Series(solar_cf, copy=True).astype(float).clip(0.0, 1.0)
    s.index = pd.DatetimeIndex(s.index)
    df = pd.DataFrame(
        {
            "solar_cf": s.to_numpy(dtype=float),
            "doy": s.index.dayofyear.to_numpy(dtype=int),
            "hour": s.index.hour.to_numpy(dtype=int),
        }
    )
    raw = df.groupby(["doy", "hour"], observed=False)["solar_cf"].quantile(float(quantile)).unstack("hour")
    raw = raw.reindex(index=np.arange(1, 367), columns=np.arange(24))
    raw = _fill_clear_sky_gaps(raw)
    climatology = raw.to_numpy(dtype=float)
    for hour in range(24):
        climatology[:, hour] = _circular_rolling_mean(climatology[:, hour], int(smoothing_days))
    return np.clip(climatology, 0.0, 1.0)


def solar_clear_sky_from_climatology(index: pd.DatetimeIndex, climatology: np.ndarray) -> pd.Series:
    idx = pd.DatetimeIndex(index)
    doy = idx.dayofyear.to_numpy(dtype=int) - 1
    hour = idx.hour.to_numpy(dtype=int)
    values = np.asarray(climatology, dtype=float)[doy, hour]
    return pd.Series(values, index=idx, name="solar_clear_sky")


def _fit_solar_transform_params(
    risk_driver: pd.Series,
    *,
    clear_sky_floor: float,
    clear_sky_quantile: float,
    clear_sky_smoothing_days: int,
    epsilon: float = SOLAR_EPS,
) -> SolarTransformParameters:
    x = pd.Series(risk_driver, copy=True).astype(float).dropna().clip(0.0, 1.0).to_numpy(dtype=float)
    q_low = float(np.quantile(x, 0.001))
    q_high = float(np.quantile(x, 0.999))
    spread = max(q_high - q_low, 0.10)
    pad = max(5.0 * float(epsilon), 0.02 * spread)
    alpha = max(float(epsilon), q_low - pad)
    upper = min(1.0 - float(epsilon), q_high + pad)
    if upper - alpha < 0.10:
        alpha = float(epsilon)
        upper = 1.0 - float(epsilon)
    beta = upper - alpha
    return SolarTransformParameters(
        alpha=float(alpha),
        beta=float(beta),
        epsilon=float(epsilon),
        clear_sky_floor=float(clear_sky_floor),
        clear_sky_quantile=float(clear_sky_quantile),
        clear_sky_smoothing_days=int(clear_sky_smoothing_days),
    )


def solar_risk_driver_from_physical(
    solar_cf: pd.Series,
    clear_sky_proxy: pd.Series,
    *,
    clear_sky_floor: float = SOLAR_EPS,
) -> pd.Series:
    solar = pd.Series(solar_cf, copy=True).astype(float).clip(0.0, 1.0)
    clear = pd.Series(clear_sky_proxy, copy=True).astype(float).clip(0.0, 1.0)
    safe_clear = clear.clip(lower=float(clear_sky_floor))
    risk = 1.0 - solar / safe_clear
    risk = risk.clip(0.0, 1.0)
    return risk.rename("solar_risk_driver")


def evaluate_mle_notebook(fit: SeasonalityFit, index: pd.DatetimeIndex) -> np.ndarray:
    if fit.parameterization == "solar_logit":
        X = solar_calendar_design(
            index,
            fourier_order=fit.notebook_fourier_order,
            interaction_order=fit.solar_interaction_order,
        )
    else:
        X = notebook_calendar_design(index, fourier_order=fit.notebook_fourier_order)
    exog_names = list(fit.notebook_exog_names)
    if exog_names:
        X = X.reindex(columns=exog_names, fill_value=0.0)
    beta = np.asarray(fit.ols_result.params, dtype=float)
    return X.to_numpy(dtype=float) @ beta


def latent_to_physical(fit: SeasonalityFit, index: pd.DatetimeIndex, latent: np.ndarray) -> np.ndarray:
    z = np.asarray(latent, dtype=float)
    if fit.inverse_kind == "raw_logit":
        return 1.0 / (1.0 + np.exp(-z))
    if fit.inverse_kind == "solar_clear_sky_logit":
        if fit.clear_sky_climatology is None or fit.transform_params is None:
            raise ValueError("Solar inverse requested without clear-sky climatology and transform parameters.")
        clear = solar_clear_sky_from_climatology(index, fit.clear_sky_climatology).to_numpy(dtype=float)
        xprime = 1.0 / (1.0 + np.exp(-z))
        cf = clear * (1.0 - fit.transform_params.alpha - fit.transform_params.beta * xprime)
        return np.clip(cf, 0.0, 1.0)
    return z


def evaluate_clear_sky_proxy(fit: SeasonalityFit, index: pd.DatetimeIndex) -> np.ndarray:
    if fit.clear_sky_climatology is None:
        return np.ones(len(index), dtype=float)
    return solar_clear_sky_from_climatology(index, fit.clear_sky_climatology).to_numpy(dtype=float)


def fit_harmonic_seasonality(
    series: pd.Series,
    *,
    name: str,
    origin: Optional[pd.Timestamp] = None,
    fit_mask: Optional[pd.Series] = None,
) -> SeasonalityFit:
    s = pd.Series(series, copy=True).astype(float)
    s.index = pd.DatetimeIndex(s.index)
    if fit_mask is None:
        fit_mask = pd.Series(True, index=s.index)
    fit_mask = fit_mask.reindex(s.index).fillna(False).astype(bool)
    hours = hours_from_origin(s.index, origin=origin)
    X = harmonic_design(hours)
    X_fit = X.loc[fit_mask.to_numpy(dtype=bool)]
    y_fit = s.loc[fit_mask]
    ols = sm.OLS(y_fit.to_numpy(dtype=float), X_fit.to_numpy(dtype=float)).fit()
    beta = np.asarray(ols.params, dtype=float)
    harmonic_params = HarmonicSeasonalityParameters(
        intercept=float(beta[0]),
        trend=float(beta[1]),
        quad=float(beta[2]),
        year_cos=float(beta[3]),
        year_sin=float(beta[4]),
        week_cos=float(beta[5]),
        week_sin=float(beta[6]),
        day_cos=float(beta[7]),
        day_sin=float(beta[8]),
    )
    fitted = pd.Series(evaluate_harmonic(harmonic_params, hours), index=s.index, name=f"{name}_seasonality")
    residual = (s - fitted).rename(f"{name}_residual")
    return SeasonalityFit(
        name=name,
        parameterization="harmonic",
        origin=s.index[0] if origin is None else pd.Timestamp(origin),
        series=s,
        fit_mask=fit_mask.rename("fit_mask"),
        fitted=fitted,
        residual=residual,
        harmonic_params=harmonic_params,
        paper_params=_harmonic_to_paper(harmonic_params),
        ols_result=ols,
    )


def fit_paper_phase_seasonality(
    series: pd.Series,
    *,
    name: str,
    origin: Optional[pd.Timestamp] = None,
    fit_mask: Optional[pd.Series] = None,
) -> SeasonalityFit:
    fit = fit_harmonic_seasonality(series, name=name, origin=origin, fit_mask=fit_mask)
    return SeasonalityFit(
        name=fit.name,
        parameterization="paper_phase",
        origin=fit.origin,
        series=fit.series,
        fit_mask=fit.fit_mask,
        fitted=fit.fitted,
        residual=fit.residual,
        harmonic_params=fit.harmonic_params,
        paper_params=fit.paper_params,
        ols_result=fit.ols_result,
    )


def fit_mle_notebook_seasonality(
    series: pd.Series,
    *,
    name: str,
    origin: Optional[pd.Timestamp] = None,
    fit_mask: Optional[pd.Series] = None,
    fourier_order: int = 2,
    inverse_kind: str = "identity",
    physical_series: Optional[pd.Series] = None,
) -> SeasonalityFit:
    s = pd.Series(series, copy=True).astype(float)
    s.index = pd.DatetimeIndex(s.index)
    if fit_mask is None:
        fit_mask = pd.Series(True, index=s.index)
    fit_mask = fit_mask.reindex(s.index).fillna(False).astype(bool)
    X = notebook_calendar_design(s.index, fourier_order=fourier_order)
    X_fit = X.loc[fit_mask.to_numpy(dtype=bool)]
    y_fit = s.loc[fit_mask]
    ols = sm.OLS(y_fit.to_numpy(dtype=float), X_fit.to_numpy(dtype=float)).fit()
    fitted = pd.Series(X.to_numpy(dtype=float) @ np.asarray(ols.params, dtype=float), index=s.index, name=f"{name}_seasonality")
    residual = (s - fitted).rename(f"{name}_residual")
    return SeasonalityFit(
        name=name,
        parameterization="mle_notebook",
        origin=s.index[0] if origin is None else pd.Timestamp(origin),
        series=s,
        fit_mask=fit_mask.rename("fit_mask"),
        fitted=fitted,
        residual=residual,
        harmonic_params=None,
        paper_params=None,
        notebook_fourier_order=int(fourier_order),
        notebook_exog_names=tuple(X.columns.tolist()),
        inverse_kind=str(inverse_kind),
        physical_series=physical_series,
        ols_result=ols,
    )


def fit_naive_raw_solar_logit_seasonality(
    solar_cf: pd.Series,
    *,
    name: str,
    fit_mask: Optional[pd.Series] = None,
    fourier_order: int = 2,
    epsilon: float = SOLAR_EPS,
) -> SeasonalityFit:
    solar = pd.Series(solar_cf, copy=True).astype(float).clip(float(epsilon), 1.0 - float(epsilon))
    latent = pd.Series(np.log(solar / (1.0 - solar)), index=solar.index, name=f"{name}_latent")
    fit = fit_mle_notebook_seasonality(
        latent,
        name=name,
        fit_mask=fit_mask,
        fourier_order=fourier_order,
        inverse_kind="raw_logit",
        physical_series=pd.Series(solar_cf, copy=True).astype(float).clip(0.0, 1.0),
    )
    fit.physical_series = pd.Series(solar_cf, copy=True).astype(float).clip(0.0, 1.0)
    return fit


def fit_solar_logit_seasonality(
    solar_cf: pd.Series,
    *,
    name: str,
    fit_mask: Optional[pd.Series] = None,
    clear_sky_quantile: float = 0.98,
    clear_sky_smoothing_days: int = 21,
    clear_sky_floor: float = SOLAR_EPS,
    fourier_order: int = 3,
    interaction_order: int = 2,
    epsilon: float = SOLAR_EPS,
) -> SeasonalityFit:
    solar = pd.Series(solar_cf, copy=True).astype(float).clip(0.0, 1.0)
    solar.index = pd.DatetimeIndex(solar.index)
    if fit_mask is None:
        fit_mask = pd.Series(True, index=solar.index)
    fit_mask = fit_mask.reindex(solar.index).fillna(False).astype(bool)

    climatology = estimate_solar_clear_sky_climatology(
        solar,
        quantile=float(clear_sky_quantile),
        smoothing_days=int(clear_sky_smoothing_days),
    )
    clear_sky_proxy = solar_clear_sky_from_climatology(solar.index, climatology)
    risk_driver = solar_risk_driver_from_physical(solar, clear_sky_proxy, clear_sky_floor=float(clear_sky_floor))
    transform = _fit_solar_transform_params(
        risk_driver,
        clear_sky_floor=float(clear_sky_floor),
        clear_sky_quantile=float(clear_sky_quantile),
        clear_sky_smoothing_days=int(clear_sky_smoothing_days),
        epsilon=float(epsilon),
    )
    normalized = ((risk_driver - transform.alpha) / transform.beta).clip(float(epsilon), 1.0 - float(epsilon))
    latent = pd.Series(np.log(normalized / (1.0 - normalized)), index=solar.index, name=f"{name}_latent")

    X = solar_calendar_design(solar.index, fourier_order=fourier_order, interaction_order=interaction_order)
    X_fit = X.loc[fit_mask.to_numpy(dtype=bool)]
    y_fit = latent.loc[fit_mask]
    ols = sm.OLS(y_fit.to_numpy(dtype=float), X_fit.to_numpy(dtype=float)).fit()
    fitted = pd.Series(X.to_numpy(dtype=float) @ np.asarray(ols.params, dtype=float), index=solar.index, name=f"{name}_seasonality")
    residual = (latent - fitted).rename(f"{name}_residual")
    return SeasonalityFit(
        name=name,
        parameterization="solar_logit",
        origin=solar.index[0],
        series=latent,
        fit_mask=fit_mask.rename("fit_mask"),
        fitted=fitted,
        residual=residual,
        harmonic_params=None,
        paper_params=None,
        notebook_fourier_order=int(fourier_order),
        notebook_exog_names=tuple(X.columns.tolist()),
        inverse_kind="solar_clear_sky_logit",
        physical_series=solar.rename("solar_cf"),
        risk_driver=risk_driver.rename("solar_risk_driver"),
        normalized_driver=normalized.rename("solar_xprime"),
        clear_sky_proxy=clear_sky_proxy.rename("solar_clear_sky"),
        clear_sky_climatology=climatology,
        transform_params=transform,
        solar_interaction_order=int(interaction_order),
        ols_result=ols,
    )


def paper_parameter_table(fit: SeasonalityFit) -> pd.DataFrame:
    if fit.paper_params is None:
        raise ValueError("Paper parameter table requested for a non-paper seasonality fit.")
    p = fit.paper_params
    return pd.DataFrame(
        [
            {"parameter": "c0", "value": p.c0},
            {"parameter": "c1", "value": p.c1},
            {"parameter": "c2", "value": p.c2},
            {"parameter": "c3", "value": p.amp_year},
            {"parameter": "tau0", "value": p.tau_year},
            {"parameter": "c4", "value": p.amp_week},
            {"parameter": "tau1", "value": p.tau_week},
            {"parameter": "c5", "value": p.amp_day},
            {"parameter": "tau2", "value": p.tau_day},
            {"parameter": "rmse", "value": fit.rmse},
        ]
    )


def harmonic_parameter_table(fit: SeasonalityFit) -> pd.DataFrame:
    if fit.harmonic_params is None:
        raise ValueError("Harmonic parameter table requested for a non-harmonic seasonality fit.")
    p = fit.harmonic_params
    return pd.DataFrame(
        [
            {"parameter": "intercept", "value": p.intercept},
            {"parameter": "trend", "value": p.trend},
            {"parameter": "quad", "value": p.quad},
            {"parameter": "year_cos", "value": p.year_cos},
            {"parameter": "year_sin", "value": p.year_sin},
            {"parameter": "week_cos", "value": p.week_cos},
            {"parameter": "week_sin", "value": p.week_sin},
            {"parameter": "day_cos", "value": p.day_cos},
            {"parameter": "day_sin", "value": p.day_sin},
            {"parameter": "rmse", "value": fit.rmse},
        ]
    )


def notebook_parameter_table(fit: SeasonalityFit) -> pd.DataFrame:
    names = list(fit.notebook_exog_names) or [f"beta_{i}" for i in range(len(np.asarray(fit.ols_result.params, dtype=float)))]
    rows = [{"parameter": name, "value": float(val)} for name, val in zip(names, np.asarray(fit.ols_result.params, dtype=float))]
    rows.append({"parameter": "rmse_latent", "value": fit.rmse})
    rows.append({"parameter": "rmse_physical", "value": fit.physical_rmse})
    return pd.DataFrame(rows)


def solar_transform_parameter_table(fit: SeasonalityFit) -> pd.DataFrame:
    if fit.transform_params is None:
        raise ValueError("Solar transform parameter table requested for a non-solar seasonality fit.")
    p = fit.transform_params
    rows = [
        {"parameter": "alpha", "value": p.alpha},
        {"parameter": "beta", "value": p.beta},
        {"parameter": "epsilon", "value": p.epsilon},
        {"parameter": "clear_sky_floor", "value": p.clear_sky_floor},
        {"parameter": "clear_sky_quantile", "value": p.clear_sky_quantile},
        {"parameter": "clear_sky_smoothing_days", "value": float(p.clear_sky_smoothing_days)},
    ]
    if fit.clear_sky_proxy is not None:
        rows.extend(
            [
                {"parameter": "clear_sky_mean", "value": float(fit.clear_sky_proxy.mean())},
                {"parameter": "clear_sky_max", "value": float(fit.clear_sky_proxy.max())},
            ]
        )
    return pd.DataFrame(rows)


__all__ = [
    "HarmonicSeasonalityParameters",
    "PaperSeasonalityParameters",
    "SeasonalityFit",
    "SolarTransformParameters",
    "evaluate_clear_sky_proxy",
    "evaluate_harmonic",
    "evaluate_mle_notebook",
    "evaluate_paper_phase",
    "estimate_solar_clear_sky_climatology",
    "fit_harmonic_seasonality",
    "fit_mle_notebook_seasonality",
    "fit_naive_raw_solar_logit_seasonality",
    "fit_paper_phase_seasonality",
    "fit_solar_logit_seasonality",
    "harmonic_design",
    "harmonic_parameter_table",
    "hours_from_origin",
    "latent_to_physical",
    "notebook_calendar_design",
    "notebook_parameter_table",
    "paper_parameter_table",
    "solar_calendar_design",
    "solar_clear_sky_from_climatology",
    "solar_risk_driver_from_physical",
    "solar_transform_parameter_table",
]
