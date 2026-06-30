from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

try:
    from .intensity_seasonality import (
        SeasonalityFit,
        evaluate_mle_notebook,
        fit_mle_notebook_seasonality,
    )
except ImportError:
    from intensity_seasonality import (
        SeasonalityFit,
        evaluate_mle_notebook,
        fit_mle_notebook_seasonality,
    )


@dataclass
class FinalSeasonalityFit:
    name: str
    family: str
    series: pd.Series
    fitted: pd.Series
    residual: pd.Series
    coefficients: pd.Series
    metadata: dict
    base_fit: Optional[SeasonalityFit] = None

    @property
    def rmse(self) -> float:
        err = self.residual.dropna().to_numpy(dtype=float)
        return float(np.sqrt(np.mean(err**2))) if err.size else float("nan")

    def evaluate(self, index: pd.DatetimeIndex) -> pd.Series:
        idx = pd.DatetimeIndex(index)
        if self.family == "mle_benchmark":
            if self.base_fit is None:
                raise ValueError("MLE benchmark seasonality requires base_fit.")
            values = evaluate_mle_notebook(self.base_fit, idx)
            return pd.Series(values, index=idx, name=f"{self.name}_seasonality")
        if self.family == "carma_temperature_fourier":
            values = _evaluate_carma_temperature_design(idx, self.metadata) @ self.coefficients.to_numpy(dtype=float)
            return pd.Series(values, index=idx, name=f"{self.name}_seasonality")
        if self.family == "carma_paraschiv_log_price":
            return _evaluate_paraschiv_price_in_sample(idx, self)
        raise ValueError(f"Unsupported final seasonality family: {self.family}")


def fit_mle_benchmark_seasonality_final(
    series: pd.Series,
    *,
    name: str,
    fit_mask: Optional[pd.Series] = None,
    fourier_order: int = 2,
    inverse_kind: str = "identity",
    physical_series: Optional[pd.Series] = None,
) -> FinalSeasonalityFit:
    fit = fit_mle_notebook_seasonality(
        series,
        name=name,
        fit_mask=fit_mask,
        fourier_order=fourier_order,
        inverse_kind=inverse_kind,
        physical_series=physical_series,
    )
    names = list(fit.notebook_exog_names)
    coefficients = pd.Series(np.asarray(fit.ols_result.params, dtype=float), index=names, name="coefficient")
    return FinalSeasonalityFit(
        name=name,
        family="mle_benchmark",
        series=fit.series,
        fitted=fit.fitted,
        residual=fit.residual,
        coefficients=coefficients,
        metadata={"fourier_order": int(fourier_order), "inverse_kind": str(inverse_kind)},
        base_fit=fit,
    )


def _fourier_block(x: np.ndarray, period: float, order: int) -> np.ndarray:
    omega = 2.0 * np.pi / float(period)
    cols = []
    for k in range(1, int(order) + 1):
        cols.append(np.cos(k * omega * x))
        cols.append(np.sin(k * omega * x))
    return np.column_stack(cols) if cols else np.empty((len(x), 0), dtype=float)


def _carma_temperature_design(index: pd.DatetimeIndex, *, origin: pd.Timestamp, k_day: int, k_year: int) -> tuple[np.ndarray, list[str]]:
    idx = pd.DatetimeIndex(index)
    x = ((idx - pd.Timestamp(origin)) / pd.Timedelta(hours=1)).to_numpy(dtype=float)
    day = _fourier_block(x, 24.0, int(k_day))
    year = _fourier_block(x, 365.25 * 24.0, int(k_year))
    cols = [np.ones_like(x), x]
    names = ["intercept", "trend"]
    for k in range(1, int(k_day) + 1):
        names.extend([f"day_cos_{k}", f"day_sin_{k}"])
    for k in range(1, int(k_year) + 1):
        names.extend([f"year_cos_{k}", f"year_sin_{k}"])
    cols.extend([day, year])
    interactions = []
    for i in range(day.shape[1]):
        for j in range(year.shape[1]):
            interactions.append(day[:, i] * year[:, j])
            names.append(f"inter_{i}_{j}")
    if interactions:
        cols.append(np.column_stack(interactions))
    return np.column_stack(cols), names


def _evaluate_carma_temperature_design(index: pd.DatetimeIndex, metadata: dict) -> np.ndarray:
    design, _ = _carma_temperature_design(
        index,
        origin=pd.Timestamp(metadata["origin"]),
        k_day=int(metadata["k_day"]),
        k_year=int(metadata["k_year"]),
    )
    return design


def fit_carma_temperature_fourier_seasonality_final(
    temperature: pd.Series,
    *,
    name: str = "temperature_carma_fourier_final",
    k_day: int = 3,
    k_year: int = 3,
) -> FinalSeasonalityFit:
    temp = pd.Series(temperature, copy=True).astype(float).sort_index()
    temp.index = pd.DatetimeIndex(temp.index)
    temp = temp.asfreq("h").interpolate(method="linear")
    origin = pd.Timestamp(temp.index[0])
    design, names = _carma_temperature_design(temp.index, origin=origin, k_day=k_day, k_year=k_year)
    beta, *_ = np.linalg.lstsq(design, temp.to_numpy(dtype=float), rcond=None)
    fitted = pd.Series(design @ beta, index=temp.index, name=f"{name}_seasonality")
    residual = (temp - fitted).rename(f"{name}_residual")
    coefficients = pd.Series(beta, index=names, name="coefficient")
    return FinalSeasonalityFit(
        name=name,
        family="carma_temperature_fourier",
        series=temp.rename(name),
        fitted=fitted,
        residual=residual,
        coefficients=coefficients,
        metadata={"origin": origin, "k_day": int(k_day), "k_year": int(k_year)},
    )


def _profile_class(index: pd.DatetimeIndex) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    out = np.zeros(len(idx), dtype=int)
    for i, (month, dow) in enumerate(zip(idx.month, idx.dayofweek)):
        if dow <= 4:
            out[i] = int(month)
        elif dow == 5:
            out[i] = 13 if month in [1, 2, 12] else 14 if month in [3, 4, 5] else 15 if month in [6, 7, 8] else 16
        else:
            out[i] = 17 if month in [1, 2, 12] else 18 if month in [3, 4, 5] else 19 if month in [6, 7, 8] else 20
    return out


def _f2y_design(daily_index: pd.DatetimeIndex, temperature_daily: pd.Series) -> pd.DataFrame:
    idx = pd.DatetimeIndex(daily_index)
    X = pd.DataFrame(index=idx)
    for d, lbl in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]):
        X[lbl] = (idx.dayofweek == d).astype(float)
    for m in range(2, 13):
        if m != 8:
            X[f"Month_{m}"] = (idx.month == m).astype(float)
    X["Aug_early"] = ((idx.month == 8) & (idx.day <= 15)).astype(float)
    X["Aug_late"] = ((idx.month == 8) & (idx.day > 15)).astype(float)
    t = pd.Series(temperature_daily, copy=False).reindex(idx).astype(float)
    X["HDD"] = (15.0 - t).clip(lower=0.0)
    X["CDD"] = (t - 15.0).clip(lower=0.0)
    return sm.add_constant(X, has_constant="add")


def fit_carma_paraschiv_log_price_seasonality_final(
    spot_price: pd.Series,
    temperature: pd.Series,
    *,
    name: str = "spot_paraschiv_log_price_final",
    delta_shift: float = 1000.0,
) -> FinalSeasonalityFit:
    spot_raw = pd.Series(spot_price, copy=True).astype(float).sort_index()
    spot_raw.index = pd.DatetimeIndex(spot_raw.index)
    spot_raw = spot_raw.asfreq("h").interpolate(method="linear")
    temp = pd.Series(temperature, copy=True).astype(float).sort_index()
    temp.index = pd.DatetimeIndex(temp.index)
    temp = temp.asfreq("h").interpolate(method="linear")
    log_price = np.log(spot_raw + float(delta_shift)).rename(name)

    daily_mean = log_price.resample("D").mean()
    daily_mean_hourly = pd.Series(daily_mean.reindex(log_price.index.floor("D")).to_numpy(dtype=float), index=log_price.index)
    f2d_observed = log_price / daily_mean_hourly
    classes = _profile_class(log_price.index)
    hour_dummies = pd.get_dummies(log_price.index.hour, drop_first=True).astype(float)
    X_hour = np.column_stack([np.ones(len(log_price)), hour_dummies.to_numpy(dtype=float)])
    f2d_hat = np.zeros(len(log_price), dtype=float)
    f2d_coefficients: dict[int, np.ndarray] = {}
    for class_id in np.unique(classes):
        mask = classes == int(class_id)
        coeffs, *_ = np.linalg.lstsq(X_hour[mask], log_price.to_numpy(dtype=float)[mask], rcond=None)
        predicted = X_hour[mask] @ coeffs
        scale = float(np.mean(predicted))
        f2d_hat[mask] = predicted / scale if abs(scale) > 1e-12 else 1.0
        f2d_coefficients[int(class_id)] = coeffs

    temp_day = temp.resample("D").mean()
    common_daily = pd.concat([daily_mean.rename("log_price_day"), temp_day.rename("temperature")], axis=1).dropna()
    yearly_mean = common_daily["log_price_day"].groupby(common_daily.index.year).transform("mean")
    y = (common_daily["log_price_day"] / yearly_mean).dropna()
    X_f2y = _f2y_design(pd.DatetimeIndex(y.index), common_daily["temperature"])
    f2y_model = sm.OLS(y.to_numpy(dtype=float), X_f2y.to_numpy(dtype=float)).fit()
    f2y_fit = pd.Series(X_f2y.to_numpy(dtype=float) @ np.asarray(f2y_model.params, dtype=float), index=y.index, name="f2y")
    f2y_hourly = pd.Series(f2y_fit.reindex(log_price.index.floor("D")).to_numpy(dtype=float), index=log_price.index).ffill().bfill()

    seasonal_weight = f2y_hourly * f2d_hat
    seasonal = pd.Series(float(np.mean(log_price)) * seasonal_weight, index=log_price.index, name=f"{name}_seasonality")
    residual = (log_price - seasonal).rename(f"{name}_residual")
    f2y_names = list(X_f2y.columns)
    coeff_rows = {f"f2y::{col}": float(val) for col, val in zip(f2y_names, np.asarray(f2y_model.params, dtype=float))}
    for class_id, coeffs in f2d_coefficients.items():
        coeff_rows[f"f2d_class_{class_id}::intercept"] = float(coeffs[0])
        for j, value in enumerate(coeffs[1:], start=1):
            coeff_rows[f"f2d_class_{class_id}::hour_dummy_{j}"] = float(value)

    return FinalSeasonalityFit(
        name=name,
        family="carma_paraschiv_log_price",
        series=log_price,
        fitted=seasonal,
        residual=residual,
        coefficients=pd.Series(coeff_rows, name="coefficient"),
        metadata={
            "delta_shift": float(delta_shift),
            "fitted_index": log_price.index,
            "temperature_daily": temp_day,
            "f2d_hat": pd.Series(f2d_hat, index=log_price.index, name="f2d_hat"),
            "f2y_hat": f2y_hourly.rename("f2y_hat"),
            "yearly_avg_log_price": float(np.mean(log_price)),
        },
    )


def _evaluate_paraschiv_price_in_sample(index: pd.DatetimeIndex, fit: FinalSeasonalityFit) -> pd.Series:
    idx = pd.DatetimeIndex(index)
    fitted = fit.fitted.reindex(idx)
    if fitted.isna().any():
        raise ValueError(
            "The Paraschiv final seasonality is currently evaluated on the fitted hourly support. "
            "Refit it on a sample covering the requested timestamps before using it out of sample."
        )
    return fitted.rename(f"{fit.name}_seasonality")


def final_seasonality_comparison_table(results: dict[str, FinalSeasonalityFit]) -> pd.DataFrame:
    rows = []
    for key, fit in results.items():
        rows.append(
            {
                "seasonality_key": key,
                "name": fit.name,
                "family": fit.family,
                "n_obs": int(len(fit.series.dropna())),
                "series_mean": float(fit.series.mean()),
                "fitted_mean": float(fit.fitted.mean()),
                "residual_mean": float(fit.residual.mean()),
                "residual_std": float(fit.residual.std(ddof=1)),
                "rmse": fit.rmse,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "FinalSeasonalityFit",
    "final_seasonality_comparison_table",
    "fit_carma_paraschiv_log_price_seasonality_final",
    "fit_carma_temperature_fourier_seasonality_final",
    "fit_mle_benchmark_seasonality_final",
]
