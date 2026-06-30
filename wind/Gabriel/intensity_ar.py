from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import logm
@dataclass
class AR24Fit:
    series: pd.Series
    fitted: pd.Series
    residual: pd.Series
    coeffs: np.ndarray
    innovation_std: float
    innovation_var: float
    loglikelihood: float
    aic: float
    bic: float
    continuous_companion: np.ndarray
    car_alpha: np.ndarray
    model_result: object


def _companion_from_ar(coeffs: np.ndarray) -> np.ndarray:
    a = np.asarray(coeffs, dtype=float).reshape(-1)
    p = a.size
    comp = np.zeros((p, p), dtype=float)
    comp[0, :] = a
    if p > 1:
        comp[1:, :-1] = np.eye(p - 1)
    return comp


def equivalent_car_companion(coeffs: np.ndarray, *, dt_hours: float = 1.0) -> np.ndarray:
    G = _companion_from_ar(coeffs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A = logm(G) / float(dt_hours)
    A = np.real_if_close(A, tol=1000)
    return np.asarray(A, dtype=float)


def fit_ar24_exact_mle(series: pd.Series, *, enforce_stationarity: bool = False) -> AR24Fit:
    s = pd.Series(series, copy=True).astype(float).dropna()
    p = 24
    frame = pd.DataFrame({"y": s})
    for lag in range(1, p + 1):
        frame[f"lag_{lag}"] = s.shift(lag)
    frame = frame.dropna()
    y = frame["y"].to_numpy(dtype=float)
    X = frame.drop(columns=["y"]).to_numpy(dtype=float)
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted_values = X @ coeffs
    resid = y - fitted_values
    fitted = pd.Series(fitted_values, index=frame.index, name="fitted")
    residual = pd.Series(resid, index=frame.index, name="residual")
    innovation_var = float(np.var(resid, ddof=1))
    n = len(y)
    k = len(coeffs) + 1
    sigma2 = max(innovation_var, 1e-12)
    loglik = float(-0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1.0))
    aic = float(-2.0 * loglik + 2.0 * k)
    bic = float(-2.0 * loglik + np.log(max(n, 1)) * k)
    companion = equivalent_car_companion(coeffs, dt_hours=1.0)
    car_alpha = -np.asarray(companion[-1, ::-1], dtype=float)
    return AR24Fit(
        series=s,
        fitted=fitted,
        residual=residual,
        coeffs=coeffs,
        innovation_std=float(np.sqrt(max(innovation_var, 1e-12))),
        innovation_var=innovation_var,
        loglikelihood=loglik,
        aic=aic,
        bic=bic,
        continuous_companion=companion,
        car_alpha=car_alpha,
        model_result=None,
    )


def ar_parameter_table(fit: AR24Fit, *, prefix: str) -> pd.DataFrame:
    rows = []
    for i, value in enumerate(fit.coeffs, start=1):
        rows.append({"parameter": f"a{i},{prefix}", "value": float(value), "block": "AR"})
    for i, value in enumerate(fit.car_alpha, start=1):
        rows.append({"parameter": f"alpha{i},{prefix}", "value": float(value), "block": "CAR"})
    rows.append({"parameter": f"sigma2_{prefix}", "value": float(fit.innovation_var), "block": "AR"})
    rows.append({"parameter": f"loglik_{prefix}", "value": float(fit.loglikelihood), "block": "fit"})
    rows.append({"parameter": f"aic_{prefix}", "value": float(fit.aic), "block": "fit"})
    rows.append({"parameter": f"bic_{prefix}", "value": float(fit.bic), "block": "fit"})
    return pd.DataFrame(rows)
