from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf


def flatten_simulation_paths(paths: np.ndarray, *, max_points: int | None = None, seed: int = 123) -> np.ndarray:
    arr = np.asarray(paths, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if max_points is None or arr.size <= max_points:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.size, size=int(max_points), replace=False)
    return arr[idx]


def empirical_vs_simulated_marginals(
    empirical: pd.Series | np.ndarray,
    simulated: np.ndarray,
    *,
    label: str,
    max_points: int = 20000,
) -> pd.DataFrame:
    emp = np.asarray(pd.Series(empirical).dropna(), dtype=float)
    sim = flatten_simulation_paths(simulated, max_points=max_points)
    return pd.DataFrame(
        [
            {
                "series": label,
                "empirical_mean": float(np.mean(emp)),
                "simulated_mean": float(np.mean(sim)),
                "empirical_std": float(np.std(emp, ddof=1)),
                "simulated_std": float(np.std(sim, ddof=1)),
                "ks_stat": float(ks_2samp(emp, sim).statistic),
                "wasserstein": float(wasserstein_distance(emp, sim)),
            }
        ]
    )


def joint_dependence_summary(
    empirical: pd.DataFrame,
    simulated_spot: np.ndarray,
    simulated_renewable: np.ndarray,
) -> pd.DataFrame:
    emp = empirical.dropna()
    sim_spot = flatten_simulation_paths(simulated_spot, max_points=min(simulated_spot.size, 20000))
    sim_renewable = flatten_simulation_paths(simulated_renewable, max_points=min(simulated_renewable.size, 20000))
    n = min(len(sim_spot), len(sim_renewable))
    sim_spot = sim_spot[:n]
    sim_renewable = sim_renewable[:n]
    return pd.DataFrame(
        [
            {
                "measure": "correlation",
                "empirical": float(emp.iloc[:, 0].corr(emp.iloc[:, 1])),
                "simulated": float(np.corrcoef(sim_spot, sim_renewable)[0, 1]),
            },
            {
                "measure": "spot_mean_low_renewable",
                "empirical": float(emp.loc[emp.iloc[:, 1] <= emp.iloc[:, 1].median(), emp.columns[0]].mean()),
                "simulated": float(np.mean(sim_spot[sim_renewable <= np.median(sim_renewable)])),
            },
            {
                "measure": "spot_mean_high_renewable",
                "empirical": float(emp.loc[emp.iloc[:, 1] > emp.iloc[:, 1].median(), emp.columns[0]].mean()),
                "simulated": float(np.mean(sim_spot[sim_renewable > np.median(sim_renewable)])),
            },
        ]
    )


@dataclass
class CorrFunctionComparison:
    lags: np.ndarray
    empirical: np.ndarray
    simulated_mean: np.ndarray
    rmse: float
    mae: float
    max_abs: float


def compare_correlation_function(
    empirical: pd.Series | np.ndarray,
    simulated_paths: np.ndarray,
    *,
    nlags: int = 168,
    kind: str = "acf",
) -> CorrFunctionComparison:
    emp = np.asarray(pd.Series(empirical).dropna(), dtype=float)
    if kind == "acf":
        emp_corr = acf(emp, nlags=nlags, fft=True)
        sim_corr = np.vstack([acf(np.asarray(path, dtype=float), nlags=nlags, fft=True) for path in np.asarray(simulated_paths, dtype=float)])
    else:
        emp_corr = pacf(emp, nlags=nlags, method="ywm")
        sim_corr = np.vstack([pacf(np.asarray(path, dtype=float), nlags=nlags, method="ywm") for path in np.asarray(simulated_paths, dtype=float)])
    sim_mean = np.mean(sim_corr, axis=0)
    diff = sim_mean - emp_corr
    lags = np.arange(len(emp_corr), dtype=int)
    return CorrFunctionComparison(
        lags=lags,
        empirical=emp_corr,
        simulated_mean=sim_mean,
        rmse=float(np.sqrt(np.mean(diff ** 2))),
        mae=float(np.mean(np.abs(diff))),
        max_abs=float(np.max(np.abs(diff))),
    )


def correlation_function_table(
    *,
    name: str,
    acf_cmp: CorrFunctionComparison,
    pacf_cmp: CorrFunctionComparison,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"series": name, "kind": "acf", "rmse": acf_cmp.rmse, "mae": acf_cmp.mae, "max_abs": acf_cmp.max_abs},
            {"series": name, "kind": "pacf", "rmse": pacf_cmp.rmse, "mae": pacf_cmp.mae, "max_abs": pacf_cmp.max_abs},
        ]
    )


def squared_return_series(series: pd.Series | np.ndarray, *, name: str) -> pd.Series:
    s = pd.Series(series, copy=True).astype(float).dropna()
    sq = s.diff().pow(2).dropna().rename(name)
    return sq


def rolling_covariance_proxy(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    window: int = 24 * 30,
    min_periods: int | None = None,
) -> pd.DataFrame:
    sx = pd.Series(x, copy=True).astype(float)
    sy = pd.Series(y, copy=True).astype(float)
    frame = pd.concat([sx.rename("x"), sy.rename("y")], axis=1).dropna()
    if min_periods is None:
        min_periods = int(window)
    proxy = pd.DataFrame(index=frame.index)
    proxy["s11"] = frame["x"].rolling(window, min_periods=min_periods).var(ddof=1)
    proxy["s22"] = frame["y"].rolling(window, min_periods=min_periods).var(ddof=1)
    proxy["s12"] = frame["x"].rolling(window, min_periods=min_periods).cov(frame["y"])
    denom = np.sqrt(np.maximum(proxy["s11"] * proxy["s22"], 1e-12))
    proxy["corr"] = np.divide(proxy["s12"], denom, out=np.full(len(proxy), np.nan), where=np.isfinite(denom) & (denom > 0.0))
    proxy["det"] = proxy["s11"] * proxy["s22"] - proxy["s12"] ** 2
    return proxy.dropna()


def persistence_summary_table(
    series: pd.Series | np.ndarray,
    *,
    name: str,
    acf_lags: tuple[int, ...] = (1, 6, 24, 24 * 7),
    ljung_box_lags: tuple[int, ...] = (24, 24 * 7),
) -> pd.DataFrame:
    s = pd.Series(series, copy=True).astype(float).dropna()
    max_lag = max(max(acf_lags, default=1), max(ljung_box_lags, default=1))
    corr = acf(s.to_numpy(dtype=float), nlags=max_lag, fft=True)
    row: dict[str, float | str | int] = {
        "series": name,
        "n_obs": int(len(s)),
        "mean": float(np.mean(s)),
        "std": float(np.std(s, ddof=1)),
    }
    for lag in acf_lags:
        row[f"acf_{lag}"] = float(corr[lag]) if lag < len(corr) else float("nan")
    lb = acorr_ljungbox(s.to_numpy(dtype=float), lags=list(ljung_box_lags), return_df=True)
    for lag in ljung_box_lags:
        if lag in lb.index:
            row[f"lb_stat_{lag}"] = float(lb.loc[lag, "lb_stat"])
            row[f"lb_pvalue_{lag}"] = float(lb.loc[lag, "lb_pvalue"])
    return pd.DataFrame([row])
