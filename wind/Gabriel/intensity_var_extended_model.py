from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MultivariateVARCompanionFitExtendedModel:
    lags: int
    names: list[str]
    coefs: np.ndarray
    innovation_cov: np.ndarray
    residual: pd.DataFrame
    fitted: pd.DataFrame
    state_frame: pd.DataFrame
    companion_matrix: np.ndarray
    spectral_radius: float


def fit_var_companion_extended_model(
    state_frame: pd.DataFrame,
    *,
    lags: int = 24,
    ridge: float = 1e-8,
) -> MultivariateVARCompanionFitExtendedModel:
    frame = pd.DataFrame(state_frame, copy=True).astype(float).dropna().sort_index()
    if len(frame) <= int(lags):
        raise ValueError("Not enough observations to fit the multivariate VAR companion model.")

    p = int(lags)
    d = int(frame.shape[1])
    values = frame.to_numpy(dtype=float)
    y = values[p:, :]
    x_blocks = [values[p - lag : values.shape[0] - lag, :] for lag in range(1, p + 1)]
    x = np.concatenate(x_blocks, axis=1)

    xtx = x.T @ x + float(ridge) * np.eye(x.shape[1], dtype=float)
    beta = np.linalg.solve(xtx, x.T @ y)
    fitted = x @ beta
    residual = y - fitted

    coefs = np.stack(
        [beta[d * lag : d * (lag + 1), :].T for lag in range(p)],
        axis=0,
    )

    companion = np.zeros((d * p, d * p), dtype=float)
    companion[:d, :] = np.hstack(coefs)
    if p > 1:
        companion[d:, :-d] = np.eye(d * (p - 1), dtype=float)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(companion))))

    resid_cols = [f"{name}_innovation" for name in frame.columns]
    fitted_cols = [f"{name}_fitted" for name in frame.columns]
    residual_df = pd.DataFrame(residual, index=frame.index[p:], columns=resid_cols)
    fitted_df = pd.DataFrame(fitted, index=frame.index[p:], columns=fitted_cols)
    innovation_cov = np.cov(residual.T, ddof=1)
    innovation_cov = 0.5 * (innovation_cov + innovation_cov.T)

    return MultivariateVARCompanionFitExtendedModel(
        lags=p,
        names=list(frame.columns),
        coefs=coefs,
        innovation_cov=np.asarray(innovation_cov, dtype=float),
        residual=residual_df,
        fitted=fitted_df,
        state_frame=frame,
        companion_matrix=companion,
        spectral_radius=spectral_radius,
    )


def var_parameter_table_extended_model(
    fit: MultivariateVARCompanionFitExtendedModel,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    d = len(fit.names)
    for lag in range(fit.lags):
        coef = fit.coefs[lag]
        for i in range(d):
            for j in range(d):
                rows.append(
                    {
                        "parameter": f"phi[{fit.names[i]},{fit.names[j]},lag{lag+1}]",
                        "value": float(coef[i, j]),
                        "block": "VAR",
                    }
                )
    for i in range(d):
        for j in range(d):
            rows.append(
                {
                    "parameter": f"sigma_eta[{fit.names[i]},{fit.names[j]}]",
                    "value": float(fit.innovation_cov[i, j]),
                    "block": "innovation_cov",
                }
            )
    rows.append({"parameter": "var_spectral_radius", "value": float(fit.spectral_radius), "block": "VAR"})
    return pd.DataFrame(rows)


__all__ = [
    "MultivariateVARCompanionFitExtendedModel",
    "fit_var_companion_extended_model",
    "var_parameter_table_extended_model",
]
