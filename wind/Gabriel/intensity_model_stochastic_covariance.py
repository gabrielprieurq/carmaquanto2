from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from .intensity_model import IntensityModelSpecification
    from .intensity_seasonality import (
        SeasonalityFit,
        evaluate_harmonic,
        evaluate_mle_notebook,
        evaluate_paper_phase,
        hours_from_origin,
    )
    from .intensity_wishart_stochastic_covariance import (
        WishartFullCovarianceCalibration,
        simulate_wishart_innovation_system,
    )
except ImportError:
    from intensity_model import IntensityModelSpecification
    from intensity_seasonality import (
        SeasonalityFit,
        evaluate_harmonic,
        evaluate_mle_notebook,
        evaluate_paper_phase,
        hours_from_origin,
    )
    from intensity_wishart_stochastic_covariance import (
        WishartFullCovarianceCalibration,
        simulate_wishart_innovation_system,
    )


@dataclass
class VAR24CompanionFit:
    lags: int
    coefs: np.ndarray
    innovation_cov: np.ndarray
    residual: pd.DataFrame
    fitted: pd.DataFrame
    state_pair: pd.DataFrame
    companion_matrix: np.ndarray
    spectral_radius: float


@dataclass
class IntensityStochasticCovarianceSpecification:
    name: str
    base_spec: IntensityModelSpecification
    companion_fit: VAR24CompanionFit
    wishart: WishartFullCovarianceCalibration
    initial_history: np.ndarray
    initial_spike_state: float


def _evaluate_seasonality(fit: SeasonalityFit, index: pd.DatetimeIndex) -> np.ndarray:
    if fit.parameterization == "mle_notebook":
        return evaluate_mle_notebook(fit, index)
    hours = hours_from_origin(index, origin=fit.origin)
    if fit.parameterization == "paper_phase":
        return evaluate_paper_phase(fit.paper_params, hours)
    return evaluate_harmonic(fit.harmonic_params, hours)


def fit_var24_companion(
    state_pair: pd.DataFrame,
    *,
    lags: int = 24,
    ridge: float = 1e-8,
) -> VAR24CompanionFit:
    pair = pd.DataFrame(state_pair, copy=True).astype(float).dropna().sort_index()
    pair = pair.iloc[:, :2]
    pair.columns = ["spot_state", "wind_state"]
    if len(pair) <= int(lags):
        raise ValueError("Not enough observations to fit a VAR/CAR companion model.")

    p = int(lags)
    values = pair.to_numpy(dtype=float)
    n_obs = values.shape[0] - p
    y = values[p:, :]
    x_blocks = [values[p - lag : values.shape[0] - lag, :] for lag in range(1, p + 1)]
    x = np.concatenate(x_blocks, axis=1)

    xtx = x.T @ x + float(ridge) * np.eye(x.shape[1], dtype=float)
    beta = np.linalg.solve(xtx, x.T @ y)
    fitted = x @ beta
    residual = y - fitted

    coefs = np.stack(
        [beta[2 * lag : 2 * (lag + 1), :].T for lag in range(p)],
        axis=0,
    )

    companion = np.zeros((2 * p, 2 * p), dtype=float)
    companion[:2, :] = np.hstack(coefs)
    if p > 1:
        companion[2:, :-2] = np.eye(2 * (p - 1), dtype=float)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(companion))))

    resid_df = pd.DataFrame(residual, index=pair.index[p:], columns=["spot_innovation", "wind_innovation"])
    fitted_df = pd.DataFrame(fitted, index=pair.index[p:], columns=["spot_fitted", "wind_fitted"])
    innovation_cov = np.cov(residual.T, ddof=1)
    return VAR24CompanionFit(
        lags=p,
        coefs=coefs,
        innovation_cov=np.asarray(innovation_cov, dtype=float),
        residual=resid_df,
        fitted=fitted_df,
        state_pair=pair,
        companion_matrix=companion,
        spectral_radius=spectral_radius,
    )


def build_stochastic_covariance_spec(
    base_spec: IntensityModelSpecification,
    companion_fit: VAR24CompanionFit,
    wishart_calibration: WishartFullCovarianceCalibration,
    *,
    initial_history: Optional[np.ndarray] = None,
    initial_spike_state: Optional[float] = None,
    name: Optional[str] = None,
) -> IntensityStochasticCovarianceSpecification:
    p = int(companion_fit.lags)
    history = companion_fit.state_pair.iloc[-p:].to_numpy(dtype=float) if initial_history is None else np.asarray(initial_history, dtype=float)
    if history.shape != (p, 2):
        raise ValueError(f"initial_history must have shape {(p, 2)}, got {history.shape}.")
    spike_state = float(base_spec.initial_spike_state) if initial_spike_state is None else float(initial_spike_state)
    return IntensityStochasticCovarianceSpecification(
        name=base_spec.name if name is None else str(name),
        base_spec=base_spec,
        companion_fit=companion_fit,
        wishart=wishart_calibration,
        initial_history=history,
        initial_spike_state=spike_state,
    )


def _simulate_companion_core(
    spec: IntensityStochasticCovarianceSpecification,
    *,
    index: pd.DatetimeIndex,
    num_paths: int,
    seed: Optional[int] = None,
    stochastic_covariance: bool,
    eps: float = 1e-10,
) -> dict[str, np.ndarray]:
    idx = pd.DatetimeIndex(index)
    if len(idx) < 2:
        raise ValueError("The simulation index must contain at least two timestamps.")
    step_hours = np.diff(idx.view("i8")) / 3.6e12
    dt_hours = float(np.median(step_hours))
    if not np.allclose(step_hours, dt_hours):
        raise ValueError("The simulation index must have a constant time step.")

    base = spec.base_spec
    var_fit = spec.companion_fit
    p = int(var_fit.lags)
    P = int(num_paths)
    T = len(idx)
    rng = np.random.default_rng(seed)

    spot_season = _evaluate_seasonality(base.spot_seasonality, idx)
    wind_season = _evaluate_seasonality(base.wind_seasonality, idx)
    states = np.zeros((P, T, 2), dtype=float)
    spot = np.zeros((P, T), dtype=float)
    wind_cf = np.zeros((P, T), dtype=float)
    spike_state = np.zeros((P, T), dtype=float)
    jump_pos = np.zeros((P, T), dtype=float)
    jump_neg = np.zeros((P, T), dtype=float)
    lambda_neg = np.zeros((P, T), dtype=float)

    history = np.broadcast_to(np.asarray(spec.initial_history, dtype=float), (P, p, 2)).copy()
    states[:, 0, :] = history[:, -1, :]
    spike_state[:, 0] = float(spec.initial_spike_state)
    wind_logit0 = np.clip(wind_season[0] + states[:, 0, 1], -40.0, 40.0)
    wind_cf[:, 0] = 1.0 / (1.0 + np.exp(-wind_logit0))
    spot[:, 0] = spot_season[0] + states[:, 0, 0] + spike_state[:, 0] - float(base.spot_shift)

    decay = float(np.exp(-float(base.beta) * dt_hours / 8760.0))
    lam_pos_dt = float(base.intensity.lambda_pos * dt_hours / 8760.0)
    neg_low_dt = float(base.intensity.negative_two_state.lambda_neg_low * dt_hours / 8760.0)
    neg_high_dt = float(base.intensity.negative_two_state.lambda_neg_high * dt_hours / 8760.0)
    wp_thr = float(base.intensity.negative_two_state.wp_threshold)
    lambda_neg[:, 0] = np.where(wind_cf[:, 0] <= wp_thr, neg_low_dt * 8760.0 / dt_hours, neg_high_dt * 8760.0 / dt_hours)

    if stochastic_covariance:
        wishart_fit = spec.wishart.fit
        scale_matrix = np.asarray(spec.wishart.scale_matrix, dtype=float)
        wishart_sim_std = simulate_wishart_innovation_system(
            alpha=float(wishart_fit.alpha),
            b=np.asarray(wishart_fit.b, dtype=float),
            A=np.asarray(wishart_fit.A, dtype=float),
            rho=np.asarray(spec.wishart.leverage, dtype=float),
            sigma0=np.asarray(spec.wishart.sigma0_forecast_standardized, dtype=float),
            n_steps=T - 1,
            dt_years=dt_hours / 8760.0,
            n_paths=P,
            seed=seed,
            eps=eps,
        )
        sigma_paths = scale_matrix[None, None, :, :] @ wishart_sim_std["Sigma_paths"] @ scale_matrix[None, None, :, :]
        innovations = np.einsum("ij,ptj->pti", scale_matrix, wishart_sim_std["innovation_paths"])
    else:
        innovation_cov = np.asarray(var_fit.innovation_cov, dtype=float)
        innovation_cov = 0.5 * (innovation_cov + innovation_cov.T)
        evals, evecs = np.linalg.eigh(innovation_cov)
        evals = np.clip(evals, eps, None)
        innovation_cov = (evecs * evals[None, :]) @ evecs.T
        chol = np.linalg.cholesky(innovation_cov)
        innovations = rng.normal(size=(P, T - 1, 2)) @ chol.T
        sigma_paths = np.broadcast_to(innovation_cov, (P, T, 2, 2)).copy()

    for t in range(1, T):
        next_state = np.zeros((P, 2), dtype=float)
        for lag in range(1, p + 1):
            next_state += history[:, -lag, :] @ var_fit.coefs[lag - 1].T
        next_state += innovations[:, t - 1, :]

        wind_logit = np.clip(wind_season[t] + next_state[:, 1], -40.0, 40.0)
        wind_level = 1.0 / (1.0 + np.exp(-wind_logit))
        neg_int_dt = np.where(wind_level <= wp_thr, neg_low_dt, neg_high_dt)

        next_spike = decay * spike_state[:, t - 1]
        n_pos = rng.poisson(lam=max(lam_pos_dt, 0.0), size=P)
        n_neg = rng.poisson(lam=np.maximum(neg_int_dt, 0.0), size=P)
        for pth in range(P):
            if n_pos[pth] > 0 and base.positive_jump_sizes.size > 0:
                draws = rng.choice(base.positive_jump_sizes, size=int(n_pos[pth]), replace=True)
                jump_pos[pth, t] = float(np.sum(draws))
                next_spike[pth] += float(np.sum(draws))
            if n_neg[pth] > 0 and base.negative_jump_sizes.size > 0:
                draws = rng.choice(base.negative_jump_sizes, size=int(n_neg[pth]), replace=True)
                jump_neg[pth, t] = float(np.sum(draws))
                next_spike[pth] += float(np.sum(draws))

        history[:, :-1, :] = history[:, 1:, :]
        history[:, -1, :] = next_state
        states[:, t, :] = next_state
        spike_state[:, t] = next_spike
        wind_cf[:, t] = wind_level
        spot[:, t] = spot_season[t] + next_state[:, 0] + next_spike - float(base.spot_shift)
        lambda_neg[:, t] = np.where(wind_level <= wp_thr, neg_low_dt * 8760.0 / dt_hours, neg_high_dt * 8760.0 / dt_hours)

    wishart_corr = sigma_paths[..., 0, 1] / np.sqrt(np.maximum(sigma_paths[..., 0, 0] * sigma_paths[..., 1, 1], eps))
    wishart_corr = np.clip(wishart_corr, -1.0, 1.0)
    return {
        "spot": spot,
        "wind_cf": np.clip(wind_cf, 1e-6, 1.0 - 1e-6),
        "spot_residual": states[:, :, 0],
        "wind_residual": states[:, :, 1],
        "spike_state": spike_state,
        "spot_seasonality": spot_season[None, :].repeat(P, axis=0),
        "wind_seasonality": wind_season[None, :].repeat(P, axis=0),
        "jump_pos": jump_pos,
        "jump_neg": jump_neg,
        "lambda_neg": lambda_neg,
        "Sigma_paths": sigma_paths,
        "wishart_innovations": innovations,
        "wishart_corr": wishart_corr,
        "index": idx,
    }


def simulate_intensity_paths_constant_covariance(
    spec: IntensityStochasticCovarianceSpecification,
    *,
    index: pd.DatetimeIndex,
    num_paths: int,
    seed: Optional[int] = None,
    eps: float = 1e-10,
) -> dict[str, np.ndarray]:
    return _simulate_companion_core(
        spec,
        index=index,
        num_paths=num_paths,
        seed=seed,
        stochastic_covariance=False,
        eps=eps,
    )


def simulate_intensity_paths_stochastic_covariance(
    spec: IntensityStochasticCovarianceSpecification,
    *,
    index: pd.DatetimeIndex,
    num_paths: int,
    seed: Optional[int] = None,
    eps: float = 1e-10,
) -> dict[str, np.ndarray]:
    return _simulate_companion_core(
        spec,
        index=index,
        num_paths=num_paths,
        seed=seed,
        stochastic_covariance=True,
        eps=eps,
    )


__all__ = [
    "IntensityStochasticCovarianceSpecification",
    "VAR24CompanionFit",
    "build_stochastic_covariance_spec",
    "fit_var24_companion",
    "simulate_intensity_paths_constant_covariance",
    "simulate_intensity_paths_stochastic_covariance",
]
