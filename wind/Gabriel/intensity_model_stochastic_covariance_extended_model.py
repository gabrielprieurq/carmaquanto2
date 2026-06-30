from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from .intensity_ar import AR24Fit
    from .intensity_intensity_extended_model import (
        CovariateIntensityCalibrationResult,
        EmpiricalStateTransform,
    )
    from .intensity_seasonality import (
        SeasonalityFit,
        evaluate_clear_sky_proxy,
        evaluate_harmonic,
        evaluate_mle_notebook,
        evaluate_paper_phase,
        hours_from_origin,
    )
    from .intensity_var_extended_model import MultivariateVARCompanionFitExtendedModel
    from .intensity_wishart_stochastic_covariance_extended_model import (
        WishartFullCovarianceCalibrationExtendedModel,
        simulate_wishart_innovation_system_extended_model,
    )
    from .intensity_model import IntensityModelSpecification
except ImportError:
    from intensity_ar import AR24Fit
    from intensity_intensity_extended_model import (
        CovariateIntensityCalibrationResult,
        EmpiricalStateTransform,
    )
    from intensity_seasonality import (
        SeasonalityFit,
        evaluate_clear_sky_proxy,
        evaluate_harmonic,
        evaluate_mle_notebook,
        evaluate_paper_phase,
        hours_from_origin,
    )
    from intensity_var_extended_model import MultivariateVARCompanionFitExtendedModel
    from intensity_wishart_stochastic_covariance_extended_model import (
        WishartFullCovarianceCalibrationExtendedModel,
        simulate_wishart_innovation_system_extended_model,
    )
    from intensity_model import IntensityModelSpecification


@dataclass
class ActiveCovariateLatentSpecificationExtendedModel:
    name: str
    label: str
    transform: str
    seasonality: Optional[SeasonalityFit]
    ar_fit: AR24Fit
    state_transform: EmpiricalStateTransform


@dataclass
class CovariateDrivenStochasticCovarianceSpecification:
    name: str
    base_spec: IntensityModelSpecification
    primary_covariate_name: str
    primary_covariate_label: str
    active_covariates: list[str]
    covariate_specs: dict[str, ActiveCovariateLatentSpecificationExtendedModel]
    joint_var_fit: MultivariateVARCompanionFitExtendedModel
    wishart: WishartFullCovarianceCalibrationExtendedModel
    initial_history: np.ndarray
    initial_spike_state: float
    intensity: CovariateIntensityCalibrationResult


def _evaluate_seasonality(fit: Optional[SeasonalityFit], index: pd.DatetimeIndex) -> np.ndarray:
    if fit is None:
        return np.zeros(len(index), dtype=float)
    if fit.parameterization in {"mle_notebook", "solar_logit"}:
        return evaluate_mle_notebook(fit, index)
    hours = hours_from_origin(index, origin=fit.origin)
    if fit.parameterization == "paper_phase":
        return evaluate_paper_phase(fit.paper_params, hours)
    return evaluate_harmonic(fit.harmonic_params, hours)


def _back_transform(
    values: np.ndarray,
    transform: str,
    *,
    fit: Optional[SeasonalityFit] = None,
    clear_value: Optional[float] = None,
) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    inverse_kind = "identity" if fit is None else str(getattr(fit, "inverse_kind", "identity"))
    if inverse_kind == "raw_logit" or transform == "raw_logit":
        return np.clip(1.0 / (1.0 + np.exp(-x)), 1e-6, 1.0 - 1e-6)
    if inverse_kind == "solar_clear_sky_logit" or transform == "solar_clear_sky_logit":
        if fit is None or fit.transform_params is None:
            raise ValueError("Solar clear-sky inverse requested without seasonality transform parameters.")
        if clear_value is None:
            raise ValueError("Solar clear-sky inverse requested without a clear-sky proxy value.")
        xprime = 1.0 / (1.0 + np.exp(-x))
        cf = float(clear_value) * (1.0 - fit.transform_params.alpha - fit.transform_params.beta * xprime)
        return np.clip(cf, 0.0, 1.0)
    if transform == "identity":
        return x
    if transform == "log":
        return np.exp(np.clip(x, -40.0, 40.0))
    if transform == "log1p":
        return np.expm1(np.clip(x, -40.0, 40.0))
    if transform == "signed_log1p":
        clipped = np.clip(x, -40.0, 40.0)
        return np.sign(clipped) * np.expm1(np.abs(clipped))
    raise ValueError(f"Unsupported covariate transform: {transform}")


def _simulate_companion_with_covariates(
    spec: CovariateDrivenStochasticCovarianceSpecification,
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
    var_fit = spec.joint_var_fit
    active_covs = list(spec.active_covariates)
    names = list(var_fit.names)
    d = len(names)
    p = int(var_fit.lags)
    P = int(num_paths)
    T = len(idx)
    rng = np.random.default_rng(seed)

    name_to_idx = {name: i for i, name in enumerate(names)}
    spot_idx = name_to_idx["spot_state"]
    wind_idx = name_to_idx["wind_state"]
    cov_indices = {name: name_to_idx[name] for name in active_covs}

    spot_season = _evaluate_seasonality(base.spot_seasonality, idx)
    wind_season = _evaluate_seasonality(base.wind_seasonality, idx)
    cov_seasons = {
        name: _evaluate_seasonality(spec.covariate_specs[name].seasonality, idx)
        for name in active_covs
    }
    cov_clear = {
        name: (
            evaluate_clear_sky_proxy(spec.covariate_specs[name].seasonality, idx)
            if spec.covariate_specs[name].seasonality is not None
            else np.ones(T, dtype=float)
        )
        for name in active_covs
    }

    states = np.zeros((P, T, d), dtype=float)
    spot = np.zeros((P, T), dtype=float)
    wind_cf = np.zeros((P, T), dtype=float)
    spike_state = np.zeros((P, T), dtype=float)
    jump_pos = np.zeros((P, T), dtype=float)
    jump_neg = np.zeros((P, T), dtype=float)
    lambda_neg = np.zeros((P, T), dtype=float)

    covariate_residual = {name: np.zeros((P, T), dtype=float) for name in active_covs}
    covariate_signal = {name: np.zeros((P, T), dtype=float) for name in active_covs}
    covariate_state = {name: np.zeros((P, T), dtype=float) for name in active_covs}
    covariate_value = {name: np.zeros((P, T), dtype=float) for name in active_covs}

    history = np.broadcast_to(np.asarray(spec.initial_history, dtype=float), (P, p, d)).copy()
    states[:, 0, :] = history[:, -1, :]
    spike_state[:, 0] = float(spec.initial_spike_state)
    wind_logit0 = np.clip(wind_season[0] + states[:, 0, wind_idx], -40.0, 40.0)
    wind_cf[:, 0] = 1.0 / (1.0 + np.exp(-wind_logit0))
    for name in active_covs:
        j = cov_indices[name]
        cov_signal0 = cov_seasons[name][0] + states[:, 0, j]
        covariate_residual[name][:, 0] = states[:, 0, j]
        covariate_signal[name][:, 0] = cov_signal0
        covariate_state[name][:, 0] = spec.covariate_specs[name].state_transform.transform(cov_signal0)
        covariate_value[name][:, 0] = _back_transform(
            cov_signal0,
            spec.covariate_specs[name].transform,
            fit=spec.covariate_specs[name].seasonality,
            clear_value=float(cov_clear[name][0]),
        )
    spot[:, 0] = spot_season[0] + states[:, 0, spot_idx] + spike_state[:, 0] - float(base.spot_shift)

    decay = float(np.exp(-float(base.beta) * dt_hours / 8760.0))
    lam_pos_dt = float(spec.intensity.lambda_pos * dt_hours / 8760.0)
    neg_low_dt = float(spec.intensity.negative_two_state.lambda_neg_low * dt_hours / 8760.0)
    neg_high_dt = float(spec.intensity.negative_two_state.lambda_neg_high * dt_hours / 8760.0)
    state_thr = float(spec.intensity.negative_two_state.wp_threshold)
    intensity_mode = str(getattr(spec.intensity, "driver_mode", "state"))
    primary_state0 = covariate_state[spec.primary_covariate_name][:, 0]
    primary_value0 = covariate_value[spec.primary_covariate_name][:, 0]
    primary_driver0 = primary_value0 if intensity_mode == "raw" else primary_state0
    lambda_neg[:, 0] = np.where(primary_driver0 <= state_thr, neg_low_dt * 8760.0 / dt_hours, neg_high_dt * 8760.0 / dt_hours)

    if stochastic_covariance:
        wishart_fit = spec.wishart.fit
        scale_matrix = np.asarray(spec.wishart.scale_matrix, dtype=float)
        wishart_sim_std = simulate_wishart_innovation_system_extended_model(
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
        innovations = rng.normal(size=(P, T - 1, d)) @ chol.T
        sigma_paths = np.broadcast_to(innovation_cov, (P, T, d, d)).copy()

    for t in range(1, T):
        next_state = np.zeros((P, d), dtype=float)
        for lag in range(1, p + 1):
            next_state += history[:, -lag, :] @ var_fit.coefs[lag - 1].T
        next_state += innovations[:, t - 1, :]

        wind_logit = np.clip(wind_season[t] + next_state[:, wind_idx], -40.0, 40.0)
        wind_level = 1.0 / (1.0 + np.exp(-wind_logit))
        primary_signal = cov_seasons[spec.primary_covariate_name][t] + next_state[:, cov_indices[spec.primary_covariate_name]]
        primary_state = spec.covariate_specs[spec.primary_covariate_name].state_transform.transform(primary_signal)
        primary_value = _back_transform(
            primary_signal,
            spec.covariate_specs[spec.primary_covariate_name].transform,
            fit=spec.covariate_specs[spec.primary_covariate_name].seasonality,
            clear_value=float(cov_clear[spec.primary_covariate_name][t]),
        )
        primary_driver = primary_value if intensity_mode == "raw" else primary_state
        neg_int_dt = np.where(primary_driver <= state_thr, neg_low_dt, neg_high_dt)

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
        spot[:, t] = spot_season[t] + next_state[:, spot_idx] + next_spike - float(base.spot_shift)
        lambda_neg[:, t] = np.where(primary_driver <= state_thr, neg_low_dt * 8760.0 / dt_hours, neg_high_dt * 8760.0 / dt_hours)
        for name in active_covs:
            j = cov_indices[name]
            signal = cov_seasons[name][t] + next_state[:, j]
            covariate_residual[name][:, t] = next_state[:, j]
            covariate_signal[name][:, t] = signal
            covariate_state[name][:, t] = spec.covariate_specs[name].state_transform.transform(signal)
            covariate_value[name][:, t] = _back_transform(
                signal,
                spec.covariate_specs[name].transform,
                fit=spec.covariate_specs[name].seasonality,
                clear_value=float(cov_clear[name][t]),
            )

    out: dict[str, np.ndarray] = {
        "spot": spot,
        "wind_cf": np.clip(wind_cf, 1e-6, 1.0 - 1e-6),
        "spot_residual": states[:, :, spot_idx],
        "wind_residual": states[:, :, wind_idx],
        "spike_state": spike_state,
        "spot_seasonality": spot_season[None, :].repeat(P, axis=0),
        "wind_seasonality": wind_season[None, :].repeat(P, axis=0),
        "jump_pos": jump_pos,
        "jump_neg": jump_neg,
        "lambda_neg": lambda_neg,
        "Sigma_paths": sigma_paths,
        "wishart_innovations": innovations,
        "index": idx,
        "active_covariates": np.array(active_covs, dtype=object),
        "primary_covariate_name": np.array(spec.primary_covariate_name, dtype=object),
    }
    for name in active_covs:
        out[f"covariate_residual__{name}"] = covariate_residual[name]
        out[f"covariate_signal__{name}"] = covariate_signal[name]
        out[f"covariate_state__{name}"] = covariate_state[name]
        out[f"covariate_value__{name}"] = covariate_value[name]
        out[f"covariate_clear_sky__{name}"] = cov_clear[name][None, :].repeat(P, axis=0)

    primary = spec.primary_covariate_name
    out["covariate_residual"] = covariate_residual[primary]
    out["covariate_signal"] = covariate_signal[primary]
    out["covariate_state"] = covariate_state[primary]
    out["covariate_value"] = covariate_value[primary]
    out["covariate_clear_sky"] = cov_clear[primary][None, :].repeat(P, axis=0)
    return out


def simulate_intensity_paths_constant_covariance_extended_model(
    spec: CovariateDrivenStochasticCovarianceSpecification,
    *,
    index: pd.DatetimeIndex,
    num_paths: int,
    seed: Optional[int] = None,
    eps: float = 1e-10,
) -> dict[str, np.ndarray]:
    return _simulate_companion_with_covariates(
        spec,
        index=index,
        num_paths=num_paths,
        seed=seed,
        stochastic_covariance=False,
        eps=eps,
    )


def simulate_intensity_paths_stochastic_covariance_extended_model(
    spec: CovariateDrivenStochasticCovarianceSpecification,
    *,
    index: pd.DatetimeIndex,
    num_paths: int,
    seed: Optional[int] = None,
    eps: float = 1e-10,
) -> dict[str, np.ndarray]:
    return _simulate_companion_with_covariates(
        spec,
        index=index,
        num_paths=num_paths,
        seed=seed,
        stochastic_covariance=True,
        eps=eps,
    )


__all__ = [
    "ActiveCovariateLatentSpecificationExtendedModel",
    "CovariateDrivenStochasticCovarianceSpecification",
    "simulate_intensity_paths_constant_covariance_extended_model",
    "simulate_intensity_paths_stochastic_covariance_extended_model",
]
