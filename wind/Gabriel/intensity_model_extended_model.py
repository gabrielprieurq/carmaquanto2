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
    from .intensity_model import IntensityModelSpecification
    from .intensity_seasonality import (
        SeasonalityFit,
        evaluate_clear_sky_proxy,
        evaluate_harmonic,
        evaluate_mle_notebook,
        evaluate_paper_phase,
        hours_from_origin,
    )
except ImportError:
    from intensity_ar import AR24Fit
    from intensity_intensity_extended_model import (
        CovariateIntensityCalibrationResult,
        EmpiricalStateTransform,
    )
    from intensity_model import IntensityModelSpecification
    from intensity_seasonality import (
        SeasonalityFit,
        evaluate_clear_sky_proxy,
        evaluate_harmonic,
        evaluate_mle_notebook,
        evaluate_paper_phase,
        hours_from_origin,
    )


@dataclass
class CovariateDrivenIntensityModelSpecification:
    name: str
    base_spec: IntensityModelSpecification
    covariate_name: str
    covariate_label: str
    covariate_transform: str
    covariate_seasonality: Optional[SeasonalityFit]
    covariate_ar: AR24Fit
    covariate_state_transform: EmpiricalStateTransform
    initial_covariate_lags: np.ndarray
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


def simulate_intensity_paths_extended_model(
    spec: CovariateDrivenIntensityModelSpecification,
    *,
    index: pd.DatetimeIndex,
    num_paths: int,
    seed: Optional[int] = None,
) -> dict[str, np.ndarray]:
    idx = pd.DatetimeIndex(index)
    P = int(num_paths)
    T = len(idx)
    p = 24
    rng = np.random.default_rng(seed)

    base = spec.base_spec
    spot_season = _evaluate_seasonality(base.spot_seasonality, idx)
    wind_season = _evaluate_seasonality(base.wind_seasonality, idx)
    cov_season = _evaluate_seasonality(spec.covariate_seasonality, idx)
    cov_clear = (
        evaluate_clear_sky_proxy(spec.covariate_seasonality, idx)
        if spec.covariate_seasonality is not None
        else np.ones(T, dtype=float)
    )

    spot_hist = np.zeros((P, T, p), dtype=float)
    wind_hist = np.zeros((P, T, p), dtype=float)
    cov_hist = np.zeros((P, T, p), dtype=float)
    spot_resid = np.zeros((P, T), dtype=float)
    wind_resid = np.zeros((P, T), dtype=float)
    cov_resid = np.zeros((P, T), dtype=float)
    cov_signal = np.zeros((P, T), dtype=float)
    cov_state = np.zeros((P, T), dtype=float)
    cov_value = np.zeros((P, T), dtype=float)
    spike_state = np.zeros((P, T), dtype=float)
    spot = np.zeros((P, T), dtype=float)
    wind_cf = np.zeros((P, T), dtype=float)
    lambda_neg = np.zeros((P, T), dtype=float)
    jump_pos = np.zeros((P, T), dtype=float)
    jump_neg = np.zeros((P, T), dtype=float)

    coeff_spot = np.asarray(base.spot_ar.coeffs, dtype=float)
    coeff_wind = np.asarray(base.wind_ar.coeffs, dtype=float)
    coeff_cov = np.asarray(spec.covariate_ar.coeffs, dtype=float)
    sigma_spot = float(base.spot_ar.innovation_std)
    sigma_wind = float(base.wind_ar.innovation_std)
    sigma_cov = float(spec.covariate_ar.innovation_std)
    rho = float(np.clip(spec.intensity.rho, -0.999, 0.999))
    chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]], dtype=float))
    decay = float(np.exp(-float(base.beta) / 8760.0))
    lam_pos_dt = float(spec.intensity.lambda_pos / 8760.0)
    neg_low_dt = float(spec.intensity.negative_two_state.lambda_neg_low / 8760.0)
    neg_high_dt = float(spec.intensity.negative_two_state.lambda_neg_high / 8760.0)
    state_thr = float(spec.intensity.negative_two_state.wp_threshold)
    intensity_mode = str(getattr(spec.intensity, "driver_mode", "state"))

    spot_hist[:, 0, :] = base.initial_spot_lags[None, :]
    wind_hist[:, 0, :] = base.initial_wind_lags[None, :]
    cov_hist[:, 0, :] = spec.initial_covariate_lags[None, :]
    spot_resid[:, 0] = base.initial_spot_lags[0]
    wind_resid[:, 0] = base.initial_wind_lags[0]
    cov_resid[:, 0] = spec.initial_covariate_lags[0]
    spike_state[:, 0] = float(base.initial_spike_state)

    wind_logit0 = np.clip(wind_season[0] + wind_resid[:, 0], -40.0, 40.0)
    wind_cf[:, 0] = 1.0 / (1.0 + np.exp(-wind_logit0))
    cov_signal[:, 0] = cov_season[0] + cov_resid[:, 0]
    cov_state[:, 0] = spec.covariate_state_transform.transform(cov_signal[:, 0])
    cov_value[:, 0] = _back_transform(
        cov_signal[:, 0],
        spec.covariate_transform,
        fit=spec.covariate_seasonality,
        clear_value=float(cov_clear[0]),
    )
    spot[:, 0] = spot_season[0] + spot_resid[:, 0] + spike_state[:, 0] - float(base.spot_shift)
    intensity_driver0 = cov_value[:, 0] if intensity_mode == "raw" else cov_state[:, 0]
    lambda_neg[:, 0] = np.where(intensity_driver0 <= state_thr, neg_low_dt * 8760.0, neg_high_dt * 8760.0)

    for t in range(1, T):
        z = rng.normal(size=(P, 2)) @ chol.T
        eps_spot = sigma_spot * z[:, 0]
        eps_wind = sigma_wind * z[:, 1]
        eps_cov = sigma_cov * rng.normal(size=P)
        spot_next = spot_hist[:, t - 1, :] @ coeff_spot + eps_spot
        wind_next = wind_hist[:, t - 1, :] @ coeff_wind + eps_wind
        cov_next = cov_hist[:, t - 1, :] @ coeff_cov + eps_cov

        wind_logit = np.clip(wind_season[t] + wind_next, -40.0, 40.0)
        wind_level = 1.0 / (1.0 + np.exp(-wind_logit))
        cov_signal_t = cov_season[t] + cov_next
        cov_state_t = spec.covariate_state_transform.transform(cov_signal_t)
        cov_value_t = _back_transform(
            cov_signal_t,
            spec.covariate_transform,
            fit=spec.covariate_seasonality,
            clear_value=float(cov_clear[t]),
        )
        intensity_driver_t = cov_value_t if intensity_mode == "raw" else cov_state_t
        neg_int_dt = np.where(intensity_driver_t <= state_thr, neg_low_dt, neg_high_dt)

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

        spot_hist[:, t, 0] = spot_next
        wind_hist[:, t, 0] = wind_next
        cov_hist[:, t, 0] = cov_next
        if p > 1:
            spot_hist[:, t, 1:] = spot_hist[:, t - 1, :-1]
            wind_hist[:, t, 1:] = wind_hist[:, t - 1, :-1]
            cov_hist[:, t, 1:] = cov_hist[:, t - 1, :-1]

        spot_resid[:, t] = spot_next
        wind_resid[:, t] = wind_next
        cov_resid[:, t] = cov_next
        cov_signal[:, t] = cov_signal_t
        cov_state[:, t] = cov_state_t
        cov_value[:, t] = cov_value_t
        spike_state[:, t] = next_spike
        wind_cf[:, t] = wind_level
        spot[:, t] = spot_season[t] + spot_next + next_spike - float(base.spot_shift)
        lambda_neg[:, t] = np.where(intensity_driver_t <= state_thr, neg_low_dt * 8760.0, neg_high_dt * 8760.0)

    return {
        "spot": spot,
        "wind_cf": np.clip(wind_cf, 1e-6, 1.0 - 1e-6),
        "spot_residual": spot_resid,
        "wind_residual": wind_resid,
        "spike_state": spike_state,
        "spot_seasonality": spot_season[None, :].repeat(P, axis=0),
        "wind_seasonality": wind_season[None, :].repeat(P, axis=0),
        "jump_pos": jump_pos,
        "jump_neg": jump_neg,
        "lambda_neg": lambda_neg,
        "covariate_residual": cov_resid,
        "covariate_signal": cov_signal,
        "covariate_state": cov_state,
        "covariate_value": cov_value,
        "covariate_clear_sky": cov_clear[None, :].repeat(P, axis=0),
        "covariate_name": spec.covariate_name,
        "index": idx,
    }


__all__ = [
    "CovariateDrivenIntensityModelSpecification",
    "simulate_intensity_paths_extended_model",
]
