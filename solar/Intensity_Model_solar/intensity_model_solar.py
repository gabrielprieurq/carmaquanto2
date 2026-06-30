from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from .intensity_ar_solar import AR24Fit
    from .intensity_intensity_solar import IntensityCalibrationResult
    from .intensity_seasonality_solar import (
        SeasonalityFit,
        evaluate_clear_sky_proxy,
        evaluate_harmonic,
        evaluate_mle_notebook,
        evaluate_paper_phase,
        hours_from_origin,
    )
except ImportError:
    from intensity_ar_solar import AR24Fit
    from intensity_intensity_solar import IntensityCalibrationResult
    from intensity_seasonality_solar import (
        SeasonalityFit,
        evaluate_clear_sky_proxy,
        evaluate_harmonic,
        evaluate_mle_notebook,
        evaluate_paper_phase,
        hours_from_origin,
    )


@dataclass
class IntensityModelSpecification:
    name: str
    spot_shift: float
    spot_seasonality: SeasonalityFit
    solar_seasonality: SeasonalityFit
    spot_ar: AR24Fit
    solar_ar: AR24Fit
    beta: float
    positive_jump_sizes: np.ndarray
    negative_jump_sizes: np.ndarray
    intensity: IntensityCalibrationResult
    initial_spot_lags: np.ndarray
    initial_solar_lags: np.ndarray
    initial_spike_state: float


def _evaluate_seasonality(fit: SeasonalityFit, index: pd.DatetimeIndex) -> np.ndarray:
    if fit.parameterization in {"mle_notebook", "solar_logit"}:
        return evaluate_mle_notebook(fit, index)
    hours = hours_from_origin(index, origin=fit.origin)
    if fit.parameterization == "paper_phase":
        return evaluate_paper_phase(fit.paper_params, hours)
    return evaluate_harmonic(fit.harmonic_params, hours)


def _latent_to_physical(fit: SeasonalityFit, clear_values: np.ndarray, latent: np.ndarray) -> np.ndarray:
    z = np.asarray(latent, dtype=float)
    if fit.inverse_kind == "raw_logit":
        return np.clip(1.0 / (1.0 + np.exp(-z)), 1e-6, 1.0 - 1e-6)
    if fit.inverse_kind == "solar_clear_sky_logit":
        if fit.transform_params is None:
            raise ValueError("Solar inverse requested without transform parameters.")
        xprime = 1.0 / (1.0 + np.exp(-z))
        cf = clear_values * (1.0 - fit.transform_params.alpha - fit.transform_params.beta * xprime)
        return np.clip(cf, 0.0, 1.0)
    return z


def simulate_intensity_paths(
    spec: IntensityModelSpecification,
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

    spot_season = _evaluate_seasonality(spec.spot_seasonality, idx)
    solar_latent_season = _evaluate_seasonality(spec.solar_seasonality, idx)
    solar_clear = evaluate_clear_sky_proxy(spec.solar_seasonality, idx)

    spot_hist = np.zeros((P, T, p), dtype=float)
    solar_hist = np.zeros((P, T, p), dtype=float)
    spot_resid = np.zeros((P, T), dtype=float)
    solar_resid = np.zeros((P, T), dtype=float)
    spike_state = np.zeros((P, T), dtype=float)
    spot = np.zeros((P, T), dtype=float)
    solar_cf = np.zeros((P, T), dtype=float)
    lambda_neg = np.zeros((P, T), dtype=float)
    jump_pos = np.zeros((P, T), dtype=float)
    jump_neg = np.zeros((P, T), dtype=float)

    coeff_spot = np.asarray(spec.spot_ar.coeffs, dtype=float)
    coeff_solar = np.asarray(spec.solar_ar.coeffs, dtype=float)
    sigma_spot = float(spec.spot_ar.innovation_std)
    sigma_solar = float(spec.solar_ar.innovation_std)
    rho = float(np.clip(spec.intensity.rho, -0.999, 0.999))
    chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]], dtype=float))
    decay = float(np.exp(-float(spec.beta) / 8760.0))
    lam_pos_dt = float(spec.intensity.lambda_pos / 8760.0)
    neg_low_dt = float(spec.intensity.negative_two_state.lambda_neg_low / 8760.0)
    neg_high_dt = float(spec.intensity.negative_two_state.lambda_neg_high / 8760.0)
    solar_thr = float(spec.intensity.negative_two_state.renewable_threshold)

    spot_hist[:, 0, :] = spec.initial_spot_lags[None, :]
    solar_hist[:, 0, :] = spec.initial_solar_lags[None, :]
    spot_resid[:, 0] = spec.initial_spot_lags[0]
    solar_resid[:, 0] = spec.initial_solar_lags[0]
    spike_state[:, 0] = float(spec.initial_spike_state)
    solar_latent0 = solar_latent_season[0] + solar_resid[:, 0]
    solar_cf[:, 0] = _latent_to_physical(spec.solar_seasonality, np.full(P, solar_clear[0], dtype=float), solar_latent0)
    spot[:, 0] = spot_season[0] + spot_resid[:, 0] + spike_state[:, 0] - float(spec.spot_shift)
    lambda_neg[:, 0] = np.where(solar_cf[:, 0] <= solar_thr, neg_low_dt * 8760.0, neg_high_dt * 8760.0)

    for t in range(1, T):
        z = rng.normal(size=(P, 2)) @ chol.T
        eps_spot = sigma_spot * z[:, 0]
        eps_solar = sigma_solar * z[:, 1]
        spot_next = spot_hist[:, t - 1, :] @ coeff_spot + eps_spot
        solar_next = solar_hist[:, t - 1, :] @ coeff_solar + eps_solar

        solar_latent = solar_latent_season[t] + solar_next
        solar_level = _latent_to_physical(
            spec.solar_seasonality,
            np.full(P, solar_clear[t], dtype=float),
            solar_latent,
        )
        neg_int_dt = np.where(solar_level <= solar_thr, neg_low_dt, neg_high_dt)

        next_spike = decay * spike_state[:, t - 1]
        n_pos = rng.poisson(lam=lam_pos_dt, size=P)
        n_neg = rng.poisson(lam=neg_int_dt, size=P)

        for pth in range(P):
            if n_pos[pth] > 0 and spec.positive_jump_sizes.size > 0:
                draws = rng.choice(spec.positive_jump_sizes, size=int(n_pos[pth]), replace=True)
                jump_pos[pth, t] = float(np.sum(draws))
                next_spike[pth] += float(np.sum(draws))
            if n_neg[pth] > 0 and spec.negative_jump_sizes.size > 0:
                draws = rng.choice(spec.negative_jump_sizes, size=int(n_neg[pth]), replace=True)
                jump_neg[pth, t] = float(np.sum(draws))
                next_spike[pth] += float(np.sum(draws))

        spot_hist[:, t, 0] = spot_next
        solar_hist[:, t, 0] = solar_next
        if p > 1:
            spot_hist[:, t, 1:] = spot_hist[:, t - 1, :-1]
            solar_hist[:, t, 1:] = solar_hist[:, t - 1, :-1]

        spot_resid[:, t] = spot_next
        solar_resid[:, t] = solar_next
        spike_state[:, t] = next_spike
        solar_cf[:, t] = solar_level
        spot[:, t] = spot_season[t] + spot_next + next_spike - float(spec.spot_shift)
        lambda_neg[:, t] = np.where(solar_level <= solar_thr, neg_low_dt * 8760.0, neg_high_dt * 8760.0)

    return {
        "spot": spot,
        "solar_cf": np.clip(solar_cf, 0.0, 1.0),
        "spot_residual": spot_resid,
        "solar_residual": solar_resid,
        "spike_state": spike_state,
        "spot_seasonality": spot_season[None, :].repeat(P, axis=0),
        "solar_latent_seasonality": solar_latent_season[None, :].repeat(P, axis=0),
        "solar_clear_sky": solar_clear[None, :].repeat(P, axis=0),
        "jump_pos": jump_pos,
        "jump_neg": jump_neg,
        "lambda_neg": lambda_neg,
        "index": idx,
    }


__all__ = [
    "IntensityModelSpecification",
    "simulate_intensity_paths",
]
