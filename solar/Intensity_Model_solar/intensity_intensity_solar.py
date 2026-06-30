from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    x = np.asarray(u, dtype=float)
    out = 0.75 * (1.0 - x**2)
    out[np.abs(x) > 1.0] = 0.0
    return out


@dataclass
class KernelIntensityEstimate:
    grid: np.ndarray
    bandwidth: float
    values: np.ndarray
    objective: float
    bandwidth_objectives: Dict[float, float]


@dataclass
class TwoStateNegativeIntensityFit:
    lambda_neg_low: float
    lambda_neg_high: float
    renewable_threshold: float
    loss: float


@dataclass
class IntensityCalibrationResult:
    negative_kernel: KernelIntensityEstimate
    positive_kernel: KernelIntensityEstimate
    negative_two_state: TwoStateNegativeIntensityFit
    lambda_pos: float
    rho: float
    event_table: pd.DataFrame
    grid_min: float
    grid_max: float


def _paper_kernel_matrix(x: np.ndarray, grid: np.ndarray, h: float) -> np.ndarray:
    scaled = (grid[:, None] - x[None, :]) / float(h)
    return epanechnikov_kernel(scaled) / float(h)


def _trapz(values: np.ndarray, grid: np.ndarray) -> float:
    vals = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.trapz(vals, np.asarray(grid, dtype=float)))


def estimate_intensity_kernel_paper(
    *,
    covariate: pd.Series,
    events: pd.Series,
    grid: np.ndarray,
    h_min: float,
    h_max: float = 0.4,
    h_step: float = 0.01,
    dt_years: float = 1.0 / 8760.0,
    fixed_bandwidth: float | None = None,
) -> KernelIntensityEstimate:
    cov = pd.Series(covariate, copy=True).astype(float)
    ev = pd.Series(events, index=cov.index).fillna(0.0).astype(float)
    inside = (cov >= float(np.min(grid))) & (cov <= float(np.max(grid)))
    cov = cov.loc[inside]
    ev = ev.loc[inside]
    x = cov.to_numpy(dtype=float)
    y = ev.to_numpy(dtype=float)
    grid = np.asarray(grid, dtype=float)
    bandwidths = np.arange(float(h_min), float(h_max) + 1e-12, float(h_step))
    if x.size == 0:
        return KernelIntensityEstimate(
            grid=grid,
            bandwidth=float(h_min),
            values=np.zeros_like(grid),
            objective=0.0,
            bandwidth_objectives={float(h_min): 0.0},
        )
    h_ref = float(h_min)
    objectives: Dict[float, float] = {}

    def compute_kernel_objects(h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        K = _paper_kernel_matrix(x, grid, float(h))
        denom = np.sum(K, axis=1)
        numer = K @ y
        qhat = np.divide(numer, dt_years * np.maximum(denom, 1e-12), out=np.zeros_like(grid), where=denom > 0.0)
        var_num = (K**2) @ y
        var_vals = np.divide(
            var_num,
            (dt_years**2) * (len(x) ** 2) * np.maximum(denom, 1e-12) ** 2,
            out=np.zeros_like(grid),
            where=denom > 0.0,
        )
        return K, denom, qhat, var_vals

    K_ref, denom_ref, qhat_ref, var_ref = compute_kernel_objects(h_ref)
    best_h = h_ref
    best_qhat = qhat_ref
    best_obj = np.inf
    vref = _trapz(var_ref, grid)
    qhat_by_bandwidth: Dict[float, np.ndarray] = {h_ref: qhat_ref}
    for h in bandwidths:
        h = float(h)
        K_h, denom_h, qhat_h, var_h = compute_kernel_objects(h)
        qhat_by_bandwidth[h] = qhat_h
        diff_term = _trapz((qhat_h - qhat_ref) ** 2, grid)
        vh = _trapz(var_h, grid)
        cross_num = np.sum(K_h * K_ref * y[None, :], axis=1)
        cross_vals = np.divide(
            cross_num,
            (dt_years**2) * (len(x) ** 2) * np.maximum(denom_h, 1e-12) * np.maximum(denom_ref, 1e-12),
            out=np.zeros_like(grid),
            where=(denom_h > 0.0) & (denom_ref > 0.0),
        )
        vhref = _trapz(cross_vals, grid)
        objectives[h] = float(diff_term + (vh - vref + 2.0 * vhref))
        if objectives[h] < best_obj:
            best_obj = objectives[h]
            best_h = h
            best_qhat = qhat_h
    if fixed_bandwidth is not None:
        fixed_h = float(fixed_bandwidth)
        if fixed_h not in qhat_by_bandwidth:
            _, _, qhat_fixed, _ = compute_kernel_objects(fixed_h)
            qhat_by_bandwidth[fixed_h] = qhat_fixed
            objectives[fixed_h] = objectives.get(fixed_h, float("nan"))
        best_h = fixed_h
        best_qhat = qhat_by_bandwidth[fixed_h]
        best_obj = float(objectives.get(fixed_h, np.nan))
    return KernelIntensityEstimate(
        grid=grid,
        bandwidth=float(best_h),
        values=best_qhat,
        objective=float(best_obj),
        bandwidth_objectives=objectives,
    )


def fit_two_state_negative_intensity(
    kernel_est: KernelIntensityEstimate,
    *,
    monotonic: str | None = "increasing",
) -> TwoStateNegativeIntensityFit:
    x = np.asarray(kernel_est.grid, dtype=float)
    q = np.asarray(kernel_est.values, dtype=float)
    keep = np.isfinite(x) & np.isfinite(q)
    x = x[keep]
    q = q[keep]
    if x.size < 6:
        thr = float(np.nanmedian(kernel_est.grid)) if np.isfinite(np.nanmedian(kernel_est.grid)) else 0.0
        mean_q = float(np.nanmean(kernel_est.values)) if np.isfinite(np.nanmean(kernel_est.values)) else 1e-8
        mean_q = float(max(mean_q, 1e-8))
        return TwoStateNegativeIntensityFit(
            lambda_neg_low=mean_q,
            lambda_neg_high=mean_q,
            renewable_threshold=thr,
            loss=float("inf"),
        )
    best = (np.inf, float(np.median(x)), float(np.mean(q)), float(np.mean(q)))
    for thr in x[(x > x.min()) & (x < x.max())]:
        left = x <= thr
        right = x > thr
        if left.sum() < 3 or right.sum() < 3:
            continue
        lam_lo = float(np.mean(q[left]))
        lam_hi = float(np.mean(q[right]))
        if monotonic == "increasing" and lam_hi < lam_lo:
            continue
        if monotonic == "decreasing" and lam_hi > lam_lo:
            continue
        fit = np.where(left, lam_lo, lam_hi)
        loss = _trapz((q - fit) ** 2, x)
        if loss < best[0]:
            best = (loss, float(thr), lam_lo, lam_hi)
    loss, thr, lam_lo, lam_hi = best
    return TwoStateNegativeIntensityFit(
        lambda_neg_low=float(max(lam_lo, 1e-8)),
        lambda_neg_high=float(max(lam_hi, 1e-8)),
        renewable_threshold=float(thr),
        loss=float(loss),
    )


def suggested_kernel_grid(renewable_cf: pd.Series, *, num_points: int = 181) -> np.ndarray:
    x = pd.Series(renewable_cf, copy=True).astype(float).dropna().clip(0.0, 1.0).to_numpy(dtype=float)
    if x.size == 0:
        return np.linspace(0.05, 0.95, int(num_points))
    low = float(np.quantile(x, 0.01))
    high = float(np.quantile(x, 0.995))
    low = max(0.0, low)
    high = min(0.99, high)
    if high - low < 0.10:
        low = max(0.0, float(np.min(x)))
        high = min(0.99, float(np.max(x)))
    if high - low < 0.05:
        low = 0.0
        high = 0.99
    return np.linspace(low, high, int(num_points))


def _adaptive_bandwidths(grid: np.ndarray) -> tuple[float, float, float]:
    width = float(np.max(grid) - np.min(grid))
    h_min = max(0.01, 0.05 * width)
    neg_bw = min(0.40, max(0.05, 0.30 * width))
    pos_bw = min(0.40, max(0.05, 0.35 * width))
    return h_min, neg_bw, pos_bw


def calibrate_intensity_functions(
    *,
    renewable_cf: pd.Series,
    jump_mask: pd.Series,
    jump_increment: pd.Series,
    spot_residual: pd.Series,
    renewable_residual: pd.Series,
    grid: np.ndarray | None = None,
) -> IntensityCalibrationResult:
    pre_jump_covariate = pd.Series(renewable_cf, copy=True).shift(1).reindex(jump_mask.index).ffill()
    neg_events = pd.Series(
        jump_mask.to_numpy(dtype=bool) & (jump_increment.to_numpy(dtype=float) < 0.0),
        index=jump_mask.index,
        name="negative_jump_event",
    )
    pos_events = pd.Series(
        jump_mask.to_numpy(dtype=bool) & (jump_increment.to_numpy(dtype=float) > 0.0),
        index=jump_mask.index,
        name="positive_jump_event",
    )
    if grid is None:
        grid = suggested_kernel_grid(pre_jump_covariate)
    grid = np.asarray(grid, dtype=float)
    h_min, neg_bw, pos_bw = _adaptive_bandwidths(grid)
    neg_kernel = estimate_intensity_kernel_paper(
        covariate=pre_jump_covariate,
        events=neg_events.astype(float),
        grid=grid,
        h_min=h_min,
        fixed_bandwidth=neg_bw,
    )
    pos_kernel = estimate_intensity_kernel_paper(
        covariate=pre_jump_covariate,
        events=pos_events.astype(float),
        grid=grid,
        h_min=h_min,
        fixed_bandwidth=pos_bw,
    )
    neg_two_state = fit_two_state_negative_intensity(neg_kernel, monotonic="increasing")
    lambda_pos = float(pos_events.sum()) / max(len(pos_events) / 8760.0, 1e-12)
    common_index = pd.DatetimeIndex(spot_residual.index).intersection(pd.DatetimeIndex(renewable_residual.index))
    rho = float(
        np.corrcoef(
            pd.Series(spot_residual).loc[common_index].to_numpy(dtype=float),
            pd.Series(renewable_residual).loc[common_index].to_numpy(dtype=float),
        )[0, 1]
    )
    if not np.isfinite(rho):
        rho = 0.0

    event_table = pd.DataFrame(
        [
            {
                "event_type": "positive",
                "count": int(pos_events.sum()),
                "intensity_year_inv": lambda_pos,
                "covariate_support_min": float(grid.min()),
                "covariate_support_max": float(grid.max()),
            },
            {
                "event_type": "negative",
                "count": int(neg_events.sum()),
                "intensity_year_inv_low": neg_two_state.lambda_neg_low,
                "intensity_year_inv_high": neg_two_state.lambda_neg_high,
                "threshold": neg_two_state.renewable_threshold,
                "covariate_support_min": float(grid.min()),
                "covariate_support_max": float(grid.max()),
            },
        ]
    )
    return IntensityCalibrationResult(
        negative_kernel=neg_kernel,
        positive_kernel=pos_kernel,
        negative_two_state=neg_two_state,
        lambda_pos=lambda_pos,
        rho=rho,
        event_table=event_table,
        grid_min=float(grid.min()),
        grid_max=float(grid.max()),
    )


__all__ = [
    "IntensityCalibrationResult",
    "KernelIntensityEstimate",
    "TwoStateNegativeIntensityFit",
    "calibrate_intensity_functions",
    "estimate_intensity_kernel_paper",
    "epanechnikov_kernel",
    "fit_two_state_negative_intensity",
    "suggested_kernel_grid",
]
