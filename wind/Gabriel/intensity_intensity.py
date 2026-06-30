from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

_TRAPEZOID = getattr(np, "trapezoid", None)
if _TRAPEZOID is None:
    _TRAPEZOID = np.trapz


def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    x = np.asarray(u, dtype=float)
    out = 0.75 * (1.0 - x ** 2)
    out[np.abs(x) > 1.0] = 0.0
    return out


@dataclass
class KernelIntensityEstimate:
    grid: np.ndarray
    bandwidth: float
    values: np.ndarray
    objective: float
    bandwidth_objectives: Dict[float, float]
    support_mask: np.ndarray | None = None
    effective_sample_size: np.ndarray | None = None
    variance_proxy: np.ndarray | None = None
    polynomial_degree: int = 0
    selection_alpha: float = np.nan


@dataclass
class TwoStateNegativeIntensityFit:
    lambda_neg_low: float
    lambda_neg_high: float
    wp_threshold: float
    loss: float


@dataclass
class IntensityCalibrationResult:
    negative_kernel: KernelIntensityEstimate
    positive_kernel: KernelIntensityEstimate
    negative_two_state: TwoStateNegativeIntensityFit
    lambda_pos: float
    rho: float
    event_table: pd.DataFrame


def _paper_kernel_matrix(x: np.ndarray, grid: np.ndarray, h: float) -> np.ndarray:
    scaled = (grid[:, None] - x[None, :]) / float(h)
    return epanechnikov_kernel(scaled) / float(h)


def _trapz(values: np.ndarray, grid: np.ndarray) -> float:
    vals = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return float(_TRAPEZOID(vals, np.asarray(grid, dtype=float)))


def _trapz_masked(values: np.ndarray, grid: np.ndarray, mask: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    x = np.asarray(grid, dtype=float)
    m = np.asarray(mask, dtype=bool) & np.isfinite(vals) & np.isfinite(x)
    if m.sum() < 2:
        return float("inf")
    return float(_TRAPEZOID(vals[m], x[m]))


def _local_polynomial_smoother(
    x: np.ndarray,
    grid: np.ndarray,
    h: float,
    *,
    degree: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if degree not in {0, 1}:
        raise ValueError(f"Unsupported local polynomial degree: {degree}")

    grid = np.asarray(grid, dtype=float)
    x = np.asarray(x, dtype=float)
    kernel = _paper_kernel_matrix(x, grid, float(h))
    local_obs = np.sum(kernel > 0.0, axis=1)

    if degree == 0:
        denom = np.sum(kernel, axis=1)
        valid = denom > 1e-12
        smoother = np.zeros_like(kernel)
        smoother[valid] = kernel[valid] / denom[valid, None]
    else:
        scaled = (x[None, :] - grid[:, None]) / float(h)
        s0 = np.sum(kernel, axis=1)
        s1 = np.sum(kernel * scaled, axis=1)
        s2 = np.sum(kernel * scaled ** 2, axis=1)
        det = s0 * s2 - s1 ** 2
        scale = np.maximum(np.abs(s0 * s2), 1.0)
        valid = det > 1e-12 * scale
        smoother = np.zeros_like(kernel)
        smoother[valid] = ((s2[valid, None] - s1[valid, None] * scaled[valid]) * kernel[valid]) / det[valid, None]

    weight_sq = np.sum(smoother ** 2, axis=1)
    ess = np.zeros(len(grid), dtype=float)
    good = valid & (weight_sq > 1e-12)
    ess[good] = 1.0 / weight_sq[good]
    return smoother, valid, ess, local_obs


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
        return KernelIntensityEstimate(grid=grid, bandwidth=float(h_min), values=np.zeros_like(grid), objective=0.0, bandwidth_objectives={float(h_min): 0.0})
    h_ref = float(h_min)
    objectives: Dict[float, float] = {}

    def compute_kernel_objects(h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        K = _paper_kernel_matrix(x, grid, float(h))
        denom = np.sum(K, axis=1)
        numer = K @ y
        qhat = np.divide(numer, dt_years * np.maximum(denom, 1e-12), out=np.zeros_like(grid), where=denom > 0.0)
        var_num = (K ** 2) @ y
        var_vals = np.divide(
            var_num,
            (dt_years ** 2) * (len(x) ** 2) * np.maximum(denom, 1e-12) ** 2,
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
            (dt_years ** 2) * (len(x) ** 2) * np.maximum(denom_h, 1e-12) * np.maximum(denom_ref, 1e-12),
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
        support_mask=np.isfinite(best_qhat),
        effective_sample_size=None,
        variance_proxy=None,
        polynomial_degree=0,
        selection_alpha=1.0,
    )


def estimate_intensity_local_polynomial_paper(
    *,
    covariate: pd.Series,
    events: pd.Series,
    grid: np.ndarray,
    bandwidths: np.ndarray,
    degree: int = 1,
    alpha: float = 1.0,
    dt_years: float = 1.0 / 8760.0,
    fixed_bandwidth: float | None = None,
    min_effective_sample_size: float = 24.0,
    min_local_observations: int = 12,
) -> KernelIntensityEstimate:
    cov = pd.Series(covariate, copy=True).astype(float)
    ev = pd.Series(events, index=cov.index).fillna(0.0).astype(float)
    grid = np.asarray(grid, dtype=float)
    inside = (cov >= float(np.min(grid))) & (cov <= float(np.max(grid)))
    cov = cov.loc[inside]
    ev = ev.loc[inside]
    x = cov.to_numpy(dtype=float)
    y = ev.to_numpy(dtype=float)

    bandwidths = np.asarray(bandwidths, dtype=float)
    bandwidths = np.unique(bandwidths[np.isfinite(bandwidths) & (bandwidths > 0.0)])
    if bandwidths.size == 0:
        raise ValueError("bandwidths must contain at least one positive finite value.")

    if x.size == 0:
        return KernelIntensityEstimate(
            grid=grid,
            bandwidth=float(np.min(bandwidths)),
            values=np.full_like(grid, np.nan, dtype=float),
            objective=np.nan,
            bandwidth_objectives={},
            support_mask=np.zeros_like(grid, dtype=bool),
            effective_sample_size=np.zeros_like(grid, dtype=float),
            variance_proxy=np.full_like(grid, np.nan, dtype=float),
            polynomial_degree=int(degree),
            selection_alpha=float(alpha),
        )

    objects: Dict[float, dict[str, np.ndarray]] = {}
    objectives: Dict[float, float] = {}

    for h in bandwidths:
        smoother, valid, ess, local_obs = _local_polynomial_smoother(x, grid, float(h), degree=int(degree))
        support = valid & (ess >= float(min_effective_sample_size)) & (local_obs >= int(min_local_observations))
        qhat = np.full(len(grid), np.nan, dtype=float)
        var_hat = np.full(len(grid), np.nan, dtype=float)
        if np.any(support):
            qhat[support] = (smoother[support] @ y) / float(dt_years)
            qhat[support] = np.maximum(qhat[support], 0.0)
            var_hat[support] = ((smoother[support] ** 2) @ y) / float(dt_years ** 2)
        objects[float(h)] = {
            "qhat": qhat,
            "var": var_hat,
            "support": support,
            "smoother": smoother,
            "ess": ess,
        }

    h_ref = float(np.min(bandwidths))
    ref = objects[h_ref]
    vref = _trapz_masked(ref["var"], grid, ref["support"])
    best_h = h_ref
    best_obj = float("inf")

    for h in bandwidths:
        h = float(h)
        cur = objects[h]
        mask = cur["support"] & ref["support"]
        diff_term = _trapz_masked((cur["qhat"] - ref["qhat"]) ** 2, grid, mask)
        vh = _trapz_masked(cur["var"], grid, cur["support"])
        cross_hat = np.full(len(grid), np.nan, dtype=float)
        if np.any(mask):
            cross_hat[mask] = ((cur["smoother"][mask] * ref["smoother"][mask]) @ y) / float(dt_years ** 2)
        vhref = _trapz_masked(cross_hat, grid, mask)
        objectives[h] = float(diff_term + float(alpha) * vh - vref + 2.0 * vhref)
        if objectives[h] < best_obj:
            best_h = h
            best_obj = objectives[h]

    if fixed_bandwidth is not None:
        fixed_h = float(fixed_bandwidth)
        if fixed_h not in objects:
            smoother, valid, ess, local_obs = _local_polynomial_smoother(x, grid, fixed_h, degree=int(degree))
            support = valid & (ess >= float(min_effective_sample_size)) & (local_obs >= int(min_local_observations))
            qhat = np.full(len(grid), np.nan, dtype=float)
            var_hat = np.full(len(grid), np.nan, dtype=float)
            if np.any(support):
                qhat[support] = (smoother[support] @ y) / float(dt_years)
                qhat[support] = np.maximum(qhat[support], 0.0)
                var_hat[support] = ((smoother[support] ** 2) @ y) / float(dt_years ** 2)
            objects[fixed_h] = {
                "qhat": qhat,
                "var": var_hat,
                "support": support,
                "smoother": smoother,
                "ess": ess,
            }
            mask = support & ref["support"]
            diff_term = _trapz_masked((qhat - ref["qhat"]) ** 2, grid, mask)
            vh = _trapz_masked(var_hat, grid, support)
            cross_hat = np.full(len(grid), np.nan, dtype=float)
            if np.any(mask):
                cross_hat[mask] = ((smoother[mask] * ref["smoother"][mask]) @ y) / float(dt_years ** 2)
            vhref = _trapz_masked(cross_hat, grid, mask)
            objectives[fixed_h] = float(diff_term + float(alpha) * vh - vref + 2.0 * vhref)
        best_h = fixed_h
        best_obj = float(objectives.get(fixed_h, np.nan))

    best = objects[float(best_h)]
    return KernelIntensityEstimate(
        grid=grid,
        bandwidth=float(best_h),
        values=best["qhat"],
        objective=float(best_obj),
        bandwidth_objectives=objectives,
        support_mask=np.asarray(best["support"], dtype=bool),
        effective_sample_size=np.asarray(best["ess"], dtype=float),
        variance_proxy=np.asarray(best["var"], dtype=float),
        polynomial_degree=int(degree),
        selection_alpha=float(alpha),
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
        return TwoStateNegativeIntensityFit(lambda_neg_low=mean_q, lambda_neg_high=mean_q, wp_threshold=thr, loss=float("inf"))
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
        wp_threshold=float(thr),
        loss=float(loss),
    )


def calibrate_intensity_functions(
    *,
    wind_cf: pd.Series,
    jump_mask: pd.Series,
    jump_increment: pd.Series,
    spot_residual: pd.Series,
    wind_residual: pd.Series,
) -> IntensityCalibrationResult:
    pre_jump_covariate = pd.Series(wind_cf, copy=True).shift(1).reindex(jump_mask.index).ffill()
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
    grid = np.linspace(0.05, 0.95, 181)
    neg_kernel = estimate_intensity_kernel_paper(
        covariate=pre_jump_covariate,
        events=neg_events.astype(float),
        grid=grid,
        h_min=0.0225,
        fixed_bandwidth=0.25,
    )
    pos_kernel = estimate_intensity_kernel_paper(
        covariate=pre_jump_covariate,
        events=pos_events.astype(float),
        grid=grid,
        h_min=0.0089,
        fixed_bandwidth=0.30,
    )
    neg_two_state = fit_two_state_negative_intensity(neg_kernel)
    lambda_pos = float(pos_events.sum()) / max(len(pos_events) / 8760.0, 1e-12)
    common_index = pd.DatetimeIndex(spot_residual.index).intersection(pd.DatetimeIndex(wind_residual.index))
    rho = float(
        np.corrcoef(
            pd.Series(spot_residual).loc[common_index].to_numpy(dtype=float),
            pd.Series(wind_residual).loc[common_index].to_numpy(dtype=float),
        )[0, 1]
    )
    if not np.isfinite(rho):
        rho = 0.0

    event_table = pd.DataFrame(
        [
            {"event_type": "positive", "count": int(pos_events.sum()), "intensity_year_inv": lambda_pos},
            {
                "event_type": "negative",
                "count": int(neg_events.sum()),
                "intensity_year_inv_low": neg_two_state.lambda_neg_low,
                "intensity_year_inv_high": neg_two_state.lambda_neg_high,
                "threshold": neg_two_state.wp_threshold,
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
    )
