from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

_TRAPEZOID = getattr(np, "trapezoid", None)
if _TRAPEZOID is None:
    _TRAPEZOID = np.trapz

try:
    from .intensity_intensity import (
        KernelIntensityEstimate,
        TwoStateNegativeIntensityFit,
        estimate_intensity_local_polynomial_paper,
        estimate_intensity_kernel_paper,
        fit_two_state_negative_intensity,
    )
except ImportError:
    from intensity_intensity import (
        KernelIntensityEstimate,
        TwoStateNegativeIntensityFit,
        estimate_intensity_local_polynomial_paper,
        estimate_intensity_kernel_paper,
        fit_two_state_negative_intensity,
    )


@dataclass
class EmpiricalStateTransform:
    name: str
    x_sorted: np.ndarray
    u_sorted: np.ndarray
    lower_clip: float = 0.01
    upper_clip: float = 0.99

    def transform(self, x: np.ndarray | pd.Series) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if self.x_sorted.size == 0:
            return np.full_like(values, 0.5, dtype=float)
        out = np.interp(
            values,
            self.x_sorted,
            self.u_sorted,
            left=float(self.u_sorted[0]),
            right=float(self.u_sorted[-1]),
        )
        return np.clip(out, float(self.lower_clip), float(self.upper_clip))


@dataclass
class CovariateIntensityCalibrationResult:
    covariate_name: str
    covariate_label: str
    state_series: pd.Series
    driver_series: pd.Series
    driver_mode: str
    driver_label: str
    negative_kernel: KernelIntensityEstimate
    positive_kernel: KernelIntensityEstimate
    negative_two_state: TwoStateNegativeIntensityFit
    lambda_pos: float
    rho: float
    event_table: pd.DataFrame
    interval_low: float
    interval_high: float
    negative_estimator_type: str
    positive_estimator_type: str
    negative_event_count: int
    positive_event_count: int
    calibration_note: str


@dataclass(frozen=True)
class RawIntensityEstimatorConfig:
    degree: int = 1
    alpha: float = 1.0
    min_event_count: int = 8
    bandwidth_count: int = 61
    max_bandwidth_fraction: float = 0.45
    min_bandwidth_expansion: float = 12.0
    min_effective_sample_size: float = 24.0
    min_local_observations: int = 18


def _trapz_masked(values: np.ndarray, grid: np.ndarray, mask: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    x = np.asarray(grid, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if m.sum() < 2:
        return float("inf")
    return float(_TRAPEZOID(vals[m], x[m]))


def _constant_kernel_estimate(
    *,
    grid: np.ndarray,
    events: pd.Series,
    dt_years: float = 1.0 / 8760.0,
    bandwidth: float = np.nan,
    objective: float = np.nan,
) -> KernelIntensityEstimate:
    ev = pd.Series(events, copy=True).fillna(0.0).astype(float)
    lam = float(ev.sum()) / max(len(ev) * dt_years, 1e-12)
    return KernelIntensityEstimate(
        grid=np.asarray(grid, dtype=float),
        bandwidth=float(bandwidth),
        values=np.full_like(np.asarray(grid, dtype=float), lam, dtype=float),
        objective=float(objective),
        bandwidth_objectives={},
    )


def _renewable_kernel_grid(covariate: pd.Series, *, num_points: int = 181) -> np.ndarray:
    x = pd.Series(covariate, copy=True).astype(float).dropna().clip(0.0, 1.0).to_numpy(dtype=float)
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


def _renewable_bandwidths(grid: np.ndarray) -> tuple[float, float, float]:
    width = float(np.max(grid) - np.min(grid))
    h_min = max(0.01, 0.05 * width)
    neg_bw = min(0.40, max(0.05, 0.30 * width))
    pos_bw = min(0.40, max(0.05, 0.35 * width))
    return h_min, neg_bw, pos_bw


def _paper_raw_bandwidth_candidates(
    *,
    span: float,
    event_count: int,
    config: RawIntensityEstimatorConfig,
) -> np.ndarray:
    span = float(max(span, 1e-12))
    count = max(int(event_count), 1)
    kernel_sup = 0.75
    kernel_l1 = 1.0
    h_min = float(max(span * kernel_sup * kernel_l1 / count, span / 500.0, 1e-6))
    h_max = float(min(span, max(config.min_bandwidth_expansion * h_min, config.max_bandwidth_fraction * span)))
    if not np.isfinite(h_max) or h_max <= h_min:
        h_max = float(min(span, max(1.5 * h_min, h_min + span / 50.0)))
    count_h = max(int(config.bandwidth_count), 5)
    return np.linspace(h_min, h_max, count_h, dtype=float)


def _estimate_intensity_raw_local_polynomial(
    *,
    covariate: pd.Series,
    events: pd.Series,
    grid: np.ndarray,
    config: RawIntensityEstimatorConfig,
    dt_years: float = 1.0 / 8760.0,
) -> tuple[KernelIntensityEstimate, str]:
    cov = pd.Series(covariate, copy=True).astype(float)
    ev = pd.Series(events, index=cov.index).fillna(0.0).astype(float)
    inside = cov.notna() & ev.notna() & (cov >= float(np.min(grid))) & (cov <= float(np.max(grid)))
    cov = cov.loc[inside]
    ev = ev.loc[inside]
    grid = np.asarray(grid, dtype=float)
    event_count = int(ev.sum())
    span = float(np.max(grid) - np.min(grid))
    if len(cov) == 0:
        return _constant_kernel_estimate(grid=grid, events=ev, dt_years=dt_years), "constant_no_support"
    if event_count < int(config.min_event_count):
        return _constant_kernel_estimate(grid=grid, events=ev, dt_years=dt_years), "constant_low_event_count"
    bandwidths = _paper_raw_bandwidth_candidates(span=span, event_count=event_count, config=config)
    if bandwidths.size == 0 or not np.all(np.isfinite(bandwidths)):
        bw = float(np.nan) if bandwidths.size == 0 else float(bandwidths[0])
        return _constant_kernel_estimate(grid=grid, events=ev, dt_years=dt_years, bandwidth=bw), "constant_invalid_bandwidth_grid"

    est = estimate_intensity_local_polynomial_paper(
        covariate=cov,
        events=ev,
        grid=grid,
        bandwidths=bandwidths,
        degree=int(config.degree),
        alpha=float(config.alpha),
        dt_years=float(dt_years),
        min_effective_sample_size=float(config.min_effective_sample_size),
        min_local_observations=int(config.min_local_observations),
    )
    values = np.asarray(est.values, dtype=float).copy()
    if np.isfinite(values).sum() < 8:
        return _constant_kernel_estimate(
            grid=grid,
            events=ev,
            dt_years=dt_years,
            bandwidth=float(est.bandwidth),
            objective=est.objective,
        ), "constant_sparse_effective_support"
    return KernelIntensityEstimate(
        grid=est.grid,
        bandwidth=est.bandwidth,
        values=values,
        objective=est.objective,
        bandwidth_objectives=est.bandwidth_objectives,
        support_mask=est.support_mask,
        effective_sample_size=est.effective_sample_size,
        variance_proxy=est.variance_proxy,
        polynomial_degree=est.polynomial_degree,
        selection_alpha=est.selection_alpha,
    ), "paper_local_polynomial_raw"


def build_empirical_state_transform(
    series: pd.Series,
    *,
    name: str,
    lower_clip: float = 0.01,
    upper_clip: float = 0.99,
) -> EmpiricalStateTransform:
    s = pd.Series(series, copy=True).astype(float).dropna().sort_values()
    if len(s) == 0:
        return EmpiricalStateTransform(
            name=str(name),
            x_sorted=np.array([0.0, 1.0], dtype=float),
            u_sorted=np.array([0.5, 0.5], dtype=float),
            lower_clip=float(lower_clip),
            upper_clip=float(upper_clip),
        )
    n = len(s)
    x_sorted = s.to_numpy(dtype=float)
    u_sorted = (np.arange(1, n + 1, dtype=float) - 0.5) / float(n)
    return EmpiricalStateTransform(
        name=str(name),
        x_sorted=x_sorted,
        u_sorted=np.clip(u_sorted, float(lower_clip), float(upper_clip)),
        lower_clip=float(lower_clip),
        upper_clip=float(upper_clip),
    )


def state_series_from_transform(
    series: pd.Series,
    transform: EmpiricalStateTransform,
    *,
    name: str,
) -> pd.Series:
    s = pd.Series(series, copy=True).astype(float)
    mask = s.notna()
    out = pd.Series(np.nan, index=s.index, name=name, dtype=float)
    out.loc[mask] = transform.transform(s.loc[mask].to_numpy(dtype=float))
    return out


def calibrate_intensity_for_covariate(
    *,
    covariate_state: pd.Series,
    covariate_raw: Optional[pd.Series] = None,
    jump_mask: pd.Series,
    jump_increment: pd.Series,
    spot_residual: pd.Series,
    wind_residual: pd.Series,
    covariate_name: str,
    covariate_label: str,
    intensity_driver_mode: str = "state",
    raw_driver_label: Optional[str] = None,
    interval_quantiles: tuple[float, float] = (0.05, 0.95),
    negative_bandwidth: float = 0.25,
    positive_bandwidth: float = 0.30,
    raw_estimator_config: Optional[RawIntensityEstimatorConfig] = None,
) -> CovariateIntensityCalibrationResult:
    raw_config = raw_estimator_config or RawIntensityEstimatorConfig()
    if intensity_driver_mode not in {"state", "raw"}:
        raise ValueError(f"Unsupported intensity_driver_mode: {intensity_driver_mode}")
    if intensity_driver_mode == "raw":
        if covariate_raw is None:
            raise ValueError("covariate_raw must be provided when intensity_driver_mode='raw'.")
        driver_series = pd.Series(covariate_raw, copy=True).astype(float)
        driver_label = str(raw_driver_label or f"{covariate_label} (raw physical units)")
    else:
        driver_series = pd.Series(covariate_state, copy=True).astype(float)
        driver_label = f"{covariate_label} (empirical-state units)"

    pre_jump_covariate = driver_series.shift(1).reindex(jump_mask.index).ffill()
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
    low_q, high_q = map(float, interval_quantiles)
    observed = pre_jump_covariate.dropna().to_numpy(dtype=float)
    if observed.size == 0:
        interval_low, interval_high = 0.05, 0.95
    else:
        interval_low, interval_high = np.quantile(observed, [low_q, high_q]).tolist()
        if not np.isfinite(interval_low) or not np.isfinite(interval_high) or interval_high <= interval_low:
            interval_low = float(np.nanmin(observed))
            interval_high = float(np.nanmax(observed))
    if not np.isfinite(interval_low) or not np.isfinite(interval_high) or interval_high <= interval_low:
        interval_low, interval_high = 0.05, 0.95
    grid = np.linspace(float(interval_low), float(interval_high), 181)
    if intensity_driver_mode == "raw":
        monotonic = None
        neg_kernel, negative_estimator_type = _estimate_intensity_raw_local_polynomial(
            covariate=pre_jump_covariate,
            events=neg_events.astype(float),
            grid=grid,
            config=raw_config,
            dt_years=1.0 / 8760.0,
        )
        pos_kernel, positive_estimator_type = _estimate_intensity_raw_local_polynomial(
            covariate=pre_jump_covariate,
            events=pos_events.astype(float),
            grid=grid,
            config=raw_config,
            dt_years=1.0 / 8760.0,
        )
    else:
        neg_h_min = 0.0225
        neg_h_max = 0.4
        neg_h_step = 0.01
        pos_h_min = 0.0089
        pos_h_max = 0.4
        pos_h_step = 0.01
        neg_fixed = float(negative_bandwidth)
        pos_fixed = float(positive_bandwidth)
        monotonic = "increasing"
        neg_kernel = estimate_intensity_kernel_paper(
            covariate=pre_jump_covariate,
            events=neg_events.astype(float),
            grid=grid,
            h_min=neg_h_min,
            h_max=neg_h_max,
            h_step=neg_h_step,
            fixed_bandwidth=neg_fixed,
        )
        pos_kernel = estimate_intensity_kernel_paper(
            covariate=pre_jump_covariate,
            events=pos_events.astype(float),
            grid=grid,
            h_min=pos_h_min,
            h_max=pos_h_max,
            h_step=pos_h_step,
            fixed_bandwidth=pos_fixed,
        )
        negative_estimator_type = "paper_kernel_state"
        positive_estimator_type = "paper_kernel_state"
    if negative_estimator_type.startswith("constant_"):
        finite_grid = np.asarray(neg_kernel.grid, dtype=float)
        finite_grid = finite_grid[np.isfinite(finite_grid)]
        thr = float(np.median(finite_grid)) if finite_grid.size > 0 else 0.0
        lam_const = float(np.nanmean(neg_kernel.values)) if np.isfinite(np.nanmean(neg_kernel.values)) else 0.0
        neg_two_state = TwoStateNegativeIntensityFit(
            lambda_neg_low=float(max(lam_const, 1e-8)),
            lambda_neg_high=float(max(lam_const, 1e-8)),
            wp_threshold=thr,
            loss=0.0,
        )
    else:
        neg_two_state = fit_two_state_negative_intensity(neg_kernel, monotonic=monotonic)
    lambda_pos = float(pos_events.sum()) / max(len(pos_events) / 8760.0, 1e-12)
    inside_interval = pre_jump_covariate.notna() & (pre_jump_covariate >= float(interval_low)) & (pre_jump_covariate <= float(interval_high))
    calibration_note = (
        (
            "Raw-driver intensity calibration with the Chapter-4 Epanechnikov local polynomial estimator "
            f"(degree={raw_config.degree}) and Goldenshluger-Lepski bandwidth selection with alpha={raw_config.alpha:.1f}."
        )
        if intensity_driver_mode == "raw"
        else "Empirical-state intensity calibration with the wind-style paper bandwidth setup."
    )
    if negative_estimator_type.startswith("constant_"):
        calibration_note += " Negative-jump nonparametric calibration fell back to a constant-rate estimate because the selected sample provides too few effective negative-jump observations."
    if positive_estimator_type.startswith("constant_"):
        calibration_note += " Positive-jump nonparametric calibration fell back to a constant-rate estimate because the selected sample provides too few effective positive-jump observations."

    common_index = pd.DatetimeIndex(spot_residual.index).intersection(pd.DatetimeIndex(wind_residual.index))
    rho = float(
        np.corrcoef(
            pd.Series(spot_residual).loc[common_index].to_numpy(dtype=float),
            pd.Series(wind_residual).loc[common_index].to_numpy(dtype=float),
        )[0, 1]
    )
    if not np.isfinite(rho):
        rho = 0.0

    cov_low = pre_jump_covariate <= neg_two_state.wp_threshold
    cov_high = pre_jump_covariate > neg_two_state.wp_threshold
    event_table = pd.DataFrame(
        [
            {
                "covariate": covariate_name,
                "event_type": "positive",
                "count": int(pos_events.sum()),
                "intensity_year_inv": float(lambda_pos),
                "n_interval_obs": int(inside_interval.sum()),
                "n_interval_events": int((pos_events & inside_interval.fillna(False)).sum()),
            },
            {
                "covariate": covariate_name,
                "event_type": "negative",
                "count": int(neg_events.sum()),
                "intensity_year_inv_low": float(neg_two_state.lambda_neg_low),
                "intensity_year_inv_high": float(neg_two_state.lambda_neg_high),
                "threshold": float(neg_two_state.wp_threshold),
                "n_interval_obs": int(inside_interval.sum()),
                "n_interval_events": int((neg_events & inside_interval.fillna(False)).sum()),
                "n_low_state_obs": int(cov_low.sum()),
                "n_high_state_obs": int(cov_high.sum()),
                "n_low_state_neg_events": int((neg_events & cov_low.fillna(False)).sum()),
                "n_high_state_neg_events": int((neg_events & cov_high.fillna(False)).sum()),
            },
        ]
    )
    return CovariateIntensityCalibrationResult(
        covariate_name=str(covariate_name),
        covariate_label=str(covariate_label),
        state_series=pd.Series(covariate_state, copy=True).astype(float),
        driver_series=driver_series,
        driver_mode=str(intensity_driver_mode),
        driver_label=str(driver_label),
        negative_kernel=neg_kernel,
        positive_kernel=pos_kernel,
        negative_two_state=neg_two_state,
        lambda_pos=float(lambda_pos),
        rho=float(rho),
        event_table=event_table,
        interval_low=float(interval_low),
        interval_high=float(interval_high),
        negative_estimator_type=str(negative_estimator_type),
        positive_estimator_type=str(positive_estimator_type),
        negative_event_count=int(neg_events.sum()),
        positive_event_count=int(pos_events.sum()),
        calibration_note=str(calibration_note),
    )


def calibrate_intensity_for_renewable_covariate(
    *,
    covariate_state: pd.Series,
    renewable_raw: pd.Series,
    jump_mask: pd.Series,
    jump_increment: pd.Series,
    spot_residual: pd.Series,
    correlation_residual: pd.Series,
    covariate_name: str,
    covariate_label: str,
) -> CovariateIntensityCalibrationResult:
    driver_series = pd.Series(renewable_raw, copy=True).astype(float).clip(0.0, 1.0)
    pre_jump_covariate = driver_series.shift(1).reindex(jump_mask.index).ffill()
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
    grid = _renewable_kernel_grid(pre_jump_covariate)
    h_min, neg_bw, pos_bw = _renewable_bandwidths(grid)
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

    common_index = pd.DatetimeIndex(spot_residual.index).intersection(pd.DatetimeIndex(correlation_residual.index))
    rho = float(
        np.corrcoef(
            pd.Series(spot_residual).loc[common_index].to_numpy(dtype=float),
            pd.Series(correlation_residual).loc[common_index].to_numpy(dtype=float),
        )[0, 1]
    )
    if not np.isfinite(rho):
        rho = 0.0

    cov_low = pre_jump_covariate <= neg_two_state.wp_threshold
    cov_high = pre_jump_covariate > neg_two_state.wp_threshold
    event_table = pd.DataFrame(
        [
            {
                "covariate": covariate_name,
                "event_type": "positive",
                "count": int(pos_events.sum()),
                "intensity_year_inv": float(lambda_pos),
                "covariate_support_min": float(grid.min()),
                "covariate_support_max": float(grid.max()),
            },
            {
                "covariate": covariate_name,
                "event_type": "negative",
                "count": int(neg_events.sum()),
                "intensity_year_inv_low": float(neg_two_state.lambda_neg_low),
                "intensity_year_inv_high": float(neg_two_state.lambda_neg_high),
                "threshold": float(neg_two_state.wp_threshold),
                "covariate_support_min": float(grid.min()),
                "covariate_support_max": float(grid.max()),
                "n_low_state_obs": int(cov_low.sum()),
                "n_high_state_obs": int(cov_high.sum()),
                "n_low_state_neg_events": int((neg_events & cov_low.fillna(False)).sum()),
                "n_high_state_neg_events": int((neg_events & cov_high.fillna(False)).sum()),
            },
        ]
    )
    return CovariateIntensityCalibrationResult(
        covariate_name=str(covariate_name),
        covariate_label=str(covariate_label),
        state_series=pd.Series(covariate_state, copy=True).astype(float),
        driver_series=driver_series,
        driver_mode="raw",
        driver_label=f"{covariate_label} (raw physical units)",
        negative_kernel=neg_kernel,
        positive_kernel=pos_kernel,
        negative_two_state=neg_two_state,
        lambda_pos=float(lambda_pos),
        rho=float(rho),
        event_table=event_table,
        interval_low=float(grid.min()),
        interval_high=float(grid.max()),
        negative_estimator_type="paper_kernel_raw_renewable",
        positive_estimator_type="paper_kernel_raw_renewable",
        negative_event_count=int(neg_events.sum()),
        positive_event_count=int(pos_events.sum()),
        calibration_note=(
            "Raw renewable intensity calibration on the lagged physical capacity factor, "
            "using the same adaptive-support fixed-bandwidth methodology as the standalone solar model."
        ),
    )


def intensity_comparison_table(
    results: Dict[str, CovariateIntensityCalibrationResult],
) -> pd.DataFrame:
    rows = []
    for key, res in results.items():
        rows.append(
            {
                "covariate": key,
                "label": res.covariate_label,
                "lambda_pos": float(res.lambda_pos),
                "lambda_neg_low": float(res.negative_two_state.lambda_neg_low),
                "lambda_neg_high": float(res.negative_two_state.lambda_neg_high),
                "threshold": float(res.negative_two_state.wp_threshold),
                "negative_bandwidth": float(res.negative_kernel.bandwidth),
                "positive_bandwidth": float(res.positive_kernel.bandwidth),
                "negative_degree": int(getattr(res.negative_kernel, "polynomial_degree", 0)),
                "positive_degree": int(getattr(res.positive_kernel, "polynomial_degree", 0)),
                "negative_selection_alpha": float(getattr(res.negative_kernel, "selection_alpha", np.nan)),
                "positive_selection_alpha": float(getattr(res.positive_kernel, "selection_alpha", np.nan)),
                "rho": float(res.rho),
                "kernel_loss": float(res.negative_two_state.loss),
                "driver_mode": res.driver_mode,
                "driver_mean": float(res.driver_series.mean()),
                "driver_std": float(res.driver_series.std(ddof=1)),
                "interval_low": float(res.interval_low),
                "interval_high": float(res.interval_high),
                "negative_estimator_type": res.negative_estimator_type,
                "positive_estimator_type": res.positive_estimator_type,
                "negative_event_count": int(res.negative_event_count),
                "positive_event_count": int(res.positive_event_count),
                "calibration_note": res.calibration_note,
                "state_mean": float(res.state_series.mean()),
                "state_std": float(res.state_series.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "CovariateIntensityCalibrationResult",
    "EmpiricalStateTransform",
    "RawIntensityEstimatorConfig",
    "build_empirical_state_transform",
    "calibrate_intensity_for_covariate",
    "calibrate_intensity_for_renewable_covariate",
    "intensity_comparison_table",
    "state_series_from_transform",
]
