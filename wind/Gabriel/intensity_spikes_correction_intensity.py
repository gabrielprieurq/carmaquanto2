from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable

import numpy as np
import pandas as pd

try:
    from .intensity_spikes import calibration_years, multipower_variation
except ImportError:
    from intensity_spikes import calibration_years, multipower_variation


@dataclass(frozen=True)
class SpikeDetectionCorrectionConfig:
    shift_min_level: float = 1.0
    threshold_scale: float = 5.0
    threshold_power: float = 0.49
    multipower_order: int = 20
    follow_up_drop: int = 10
    positive_recovery_window: int = 3
    negative_recovery_window: int = 12
    positive_recovery_fraction: float = 0.35
    negative_recovery_fraction: float = 0.35
    episode_merge_gap: int = 12
    beta_ratio_cap: float = 0.98


DEFAULT_SPIKE_CORRECTION_CONFIG = SpikeDetectionCorrectionConfig()


@dataclass
class SpikeDetectionCorrectionResult:
    spot_shift: float
    shifted_spot: pd.Series
    returns: pd.Series
    sigma_bar_by_year: Dict[int, float]
    threshold_by_year: Dict[int, float]
    extreme_mask: pd.Series
    jump_mask: pd.Series
    jump_increment: pd.Series
    filtered_mask: pd.Series
    positive_jump_sizes: np.ndarray
    negative_jump_sizes: np.ndarray
    beta: float
    beta_samples: np.ndarray
    decay: float
    candidate_mask: pd.Series
    representative_extreme_mask: pd.Series
    recovery_ratio: pd.Series
    recovery_horizon_hours: pd.Series
    config: SpikeDetectionCorrectionConfig

    @property
    def sample_years(self) -> float:
        return calibration_years(self.shifted_spot.index)


def _yearly_thresholds(
    shifted: pd.Series,
    *,
    threshold_scale: float,
    threshold_power: float,
    multipower_order: int,
) -> tuple[pd.Series, Dict[int, float], Dict[int, float], pd.Series]:
    returns = shifted.diff().rename("spot_return")
    dt_years = 1.0 / 8760.0
    sigma_bar_by_year: Dict[int, float] = {}
    threshold_by_year: Dict[int, float] = {}
    threshold_series = pd.Series(np.nan, index=shifted.index, name="threshold")
    extreme_mask = pd.Series(False, index=shifted.index, name="extreme_mask")

    for year in shifted.index.year.unique():
        loc = shifted.index.year == year
        ret_year = returns.loc[loc].dropna()
        sigma_bar = multipower_variation(ret_year, order=multipower_order)
        sigma_bar_by_year[int(year)] = float(sigma_bar)
        threshold = float(threshold_scale * sigma_bar * (dt_years ** threshold_power))
        threshold_by_year[int(year)] = threshold
        threshold_series.loc[ret_year.index] = threshold
        extreme_mask.loc[ret_year.index] = (ret_year.abs() >= threshold).to_numpy(dtype=bool)
    return returns, sigma_bar_by_year, threshold_by_year, extreme_mask


def _extreme_clusters(
    returns: pd.Series,
    extreme_mask: pd.Series,
    *,
    merge_gap: int,
) -> list[list[int]]:
    values = returns.to_numpy(dtype=float)
    extreme_positions = np.flatnonzero(extreme_mask.to_numpy(dtype=bool))
    clusters: list[list[int]] = []
    i = 0
    while i < len(extreme_positions):
        cluster = [int(extreme_positions[i])]
        prev = int(extreme_positions[i])
        sign = float(np.sign(values[prev]))
        j = i + 1
        while j < len(extreme_positions):
            pos = int(extreme_positions[j])
            if pos - prev > int(merge_gap):
                break
            if float(np.sign(values[pos])) != sign:
                break
            cluster.append(pos)
            prev = pos
            j += 1
        clusters.append(cluster)
        i = j
    return clusters


def _best_opposite_recovery(
    returns: np.ndarray,
    position: int,
    *,
    sign: float,
    horizon: int,
) -> tuple[float, int | None]:
    start = int(position) + 1
    stop = min(start + int(horizon), len(returns))
    future = returns[start:stop]
    if future.size == 0:
        return 0.0, None
    cumulative = np.cumsum(future)
    opposite = cumulative if sign < 0.0 else -cumulative
    best_idx = int(np.argmax(opposite))
    best_recovery = float(max(opposite[best_idx], 0.0))
    if best_recovery <= 0.0:
        return 0.0, None
    return best_recovery, best_idx + 1


def detect_spikes_correction_intensity(
    spot: pd.Series,
    *,
    config: SpikeDetectionCorrectionConfig = DEFAULT_SPIKE_CORRECTION_CONFIG,
) -> SpikeDetectionCorrectionResult:
    raw = pd.Series(spot, copy=True).astype(float).dropna().sort_index()
    shift = max(float(config.shift_min_level) - float(raw.min()), 0.0)
    shifted = (raw + shift).rename("shifted_spot")

    returns, sigma_bar_by_year, threshold_by_year, extreme_mask = _yearly_thresholds(
        shifted,
        threshold_scale=float(config.threshold_scale),
        threshold_power=float(config.threshold_power),
        multipower_order=int(config.multipower_order),
    )

    representative_extreme_mask = pd.Series(False, index=shifted.index, name="representative_extreme_mask")
    candidate_mask = pd.Series(False, index=shifted.index, name="candidate_mask")
    jump_mask = pd.Series(False, index=shifted.index, name="jump_mask")
    recovery_ratio = pd.Series(np.nan, index=shifted.index, name="recovery_ratio")
    recovery_horizon = pd.Series(np.nan, index=shifted.index, name="recovery_horizon_hours")

    values = returns.to_numpy(dtype=float)
    blocked_until = -1
    clusters = _extreme_clusters(returns, extreme_mask, merge_gap=int(config.episode_merge_gap))

    for cluster in clusters:
        representative = max(cluster, key=lambda pos: abs(values[pos]))
        representative_extreme_mask.iloc[representative] = True
        if representative <= blocked_until or not np.isfinite(values[representative]) or values[representative] == 0.0:
            continue

        sign = float(np.sign(values[representative]))
        horizon = int(config.negative_recovery_window if sign < 0.0 else config.positive_recovery_window)
        fraction = float(config.negative_recovery_fraction if sign < 0.0 else config.positive_recovery_fraction)
        best_recovery, best_h = _best_opposite_recovery(values, representative, sign=sign, horizon=horizon)
        ratio = best_recovery / max(abs(values[representative]), 1e-12)
        recovery_ratio.iloc[representative] = ratio
        recovery_horizon.iloc[representative] = float(best_h) if best_h is not None else np.nan
        if best_h is None or ratio < fraction:
            continue

        candidate_mask.iloc[representative] = True
        jump_mask.iloc[representative] = True
        blocked_until = representative + max(int(config.episode_merge_gap), int(best_h))

    filtered_mask = extreme_mask.copy()
    extreme_positions = np.flatnonzero(extreme_mask.to_numpy(dtype=bool))
    for pos in extreme_positions:
        hi = min(pos + int(config.follow_up_drop) + 1, len(filtered_mask))
        filtered_mask.iloc[pos:hi] = True
    filtered_mask = filtered_mask.rename("filtered_mask")

    jump_increment = pd.Series(0.0, index=shifted.index, name="jump_increment")
    jump_increment.loc[jump_mask] = returns.loc[jump_mask].to_numpy(dtype=float)
    pos_sizes = jump_increment.loc[jump_increment > 0.0].to_numpy(dtype=float)
    neg_sizes = jump_increment.loc[jump_increment < 0.0].to_numpy(dtype=float)

    beta_samples: list[float] = []
    jump_positions = np.flatnonzero(jump_mask.to_numpy(dtype=bool))
    dt_years = 1.0 / 8760.0
    beta_cap = float(np.clip(config.beta_ratio_cap, 1e-6, 1.0 - 1e-6))
    for pos in jump_positions:
        h = recovery_horizon.iloc[pos]
        ratio = recovery_ratio.iloc[pos]
        if not np.isfinite(h) or not np.isfinite(ratio) or h <= 0.0:
            continue
        effective_ratio = float(np.clip(ratio, 1e-8, beta_cap))
        beta_samples.append(-math.log(1.0 - effective_ratio) / (float(h) * dt_years))
    beta_array = np.asarray(beta_samples, dtype=float)
    beta = float(np.median(beta_array)) if beta_array.size > 0 else 7718.84
    decay = float(np.exp(-beta * dt_years))

    return SpikeDetectionCorrectionResult(
        spot_shift=float(shift),
        shifted_spot=shifted,
        returns=returns,
        sigma_bar_by_year=sigma_bar_by_year,
        threshold_by_year=threshold_by_year,
        extreme_mask=extreme_mask,
        jump_mask=jump_mask,
        jump_increment=jump_increment,
        filtered_mask=filtered_mask,
        positive_jump_sizes=pos_sizes,
        negative_jump_sizes=neg_sizes,
        beta=beta,
        beta_samples=beta_array,
        decay=decay,
        candidate_mask=candidate_mask,
        representative_extreme_mask=representative_extreme_mask,
        recovery_ratio=recovery_ratio,
        recovery_horizon_hours=recovery_horizon,
        config=config,
    )


def reconstruct_spike_path_correction_intensity(
    jump_increment: pd.Series,
    *,
    beta: float,
) -> pd.Series:
    s = pd.Series(jump_increment, copy=True).astype(float).sort_index()
    out = pd.Series(0.0, index=s.index, name="spike_state")
    decay = float(np.exp(-float(beta) / 8760.0))
    for i in range(1, len(s)):
        out.iloc[i] = decay * out.iloc[i - 1] + float(s.iloc[i])
    return out


def spike_summary_table_correction_intensity(spikes: SpikeDetectionCorrectionResult) -> pd.DataFrame:
    rows = []
    returns = pd.Series(spikes.returns, copy=False)
    for year, threshold in spikes.threshold_by_year.items():
        year_mask = returns.index.year == year
        jump_year = spikes.jump_mask.loc[year_mask]
        recovery_year = spikes.recovery_ratio.loc[year_mask]
        horizon_year = spikes.recovery_horizon_hours.loc[year_mask]
        rows.append(
            {
                "year": int(year),
                "sigma_bar": float(spikes.sigma_bar_by_year[year]),
                "threshold": float(threshold),
                "n_extremes": int(spikes.extreme_mask.loc[year_mask].sum()),
                "n_negative_extremes": int((spikes.extreme_mask.loc[year_mask] & (returns.loc[year_mask] < 0.0)).sum()),
                "n_positive_extremes": int((spikes.extreme_mask.loc[year_mask] & (returns.loc[year_mask] > 0.0)).sum()),
                "n_jumps": int(jump_year.sum()),
                "n_negative_jumps": int((jump_year & (returns.loc[year_mask] < 0.0)).sum()),
                "n_positive_jumps": int((jump_year & (returns.loc[year_mask] > 0.0)).sum()),
                "mean_recovery_ratio": float(recovery_year.loc[jump_year.index[jump_year]].mean()) if jump_year.any() else float("nan"),
                "mean_recovery_horizon_hours": float(horizon_year.loc[jump_year.index[jump_year]].mean()) if jump_year.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def jump_detection_sensitivity_table_correction_intensity(
    spot: pd.Series,
    *,
    positive_windows: Iterable[int] = (1, 3, 6),
    negative_windows: Iterable[int] = (6, 12, 18),
    positive_fractions: Iterable[float] = (0.25, 0.35, 0.50),
    negative_fractions: Iterable[float] = (0.25, 0.35, 0.50),
    merge_gaps: Iterable[int] = (6, 12),
    base_config: SpikeDetectionCorrectionConfig = DEFAULT_SPIKE_CORRECTION_CONFIG,
) -> pd.DataFrame:
    rows = []
    for pos_window in positive_windows:
        for neg_window in negative_windows:
            for pos_fraction in positive_fractions:
                for neg_fraction in negative_fractions:
                    for merge_gap in merge_gaps:
                        config = SpikeDetectionCorrectionConfig(
                            shift_min_level=float(base_config.shift_min_level),
                            threshold_scale=float(base_config.threshold_scale),
                            threshold_power=float(base_config.threshold_power),
                            multipower_order=int(base_config.multipower_order),
                            follow_up_drop=int(base_config.follow_up_drop),
                            positive_recovery_window=int(pos_window),
                            negative_recovery_window=int(neg_window),
                            positive_recovery_fraction=float(pos_fraction),
                            negative_recovery_fraction=float(neg_fraction),
                            episode_merge_gap=int(merge_gap),
                            beta_ratio_cap=float(base_config.beta_ratio_cap),
                        )
                        spikes = detect_spikes_correction_intensity(spot, config=config)
                        negative_jumps = int((spikes.jump_increment < 0.0).sum())
                        positive_jumps = int((spikes.jump_increment > 0.0).sum())
                        neg_recovery = spikes.recovery_ratio.loc[spikes.jump_increment < 0.0]
                        pos_recovery = spikes.recovery_ratio.loc[spikes.jump_increment > 0.0]
                        mean_neg_recovery = float(neg_recovery.mean()) if len(neg_recovery) else float("nan")
                        mean_pos_recovery = float(pos_recovery.mean()) if len(pos_recovery) else float("nan")
                        score = (
                            2.5 * math.log1p(negative_jumps)
                            + 1.0 * math.log1p(positive_jumps)
                            + min(float(np.nan_to_num(mean_neg_recovery, nan=0.0)), 1.0)
                            + 0.5 * min(float(np.nan_to_num(mean_pos_recovery, nan=0.0)), 1.0)
                        )
                        rows.append(
                            {
                                "positive_recovery_window": int(pos_window),
                                "negative_recovery_window": int(neg_window),
                                "positive_recovery_fraction": float(pos_fraction),
                                "negative_recovery_fraction": float(neg_fraction),
                                "episode_merge_gap": int(merge_gap),
                                "negative_jump_count": negative_jumps,
                                "positive_jump_count": positive_jumps,
                                "beta_sample_count": int(len(spikes.beta_samples)),
                                "mean_negative_recovery_ratio": mean_neg_recovery,
                                "mean_positive_recovery_ratio": mean_pos_recovery,
                                "selection_score": float(score),
                            }
                        )
    return pd.DataFrame(rows).sort_values(
        ["selection_score", "negative_jump_count", "positive_jump_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


__all__ = [
    "DEFAULT_SPIKE_CORRECTION_CONFIG",
    "SpikeDetectionCorrectionConfig",
    "SpikeDetectionCorrectionResult",
    "detect_spikes_correction_intensity",
    "jump_detection_sensitivity_table_correction_intensity",
    "reconstruct_spike_path_correction_intensity",
    "spike_summary_table_correction_intensity",
]
