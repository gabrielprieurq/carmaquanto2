from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def calibration_years(index: pd.DatetimeIndex) -> float:
    idx = pd.DatetimeIndex(index)
    if len(idx) <= 1:
        return 0.0
    return float((idx[-1] - idx[0]).total_seconds() / (365.25 * 24.0 * 3600.0))


def multipower_variation(returns: pd.Series, order: int = 20) -> float:
    r = np.abs(np.asarray(returns.dropna(), dtype=float))
    dt_years = 1.0 / 8760.0
    if r.size <= order:
        return float(np.nanstd(r, ddof=1) / math.sqrt(dt_years)) if r.size > 1 else 0.0
    power = 2.0 / float(order)
    mu = (2.0 ** (power / 2.0)) * math.gamma((power + 1.0) / 2.0) / math.sqrt(math.pi)
    windows = sliding_window_view(r, order)
    block_products = np.prod(np.power(windows, power), axis=1)
    integrated_var = float(np.sum(block_products) / (mu ** order))
    sample_years = max(float(r.size) * dt_years, 1e-12)
    return float(math.sqrt(max(integrated_var / sample_years, 0.0)))


@dataclass
class SpikeDetectionResult:
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

    @property
    def sample_years(self) -> float:
        return calibration_years(self.shifted_spot.index)


def detect_spikes_paper(
    spot: pd.Series,
    *,
    shift_min_level: float = 1.0,
    threshold_scale: float = 5.0,
    threshold_power: float = 0.49,
    multipower_order: int = 20,
    follow_up_drop: int = 10,
) -> SpikeDetectionResult:
    raw = pd.Series(spot, copy=True).astype(float).dropna().sort_index()
    shift = max(float(shift_min_level) - float(raw.min()), 0.0)
    shifted = (raw + shift).rename("shifted_spot")
    returns = shifted.diff().rename("spot_return")

    dt_years = 1.0 / 8760.0
    sigma_bar_by_year: Dict[int, float] = {}
    threshold_by_year: Dict[int, float] = {}
    extreme_mask = pd.Series(False, index=shifted.index, name="extreme_mask")
    jump_mask = pd.Series(False, index=shifted.index, name="jump_mask")

    for year in shifted.index.year.unique():
        loc = shifted.index.year == year
        ret_year = returns.loc[loc].dropna()
        sigma_bar = multipower_variation(ret_year, order=multipower_order)
        sigma_bar_by_year[int(year)] = sigma_bar
        threshold = threshold_scale * sigma_bar * (dt_years ** threshold_power)
        threshold_by_year[int(year)] = threshold
        idx = ret_year.index
        extreme = ret_year.abs() >= threshold
        jump = extreme & (ret_year * ret_year.shift(-1) < 0.0)
        extreme_mask.loc[idx] = extreme.fillna(False).to_numpy(dtype=bool)
        jump_mask.loc[idx] = jump.fillna(False).to_numpy(dtype=bool)

    filtered_mask = extreme_mask.copy()
    extreme_positions = np.flatnonzero(extreme_mask.to_numpy(dtype=bool))
    for pos in extreme_positions:
        hi = min(pos + follow_up_drop + 1, len(filtered_mask))
        filtered_mask.iloc[pos:hi] = True
    filtered_mask = filtered_mask.rename("filtered_mask")

    jump_increment = pd.Series(0.0, index=shifted.index, name="jump_increment")
    jump_increment.loc[jump_mask] = returns.loc[jump_mask].to_numpy(dtype=float)
    pos_sizes = jump_increment.loc[jump_increment > 0.0].to_numpy(dtype=float)
    neg_sizes = jump_increment.loc[jump_increment < 0.0].to_numpy(dtype=float)

    beta_samples = []
    jump_positions = np.flatnonzero(jump_mask.to_numpy(dtype=bool))
    for pos in jump_positions:
        if pos + 1 >= len(returns):
            continue
        jump_size = float(returns.iloc[pos])
        next_ret = float(returns.iloc[pos + 1])
        if jump_size == 0.0:
            continue
        ratio = -next_ret / jump_size
        if 1e-8 < ratio < 1.0 - 1e-8:
            beta_samples.append(-math.log(1.0 - ratio) / dt_years)
    beta_samples = np.asarray(beta_samples, dtype=float)
    beta = float(np.median(beta_samples)) if beta_samples.size > 0 else 7718.84
    decay = float(np.exp(-beta * dt_years))

    return SpikeDetectionResult(
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
        beta_samples=beta_samples,
        decay=decay,
    )


def reconstruct_spike_path(
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
