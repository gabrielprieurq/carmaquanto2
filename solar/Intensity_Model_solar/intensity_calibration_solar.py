from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from .intensity_ar_solar import AR24Fit, ar_parameter_table, fit_ar24_exact_mle
    from .intensity_data_solar import JointSolarMarketData, load_joint_market_data_solar
    from .intensity_intensity_solar import IntensityCalibrationResult, calibrate_intensity_functions
    from .intensity_model_solar import IntensityModelSpecification
    from .intensity_seasonality_solar import (
        SeasonalityFit,
        fit_mle_notebook_seasonality,
        fit_naive_raw_solar_logit_seasonality,
        fit_paper_phase_seasonality,
        fit_solar_logit_seasonality,
        notebook_parameter_table,
        paper_parameter_table,
        solar_transform_parameter_table,
    )
    from .intensity_spikes_solar import SpikeDetectionResult, detect_spikes_paper, reconstruct_spike_path
except ImportError:
    from intensity_ar_solar import AR24Fit, ar_parameter_table, fit_ar24_exact_mle
    from intensity_data_solar import JointSolarMarketData, load_joint_market_data_solar
    from intensity_intensity_solar import IntensityCalibrationResult, calibrate_intensity_functions
    from intensity_model_solar import IntensityModelSpecification
    from intensity_seasonality_solar import (
        SeasonalityFit,
        fit_mle_notebook_seasonality,
        fit_naive_raw_solar_logit_seasonality,
        fit_paper_phase_seasonality,
        fit_solar_logit_seasonality,
        notebook_parameter_table,
        paper_parameter_table,
        solar_transform_parameter_table,
    )
    from intensity_spikes_solar import SpikeDetectionResult, detect_spikes_paper, reconstruct_spike_path


@dataclass
class SpotCalibrationVariant:
    name: str
    seasonality: SeasonalityFit
    spike_state: pd.Series
    continuous_series: pd.Series
    ar_fit: AR24Fit
    filtered_series_for_ar: pd.Series


@dataclass
class SolarCalibrationVariant:
    name: str
    seasonality: SeasonalityFit
    ar_fit: AR24Fit


@dataclass
class IntensityCalibrationWorkspace:
    data: JointSolarMarketData
    spike_detection: SpikeDetectionResult
    spot_variants: Dict[str, SpotCalibrationVariant]
    solar_variants: Dict[str, SolarCalibrationVariant]
    intensity_variants: Dict[str, IntensityCalibrationResult]
    model_specs: Dict[str, IntensityModelSpecification]


def _build_spot_variant(
    *,
    name: str,
    spot: pd.Series,
    spike_detection: SpikeDetectionResult,
    spike_state: pd.Series,
    fit_mask: pd.Series,
    mode: str,
) -> SpotCalibrationVariant:
    if mode == "paper_phase":
        seasonality = fit_paper_phase_seasonality(spot, name=name, fit_mask=fit_mask)
    elif mode == "mle_notebook":
        seasonality = fit_mle_notebook_seasonality(spot, name=name, fit_mask=fit_mask, fourier_order=2)
    else:
        raise ValueError(f"Unsupported spot seasonality mode: {mode}")
    continuous = (spot - seasonality.fitted - spike_state).rename(f"{name}_continuous")
    ar_series = continuous.loc[~spike_detection.filtered_mask].dropna()
    ar_fit = fit_ar24_exact_mle(ar_series)
    return SpotCalibrationVariant(
        name=name,
        seasonality=seasonality,
        spike_state=spike_state,
        continuous_series=continuous,
        ar_fit=ar_fit,
        filtered_series_for_ar=ar_series,
    )


def _build_solar_variant(
    *,
    name: str,
    solar_cf: pd.Series,
    mode: str,
) -> SolarCalibrationVariant:
    if mode == "solar_clearsky":
        seasonality = fit_solar_logit_seasonality(solar_cf, name=name)
    elif mode == "raw_logit_mle":
        seasonality = fit_naive_raw_solar_logit_seasonality(solar_cf, name=name, fourier_order=2)
    else:
        raise ValueError(f"Unsupported solar seasonality mode: {mode}")
    ar_fit = fit_ar24_exact_mle(seasonality.residual.dropna())
    return SolarCalibrationVariant(name=name, seasonality=seasonality, ar_fit=ar_fit)


def _replace_rho(base: IntensityCalibrationResult, rho: float) -> IntensityCalibrationResult:
    return IntensityCalibrationResult(
        negative_kernel=base.negative_kernel,
        positive_kernel=base.positive_kernel,
        negative_two_state=base.negative_two_state,
        lambda_pos=base.lambda_pos,
        rho=float(rho),
        event_table=base.event_table.copy(),
        grid_min=float(base.grid_min),
        grid_max=float(base.grid_max),
    )


def run_intensity_model_calibration_solar(
    *,
    spot_path: str = "Vb-Academy PPAs Application/DayAheadPrices_2021_2025.csv",
    solar_path: str = "enwex_GER_solar_v25_combined.csv",
    start: Optional[str | pd.Timestamp] = None,
    end: Optional[str | pd.Timestamp] = None,
    calibration_start: Optional[str | pd.Timestamp] = None,
    calibration_end: Optional[str | pd.Timestamp] = None,
) -> IntensityCalibrationWorkspace:
    if start is None and calibration_start is not None:
        start = calibration_start
    if end is None and calibration_end is not None:
        end = calibration_end
    data = load_joint_market_data_solar(spot_path=spot_path, solar_path=solar_path, start=start, end=end)
    spike_detection = detect_spikes_paper(data.spot)
    spike_state = reconstruct_spike_path(spike_detection.jump_increment, beta=spike_detection.beta)

    spot_variants = {
        "paper": _build_spot_variant(
            name="spot_paper",
            spot=data.spot,
            spike_detection=spike_detection,
            spike_state=spike_state,
            fit_mask=~spike_detection.filtered_mask,
            mode="paper_phase",
        ),
        "mle_notebook_filtered": _build_spot_variant(
            name="spot_mle_filtered",
            spot=data.spot,
            spike_detection=spike_detection,
            spike_state=spike_state,
            fit_mask=~spike_detection.filtered_mask,
            mode="mle_notebook",
        ),
        "mle_notebook_full": _build_spot_variant(
            name="spot_mle_full",
            spot=data.spot,
            spike_detection=spike_detection,
            spike_state=spike_state,
            fit_mask=pd.Series(True, index=data.spot.index),
            mode="mle_notebook",
        ),
    }

    solar_variants = {
        "solar_clearsky": _build_solar_variant(name="solar_clearsky", solar_cf=data.solar_cf, mode="solar_clearsky"),
        "mle_notebook_raw": _build_solar_variant(name="solar_mle_raw", solar_cf=data.solar_cf, mode="raw_logit_mle"),
    }

    base_intensity = calibrate_intensity_functions(
        renewable_cf=data.solar_cf,
        jump_mask=spike_detection.jump_mask,
        jump_increment=spike_detection.jump_increment,
        spot_residual=spot_variants["paper"].ar_fit.residual,
        renewable_residual=solar_variants["solar_clearsky"].ar_fit.residual,
    )
    rho_index = pd.DatetimeIndex(spot_variants["mle_notebook_full"].ar_fit.residual.index).intersection(
        pd.DatetimeIndex(solar_variants["mle_notebook_raw"].ar_fit.residual.index)
    )
    if len(rho_index) == 0:
        mle_rho = 0.0
    else:
        mle_rho = float(
            np.corrcoef(
                spot_variants["mle_notebook_full"].ar_fit.residual.loc[rho_index].to_numpy(dtype=float),
                solar_variants["mle_notebook_raw"].ar_fit.residual.loc[rho_index].to_numpy(dtype=float),
            )[0, 1]
        )
        if not np.isfinite(mle_rho):
            mle_rho = 0.0
    intensity_variants = {
        "solar_clearsky": base_intensity,
        "mle_notebook_raw": _replace_rho(base_intensity, mle_rho),
    }

    model_specs = {
        "solar_clearsky": IntensityModelSpecification(
            name="solar_clearsky",
            spot_shift=0.0,
            spot_seasonality=spot_variants["paper"].seasonality,
            solar_seasonality=solar_variants["solar_clearsky"].seasonality,
            spot_ar=spot_variants["paper"].ar_fit,
            solar_ar=solar_variants["solar_clearsky"].ar_fit,
            beta=spike_detection.beta,
            positive_jump_sizes=spike_detection.positive_jump_sizes,
            negative_jump_sizes=spike_detection.negative_jump_sizes,
            intensity=intensity_variants["solar_clearsky"],
            initial_spot_lags=spot_variants["paper"].continuous_series.dropna().iloc[-24:].to_numpy(dtype=float)[::-1],
            initial_solar_lags=solar_variants["solar_clearsky"].seasonality.residual.dropna().iloc[-24:].to_numpy(dtype=float)[::-1],
            initial_spike_state=float(spike_state.iloc[-1]),
        ),
        "mle_notebook_raw": IntensityModelSpecification(
            name="mle_notebook_raw",
            spot_shift=0.0,
            spot_seasonality=spot_variants["mle_notebook_full"].seasonality,
            solar_seasonality=solar_variants["mle_notebook_raw"].seasonality,
            spot_ar=spot_variants["mle_notebook_full"].ar_fit,
            solar_ar=solar_variants["mle_notebook_raw"].ar_fit,
            beta=spike_detection.beta,
            positive_jump_sizes=spike_detection.positive_jump_sizes,
            negative_jump_sizes=spike_detection.negative_jump_sizes,
            intensity=intensity_variants["mle_notebook_raw"],
            initial_spot_lags=spot_variants["mle_notebook_full"].continuous_series.dropna().iloc[-24:].to_numpy(dtype=float)[::-1],
            initial_solar_lags=solar_variants["mle_notebook_raw"].seasonality.residual.dropna().iloc[-24:].to_numpy(dtype=float)[::-1],
            initial_spike_state=float(spike_state.iloc[-1]),
        ),
    }

    return IntensityCalibrationWorkspace(
        data=data,
        spike_detection=spike_detection,
        spot_variants=spot_variants,
        solar_variants=solar_variants,
        intensity_variants=intensity_variants,
        model_specs=model_specs,
    )


def sample_summary_table(data: JointSolarMarketData) -> pd.DataFrame:
    active_share = float((data.solar_cf > 0.01).mean())
    return pd.DataFrame(
        [
            {
                "start": data.start,
                "end": data.end,
                "n_hours": int(len(data.spot)),
                "sample_years": float((data.end - data.start).total_seconds() / (365.25 * 24.0 * 3600.0)),
                "spot_min": float(data.spot.min()),
                "spot_max": float(data.spot.max()),
                "solar_min": float(data.solar_cf.min()),
                "solar_max": float(data.solar_cf.max()),
                "solar_mean": float(data.solar_cf.mean()),
                "solar_active_share_gt_1pct": active_share,
            }
        ]
    )


def spike_summary_table(spikes: SpikeDetectionResult) -> pd.DataFrame:
    rows = []
    for year, threshold in spikes.threshold_by_year.items():
        rows.append(
            {
                "year": int(year),
                "sigma_bar": float(spikes.sigma_bar_by_year[year]),
                "threshold": float(threshold),
                "n_extremes": int(spikes.extreme_mask.loc[spikes.extreme_mask.index.year == year].sum()),
                "n_jumps": int(spikes.jump_mask.loc[spikes.jump_mask.index.year == year].sum()),
            }
        )
    return pd.DataFrame(rows)


def seasonality_comparison_table(
    *,
    reference_fit: SeasonalityFit,
    alternative_fit: SeasonalityFit,
) -> pd.DataFrame:
    aligned_index = reference_fit.series.index.intersection(alternative_fit.series.index)
    series = reference_fit.series.loc[aligned_index]
    ref_fit = reference_fit.fitted.loc[aligned_index]
    alt_fit = alternative_fit.fitted.loc[aligned_index]
    diff = ref_fit - alt_fit
    common_mask = reference_fit.fit_mask.loc[aligned_index].astype(bool) & alternative_fit.fit_mask.loc[aligned_index].astype(bool)
    ref_rmse_common = (
        float(np.sqrt(np.mean(np.square((series.loc[common_mask] - ref_fit.loc[common_mask]).to_numpy(dtype=float)))))
        if common_mask.any()
        else float("nan")
    )
    alt_rmse_common = (
        float(np.sqrt(np.mean(np.square((series.loc[common_mask] - alt_fit.loc[common_mask]).to_numpy(dtype=float)))))
        if common_mask.any()
        else float("nan")
    )
    ref_rmse_full = float(np.sqrt(np.mean(np.square((series - ref_fit).to_numpy(dtype=float)))))
    alt_rmse_full = float(np.sqrt(np.mean(np.square((series - alt_fit).to_numpy(dtype=float)))))
    return pd.DataFrame(
        [
            {
                "reference_model": reference_fit.name,
                "alternative_model": alternative_fit.name,
                "reference_rmse_common_support": ref_rmse_common,
                "alternative_rmse_common_support": alt_rmse_common,
                "reference_rmse_full_sample": ref_rmse_full,
                "alternative_rmse_full_sample": alt_rmse_full,
                "reference_physical_rmse": float(reference_fit.physical_rmse),
                "alternative_physical_rmse": float(alternative_fit.physical_rmse),
                "fit_max_abs_difference": float(np.max(np.abs(diff.to_numpy(dtype=float)))),
                "fit_rmse_difference": float(np.sqrt(np.mean(np.square(diff.to_numpy(dtype=float))))),
            }
        ]
    )


def parameter_tables(workspace: IntensityCalibrationWorkspace) -> Dict[str, pd.DataFrame]:
    tables = {
        "spot_paper_seasonality": paper_parameter_table(workspace.spot_variants["paper"].seasonality),
        "spot_mle_filtered_seasonality": notebook_parameter_table(workspace.spot_variants["mle_notebook_filtered"].seasonality),
        "spot_mle_full_seasonality": notebook_parameter_table(workspace.spot_variants["mle_notebook_full"].seasonality),
        "solar_clearsky_seasonality": notebook_parameter_table(workspace.solar_variants["solar_clearsky"].seasonality),
        "solar_clearsky_transform": solar_transform_parameter_table(workspace.solar_variants["solar_clearsky"].seasonality),
        "solar_mle_raw_seasonality": notebook_parameter_table(workspace.solar_variants["mle_notebook_raw"].seasonality),
        "spot_paper_ar24": ar_parameter_table(workspace.spot_variants["paper"].ar_fit, prefix="1"),
        "solar_clearsky_ar24": ar_parameter_table(workspace.solar_variants["solar_clearsky"].ar_fit, prefix="2"),
        "spot_mle_full_ar24": ar_parameter_table(workspace.spot_variants["mle_notebook_full"].ar_fit, prefix="1m"),
        "solar_mle_raw_ar24": ar_parameter_table(workspace.solar_variants["mle_notebook_raw"].ar_fit, prefix="2m"),
    }
    return tables


__all__ = [
    "IntensityCalibrationWorkspace",
    "SolarCalibrationVariant",
    "SpotCalibrationVariant",
    "parameter_tables",
    "run_intensity_model_calibration_solar",
    "sample_summary_table",
    "seasonality_comparison_table",
    "spike_summary_table",
]
