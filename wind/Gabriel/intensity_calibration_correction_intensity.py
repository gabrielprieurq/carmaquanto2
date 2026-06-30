from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from .intensity_calibration import (
        IntensityCalibrationWorkspace,
        SpotCalibrationVariant,
        WindCalibrationVariant,
        _build_spot_variant,
        _build_wind_variant,
        _replace_rho,
        parameter_tables,
        sample_summary_table,
        seasonality_comparison_table,
    )
    from .intensity_data import DEFAULT_SPOT_PATH, DEFAULT_WIND_PATH, JointMarketData, load_joint_market_data
    from .intensity_intensity import IntensityCalibrationResult, calibrate_intensity_functions
    from .intensity_model import IntensityModelSpecification
    from .intensity_spikes_correction_intensity import (
        DEFAULT_SPIKE_CORRECTION_CONFIG,
        SpikeDetectionCorrectionConfig,
        SpikeDetectionCorrectionResult,
        detect_spikes_correction_intensity,
        reconstruct_spike_path_correction_intensity,
        spike_summary_table_correction_intensity,
    )
except ImportError:
    from intensity_calibration import (
        IntensityCalibrationWorkspace,
        SpotCalibrationVariant,
        WindCalibrationVariant,
        _build_spot_variant,
        _build_wind_variant,
        _replace_rho,
        parameter_tables,
        sample_summary_table,
        seasonality_comparison_table,
    )
    from intensity_data import DEFAULT_SPOT_PATH, DEFAULT_WIND_PATH, JointMarketData, load_joint_market_data
    from intensity_intensity import IntensityCalibrationResult, calibrate_intensity_functions
    from intensity_model import IntensityModelSpecification
    from intensity_spikes_correction_intensity import (
        DEFAULT_SPIKE_CORRECTION_CONFIG,
        SpikeDetectionCorrectionConfig,
        SpikeDetectionCorrectionResult,
        detect_spikes_correction_intensity,
        reconstruct_spike_path_correction_intensity,
        spike_summary_table_correction_intensity,
    )


@dataclass
class SpikeDetectionComparisonCorrectionIntensity:
    baseline_jump_summary: pd.DataFrame
    yearly_jump_summary: pd.DataFrame


def run_intensity_model_calibration_correction_intensity(
    *,
    spot_path: str = str(DEFAULT_SPOT_PATH),
    wind_path: str = str(DEFAULT_WIND_PATH),
    start: Optional[str | pd.Timestamp] = None,
    end: Optional[str | pd.Timestamp] = None,
    calibration_start: Optional[str | pd.Timestamp] = None,
    calibration_end: Optional[str | pd.Timestamp] = None,
    spike_correction_config: SpikeDetectionCorrectionConfig = DEFAULT_SPIKE_CORRECTION_CONFIG,
) -> IntensityCalibrationWorkspace:
    if start is None and calibration_start is not None:
        start = calibration_start
    if end is None and calibration_end is not None:
        end = calibration_end

    data = load_joint_market_data(spot_path=spot_path, wind_path=wind_path, start=start, end=end)
    spike_detection = detect_spikes_correction_intensity(data.spot, config=spike_correction_config)
    spike_state = reconstruct_spike_path_correction_intensity(spike_detection.jump_increment, beta=spike_detection.beta)

    spot_variants = {
        "paper": _build_spot_variant(
            name="spot_paper_correction_intensity",
            spot=data.spot,
            spike_detection=spike_detection,
            spike_state=spike_state,
            fit_mask=~spike_detection.filtered_mask,
            mode="paper_phase",
        ),
        "mle_notebook_filtered": _build_spot_variant(
            name="spot_mle_filtered_correction_intensity",
            spot=data.spot,
            spike_detection=spike_detection,
            spike_state=spike_state,
            fit_mask=~spike_detection.filtered_mask,
            mode="mle_notebook",
        ),
        "mle_notebook_full": _build_spot_variant(
            name="spot_mle_full_correction_intensity",
            spot=data.spot,
            spike_detection=spike_detection,
            spike_state=spike_state,
            fit_mask=pd.Series(True, index=data.spot.index),
            mode="mle_notebook",
        ),
    }
    spot_variants["harmonic_filtered"] = spot_variants["mle_notebook_filtered"]
    spot_variants["harmonic_full"] = spot_variants["mle_notebook_full"]

    wind_variants = {
        "paper": _build_wind_variant(name="wind_paper_correction_intensity", wind_logit=data.wind_logit, mode="paper_phase"),
        "mle_notebook": _build_wind_variant(name="wind_mle_correction_intensity", wind_logit=data.wind_logit, mode="mle_notebook"),
    }
    wind_variants["harmonic"] = wind_variants["mle_notebook"]

    base_intensity = calibrate_intensity_functions(
        wind_cf=data.wind_cf,
        jump_mask=spike_detection.jump_mask,
        jump_increment=spike_detection.jump_increment,
        spot_residual=spot_variants["paper"].ar_fit.residual,
        wind_residual=wind_variants["paper"].ar_fit.residual,
    )
    rho_index = pd.DatetimeIndex(spot_variants["mle_notebook_full"].ar_fit.residual.index).intersection(
        pd.DatetimeIndex(wind_variants["mle_notebook"].ar_fit.residual.index)
    )
    mle_rho = float(
        np.corrcoef(
            spot_variants["mle_notebook_full"].ar_fit.residual.loc[rho_index].to_numpy(dtype=float),
            wind_variants["mle_notebook"].ar_fit.residual.loc[rho_index].to_numpy(dtype=float),
        )[0, 1]
    )
    if not np.isfinite(mle_rho):
        mle_rho = 0.0
    intensity_mle = _replace_rho(base_intensity, mle_rho)
    intensity_variants = {"paper": base_intensity, "mle_notebook": intensity_mle}
    intensity_variants["harmonic_full"] = intensity_mle

    model_specs = {
        "paper": IntensityModelSpecification(
            name="paper_correction_intensity",
            spot_shift=0.0,
            spot_seasonality=spot_variants["paper"].seasonality,
            wind_seasonality=wind_variants["paper"].seasonality,
            spot_ar=spot_variants["paper"].ar_fit,
            wind_ar=wind_variants["paper"].ar_fit,
            beta=spike_detection.beta,
            positive_jump_sizes=spike_detection.positive_jump_sizes,
            negative_jump_sizes=spike_detection.negative_jump_sizes,
            intensity=base_intensity,
            initial_spot_lags=spot_variants["paper"].continuous_series.dropna().iloc[-24:].to_numpy(dtype=float)[::-1],
            initial_wind_lags=wind_variants["paper"].seasonality.residual.dropna().iloc[-24:].to_numpy(dtype=float)[::-1],
            initial_spike_state=float(spike_state.iloc[-1]),
        ),
        "mle_notebook": IntensityModelSpecification(
            name="mle_notebook_correction_intensity",
            spot_shift=0.0,
            spot_seasonality=spot_variants["mle_notebook_full"].seasonality,
            wind_seasonality=wind_variants["mle_notebook"].seasonality,
            spot_ar=spot_variants["mle_notebook_full"].ar_fit,
            wind_ar=wind_variants["mle_notebook"].ar_fit,
            beta=spike_detection.beta,
            positive_jump_sizes=spike_detection.positive_jump_sizes,
            negative_jump_sizes=spike_detection.negative_jump_sizes,
            intensity=intensity_mle,
            initial_spot_lags=spot_variants["mle_notebook_full"].continuous_series.dropna().iloc[-24:].to_numpy(dtype=float)[::-1],
            initial_wind_lags=wind_variants["mle_notebook"].seasonality.residual.dropna().iloc[-24:].to_numpy(dtype=float)[::-1],
            initial_spike_state=float(spike_state.iloc[-1]),
        ),
    }
    model_specs["harmonic_full"] = model_specs["mle_notebook"]

    return IntensityCalibrationWorkspace(
        data=data,
        spike_detection=spike_detection,
        spot_variants=spot_variants,
        wind_variants=wind_variants,
        intensity_variants=intensity_variants,
        model_specs=model_specs,
    )


def jump_detection_comparison_table_correction_intensity(
    baseline_spikes,
    corrected_spikes: SpikeDetectionCorrectionResult,
    *,
    spot: Optional[pd.Series] = None,
) -> pd.DataFrame:
    baseline_returns = pd.Series(baseline_spikes.returns, copy=False)
    corrected_returns = pd.Series(corrected_spikes.returns, copy=False)
    spot_series = pd.Series(spot, copy=False).reindex(corrected_returns.index) if spot is not None else None
    rows = [
        {
            "metric": "negative_price_hours",
            "baseline": int((spot_series < 0.0).sum()) if spot_series is not None else np.nan,
            "corrected": int((spot_series < 0.0).sum()) if spot_series is not None else np.nan,
        },
        {
            "metric": "negative_extreme_returns",
            "baseline": int((baseline_spikes.extreme_mask.loc[baseline_returns.index] & (baseline_returns < 0.0)).sum()),
            "corrected": int((corrected_spikes.extreme_mask.loc[corrected_returns.index] & (corrected_returns < 0.0)).sum()),
        },
        {
            "metric": "positive_extreme_returns",
            "baseline": int((baseline_spikes.extreme_mask.loc[baseline_returns.index] & (baseline_returns > 0.0)).sum()),
            "corrected": int((corrected_spikes.extreme_mask.loc[corrected_returns.index] & (corrected_returns > 0.0)).sum()),
        },
        {
            "metric": "negative_detected_jumps",
            "baseline": int((baseline_spikes.jump_increment < 0.0).sum()),
            "corrected": int((corrected_spikes.jump_increment < 0.0).sum()),
        },
        {
            "metric": "positive_detected_jumps",
            "baseline": int((baseline_spikes.jump_increment > 0.0).sum()),
            "corrected": int((corrected_spikes.jump_increment > 0.0).sum()),
        },
        {
            "metric": "beta_sample_count",
            "baseline": int(len(getattr(baseline_spikes, "beta_samples", []))),
            "corrected": int(len(corrected_spikes.beta_samples)),
        },
        {
            "metric": "beta",
            "baseline": float(baseline_spikes.beta),
            "corrected": float(corrected_spikes.beta),
        },
    ]
    out = pd.DataFrame(rows)
    out["delta_corrected_minus_baseline"] = out["corrected"] - out["baseline"]
    return out


def jump_detection_yearly_comparison_table_correction_intensity(
    baseline_spikes,
    corrected_spikes: SpikeDetectionCorrectionResult,
    *,
    spot: Optional[pd.Series] = None,
) -> pd.DataFrame:
    years = sorted(set(baseline_spikes.threshold_by_year).union(corrected_spikes.threshold_by_year))
    baseline_returns = pd.Series(baseline_spikes.returns, copy=False)
    corrected_returns = pd.Series(corrected_spikes.returns, copy=False)
    spot_series = pd.Series(spot, copy=False) if spot is not None else None
    rows = []
    for year in years:
        base_year = baseline_returns.index.year == year
        corr_year = corrected_returns.index.year == year
        rows.append(
            {
                "year": int(year),
                "negative_price_hours": int((spot_series.loc[spot_series.index.year == year] < 0.0).sum()) if spot_series is not None else np.nan,
                "baseline_negative_extremes": int((baseline_spikes.extreme_mask.loc[baseline_returns.index[base_year]] & (baseline_returns.loc[base_year] < 0.0)).sum()),
                "corrected_negative_extremes": int((corrected_spikes.extreme_mask.loc[corrected_returns.index[corr_year]] & (corrected_returns.loc[corr_year] < 0.0)).sum()),
                "baseline_negative_jumps": int((baseline_spikes.jump_increment.loc[baseline_returns.index[base_year]] < 0.0).sum()),
                "corrected_negative_jumps": int((corrected_spikes.jump_increment.loc[corrected_returns.index[corr_year]] < 0.0).sum()),
                "baseline_positive_jumps": int((baseline_spikes.jump_increment.loc[baseline_returns.index[base_year]] > 0.0).sum()),
                "corrected_positive_jumps": int((corrected_spikes.jump_increment.loc[corrected_returns.index[corr_year]] > 0.0).sum()),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_SPIKE_CORRECTION_CONFIG",
    "SpikeDetectionComparisonCorrectionIntensity",
    "SpikeDetectionCorrectionConfig",
    "SpikeDetectionCorrectionResult",
    "detect_spikes_correction_intensity",
    "jump_detection_comparison_table_correction_intensity",
    "jump_detection_yearly_comparison_table_correction_intensity",
    "parameter_tables",
    "run_intensity_model_calibration_correction_intensity",
    "sample_summary_table",
    "seasonality_comparison_table",
    "spike_summary_table_correction_intensity",
]
