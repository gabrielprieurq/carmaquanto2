from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from .intensity_ar import AR24Fit, ar_parameter_table, fit_ar24_exact_mle
    from .intensity_data import DEFAULT_SPOT_PATH, DEFAULT_WIND_PATH, JointMarketData, load_joint_market_data
    from .intensity_intensity import (
        IntensityCalibrationResult,
        calibrate_intensity_functions,
    )
    from .intensity_model import IntensityModelSpecification
    from .intensity_seasonality import (
        SeasonalityFit,
        fit_mle_notebook_seasonality,
        paper_parameter_table,
        fit_paper_phase_seasonality,
        notebook_parameter_table,
    )
    from .intensity_spikes import SpikeDetectionResult, detect_spikes_paper, reconstruct_spike_path
except ImportError:
    from intensity_ar import AR24Fit, ar_parameter_table, fit_ar24_exact_mle
    from intensity_data import DEFAULT_SPOT_PATH, DEFAULT_WIND_PATH, JointMarketData, load_joint_market_data
    from intensity_intensity import (
        IntensityCalibrationResult,
        calibrate_intensity_functions,
    )
    from intensity_model import IntensityModelSpecification
    from intensity_seasonality import (
        SeasonalityFit,
        fit_mle_notebook_seasonality,
        paper_parameter_table,
        fit_paper_phase_seasonality,
        notebook_parameter_table,
    )
    from intensity_spikes import SpikeDetectionResult, detect_spikes_paper, reconstruct_spike_path


@dataclass
class SpotCalibrationVariant:
    name: str
    seasonality: SeasonalityFit
    spike_state: pd.Series
    continuous_series: pd.Series
    ar_fit: AR24Fit
    filtered_series_for_ar: pd.Series


@dataclass
class WindCalibrationVariant:
    name: str
    seasonality: SeasonalityFit
    ar_fit: AR24Fit


@dataclass
class IntensityCalibrationWorkspace:
    data: JointMarketData
    spike_detection: SpikeDetectionResult
    spot_variants: Dict[str, SpotCalibrationVariant]
    wind_variants: Dict[str, WindCalibrationVariant]
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


def _build_wind_variant(
    *,
    name: str,
    wind_logit: pd.Series,
    mode: str,
) -> WindCalibrationVariant:
    if mode == "paper_phase":
        seasonality = fit_paper_phase_seasonality(wind_logit, name=name)
    elif mode == "mle_notebook":
        seasonality = fit_mle_notebook_seasonality(wind_logit, name=name, fourier_order=2)
    else:
        raise ValueError(f"Unsupported wind seasonality mode: {mode}")
    ar_fit = fit_ar24_exact_mle(seasonality.residual.dropna())
    return WindCalibrationVariant(name=name, seasonality=seasonality, ar_fit=ar_fit)


def _replace_rho(base: IntensityCalibrationResult, rho: float) -> IntensityCalibrationResult:
    return IntensityCalibrationResult(
        negative_kernel=base.negative_kernel,
        positive_kernel=base.positive_kernel,
        negative_two_state=base.negative_two_state,
        lambda_pos=base.lambda_pos,
        rho=float(rho),
        event_table=base.event_table.copy(),
    )


def run_intensity_model_calibration(
    *,
    spot_path: str = str(DEFAULT_SPOT_PATH),
    wind_path: str = str(DEFAULT_WIND_PATH),
    start: Optional[str | pd.Timestamp] = None,
    end: Optional[str | pd.Timestamp] = None,
    calibration_start: Optional[str | pd.Timestamp] = None,
    calibration_end: Optional[str | pd.Timestamp] = None,
) -> IntensityCalibrationWorkspace:
    if start is None and calibration_start is not None:
        start = calibration_start
    if end is None and calibration_end is not None:
        end = calibration_end
    data = load_joint_market_data(spot_path=spot_path, wind_path=wind_path, start=start, end=end)
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
    # Backward-compatible aliases for older notebook cells.
    spot_variants["harmonic_filtered"] = spot_variants["mle_notebook_filtered"]
    spot_variants["harmonic_full"] = spot_variants["mle_notebook_full"]

    wind_variants = {
        "paper": _build_wind_variant(name="wind_paper", wind_logit=data.wind_logit, mode="paper_phase"),
        "mle_notebook": _build_wind_variant(name="wind_mle", wind_logit=data.wind_logit, mode="mle_notebook"),
    }
    # Backward-compatible alias for older notebook cells.
    wind_variants["harmonic"] = wind_variants["mle_notebook"]

    base_intensity = calibrate_intensity_functions(
        wind_cf=data.wind_cf,
        jump_mask=spike_detection.jump_mask,
        jump_increment=spike_detection.jump_increment,
        spot_residual=spot_variants["paper"].ar_fit.residual,
        wind_residual=wind_variants["paper"].ar_fit.residual,
    )
    intensity_paper = base_intensity
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
    intensity_variants = {"paper": intensity_paper, "mle_notebook": intensity_mle}
    # Backward-compatible alias for older notebook cells.
    intensity_variants["harmonic_full"] = intensity_mle

    model_specs = {
        "paper": IntensityModelSpecification(
            name="paper",
            spot_shift=0.0,
            spot_seasonality=spot_variants["paper"].seasonality,
            wind_seasonality=wind_variants["paper"].seasonality,
            spot_ar=spot_variants["paper"].ar_fit,
            wind_ar=wind_variants["paper"].ar_fit,
            beta=spike_detection.beta,
            positive_jump_sizes=spike_detection.positive_jump_sizes,
            negative_jump_sizes=spike_detection.negative_jump_sizes,
            intensity=intensity_paper,
            initial_spot_lags=spot_variants["paper"].continuous_series.dropna().iloc[-24:].to_numpy(dtype=float)[::-1],
            initial_wind_lags=wind_variants["paper"].seasonality.residual.dropna().iloc[-24:].to_numpy(dtype=float)[::-1],
            initial_spike_state=float(spike_state.iloc[-1]),
        ),
        "mle_notebook": IntensityModelSpecification(
            name="mle_notebook",
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
    # Backward-compatible alias for older notebook cells.
    model_specs["harmonic_full"] = model_specs["mle_notebook"]

    return IntensityCalibrationWorkspace(
        data=data,
        spike_detection=spike_detection,
        spot_variants=spot_variants,
        wind_variants=wind_variants,
        intensity_variants=intensity_variants,
        model_specs=model_specs,
    )


def sample_summary_table(data: JointMarketData) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "start": data.start,
                "end": data.end,
                "n_hours": int(len(data.spot)),
                "sample_years": float((data.end - data.start).total_seconds() / (365.25 * 24.0 * 3600.0)),
                "spot_min": float(data.spot.min()),
                "spot_max": float(data.spot.max()),
                "wind_min": float(data.wind_cf.min()),
                "wind_max": float(data.wind_cf.max()),
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
    reference_fit: Optional[SeasonalityFit] = None,
    alternative_fit: Optional[SeasonalityFit] = None,
    paper_fit: Optional[SeasonalityFit] = None,
    harmonic_fit: Optional[SeasonalityFit] = None,
) -> pd.DataFrame:
    if reference_fit is None and paper_fit is not None:
        reference_fit = paper_fit
    if alternative_fit is None and harmonic_fit is not None:
        alternative_fit = harmonic_fit
    if reference_fit is None or alternative_fit is None:
        raise TypeError(
            "seasonality_comparison_table() requires either "
            "`reference_fit` and `alternative_fit`, or the legacy "
            "`paper_fit` and `harmonic_fit` arguments."
        )
    aligned_index = reference_fit.series.index.intersection(alternative_fit.series.index)
    series = reference_fit.series.loc[aligned_index]
    ref_fit = reference_fit.fitted.loc[aligned_index]
    alt_fit = alternative_fit.fitted.loc[aligned_index]
    diff = ref_fit - alt_fit
    common_mask = reference_fit.fit_mask.loc[aligned_index].astype(bool) & alternative_fit.fit_mask.loc[aligned_index].astype(bool)
    ref_rmse_common = float(np.sqrt(np.mean(np.square((series.loc[common_mask] - ref_fit.loc[common_mask]).to_numpy(dtype=float))))) if common_mask.any() else float("nan")
    alt_rmse_common = float(np.sqrt(np.mean(np.square((series.loc[common_mask] - alt_fit.loc[common_mask]).to_numpy(dtype=float))))) if common_mask.any() else float("nan")
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
        "wind_paper_seasonality": paper_parameter_table(workspace.wind_variants["paper"].seasonality),
        "wind_mle_seasonality": notebook_parameter_table(workspace.wind_variants["mle_notebook"].seasonality),
        "spot_paper_ar24": ar_parameter_table(workspace.spot_variants["paper"].ar_fit, prefix="1"),
        "wind_paper_ar24": ar_parameter_table(workspace.wind_variants["paper"].ar_fit, prefix="2"),
        "spot_mle_full_ar24": ar_parameter_table(workspace.spot_variants["mle_notebook_full"].ar_fit, prefix="1m"),
        "wind_mle_ar24": ar_parameter_table(workspace.wind_variants["mle_notebook"].ar_fit, prefix="2m"),
    }
    # Backward-compatible aliases for older notebook cells.
    tables["spot_harmonic_filtered_seasonality"] = tables["spot_mle_filtered_seasonality"]
    tables["spot_harmonic_full_seasonality"] = tables["spot_mle_full_seasonality"]
    tables["wind_harmonic_seasonality"] = tables["wind_mle_seasonality"]
    tables["spot_harmonic_full_ar24"] = tables["spot_mle_full_ar24"]
    tables["wind_harmonic_ar24"] = tables["wind_mle_ar24"]
    return tables
