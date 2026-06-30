from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from .intensity_calibration_correction_intensity import (
        DEFAULT_SPIKE_CORRECTION_CONFIG,
        SpikeDetectionCorrectionConfig,
        jump_detection_comparison_table_correction_intensity,
        jump_detection_yearly_comparison_table_correction_intensity,
        run_intensity_model_calibration_correction_intensity,
    )
    from .intensity_calibration_extended_model import (
        CovariateCalibrationVariantExtendedModel,
        ExtendedIntensityCalibrationWorkspace,
        _build_level_covariate_variant,
        _build_residual_proxy_variant,
        _build_solar_clear_sky_variant,
        _raw_intensity_default_labels,
        _temperature_covariate_label,
        covariate_parameter_tables_extended_model,
        covariate_sample_summary_table,
        extended_sample_summary_table,
        run_intensity_model_calibration_extended_model,
    )
    from .intensity_data import DEFAULT_SPOT_PATH, DEFAULT_WIND_PATH
    from .intensity_data_extended_model import (
        DEFAULT_LOAD_PATH,
        DEFAULT_REAL_LOAD_PATH,
        DEFAULT_SOLAR_PATH,
        DEFAULT_TEMPERATURE_PATH,
        ExtendedJointMarketData,
        load_extended_market_data,
    )
    from .intensity_intensity_extended_model import (
        CovariateIntensityCalibrationResult,
        RawIntensityEstimatorConfig,
        calibrate_intensity_for_covariate,
        calibrate_intensity_for_renewable_covariate,
        intensity_comparison_table,
    )
    from .intensity_model_extended_model import CovariateDrivenIntensityModelSpecification
    from .intensity_temperature_extended_model import DEFAULT_OPENMETEO_TEMPERATURE_PATH
    from .intensity_entsoe_extended_model import load_entsoe_de_lu_bundle_from_csv
except ImportError:
    from intensity_calibration_correction_intensity import (
        DEFAULT_SPIKE_CORRECTION_CONFIG,
        SpikeDetectionCorrectionConfig,
        jump_detection_comparison_table_correction_intensity,
        jump_detection_yearly_comparison_table_correction_intensity,
        run_intensity_model_calibration_correction_intensity,
    )
    from intensity_calibration_extended_model import (
        CovariateCalibrationVariantExtendedModel,
        ExtendedIntensityCalibrationWorkspace,
        _build_level_covariate_variant,
        _build_residual_proxy_variant,
        _build_solar_clear_sky_variant,
        _raw_intensity_default_labels,
        _temperature_covariate_label,
        covariate_parameter_tables_extended_model,
        covariate_sample_summary_table,
        extended_sample_summary_table,
        run_intensity_model_calibration_extended_model,
    )
    from intensity_data import DEFAULT_SPOT_PATH, DEFAULT_WIND_PATH
    from intensity_data_extended_model import (
        DEFAULT_LOAD_PATH,
        DEFAULT_REAL_LOAD_PATH,
        DEFAULT_SOLAR_PATH,
        DEFAULT_TEMPERATURE_PATH,
        ExtendedJointMarketData,
        load_extended_market_data,
    )
    from intensity_intensity_extended_model import (
        CovariateIntensityCalibrationResult,
        RawIntensityEstimatorConfig,
        calibrate_intensity_for_covariate,
        calibrate_intensity_for_renewable_covariate,
        intensity_comparison_table,
    )
    from intensity_model_extended_model import CovariateDrivenIntensityModelSpecification
    from intensity_temperature_extended_model import DEFAULT_OPENMETEO_TEMPERATURE_PATH
    from intensity_entsoe_extended_model import load_entsoe_de_lu_bundle_from_csv


@dataclass
class ExtendedIntensityCorrectionEvaluation:
    baseline: ExtendedIntensityCalibrationWorkspace
    corrected: ExtendedIntensityCalibrationWorkspace
    jump_detection_summary: pd.DataFrame
    yearly_jump_detection_summary: pd.DataFrame
    covariate_intensity_summary: pd.DataFrame
    resolved_start: Optional[pd.Timestamp] = None
    resolved_end: Optional[pd.Timestamp] = None


@dataclass
class ResolvedCalibrationIntervalCorrectionIntensity:
    requested_start: Optional[pd.Timestamp]
    requested_end: Optional[pd.Timestamp]
    resolved_start: Optional[pd.Timestamp]
    resolved_end: Optional[pd.Timestamp]
    used_cache_bounds: bool
    cache_start: Optional[pd.Timestamp] = None
    cache_end: Optional[pd.Timestamp] = None


def _normalize_optional_timestamp(ts: Optional[str | pd.Timestamp]) -> Optional[pd.Timestamp]:
    if ts is None:
        return None
    out = pd.Timestamp(ts)
    if out.tzinfo is not None:
        out = out.tz_convert("UTC").tz_localize(None)
    return out


def resolve_safe_real_load_interval_correction_intensity(
    *,
    use_real_entsoe_load: bool,
    auto_download_real_load: bool,
    real_load_path: str = str(DEFAULT_REAL_LOAD_PATH),
    start: Optional[str | pd.Timestamp] = None,
    end: Optional[str | pd.Timestamp] = None,
) -> ResolvedCalibrationIntervalCorrectionIntensity:
    requested_start = _normalize_optional_timestamp(start)
    requested_end = _normalize_optional_timestamp(end)
    if not use_real_entsoe_load or auto_download_real_load:
        return ResolvedCalibrationIntervalCorrectionIntensity(
            requested_start=requested_start,
            requested_end=requested_end,
            resolved_start=requested_start,
            resolved_end=requested_end,
            used_cache_bounds=False,
        )

    cache_path = Path(real_load_path)
    if not cache_path.exists():
        return ResolvedCalibrationIntervalCorrectionIntensity(
            requested_start=requested_start,
            requested_end=requested_end,
            resolved_start=requested_start,
            resolved_end=requested_end,
            used_cache_bounds=False,
        )

    bundle = load_entsoe_de_lu_bundle_from_csv(cache_path)
    cache_start = pd.Timestamp(bundle.actual_load.index.min())
    cache_end = pd.Timestamp(bundle.actual_load.index.max())
    resolved_start = cache_start if requested_start is None else max(requested_start, cache_start)
    resolved_end = cache_end if requested_end is None else min(requested_end, cache_end)
    if resolved_start > resolved_end:
        raise ValueError(
            "The requested calibration interval does not intersect the cached ENTSO-E interval "
            f"[{cache_start}, {cache_end}]."
        )
    return ResolvedCalibrationIntervalCorrectionIntensity(
        requested_start=requested_start,
        requested_end=requested_end,
        resolved_start=resolved_start,
        resolved_end=resolved_end,
        used_cache_bounds=True,
        cache_start=cache_start,
        cache_end=cache_end,
    )


def run_intensity_model_calibration_extended_model_correction_intensity(
    *,
    spot_path: str = str(DEFAULT_SPOT_PATH),
    wind_path: str = str(DEFAULT_WIND_PATH),
    load_path: str = str(DEFAULT_LOAD_PATH),
    real_load_path: str = str(DEFAULT_REAL_LOAD_PATH),
    temperature_path: str = str(DEFAULT_OPENMETEO_TEMPERATURE_PATH),
    solar_path: str = str(DEFAULT_SOLAR_PATH),
    use_real_entsoe_load: bool = False,
    auto_download_real_load: bool = False,
    entsoe_api_key: Optional[str] = None,
    use_historical_temperature: bool = True,
    temperature_source: Optional[str] = None,
    auto_download_temperature: bool = False,
    active_covariates: Optional[list[str]] = None,
    raw_intensity_covariates: Optional[list[str]] = None,
    raw_intensity_estimator_configs: Optional[Dict[str, RawIntensityEstimatorConfig]] = None,
    spike_correction_config: SpikeDetectionCorrectionConfig = DEFAULT_SPIKE_CORRECTION_CONFIG,
    start: Optional[str | pd.Timestamp] = None,
    end: Optional[str | pd.Timestamp] = None,
) -> ExtendedIntensityCalibrationWorkspace:
    resolved_interval = resolve_safe_real_load_interval_correction_intensity(
        use_real_entsoe_load=use_real_entsoe_load,
        auto_download_real_load=auto_download_real_load,
        real_load_path=real_load_path,
        start=start,
        end=end,
    )
    data = load_extended_market_data(
        spot_path=spot_path,
        wind_path=wind_path,
        load_path=load_path,
        real_load_path=real_load_path,
        temperature_path=temperature_path,
        solar_path=solar_path,
        use_real_entsoe_load=use_real_entsoe_load,
        auto_download_real_load=auto_download_real_load,
        entsoe_api_key=entsoe_api_key,
        use_historical_temperature=use_historical_temperature,
        temperature_source=temperature_source,
        auto_download_temperature=auto_download_temperature,
        start=resolved_interval.resolved_start,
        end=resolved_interval.resolved_end,
    )
    base_workspace = run_intensity_model_calibration_correction_intensity(
        spot_path=spot_path,
        wind_path=wind_path,
        start=data.start,
        end=data.end,
        spike_correction_config=spike_correction_config,
    )

    covariate_variants: Dict[str, CovariateCalibrationVariantExtendedModel] = {
        "load_forecast_proxy": _build_level_covariate_variant(
            name="load_forecast_proxy",
            label=("DE/LU day-ahead total load forecast (ENTSO-E, log-level)" if data.use_real_entsoe_load else "DE load forecast proxy (log-level)"),
            raw_series=data.de_load,
            transform_type="log",
        ),
        "temperature_history": _build_level_covariate_variant(
            name="temperature_history",
            label=_temperature_covariate_label(data.temperature_source),
            raw_series=data.temperature,
            transform_type="identity",
        ),
        "solar_clear_sky": _build_solar_clear_sky_variant(
            name="solar_clear_sky",
            label="Solar capacity factor (clear-sky latent model)",
            raw_series=data.solar_cf,
        ),
    }
    if data.use_real_entsoe_load and data.residual_load_forecast.notna().any():
        covariate_variants["residual_forecast_proxy"] = _build_level_covariate_variant(
            name="residual_forecast_proxy",
            label="DE/LU residual load forecast (ENTSO-E load forecast minus ENTSO-E wind/solar forecast, level)",
            raw_series=data.residual_load_forecast,
            transform_type="identity",
        )
    else:
        covariate_variants["residual_forecast_proxy"] = _build_residual_proxy_variant(
            name="residual_forecast_proxy",
            label="Residual load forecast proxy (deseasonalized log-load residual)",
            residual_series=covariate_variants["load_forecast_proxy"].mle_seasonality.residual,
        )

    raw_intensity_keys = (
        {"load_forecast_proxy", "temperature_history", "residual_forecast_proxy"}
        if raw_intensity_covariates is None
        else {str(x) for x in raw_intensity_covariates}
    )
    raw_labels = _raw_intensity_default_labels(data)
    raw_configs = dict(raw_intensity_estimator_configs or {})
    for key in raw_intensity_keys:
        raw_configs.setdefault(key, RawIntensityEstimatorConfig())

    intensity_by_covariate: Dict[str, CovariateIntensityCalibrationResult] = {}
    for key, variant in covariate_variants.items():
        if key == "solar_clear_sky":
            intensity_by_covariate[key] = calibrate_intensity_for_renewable_covariate(
                covariate_state=variant.state_series,
                renewable_raw=variant.raw_series,
                jump_mask=base_workspace.spike_detection.jump_mask,
                jump_increment=base_workspace.spike_detection.jump_increment,
                spot_residual=base_workspace.spot_variants["mle_notebook_full"].ar_fit.residual,
                correlation_residual=base_workspace.wind_variants["mle_notebook"].ar_fit.residual,
                covariate_name=key,
                covariate_label=variant.label,
            )
            continue

        use_raw_intensity = key in raw_intensity_keys
        intensity_by_covariate[key] = calibrate_intensity_for_covariate(
            covariate_state=variant.state_series,
            covariate_raw=variant.raw_series,
            jump_mask=base_workspace.spike_detection.jump_mask,
            jump_increment=base_workspace.spike_detection.jump_increment,
            spot_residual=base_workspace.spot_variants["mle_notebook_full"].ar_fit.residual,
            wind_residual=base_workspace.wind_variants["mle_notebook"].ar_fit.residual,
            covariate_name=key,
            covariate_label=variant.label,
            intensity_driver_mode=("raw" if use_raw_intensity else "state"),
            raw_driver_label=raw_labels.get(key, variant.label),
            interval_quantiles=((0.01, 0.99) if use_raw_intensity else (0.05, 0.95)),
            raw_estimator_config=raw_configs.get(key),
        )

    model_specs: Dict[str, CovariateDrivenIntensityModelSpecification] = {}
    base_spec = base_workspace.model_specs["mle_notebook"]
    active_list = list(covariate_variants) if active_covariates is None else [str(x) for x in active_covariates]
    unknown = sorted(set(active_list).difference(covariate_variants))
    if unknown:
        raise KeyError(f"Unknown active_covariates: {unknown}")
    for key in active_list:
        variant = covariate_variants[key]
        if len(variant.ar_fit.series.dropna()) < 24:
            raise ValueError(f"Covariate {key} does not have enough observations to initialize an AR(24) history.")
        model_specs[key] = CovariateDrivenIntensityModelSpecification(
            name=f"{key}_extended_model_correction_intensity",
            base_spec=base_spec,
            covariate_name=key,
            covariate_label=variant.label,
            covariate_transform=variant.transform_type,
            covariate_seasonality=variant.operational_seasonality,
            covariate_ar=variant.ar_fit,
            covariate_state_transform=variant.state_transform,
            initial_covariate_lags=variant.ar_fit.series.dropna().iloc[-24:].to_numpy(dtype=float)[::-1],
            intensity=intensity_by_covariate[key],
        )

    return ExtendedIntensityCalibrationWorkspace(
        data=data,
        base_workspace=base_workspace,
        active_covariates=active_list,
        covariate_variants=covariate_variants,
        intensity_by_covariate=intensity_by_covariate,
        model_specs=model_specs,
    )


def intensity_parameter_table_extended_model_correction_intensity(
    workspace: ExtendedIntensityCalibrationWorkspace,
) -> pd.DataFrame:
    return intensity_comparison_table(workspace.intensity_by_covariate)


def covariate_intensity_comparison_table_correction_intensity(
    baseline: ExtendedIntensityCalibrationWorkspace,
    corrected: ExtendedIntensityCalibrationWorkspace,
) -> pd.DataFrame:
    keys = sorted(set(baseline.intensity_by_covariate).union(corrected.intensity_by_covariate))
    rows = []
    for key in keys:
        base = baseline.intensity_by_covariate[key]
        corr = corrected.intensity_by_covariate[key]
        rows.append(
            {
                "covariate": key,
                "baseline_negative_events": int(base.negative_event_count),
                "corrected_negative_events": int(corr.negative_event_count),
                "baseline_positive_events": int(base.positive_event_count),
                "corrected_positive_events": int(corr.positive_event_count),
                "negative_event_gain": int(corr.negative_event_count - base.negative_event_count),
                "positive_event_gain": int(corr.positive_event_count - base.positive_event_count),
                "baseline_negative_estimator": base.negative_estimator_type,
                "corrected_negative_estimator": corr.negative_estimator_type,
                "baseline_positive_estimator": base.positive_estimator_type,
                "corrected_positive_estimator": corr.positive_estimator_type,
                "baseline_negative_bandwidth": float(base.negative_kernel.bandwidth),
                "corrected_negative_bandwidth": float(corr.negative_kernel.bandwidth),
                "baseline_positive_bandwidth": float(base.positive_kernel.bandwidth),
                "corrected_positive_bandwidth": float(corr.positive_kernel.bandwidth),
                "baseline_lambda_neg_low": float(base.negative_two_state.lambda_neg_low),
                "corrected_lambda_neg_low": float(corr.negative_two_state.lambda_neg_low),
                "baseline_lambda_neg_high": float(base.negative_two_state.lambda_neg_high),
                "corrected_lambda_neg_high": float(corr.negative_two_state.lambda_neg_high),
                "baseline_lambda_pos": float(base.lambda_pos),
                "corrected_lambda_pos": float(corr.lambda_pos),
                "baseline_threshold": float(base.negative_two_state.wp_threshold),
                "corrected_threshold": float(corr.negative_two_state.wp_threshold),
            }
        )
    return pd.DataFrame(rows)


def evaluate_intensity_correction_against_baseline(
    *,
    spot_path: str = str(DEFAULT_SPOT_PATH),
    wind_path: str = str(DEFAULT_WIND_PATH),
    load_path: str = str(DEFAULT_LOAD_PATH),
    real_load_path: str = str(DEFAULT_REAL_LOAD_PATH),
    temperature_path: str = str(DEFAULT_TEMPERATURE_PATH),
    solar_path: str = str(DEFAULT_SOLAR_PATH),
    use_real_entsoe_load: bool = False,
    auto_download_real_load: bool = False,
    entsoe_api_key: Optional[str] = None,
    use_historical_temperature: bool = True,
    temperature_source: Optional[str] = None,
    auto_download_temperature: bool = False,
    active_covariates: Optional[list[str]] = None,
    raw_intensity_covariates: Optional[list[str]] = None,
    raw_intensity_estimator_configs: Optional[Dict[str, RawIntensityEstimatorConfig]] = None,
    spike_correction_config: SpikeDetectionCorrectionConfig = DEFAULT_SPIKE_CORRECTION_CONFIG,
    start: Optional[str | pd.Timestamp] = None,
    end: Optional[str | pd.Timestamp] = None,
) -> ExtendedIntensityCorrectionEvaluation:
    resolved_interval = resolve_safe_real_load_interval_correction_intensity(
        use_real_entsoe_load=use_real_entsoe_load,
        auto_download_real_load=auto_download_real_load,
        real_load_path=real_load_path,
        start=start,
        end=end,
    )
    baseline = run_intensity_model_calibration_extended_model(
        spot_path=spot_path,
        wind_path=wind_path,
        load_path=load_path,
        real_load_path=real_load_path,
        temperature_path=temperature_path,
        solar_path=solar_path,
        use_real_entsoe_load=use_real_entsoe_load,
        auto_download_real_load=auto_download_real_load,
        entsoe_api_key=entsoe_api_key,
        use_historical_temperature=use_historical_temperature,
        temperature_source=temperature_source,
        auto_download_temperature=auto_download_temperature,
        active_covariates=active_covariates,
        raw_intensity_covariates=raw_intensity_covariates,
        raw_intensity_estimator_configs=raw_intensity_estimator_configs,
        start=resolved_interval.resolved_start,
        end=resolved_interval.resolved_end,
    )
    corrected = run_intensity_model_calibration_extended_model_correction_intensity(
        spot_path=spot_path,
        wind_path=wind_path,
        load_path=load_path,
        real_load_path=real_load_path,
        temperature_path=temperature_path,
        solar_path=solar_path,
        use_real_entsoe_load=use_real_entsoe_load,
        auto_download_real_load=auto_download_real_load,
        entsoe_api_key=entsoe_api_key,
        use_historical_temperature=use_historical_temperature,
        temperature_source=temperature_source,
        auto_download_temperature=auto_download_temperature,
        active_covariates=active_covariates,
        raw_intensity_covariates=raw_intensity_covariates,
        raw_intensity_estimator_configs=raw_intensity_estimator_configs,
        spike_correction_config=spike_correction_config,
        start=resolved_interval.resolved_start,
        end=resolved_interval.resolved_end,
    )
    jump_detection_summary = jump_detection_comparison_table_correction_intensity(
        baseline.base_workspace.spike_detection,
        corrected.base_workspace.spike_detection,
        spot=corrected.data.spot,
    )
    yearly_jump_detection_summary = jump_detection_yearly_comparison_table_correction_intensity(
        baseline.base_workspace.spike_detection,
        corrected.base_workspace.spike_detection,
        spot=corrected.data.spot,
    )
    covariate_intensity_summary = covariate_intensity_comparison_table_correction_intensity(baseline, corrected)
    return ExtendedIntensityCorrectionEvaluation(
        baseline=baseline,
        corrected=corrected,
        jump_detection_summary=jump_detection_summary,
        yearly_jump_detection_summary=yearly_jump_detection_summary,
        covariate_intensity_summary=covariate_intensity_summary,
        resolved_start=resolved_interval.resolved_start,
        resolved_end=resolved_interval.resolved_end,
    )


__all__ = [
    "DEFAULT_SPIKE_CORRECTION_CONFIG",
    "ExtendedIntensityCorrectionEvaluation",
    "ResolvedCalibrationIntervalCorrectionIntensity",
    "SpikeDetectionCorrectionConfig",
    "covariate_intensity_comparison_table_correction_intensity",
    "covariate_parameter_tables_extended_model",
    "covariate_sample_summary_table",
    "evaluate_intensity_correction_against_baseline",
    "extended_sample_summary_table",
    "intensity_parameter_table_extended_model_correction_intensity",
    "resolve_safe_real_load_interval_correction_intensity",
    "run_intensity_model_calibration_extended_model_correction_intensity",
]
