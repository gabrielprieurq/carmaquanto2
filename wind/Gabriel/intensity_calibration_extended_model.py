from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from .intensity_ar import AR24Fit, ar_parameter_table, fit_ar24_exact_mle
    from .intensity_calibration import IntensityCalibrationWorkspace, run_intensity_model_calibration
    from .intensity_data import DEFAULT_SPOT_PATH, DEFAULT_WIND_PATH
    from .intensity_data_extended_model import (
        DEFAULT_LOAD_PATH,
        DEFAULT_REAL_LOAD_PATH,
        DEFAULT_SOLAR_PATH,
        DEFAULT_TEMPERATURE_PATH,
        ExtendedJointMarketData,
        extended_sample_summary_table,
        load_extended_market_data,
    )
    from .intensity_temperature_extended_model import DEFAULT_OPENMETEO_TEMPERATURE_PATH
    from .intensity_entsoe_extended_model import load_entsoe_de_lu_bundle_from_csv
    from .intensity_intensity_extended_model import (
        CovariateIntensityCalibrationResult,
        EmpiricalStateTransform,
        RawIntensityEstimatorConfig,
        build_empirical_state_transform,
        calibrate_intensity_for_renewable_covariate,
        calibrate_intensity_for_covariate,
        intensity_comparison_table,
        state_series_from_transform,
    )
    from .intensity_model_extended_model import CovariateDrivenIntensityModelSpecification
    from .intensity_seasonality import (
        SeasonalityFit,
        fit_mle_notebook_seasonality,
        fit_naive_raw_solar_logit_seasonality,
        fit_paper_phase_seasonality,
        fit_solar_logit_seasonality,
        notebook_parameter_table,
        paper_parameter_table,
        solar_transform_parameter_table,
    )
except ImportError:
    from intensity_ar import AR24Fit, ar_parameter_table, fit_ar24_exact_mle
    from intensity_calibration import IntensityCalibrationWorkspace, run_intensity_model_calibration
    from intensity_data import DEFAULT_SPOT_PATH, DEFAULT_WIND_PATH
    from intensity_data_extended_model import (
        DEFAULT_LOAD_PATH,
        DEFAULT_REAL_LOAD_PATH,
        DEFAULT_SOLAR_PATH,
        DEFAULT_TEMPERATURE_PATH,
        ExtendedJointMarketData,
        extended_sample_summary_table,
        load_extended_market_data,
    )
    from intensity_temperature_extended_model import DEFAULT_OPENMETEO_TEMPERATURE_PATH
    from intensity_entsoe_extended_model import load_entsoe_de_lu_bundle_from_csv
    from intensity_intensity_extended_model import (
        CovariateIntensityCalibrationResult,
        EmpiricalStateTransform,
        RawIntensityEstimatorConfig,
        build_empirical_state_transform,
        calibrate_intensity_for_renewable_covariate,
        calibrate_intensity_for_covariate,
        intensity_comparison_table,
        state_series_from_transform,
    )
    from intensity_model_extended_model import CovariateDrivenIntensityModelSpecification
    from intensity_seasonality import (
        SeasonalityFit,
        fit_mle_notebook_seasonality,
        fit_naive_raw_solar_logit_seasonality,
        fit_paper_phase_seasonality,
        fit_solar_logit_seasonality,
        notebook_parameter_table,
        paper_parameter_table,
        solar_transform_parameter_table,
    )


@dataclass
class CovariateCalibrationVariantExtendedModel:
    name: str
    label: str
    raw_series: pd.Series
    transformed_series: pd.Series
    transform_type: str
    paper_seasonality: Optional[SeasonalityFit]
    mle_seasonality: Optional[SeasonalityFit]
    operational_seasonality: Optional[SeasonalityFit]
    ar_fit: AR24Fit
    continuous_series: pd.Series
    driver_series: pd.Series
    state_transform: EmpiricalStateTransform
    state_series: pd.Series


@dataclass
class ExtendedIntensityCalibrationWorkspace:
    data: ExtendedJointMarketData
    base_workspace: IntensityCalibrationWorkspace
    active_covariates: list[str]
    covariate_variants: Dict[str, CovariateCalibrationVariantExtendedModel]
    intensity_by_covariate: Dict[str, CovariateIntensityCalibrationResult]
    model_specs: Dict[str, CovariateDrivenIntensityModelSpecification]


@dataclass
class ResolvedCalibrationIntervalExtendedModel:
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


def resolve_safe_real_load_interval_extended_model(
    *,
    use_real_entsoe_load: bool,
    auto_download_real_load: bool,
    real_load_path: str = str(DEFAULT_REAL_LOAD_PATH),
    start: Optional[str | pd.Timestamp] = None,
    end: Optional[str | pd.Timestamp] = None,
) -> ResolvedCalibrationIntervalExtendedModel:
    requested_start = _normalize_optional_timestamp(start)
    requested_end = _normalize_optional_timestamp(end)
    if not use_real_entsoe_load or auto_download_real_load:
        return ResolvedCalibrationIntervalExtendedModel(
            requested_start=requested_start,
            requested_end=requested_end,
            resolved_start=requested_start,
            resolved_end=requested_end,
            used_cache_bounds=False,
        )

    cache_path = Path(real_load_path)
    if not cache_path.exists():
        return ResolvedCalibrationIntervalExtendedModel(
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
    return ResolvedCalibrationIntervalExtendedModel(
        requested_start=requested_start,
        requested_end=requested_end,
        resolved_start=resolved_start,
        resolved_end=resolved_end,
        used_cache_bounds=True,
        cache_start=cache_start,
        cache_end=cache_end,
    )


def _identity_series(series: pd.Series, *, name: str) -> pd.Series:
    return pd.Series(series, copy=True).astype(float).rename(name)


def _log_series(series: pd.Series, *, name: str) -> pd.Series:
    s = pd.Series(series, copy=True).astype(float)
    return pd.Series(np.log(np.clip(s.to_numpy(dtype=float), 1e-6, None)), index=s.index, name=name)


def _log1p_series(series: pd.Series, *, name: str) -> pd.Series:
    s = pd.Series(series, copy=True).astype(float)
    return pd.Series(np.log1p(np.clip(s.to_numpy(dtype=float), 0.0, None)), index=s.index, name=name)


def _signed_log1p_series(series: pd.Series, *, name: str) -> pd.Series:
    s = pd.Series(series, copy=True).astype(float)
    x = s.to_numpy(dtype=float)
    return pd.Series(np.sign(x) * np.log1p(np.abs(x)), index=s.index, name=name)


def _build_level_covariate_variant(
    *,
    name: str,
    label: str,
    raw_series: pd.Series,
    transform_type: str,
) -> CovariateCalibrationVariantExtendedModel:
    if transform_type == "identity":
        transformed = _identity_series(raw_series, name=f"{name}_transformed")
    elif transform_type == "log":
        transformed = _log_series(raw_series, name=f"{name}_transformed")
    elif transform_type == "log1p":
        transformed = _log1p_series(raw_series, name=f"{name}_transformed")
    elif transform_type == "signed_log1p":
        transformed = _signed_log1p_series(raw_series, name=f"{name}_transformed")
    else:
        raise ValueError(f"Unsupported transform type: {transform_type}")

    paper_fit = fit_paper_phase_seasonality(transformed, name=f"{name}_paper")
    mle_fit = fit_mle_notebook_seasonality(transformed, name=f"{name}_mle", fourier_order=2)
    ar_fit = fit_ar24_exact_mle(mle_fit.residual.dropna())
    state_transform = build_empirical_state_transform(transformed, name=f"{name}_state")
    state_series = state_series_from_transform(transformed, state_transform, name=f"{name}_state")
    return CovariateCalibrationVariantExtendedModel(
        name=name,
        label=label,
        raw_series=pd.Series(raw_series, copy=True).astype(float).rename(name),
        transformed_series=transformed,
        transform_type=transform_type,
        paper_seasonality=paper_fit,
        mle_seasonality=mle_fit,
        operational_seasonality=mle_fit,
        ar_fit=ar_fit,
        continuous_series=pd.Series(mle_fit.residual, copy=True).astype(float).rename(f"{name}_continuous"),
        driver_series=transformed,
        state_transform=state_transform,
        state_series=state_series,
    )


def _build_residual_proxy_variant(
    *,
    name: str,
    label: str,
    residual_series: pd.Series,
) -> CovariateCalibrationVariantExtendedModel:
    residual = pd.Series(residual_series, copy=True).astype(float).rename(name)
    ar_fit = fit_ar24_exact_mle(residual.dropna())
    state_transform = build_empirical_state_transform(residual, name=f"{name}_state")
    state_series = state_series_from_transform(residual, state_transform, name=f"{name}_state")
    return CovariateCalibrationVariantExtendedModel(
        name=name,
        label=label,
        raw_series=residual,
        transformed_series=residual,
        transform_type="identity",
        paper_seasonality=None,
        mle_seasonality=None,
        operational_seasonality=None,
        ar_fit=ar_fit,
        continuous_series=residual,
        driver_series=residual,
        state_transform=state_transform,
        state_series=state_series,
    )


def _build_solar_clear_sky_variant(
    *,
    name: str,
    label: str,
    raw_series: pd.Series,
) -> CovariateCalibrationVariantExtendedModel:
    solar_cf = pd.Series(raw_series, copy=True).astype(float).clip(0.0, 1.0).rename(name)
    solar_fit = fit_solar_logit_seasonality(solar_cf, name=f"{name}_operational")
    raw_logit_fit = fit_naive_raw_solar_logit_seasonality(solar_cf, name=f"{name}_benchmark_raw", fourier_order=2)
    ar_fit = fit_ar24_exact_mle(solar_fit.residual.dropna())
    transformed = pd.Series(solar_fit.series, copy=True).astype(float).rename(f"{name}_latent")
    state_transform = build_empirical_state_transform(transformed, name=f"{name}_state")
    state_series = state_series_from_transform(transformed, state_transform, name=f"{name}_state")
    return CovariateCalibrationVariantExtendedModel(
        name=name,
        label=label,
        raw_series=solar_cf,
        transformed_series=transformed,
        transform_type="solar_clear_sky_logit",
        paper_seasonality=None,
        mle_seasonality=raw_logit_fit,
        operational_seasonality=solar_fit,
        ar_fit=ar_fit,
        continuous_series=pd.Series(solar_fit.residual, copy=True).astype(float).rename(f"{name}_continuous"),
        driver_series=solar_cf,
        state_transform=state_transform,
        state_series=state_series,
    )


def _temperature_covariate_label(source: Optional[str]) -> str:
    if str(source or "").lower() == "enwex":
        return "ENWEX German temperature (level)"
    return "Historical Open-Meteo temperature (level)"


def _raw_intensity_default_labels(data: ExtendedJointMarketData) -> Dict[str, str]:
    return {
        "load_forecast_proxy": (
            "DE/LU day-ahead total load forecast (ENTSO-E, level)"
            if data.use_real_entsoe_load
            else "DE load forecast proxy (level)"
        ),
        "residual_forecast_proxy": (
            "DE/LU residual load forecast (ENTSO-E load forecast minus ENTSO-E wind/solar forecast, level)"
            if data.use_real_entsoe_load and data.residual_load_forecast.notna().any()
            else "Residual load forecast proxy (deseasonalized load residual)"
        ),
        "temperature_history": _temperature_covariate_label(data.temperature_source),
    }


def run_intensity_model_calibration_extended_model(
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
    start: Optional[str | pd.Timestamp] = None,
    end: Optional[str | pd.Timestamp] = None,
) -> ExtendedIntensityCalibrationWorkspace:
    resolved_interval = resolve_safe_real_load_interval_extended_model(
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
    base_workspace = run_intensity_model_calibration(
        spot_path=spot_path,
        wind_path=wind_path,
        start=data.start,
        end=data.end,
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
        {"temperature_history", "residual_forecast_proxy"}
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
            name=f"{key}_extended_model",
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


def covariate_sample_summary_table(workspace: ExtendedIntensityCalibrationWorkspace) -> pd.DataFrame:
    rows = []
    for key, variant in workspace.covariate_variants.items():
        rows.append(
            {
                "covariate": key,
                "label": variant.label,
                "active": bool(key in workspace.active_covariates),
                "n_obs": int(len(variant.raw_series.dropna())),
                "raw_mean": float(variant.raw_series.mean()),
                "raw_std": float(variant.raw_series.std(ddof=1)),
                "transformed_mean": float(variant.transformed_series.mean()),
                "transformed_std": float(variant.transformed_series.std(ddof=1)),
                "continuous_mean": float(variant.continuous_series.mean()),
                "continuous_std": float(variant.continuous_series.std(ddof=1)),
                "state_mean": float(variant.state_series.mean()),
                "state_std": float(variant.state_series.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def covariate_parameter_tables_extended_model(
    workspace: ExtendedIntensityCalibrationWorkspace,
) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for key, variant in workspace.covariate_variants.items():
        if variant.paper_seasonality is not None:
            tables[f"{key}_paper_seasonality"] = paper_parameter_table(variant.paper_seasonality)
        if variant.mle_seasonality is not None:
            tables[f"{key}_mle_seasonality"] = notebook_parameter_table(variant.mle_seasonality)
        if variant.operational_seasonality is not None and variant.operational_seasonality is not variant.mle_seasonality:
            tables[f"{key}_operational_seasonality"] = notebook_parameter_table(variant.operational_seasonality)
            if getattr(variant.operational_seasonality, "transform_params", None) is not None:
                tables[f"{key}_solar_transform"] = solar_transform_parameter_table(variant.operational_seasonality)
        tables[f"{key}_ar24"] = ar_parameter_table(variant.ar_fit, prefix=key)
    return tables


def intensity_parameter_table_extended_model(
    workspace: ExtendedIntensityCalibrationWorkspace,
) -> pd.DataFrame:
    return intensity_comparison_table(workspace.intensity_by_covariate)


__all__ = [
    "CovariateCalibrationVariantExtendedModel",
    "ExtendedIntensityCalibrationWorkspace",
    "ResolvedCalibrationIntervalExtendedModel",
    "covariate_parameter_tables_extended_model",
    "covariate_sample_summary_table",
    "extended_sample_summary_table",
    "intensity_parameter_table_extended_model",
    "resolve_safe_real_load_interval_extended_model",
    "run_intensity_model_calibration_extended_model",
]
