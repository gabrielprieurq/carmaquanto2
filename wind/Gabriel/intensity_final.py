from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from .intensity_ar import AR24Fit, fit_ar24_exact_mle
    from .intensity_calibration import IntensityCalibrationWorkspace
    from .intensity_calibration import SpotCalibrationVariant
    from .intensity_calibration_correction_intensity import DEFAULT_SPIKE_CORRECTION_CONFIG, SpikeDetectionCorrectionConfig
    from .intensity_calibration_extended_model import (
        ExtendedIntensityCalibrationWorkspace,
        covariate_sample_summary_table,
    )
    from .intensity_calibration_extended_model_correction_intensity import (
        ExtendedIntensityCorrectionEvaluation,
        evaluate_intensity_correction_against_baseline,
    )
    from .intensity_calibration_stochastic_covariance import (
        run_stochastic_covariance_extension,
        simulate_constant_vs_stochastic_covariance,
        stochastic_covariance_parameter_table,
        stochastic_covariance_summary_table,
    )
    from .intensity_calibration_stochastic_covariance_extended_model import (
        ExtendedStochasticCovarianceWorkspace,
        run_stochastic_covariance_extension_extended_model,
        simulate_constant_vs_stochastic_covariance_extended_model,
        stochastic_covariance_parameter_table_extended_model,
        stochastic_covariance_summary_table_extended_model,
    )
    from .intensity_data import DEFAULT_SPOT_PATH, DEFAULT_WIND_PATH
    from .intensity_data_extended_model import (
        DEFAULT_LOAD_PATH,
        DEFAULT_REAL_LOAD_PATH,
        DEFAULT_SOLAR_PATH,
    )
    from .intensity_diagnostics import empirical_vs_simulated_marginals, joint_dependence_summary
    from .intensity_intensity_extended_model import (
        CovariateIntensityCalibrationResult,
        RawIntensityEstimatorConfig,
        build_empirical_state_transform,
        calibrate_intensity_for_renewable_covariate,
        intensity_comparison_table,
        state_series_from_transform,
    )
    from .intensity_model import simulate_intensity_paths
    from .intensity_model_extended_model import simulate_intensity_paths_extended_model
    from .intensity_seasonality_final import (
        FinalSeasonalityFit,
        fit_carma_paraschiv_log_price_seasonality_final,
    )
    from .intensity_temperature_extended_model import DEFAULT_OPENMETEO_TEMPERATURE_PATH
except ImportError:
    from intensity_ar import AR24Fit, fit_ar24_exact_mle
    from intensity_calibration import IntensityCalibrationWorkspace
    from intensity_calibration import SpotCalibrationVariant
    from intensity_calibration_correction_intensity import DEFAULT_SPIKE_CORRECTION_CONFIG, SpikeDetectionCorrectionConfig
    from intensity_calibration_extended_model import (
        ExtendedIntensityCalibrationWorkspace,
        covariate_sample_summary_table,
    )
    from intensity_calibration_extended_model_correction_intensity import (
        ExtendedIntensityCorrectionEvaluation,
        evaluate_intensity_correction_against_baseline,
    )
    from intensity_calibration_stochastic_covariance import (
        run_stochastic_covariance_extension,
        simulate_constant_vs_stochastic_covariance,
        stochastic_covariance_parameter_table,
        stochastic_covariance_summary_table,
    )
    from intensity_calibration_stochastic_covariance_extended_model import (
        ExtendedStochasticCovarianceWorkspace,
        run_stochastic_covariance_extension_extended_model,
        simulate_constant_vs_stochastic_covariance_extended_model,
        stochastic_covariance_parameter_table_extended_model,
        stochastic_covariance_summary_table_extended_model,
    )
    from intensity_data import DEFAULT_SPOT_PATH, DEFAULT_WIND_PATH
    from intensity_data_extended_model import (
        DEFAULT_LOAD_PATH,
        DEFAULT_REAL_LOAD_PATH,
        DEFAULT_SOLAR_PATH,
    )
    from intensity_diagnostics import empirical_vs_simulated_marginals, joint_dependence_summary
    from intensity_intensity_extended_model import (
        CovariateIntensityCalibrationResult,
        RawIntensityEstimatorConfig,
        build_empirical_state_transform,
        calibrate_intensity_for_renewable_covariate,
        intensity_comparison_table,
        state_series_from_transform,
    )
    from intensity_model import simulate_intensity_paths
    from intensity_model_extended_model import simulate_intensity_paths_extended_model
    from intensity_seasonality_final import (
        FinalSeasonalityFit,
        fit_carma_paraschiv_log_price_seasonality_final,
    )
    from intensity_temperature_extended_model import DEFAULT_OPENMETEO_TEMPERATURE_PATH


DEFAULT_FINAL_ACTIVE_COVARIATES = (
    "load_forecast_proxy",
    "residual_forecast_proxy",
    "temperature_history",
    "solar_clear_sky",
)
DEFAULT_FINAL_RAW_INTENSITY_COVARIATES = (
    "load_forecast_proxy",
    "residual_forecast_proxy",
    "temperature_history",
)
DEFAULT_FINAL_WISHART_WINDOW_MAP = {
    "7D": 24 * 7,
    "14D": 24 * 14,
    "30D": 24 * 30,
    "90D": 24 * 90,
}


@dataclass
class FinalIntensityCalibrationConfig:
    spot_path: str = str(DEFAULT_SPOT_PATH)
    wind_path: str = str(DEFAULT_WIND_PATH)
    load_path: str = str(DEFAULT_LOAD_PATH)
    real_load_path: str = str(DEFAULT_REAL_LOAD_PATH)
    temperature_path: str = str(DEFAULT_OPENMETEO_TEMPERATURE_PATH)
    solar_path: str = str(DEFAULT_SOLAR_PATH)
    use_real_entsoe_load: bool = True
    auto_download_real_load: bool = False
    entsoe_api_key: Optional[str] = None
    use_historical_temperature: bool = True
    temperature_source: Optional[str] = "openmeteo"
    auto_download_temperature: bool = False
    active_covariates: tuple[str, ...] = DEFAULT_FINAL_ACTIVE_COVARIATES
    raw_intensity_covariates: tuple[str, ...] = DEFAULT_FINAL_RAW_INTENSITY_COVARIATES
    raw_intensity_estimator_configs: Optional[Dict[str, RawIntensityEstimatorConfig]] = None
    spike_correction_config: SpikeDetectionCorrectionConfig = DEFAULT_SPIKE_CORRECTION_CONFIG
    spot_seasonality_mode: str = "carma_paraschiv_log_price"
    spot_log_price_shift: float = 1000.0
    start: Optional[str | pd.Timestamp] = None
    end: Optional[str | pd.Timestamp] = None
    wishart_window_map: Optional[dict[str, int]] = None
    wishart_stride_hours: int = 24
    wishart_sim_paths: int = 128
    wishart_seed: int = 20260416
    wishart_max_norm: float = 0.95
    wishart_b_convention: str = "simulator"
    wishart_eps: float = 1e-8

    def resolved_wishart_window_map(self) -> dict[str, int]:
        return dict(DEFAULT_FINAL_WISHART_WINDOW_MAP if self.wishart_window_map is None else self.wishart_window_map)


@dataclass
class FinalWindIntensityCalibration:
    state_series: pd.Series
    intensity: CovariateIntensityCalibrationResult
    model_spec: object


@dataclass
class FinalSpotSeasonalityCalibration:
    mode: str
    log_price_fit: Optional[FinalSeasonalityFit]
    level_seasonality: pd.Series
    continuous_series: pd.Series
    ar_fit: AR24Fit
    simulator_reference_seasonality: pd.Series
    log_price_shift: float


@dataclass
class FinalIntensityCalibrationResult:
    config: FinalIntensityCalibrationConfig
    evaluation: ExtendedIntensityCorrectionEvaluation
    workspace: ExtendedIntensityCalibrationWorkspace
    base_workspace: IntensityCalibrationWorkspace
    wind_reference: FinalWindIntensityCalibration
    spot_seasonality: Optional[FinalSpotSeasonalityCalibration] = None
    covariance_workspace: Optional[ExtendedStochasticCovarianceWorkspace] = None


def _safe_residual_corr(x: pd.Series, y: pd.Series) -> float:
    common = pd.DatetimeIndex(x.index).intersection(pd.DatetimeIndex(y.index))
    if len(common) < 2:
        return 0.0
    rho = float(np.corrcoef(pd.Series(x).loc[common].to_numpy(dtype=float), pd.Series(y).loc[common].to_numpy(dtype=float))[0, 1])
    return rho if np.isfinite(rho) else 0.0


def _replace_covariate_rho_final(
    intensity_by_covariate: Dict[str, CovariateIntensityCalibrationResult],
    *,
    rho: float,
) -> Dict[str, CovariateIntensityCalibrationResult]:
    return {key: replace(value, rho=float(rho)) for key, value in intensity_by_covariate.items()}


def _apply_price_seasonality_mode_final(
    base_workspace: IntensityCalibrationWorkspace,
    *,
    temperature: pd.Series,
    mode: str,
    log_price_shift: float,
) -> tuple[IntensityCalibrationWorkspace, Optional[FinalSpotSeasonalityCalibration]]:
    mode = str(mode)
    if mode in {"mle", "mle_benchmark", "mle_notebook"}:
        return base_workspace, None
    if mode != "carma_paraschiv_log_price":
        raise ValueError(f"Unsupported spot_seasonality_mode: {mode}")

    data = base_workspace.data
    old_variant = base_workspace.spot_variants["mle_notebook_full"]
    log_fit = fit_carma_paraschiv_log_price_seasonality_final(
        data.spot,
        temperature,
        name="spot_paraschiv_log_price_operational_final",
        delta_shift=float(log_price_shift),
    )
    level_seasonality = (np.exp(log_fit.fitted) - float(log_price_shift)).rename("spot_paraschiv_level_seasonality_final")
    level_seasonality = level_seasonality.reindex(data.spot.index).interpolate(method="time").ffill().bfill()
    spike_state = old_variant.spike_state.reindex(data.spot.index).interpolate(method="time").ffill().bfill()
    continuous = (data.spot - level_seasonality - spike_state).rename("spot_paraschiv_continuous_final")
    ar_series = continuous.loc[~base_workspace.spike_detection.filtered_mask.reindex(continuous.index).fillna(False)].dropna()
    ar_fit = fit_ar24_exact_mle(ar_series)
    new_variant = replace(
        old_variant,
        name="spot_paraschiv_level_final",
        continuous_series=continuous,
        ar_fit=ar_fit,
        filtered_series_for_ar=ar_series,
    )
    old_spec = base_workspace.model_specs["mle_notebook"]
    new_spec = replace(
        old_spec,
        name="mle_notebook_with_paraschiv_spot_seasonality_final",
        spot_ar=ar_fit,
        initial_spot_lags=continuous.dropna().iloc[-24:].to_numpy(dtype=float)[::-1],
    )
    spot_variants = dict(base_workspace.spot_variants)
    spot_variants["mle_notebook_full"] = new_variant
    spot_variants["harmonic_full"] = new_variant
    spot_variants["spot_paraschiv_level_final"] = new_variant
    model_specs = dict(base_workspace.model_specs)
    model_specs["mle_notebook"] = new_spec
    model_specs["harmonic_full"] = new_spec
    model_specs["spot_paraschiv_level_final"] = new_spec
    updated = replace(base_workspace, spot_variants=spot_variants, model_specs=model_specs)
    final_spot = FinalSpotSeasonalityCalibration(
        mode=mode,
        log_price_fit=log_fit,
        level_seasonality=level_seasonality,
        continuous_series=continuous,
        ar_fit=ar_fit,
        simulator_reference_seasonality=old_variant.seasonality.fitted.reindex(data.spot.index).interpolate(method="time").ffill().bfill(),
        log_price_shift=float(log_price_shift),
    )
    return updated, final_spot


def _calibrate_wind_reference_intensity_final(
    base_workspace: IntensityCalibrationWorkspace,
) -> FinalWindIntensityCalibration:
    data = base_workspace.data
    wind_transform = build_empirical_state_transform(
        data.wind_cf,
        name="wind_reference_capacity_factor_state_final",
        lower_clip=0.01,
        upper_clip=0.99,
    )
    wind_state = state_series_from_transform(
        data.wind_cf,
        wind_transform,
        name="wind_reference_capacity_factor_state_final",
    )
    intensity = calibrate_intensity_for_renewable_covariate(
        covariate_state=wind_state,
        renewable_raw=data.wind_cf,
        jump_mask=base_workspace.spike_detection.jump_mask,
        jump_increment=base_workspace.spike_detection.jump_increment,
        spot_residual=base_workspace.spot_variants["mle_notebook_full"].ar_fit.residual,
        correlation_residual=base_workspace.wind_variants["mle_notebook"].ar_fit.residual,
        covariate_name="wind_reference",
        covariate_label="Wind capacity factor",
    )
    model_spec = replace(
        base_workspace.model_specs["mle_notebook"],
        name="wind_reference_final",
        intensity=intensity,
    )
    return FinalWindIntensityCalibration(
        state_series=wind_state,
        intensity=intensity,
        model_spec=model_spec,
    )


def _replace_base_workspace_wind_intensity_final(
    base_workspace: IntensityCalibrationWorkspace,
    wind_reference: FinalWindIntensityCalibration,
) -> IntensityCalibrationWorkspace:
    intensity_variants = dict(base_workspace.intensity_variants)
    intensity_variants["mle_notebook"] = wind_reference.intensity
    intensity_variants["harmonic_full"] = wind_reference.intensity
    intensity_variants["wind_reference_final"] = wind_reference.intensity
    model_specs = dict(base_workspace.model_specs)
    model_specs["mle_notebook"] = wind_reference.model_spec
    model_specs["harmonic_full"] = wind_reference.model_spec
    model_specs["wind_reference_final"] = wind_reference.model_spec
    return replace(
        base_workspace,
        intensity_variants=intensity_variants,
        model_specs=model_specs,
    )


def _replace_workspace_base_final(
    workspace: ExtendedIntensityCalibrationWorkspace,
    base_workspace: IntensityCalibrationWorkspace,
    intensity_by_covariate: Optional[Dict[str, CovariateIntensityCalibrationResult]] = None,
) -> ExtendedIntensityCalibrationWorkspace:
    if intensity_by_covariate is None:
        intensity_by_covariate = workspace.intensity_by_covariate
    base_spec = base_workspace.model_specs["mle_notebook"]
    model_specs = {
        key: replace(spec, base_spec=base_spec, intensity=intensity_by_covariate.get(key, spec.intensity))
        for key, spec in workspace.model_specs.items()
    }
    return replace(
        workspace,
        base_workspace=base_workspace,
        intensity_by_covariate=dict(intensity_by_covariate),
        model_specs=model_specs,
    )


def _path_level_seasonality(final_spot: FinalSpotSeasonalityCalibration, index: pd.DatetimeIndex) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    values = final_spot.level_seasonality.reindex(idx)
    if values.isna().any():
        values = values.interpolate(method="time").ffill().bfill()
    if values.isna().any():
        raise ValueError("Final spot seasonality does not cover the requested simulation index.")
    return values.to_numpy(dtype=float)


class IntensityFinal:
    """Final additive intensity-model orchestrator.

    The class is intentionally a thin additive layer over the existing modules.
    It standardizes the wind-capacity intensity calibration so wind is treated as
    a covariate-driven renewable driver, then builds the constant-covariance and
    Wishart stochastic-covariance branches used in Section 1.9 of the PPA notes.
    """

    def __init__(self, config: FinalIntensityCalibrationConfig):
        self.config = config
        self.result: Optional[FinalIntensityCalibrationResult] = None

    @property
    def workspace(self) -> ExtendedIntensityCalibrationWorkspace:
        if self.result is None:
            raise RuntimeError("Call calibrate() before accessing the workspace.")
        return self.result.workspace

    @property
    def base_workspace(self) -> IntensityCalibrationWorkspace:
        if self.result is None:
            raise RuntimeError("Call calibrate() before accessing the base workspace.")
        return self.result.base_workspace

    @property
    def covariance_workspace(self) -> ExtendedStochasticCovarianceWorkspace:
        if self.result is None or self.result.covariance_workspace is None:
            raise RuntimeError("Call fit_covariance_extensions() before accessing covariance_workspace.")
        return self.result.covariance_workspace

    def calibrate(self) -> FinalIntensityCalibrationResult:
        cfg = self.config
        evaluation = evaluate_intensity_correction_against_baseline(
            spot_path=cfg.spot_path,
            wind_path=cfg.wind_path,
            load_path=cfg.load_path,
            real_load_path=cfg.real_load_path,
            temperature_path=cfg.temperature_path,
            solar_path=cfg.solar_path,
            use_real_entsoe_load=cfg.use_real_entsoe_load,
            auto_download_real_load=cfg.auto_download_real_load,
            entsoe_api_key=cfg.entsoe_api_key,
            use_historical_temperature=cfg.use_historical_temperature,
            temperature_source=cfg.temperature_source,
            auto_download_temperature=cfg.auto_download_temperature,
            active_covariates=list(cfg.active_covariates),
            raw_intensity_covariates=list(cfg.raw_intensity_covariates),
            raw_intensity_estimator_configs=cfg.raw_intensity_estimator_configs,
            spike_correction_config=cfg.spike_correction_config,
            start=cfg.start,
            end=cfg.end,
        )
        workspace_raw = evaluation.corrected
        base_spot_workspace, spot_seasonality = _apply_price_seasonality_mode_final(
            workspace_raw.base_workspace,
            temperature=workspace_raw.data.temperature,
            mode=cfg.spot_seasonality_mode,
            log_price_shift=float(cfg.spot_log_price_shift),
        )
        final_rho = _safe_residual_corr(
            base_spot_workspace.spot_variants["mle_notebook_full"].ar_fit.residual,
            base_spot_workspace.wind_variants["mle_notebook"].ar_fit.residual,
        )
        intensity_by_covariate = _replace_covariate_rho_final(workspace_raw.intensity_by_covariate, rho=final_rho)
        workspace_spot_final = _replace_workspace_base_final(
            workspace_raw,
            base_spot_workspace,
            intensity_by_covariate=intensity_by_covariate,
        )
        wind_reference = _calibrate_wind_reference_intensity_final(workspace_spot_final.base_workspace)
        base_workspace_final = _replace_base_workspace_wind_intensity_final(workspace_spot_final.base_workspace, wind_reference)
        workspace_final = _replace_workspace_base_final(
            workspace_spot_final,
            base_workspace_final,
            intensity_by_covariate=intensity_by_covariate,
        )
        self.result = FinalIntensityCalibrationResult(
            config=cfg,
            evaluation=evaluation,
            workspace=workspace_final,
            base_workspace=base_workspace_final,
            wind_reference=wind_reference,
            spot_seasonality=spot_seasonality,
        )
        return self.result

    def fit_covariance_extensions(self) -> FinalIntensityCalibrationResult:
        if self.result is None:
            self.calibrate()
        assert self.result is not None
        cfg = self.config
        covariance_workspace = run_stochastic_covariance_extension_extended_model(
            self.result.workspace,
            window_map=cfg.resolved_wishart_window_map(),
            stride_hours=int(cfg.wishart_stride_hours),
            sim_paths=int(cfg.wishart_sim_paths),
            random_seed=int(cfg.wishart_seed),
            max_norm=float(cfg.wishart_max_norm),
            b_convention=str(cfg.wishart_b_convention),
            eps=float(cfg.wishart_eps),
        )
        self.result = replace(self.result, covariance_workspace=covariance_workspace)
        return self.result

    def fit(self, *, covariance: bool = True) -> FinalIntensityCalibrationResult:
        self.calibrate()
        if covariance:
            self.fit_covariance_extensions()
        assert self.result is not None
        return self.result

    def intensity_table(self) -> pd.DataFrame:
        if self.result is None:
            raise RuntimeError("Call calibrate() before requesting intensity_table().")
        all_intensities: dict[str, CovariateIntensityCalibrationResult] = {
            "wind_reference": self.result.wind_reference.intensity,
        }
        all_intensities.update(self.result.workspace.intensity_by_covariate)
        return intensity_comparison_table(all_intensities)

    def spot_seasonality_table(self) -> pd.DataFrame:
        if self.result is None:
            raise RuntimeError("Call calibrate() before requesting spot_seasonality_table().")
        final_spot = self.result.spot_seasonality
        old_variant = self.result.evaluation.corrected.base_workspace.spot_variants["mle_notebook_full"]
        if final_spot is None:
            return pd.DataFrame(
                [
                    {
                        "mode": "mle_benchmark",
                        "level_seasonality_mean": float(old_variant.seasonality.fitted.mean()),
                        "continuous_std": float(old_variant.continuous_series.std(ddof=1)),
                        "innovation_std": float(old_variant.ar_fit.innovation_std),
                    }
                ]
            )
        return pd.DataFrame(
            [
                {
                    "mode": final_spot.mode,
                    "log_price_shift": float(final_spot.log_price_shift),
                    "log_seasonality_rmse": float(final_spot.log_price_fit.rmse) if final_spot.log_price_fit is not None else np.nan,
                    "level_seasonality_mean": float(final_spot.level_seasonality.mean()),
                    "level_seasonality_std": float(final_spot.level_seasonality.std(ddof=1)),
                    "continuous_mean": float(final_spot.continuous_series.mean()),
                    "continuous_std": float(final_spot.continuous_series.std(ddof=1)),
                    "innovation_std": float(final_spot.ar_fit.innovation_std),
                    "simulator_reference_mean": float(final_spot.simulator_reference_seasonality.mean()),
                }
            ]
        )

    def covariate_sample_table(self) -> pd.DataFrame:
        if self.result is None:
            raise RuntimeError("Call calibrate() before requesting covariate_sample_table().")
        wind = self.result.base_workspace.data.wind_cf
        wind_row = pd.DataFrame(
            [
                {
                    "covariate": "wind_reference",
                    "label": "Wind capacity factor",
                    "active": True,
                    "n_obs": int(len(wind.dropna())),
                    "raw_mean": float(wind.mean()),
                    "raw_std": float(wind.std(ddof=1)),
                    "transformed_mean": float(self.result.base_workspace.data.wind_logit.mean()),
                    "transformed_std": float(self.result.base_workspace.data.wind_logit.std(ddof=1)),
                    "continuous_mean": float(self.result.base_workspace.wind_variants["mle_notebook"].seasonality.residual.mean()),
                    "continuous_std": float(self.result.base_workspace.wind_variants["mle_notebook"].seasonality.residual.std(ddof=1)),
                    "state_mean": float(self.result.wind_reference.state_series.mean()),
                    "state_std": float(self.result.wind_reference.state_series.std(ddof=1)),
                }
            ]
        )
        return pd.concat([wind_row, covariate_sample_summary_table(self.result.workspace)], ignore_index=True)

    def incremental_model_table(self) -> pd.DataFrame:
        if self.result is None:
            raise RuntimeError("Call calibrate() before requesting incremental_model_table().")
        rows = [
            {
                "model_key": "wind_reference",
                "model_type": "wind_reference",
                "primary_covariate": "wind_reference",
                "active_covariates": "wind_reference",
                "covariance_dimension": 2,
                "intensity_driver_mode": self.result.wind_reference.intensity.driver_mode,
            }
        ]
        active = list(self.result.workspace.active_covariates)
        for key in active:
            intensity = self.result.workspace.intensity_by_covariate[key]
            rows.append(
                {
                    "model_key": key,
                    "model_type": "single_covariate",
                    "primary_covariate": key,
                    "active_covariates": key,
                    "covariance_dimension": 3,
                    "intensity_driver_mode": intensity.driver_mode,
                }
            )
        for n in range(1, len(active) + 1):
            subset = active[:n]
            primary = subset[-1]
            intensity = self.result.workspace.intensity_by_covariate[primary]
            rows.append(
                {
                    "model_key": f"active_set_{n}",
                    "model_type": "cumulative_active_set",
                    "primary_covariate": primary,
                    "active_covariates": ", ".join(subset),
                    "covariance_dimension": 2 + n,
                    "intensity_driver_mode": intensity.driver_mode,
                }
            )
        return pd.DataFrame(rows)

    def _apply_final_spot_seasonality_to_simulation(self, sim: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.result is None or self.result.spot_seasonality is None:
            return sim
        if "spot" not in sim or "spot_seasonality" not in sim or "index" not in sim:
            return sim
        out = dict(sim)
        idx = pd.DatetimeIndex(sim["index"])
        final_level = _path_level_seasonality(self.result.spot_seasonality, idx)
        old_level = np.asarray(sim["spot_seasonality"], dtype=float)
        if old_level.ndim == 2:
            old_path = old_level[0]
        else:
            old_path = old_level
        adjustment = final_level - old_path
        out["spot"] = np.asarray(sim["spot"], dtype=float).copy() + adjustment[None, :]
        out["spot_seasonality"] = final_level[None, :].repeat(out["spot"].shape[0], axis=0)
        return out

    def simulate_single_covariate_models(
        self,
        *,
        index: Optional[pd.DatetimeIndex] = None,
        num_paths: int = 250,
        seed: int = 20260505,
    ) -> dict[str, dict[str, np.ndarray]]:
        if self.result is None:
            raise RuntimeError("Call calibrate() before simulating models.")
        idx = pd.DatetimeIndex(index) if index is not None else pd.DatetimeIndex(self.result.workspace.data.full_index)
        sims: dict[str, dict[str, np.ndarray]] = {
            "wind_reference": self._apply_final_spot_seasonality_to_simulation(
                simulate_intensity_paths(
                    self.result.base_workspace.model_specs["mle_notebook"],
                    index=idx,
                    num_paths=int(num_paths),
                    seed=int(seed),
                )
            )
        }
        for i, key in enumerate(self.result.workspace.active_covariates, start=1):
            sims[key] = self._apply_final_spot_seasonality_to_simulation(
                simulate_intensity_paths_extended_model(
                    self.result.workspace.model_specs[key],
                    index=idx,
                    num_paths=int(num_paths),
                    seed=int(seed) + i,
                )
            )
        return sims

    def marginal_scorecard(
        self,
        simulations: dict[str, dict[str, np.ndarray]],
    ) -> pd.DataFrame:
        if self.result is None:
            raise RuntimeError("Call calibrate() before requesting marginal_scorecard().")
        data = self.result.workspace.data
        rows = []
        empirical_joint = pd.DataFrame({"spot": data.spot, "wind_cf": data.wind_cf}).dropna()
        for key, sim in simulations.items():
            rows.append(empirical_vs_simulated_marginals(data.spot, sim["spot"], label=f"spot::{key}").assign(model=key, variable="spot"))
            rows.append(empirical_vs_simulated_marginals(data.wind_cf, sim["wind_cf"], label=f"wind_cf::{key}").assign(model=key, variable="wind_cf"))
            joint = joint_dependence_summary(empirical_joint, sim["spot"], sim["wind_cf"]).assign(model=key, pair="spot_vs_wind")
            rows.append(joint.rename(columns={"measure": "series", "empirical": "empirical_mean", "simulated": "simulated_mean"}))
            if key in self.result.workspace.covariate_variants and "covariate_value" in sim:
                emp_cov = self.result.workspace.covariate_variants[key].raw_series
                rows.append(
                    empirical_vs_simulated_marginals(emp_cov, sim["covariate_value"], label=f"covariate::{key}").assign(
                        model=key,
                        variable=key,
                    )
                )
        return pd.concat(rows, ignore_index=True, sort=False)

    def simulate_covariance_models(
        self,
        *,
        index: Optional[pd.DatetimeIndex] = None,
        num_paths: int = 250,
        seed: int = 20260505,
        include_single_covariates: bool = True,
        include_active_sets: bool = True,
    ) -> dict[str, dict]:
        if self.result is None or self.result.covariance_workspace is None:
            raise RuntimeError("Call fit_covariance_extensions() before simulating covariance models.")
        idx = pd.DatetimeIndex(index) if index is not None else pd.DatetimeIndex(self.result.workspace.data.full_index)
        out: dict[str, dict] = {
            "wind_reference": simulate_constant_vs_stochastic_covariance(
                self.result.covariance_workspace.wind_reference_extension,
                index=idx,
                num_paths=int(num_paths),
                seed=int(seed),
            )
        }
        out["wind_reference"]["baseline"] = self._apply_final_spot_seasonality_to_simulation(out["wind_reference"]["baseline"])
        out["wind_reference"]["stochastic"] = self._apply_final_spot_seasonality_to_simulation(out["wind_reference"]["stochastic"])
        if include_single_covariates:
            for i, key in enumerate(self.result.workspace.active_covariates, start=1):
                out[key] = simulate_constant_vs_stochastic_covariance_extended_model(
                    self.result.covariance_workspace,
                    covariate_name=key,
                    index=idx,
                    num_paths=int(num_paths),
                    seed=int(seed) + 100 + i,
                )
                out[key]["baseline"] = self._apply_final_spot_seasonality_to_simulation(out[key]["baseline"])
                out[key]["stochastic"] = self._apply_final_spot_seasonality_to_simulation(out[key]["stochastic"])
        if include_active_sets:
            for i, key in enumerate(self.result.covariance_workspace.active_set_specs, start=1):
                out[key] = simulate_constant_vs_stochastic_covariance_extended_model(
                    self.result.covariance_workspace,
                    covariate_name=key,
                    index=idx,
                    num_paths=int(num_paths),
                    seed=int(seed) + 500 + i,
                )
                out[key]["baseline"] = self._apply_final_spot_seasonality_to_simulation(out[key]["baseline"])
                out[key]["stochastic"] = self._apply_final_spot_seasonality_to_simulation(out[key]["stochastic"])
        return out

    def covariance_summary_tables(self) -> Dict[str, pd.DataFrame]:
        if self.result is None or self.result.covariance_workspace is None:
            raise RuntimeError("Call fit_covariance_extensions() before requesting covariance tables.")
        tables: Dict[str, pd.DataFrame] = {
            "wind_reference_summary": stochastic_covariance_summary_table(self.result.covariance_workspace.wind_reference_extension),
            "wind_reference_parameters": stochastic_covariance_parameter_table(self.result.covariance_workspace.wind_reference_extension),
        }
        for key in self.result.covariance_workspace.stochastic_specs:
            tables[f"{key}_summary"] = stochastic_covariance_summary_table_extended_model(self.result.covariance_workspace, key=key)
            tables[f"{key}_parameters"] = stochastic_covariance_parameter_table_extended_model(self.result.covariance_workspace, key=key)
        for key in self.result.covariance_workspace.active_set_specs:
            tables[f"{key}_summary"] = stochastic_covariance_summary_table_extended_model(self.result.covariance_workspace, key=key)
            tables[f"{key}_parameters"] = stochastic_covariance_parameter_table_extended_model(self.result.covariance_workspace, key=key)
        return tables


def default_final_output_dir(base: str | Path | None = None) -> Path:
    if base is None:
        return Path(__file__).resolve().parent / "intensity_final_outputs"
    return Path(base)


__all__ = [
    "DEFAULT_FINAL_ACTIVE_COVARIATES",
    "DEFAULT_FINAL_RAW_INTENSITY_COVARIATES",
    "DEFAULT_FINAL_WISHART_WINDOW_MAP",
    "FinalIntensityCalibrationConfig",
    "FinalIntensityCalibrationResult",
    "FinalSpotSeasonalityCalibration",
    "FinalWindIntensityCalibration",
    "IntensityFinal",
    "default_final_output_dir",
]
