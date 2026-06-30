from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from .intensity_calibration_extended_model import ExtendedIntensityCalibrationWorkspace
    from .intensity_calibration_stochastic_covariance import (
        StochasticCovarianceExtensionWorkspace,
        leverage_summary_table,
        run_stochastic_covariance_extension,
        stochastic_covariance_parameter_table,
        stochastic_covariance_summary_table,
    )
    from .intensity_model_stochastic_covariance_extended_model import (
        ActiveCovariateLatentSpecificationExtendedModel,
        CovariateDrivenStochasticCovarianceSpecification,
        simulate_intensity_paths_constant_covariance_extended_model,
        simulate_intensity_paths_stochastic_covariance_extended_model,
    )
    from .intensity_var_extended_model import fit_var_companion_extended_model
    from .intensity_wishart_stochastic_covariance_extended_model import (
        WishartFullCovarianceCalibrationExtendedModel,
        fit_wishart_full_covariance_model_extended_model,
        wishart_parameter_table_extended_model,
    )
except ImportError:
    from intensity_calibration_extended_model import ExtendedIntensityCalibrationWorkspace
    from intensity_calibration_stochastic_covariance import (
        StochasticCovarianceExtensionWorkspace,
        leverage_summary_table,
        run_stochastic_covariance_extension,
        stochastic_covariance_parameter_table,
        stochastic_covariance_summary_table,
    )
    from intensity_model_stochastic_covariance_extended_model import (
        ActiveCovariateLatentSpecificationExtendedModel,
        CovariateDrivenStochasticCovarianceSpecification,
        simulate_intensity_paths_constant_covariance_extended_model,
        simulate_intensity_paths_stochastic_covariance_extended_model,
    )
    from intensity_var_extended_model import fit_var_companion_extended_model
    from intensity_wishart_stochastic_covariance_extended_model import (
        WishartFullCovarianceCalibrationExtendedModel,
        fit_wishart_full_covariance_model_extended_model,
        wishart_parameter_table_extended_model,
    )


@dataclass
class ExtendedStochasticCovarianceWorkspace:
    extended_workspace: ExtendedIntensityCalibrationWorkspace
    wind_reference_extension: StochasticCovarianceExtensionWorkspace
    stochastic_specs: Dict[str, CovariateDrivenStochasticCovarianceSpecification]
    active_set_specs: Dict[str, CovariateDrivenStochasticCovarianceSpecification]

    @property
    def base_stochastic_extension(self) -> StochasticCovarianceExtensionWorkspace:
        """Backward-compatible alias for older notebook cells."""
        return self.wind_reference_extension


def _regularize_hourly_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index).sort_values().unique()
    if len(idx) <= 1:
        return idx
    step_hours = np.diff(idx.view("i8")) / 3.6e12
    step_hours = step_hours[step_hours > 0.0]
    if step_hours.size == 0:
        return idx
    modal_step = float(pd.Series(np.round(step_hours, 6)).mode().iloc[0])
    if abs(modal_step - 1.0) < 1e-6:
        freq = "1h"
    else:
        rounded = max(int(round(modal_step)), 1)
        freq = f"{rounded}h"
    return pd.date_range(idx.min(), idx.max(), freq=freq)


def _joint_state_frame(
    workspace: ExtendedIntensityCalibrationWorkspace,
    *,
    active_covariates: list[str],
) -> pd.DataFrame:
    base_workspace = workspace.base_workspace
    spot_state = pd.Series(base_workspace.spot_variants["mle_notebook_full"].continuous_series, name="spot_state")
    wind_state = pd.Series(base_workspace.wind_variants["mle_notebook"].seasonality.residual, name="wind_state")
    frame = pd.concat([spot_state, wind_state], axis=1)
    for key in active_covariates:
        frame[key] = pd.Series(workspace.covariate_variants[key].continuous_series, name=key)
    return frame.dropna().sort_index()


def _proxy_row_to_matrix(proxy_row: pd.Series, names: list[str]) -> np.ndarray:
    d = len(names)
    mat = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(i, d):
            val = float(proxy_row[f"s[{names[i]},{names[j]}]"])
            mat[i, j] = val
            mat[j, i] = val
    return mat


def build_stochastic_covariance_spec_extended_model(
    workspace: ExtendedIntensityCalibrationWorkspace,
    *,
    primary_covariate_name: str,
    active_covariates: Optional[list[str]] = None,
    window_map: Optional[dict[str, int]] = None,
    stride_hours: int = 24,
    sim_paths: int = 128,
    random_seed: int = 20260416,
    max_norm: float = 0.95,
    b_convention: str = "simulator",
    eps: float = 1e-8,
) -> CovariateDrivenStochasticCovarianceSpecification:
    if primary_covariate_name not in workspace.covariate_variants:
        raise KeyError(f"Unknown primary covariate: {primary_covariate_name}")
    active_covs = [str(primary_covariate_name)] if active_covariates is None else [str(x) for x in active_covariates]
    unknown = sorted(set(active_covs).difference(workspace.covariate_variants))
    if unknown:
        raise KeyError(f"Unknown active covariates: {unknown}")
    if primary_covariate_name not in active_covs:
        raise ValueError("The primary_covariate_name must be contained in active_covariates.")

    state_frame = _joint_state_frame(workspace, active_covariates=active_covs)
    joint_var_fit = fit_var_companion_extended_model(state_frame)
    wishart = fit_wishart_full_covariance_model_extended_model(
        joint_var_fit.residual,
        target_name=f"{primary_covariate_name}_{len(active_covs)}d_var24_innovations",
        window_map=window_map,
        stride_hours=stride_hours,
        sim_paths=sim_paths,
        random_seed=random_seed,
        b_convention=b_convention,
        max_norm=max_norm,
        eps=eps,
    )

    covariate_specs = {
        key: ActiveCovariateLatentSpecificationExtendedModel(
            name=key,
            label=workspace.covariate_variants[key].label,
            transform=workspace.covariate_variants[key].transform_type,
            seasonality=workspace.covariate_variants[key].operational_seasonality,
            ar_fit=workspace.covariate_variants[key].ar_fit,
            state_transform=workspace.covariate_variants[key].state_transform,
        )
        for key in active_covs
    }
    p = int(joint_var_fit.lags)
    initial_history = joint_var_fit.state_frame.iloc[-p:].to_numpy(dtype=float)
    base_spec = workspace.base_workspace.model_specs["mle_notebook"]
    return CovariateDrivenStochasticCovarianceSpecification(
        name=f"{primary_covariate_name}_{len(active_covs)}d_stochastic_covariance_extended_model",
        base_spec=base_spec,
        primary_covariate_name=str(primary_covariate_name),
        primary_covariate_label=workspace.covariate_variants[primary_covariate_name].label,
        active_covariates=active_covs,
        covariate_specs=covariate_specs,
        joint_var_fit=joint_var_fit,
        wishart=wishart,
        initial_history=initial_history,
        initial_spike_state=float(base_spec.initial_spike_state),
        intensity=workspace.intensity_by_covariate[primary_covariate_name],
    )


def _build_rebased_spec_for_comparison(
    spec: CovariateDrivenStochasticCovarianceSpecification,
    workspace: ExtendedIntensityCalibrationWorkspace,
    *,
    index: pd.DatetimeIndex,
) -> tuple[CovariateDrivenStochasticCovarianceSpecification, pd.DatetimeIndex]:
    idx = pd.DatetimeIndex(index)
    p = int(spec.joint_var_fit.lags)
    if len(idx) <= p:
        raise ValueError("The comparison index must contain more than 24 timestamps.")

    spot_variant = workspace.base_workspace.spot_variants["mle_notebook_full"]
    start_time = max(pd.Timestamp(idx[0]), pd.Timestamp(spec.wishart.hourly_proxy.index[0]))
    compare_index = idx[idx >= start_time]
    if len(compare_index) <= p:
        raise ValueError("The overlap between the requested index and the covariance-proxy support is too short.")

    compare_start = pd.Timestamp(compare_index[0])
    compare_index = pd.date_range(compare_start, pd.Timestamp(idx[-1]), freq="1h")
    state_hist = spec.joint_var_fit.state_frame.loc[:compare_start].dropna()
    spike_hist = pd.Series(spot_variant.spike_state).loc[:compare_start].dropna()
    if len(state_hist) < p:
        raise ValueError("Not enough lag history is available to rebase the comparison simulation.")

    initial_history = state_hist.iloc[-p:].to_numpy(dtype=float)
    initial_spike_state = float(spike_hist.iloc[-1]) if len(spike_hist) else 0.0

    proxy_row = spec.wishart.hourly_proxy.loc[:compare_start].iloc[-1]
    sigma0_compare = _proxy_row_to_matrix(proxy_row, spec.wishart.variable_names)
    scale_matrix = np.asarray(spec.wishart.scale_matrix, dtype=float)
    inv_scale = np.linalg.inv(scale_matrix)
    sigma0_compare_standardized = inv_scale @ sigma0_compare @ inv_scale

    rebased = CovariateDrivenStochasticCovarianceSpecification(
        name=spec.name,
        base_spec=spec.base_spec,
        primary_covariate_name=spec.primary_covariate_name,
        primary_covariate_label=spec.primary_covariate_label,
        active_covariates=list(spec.active_covariates),
        covariate_specs=spec.covariate_specs,
        joint_var_fit=spec.joint_var_fit,
        wishart=WishartFullCovarianceCalibrationExtendedModel(
            target_name=spec.wishart.target_name,
            variable_names=spec.wishart.variable_names,
            innovation_frame=spec.wishart.innovation_frame,
            innovation_frame_standardized=spec.wishart.innovation_frame_standardized,
            scale_matrix=spec.wishart.scale_matrix,
            candidate_fits=spec.wishart.candidate_fits,
            selected_window=spec.wishart.selected_window,
            proxy=spec.wishart.proxy,
            proxy_standardized=spec.wishart.proxy_standardized,
            sigma_path=spec.wishart.sigma_path,
            sigma_path_standardized=spec.wishart.sigma_path_standardized,
            t_grid_years=spec.wishart.t_grid_years,
            fit=spec.wishart.fit,
            sigma0_fit=spec.wishart.sigma0_fit,
            sigma0_fit_standardized=spec.wishart.sigma0_fit_standardized,
            sigma0_forecast=sigma0_compare,
            sigma0_forecast_standardized=sigma0_compare_standardized,
            hourly_proxy=spec.wishart.hourly_proxy,
            hourly_proxy_standardized=spec.wishart.hourly_proxy_standardized,
            leverage=spec.wishart.leverage,
            leverage_table=spec.wishart.leverage_table,
            summary_table=spec.wishart.summary_table,
        ),
        initial_history=initial_history,
        initial_spike_state=initial_spike_state,
        intensity=spec.intensity,
    )
    return rebased, compare_index


def run_stochastic_covariance_extension_extended_model(
    workspace: ExtendedIntensityCalibrationWorkspace,
    *,
    window_map: Optional[dict[str, int]] = None,
    stride_hours: int = 24,
    sim_paths: int = 128,
    random_seed: int = 20260416,
    max_norm: float = 0.95,
    b_convention: str = "simulator",
    eps: float = 1e-8,
) -> ExtendedStochasticCovarianceWorkspace:
    wind_reference_extension = run_stochastic_covariance_extension(
        workspace.base_workspace,
        variant="mle_notebook",
        window_map=window_map,
        stride_hours=stride_hours,
        sim_paths=sim_paths,
        random_seed=random_seed,
        max_norm=max_norm,
        b_convention=b_convention,
        eps=eps,
    )

    stochastic_specs: Dict[str, CovariateDrivenStochasticCovarianceSpecification] = {}
    for i, key in enumerate(workspace.active_covariates):
        stochastic_specs[key] = build_stochastic_covariance_spec_extended_model(
            workspace,
            primary_covariate_name=key,
            active_covariates=[key],
            window_map=window_map,
            stride_hours=stride_hours,
            sim_paths=sim_paths,
            random_seed=random_seed + 500 * (i + 1),
            max_norm=max_norm,
            b_convention=b_convention,
            eps=eps,
        )

    active_set_specs: Dict[str, CovariateDrivenStochasticCovarianceSpecification] = {}
    for n in range(1, len(workspace.active_covariates) + 1):
        subset = workspace.active_covariates[:n]
        key = f"active_set_{n}"
        active_set_specs[key] = build_stochastic_covariance_spec_extended_model(
            workspace,
            primary_covariate_name=subset[-1],
            active_covariates=subset,
            window_map=window_map,
            stride_hours=stride_hours,
            sim_paths=sim_paths,
            random_seed=random_seed + 5000 + 500 * n,
            max_norm=max_norm,
            b_convention=b_convention,
            eps=eps,
        )

    return ExtendedStochasticCovarianceWorkspace(
        extended_workspace=workspace,
        wind_reference_extension=wind_reference_extension,
        stochastic_specs=stochastic_specs,
        active_set_specs=active_set_specs,
    )


def simulate_constant_vs_stochastic_covariance_extended_model(
    workspace: ExtendedStochasticCovarianceWorkspace,
    *,
    covariate_name: str,
    index: pd.DatetimeIndex,
    num_paths: int,
    seed: Optional[int] = None,
) -> dict[str, dict]:
    if covariate_name in workspace.stochastic_specs:
        spec = workspace.stochastic_specs[covariate_name]
    elif covariate_name in workspace.active_set_specs:
        spec = workspace.active_set_specs[covariate_name]
    else:
        raise KeyError(f"Unknown extended stochastic specification: {covariate_name}")
    idx = _regularize_hourly_index(pd.DatetimeIndex(index))
    rebased, compare_index = _build_rebased_spec_for_comparison(
        spec,
        workspace.extended_workspace,
        index=idx,
    )
    baseline = simulate_intensity_paths_constant_covariance_extended_model(
        rebased,
        index=compare_index,
        num_paths=num_paths,
        seed=seed,
    )
    stochastic = simulate_intensity_paths_stochastic_covariance_extended_model(
        rebased,
        index=compare_index,
        num_paths=num_paths,
        seed=None if seed is None else int(seed) + 1,
    )
    return {"baseline": baseline, "stochastic": stochastic, "compare_index": compare_index}


def stochastic_covariance_summary_table_extended_model(
    workspace: ExtendedStochasticCovarianceWorkspace,
    *,
    key: str,
) -> pd.DataFrame:
    spec = workspace.stochastic_specs.get(key, workspace.active_set_specs.get(key))
    if spec is None:
        raise KeyError(f"Unknown stochastic specification key: {key}")
    summary = spec.wishart.summary_table.copy()
    summary["selected"] = summary["window"].astype(str).eq(spec.wishart.selected_window)
    return summary


def leverage_summary_table_extended_model(
    workspace: ExtendedStochasticCovarianceWorkspace,
    *,
    key: str,
) -> pd.DataFrame:
    spec = workspace.stochastic_specs.get(key, workspace.active_set_specs.get(key))
    if spec is None:
        raise KeyError(f"Unknown stochastic specification key: {key}")
    return spec.wishart.leverage_table.copy()


def stochastic_covariance_parameter_table_extended_model(
    workspace: ExtendedStochasticCovarianceWorkspace,
    *,
    key: str,
) -> pd.DataFrame:
    spec = workspace.stochastic_specs.get(key, workspace.active_set_specs.get(key))
    if spec is None:
        raise KeyError(f"Unknown stochastic specification key: {key}")
    out = wishart_parameter_table_extended_model(spec.wishart)
    out = pd.concat(
        [
            out,
            pd.DataFrame(
                [
                    {"parameter": "var_spectral_radius", "value": float(spec.joint_var_fit.spectral_radius), "block": "companion"},
                    {"parameter": "active_dimension", "value": float(len(spec.joint_var_fit.names)), "block": "companion"},
                ]
            ),
        ],
        ignore_index=True,
    )
    return out


__all__ = [
    "ExtendedStochasticCovarianceWorkspace",
    "build_stochastic_covariance_spec_extended_model",
    "leverage_summary_table",
    "leverage_summary_table_extended_model",
    "run_stochastic_covariance_extension_extended_model",
    "simulate_constant_vs_stochastic_covariance_extended_model",
    "stochastic_covariance_parameter_table",
    "stochastic_covariance_parameter_table_extended_model",
    "stochastic_covariance_summary_table",
    "stochastic_covariance_summary_table_extended_model",
]
