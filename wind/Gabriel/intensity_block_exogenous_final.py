from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from .intensity_calibration_extended_model import ExtendedIntensityCalibrationWorkspace
    from .intensity_model_stochastic_covariance_extended_model import (
        ActiveCovariateLatentSpecificationExtendedModel,
        CovariateDrivenStochasticCovarianceSpecification,
        simulate_intensity_paths_constant_covariance_extended_model,
        simulate_intensity_paths_stochastic_covariance_extended_model,
    )
    from .intensity_var_extended_model import MultivariateVARCompanionFitExtendedModel
    from .intensity_wishart_stochastic_covariance_extended_model import (
        WishartFullCovarianceCalibrationExtendedModel,
        fit_wishart_full_covariance_model_extended_model,
        wishart_parameter_table_extended_model,
    )
except ImportError:
    from intensity_calibration_extended_model import ExtendedIntensityCalibrationWorkspace
    from intensity_model_stochastic_covariance_extended_model import (
        ActiveCovariateLatentSpecificationExtendedModel,
        CovariateDrivenStochasticCovarianceSpecification,
        simulate_intensity_paths_constant_covariance_extended_model,
        simulate_intensity_paths_stochastic_covariance_extended_model,
    )
    from intensity_var_extended_model import MultivariateVARCompanionFitExtendedModel
    from intensity_wishart_stochastic_covariance_extended_model import (
        WishartFullCovarianceCalibrationExtendedModel,
        fit_wishart_full_covariance_model_extended_model,
        wishart_parameter_table_extended_model,
    )


@dataclass
class BlockExogenousVARXDiagnosticsFinal:
    names: list[str]
    lags: int
    spectral_radius: float
    max_abs_forbidden_driver_from_spot: float
    spot_from_driver_l2: dict[str, float]
    driver_from_spot_l2: dict[str, float]
    innovation_corr: pd.DataFrame


def joint_state_frame_block_exogenous_final(
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


def _lag_design(values: np.ndarray, lags: int) -> np.ndarray:
    blocks = [values[lags - lag : values.shape[0] - lag, :] for lag in range(1, lags + 1)]
    return np.concatenate(blocks, axis=1)


def _ridge_solve(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    xtx = x.T @ x + float(ridge) * np.eye(x.shape[1], dtype=float)
    return np.linalg.solve(xtx, x.T @ y)


def fit_block_exogenous_varx_final(
    state_frame: pd.DataFrame,
    *,
    lags: int = 24,
    ridge: float = 1e-8,
) -> MultivariateVARCompanionFitExtendedModel:
    frame = pd.DataFrame(state_frame, copy=True).astype(float).dropna().sort_index()
    if "spot_state" not in frame.columns:
        raise KeyError("state_frame must contain a 'spot_state' column.")
    if frame.columns[0] != "spot_state":
        cols = ["spot_state"] + [col for col in frame.columns if col != "spot_state"]
        frame = frame.loc[:, cols]
    if frame.shape[1] < 2:
        raise ValueError("The block-exogenous model requires spot_state and at least one physical driver.")
    if len(frame) <= int(lags):
        raise ValueError("Not enough observations to fit the block-exogenous VARX model.")

    p = int(lags)
    d = int(frame.shape[1])
    q = d - 1
    values = frame.to_numpy(dtype=float)
    spot = values[:, [0]]
    drivers = values[:, 1:]

    y_spot = spot[p:, 0]
    y_driver = drivers[p:, :]
    x_full = _lag_design(values, p)
    x_driver = _lag_design(drivers, p)

    beta_spot = _ridge_solve(x_full, y_spot, ridge).reshape(-1)
    beta_driver = _ridge_solve(x_driver, y_driver, ridge)

    coefs = np.zeros((p, d, d), dtype=float)
    for lag in range(p):
        coefs[lag, 0, :] = beta_spot[lag * d : (lag + 1) * d]
        coefs[lag, 1:, 1:] = beta_driver[lag * q : (lag + 1) * q, :].T

    fitted = np.zeros((len(frame) - p, d), dtype=float)
    fitted[:, 0] = x_full @ beta_spot
    fitted[:, 1:] = x_driver @ beta_driver
    residual = values[p:, :] - fitted

    companion = np.zeros((d * p, d * p), dtype=float)
    companion[:d, :] = np.hstack(coefs)
    if p > 1:
        companion[d:, :-d] = np.eye(d * (p - 1), dtype=float)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(companion))))

    residual_cols = [f"{name}_innovation" for name in frame.columns]
    fitted_cols = [f"{name}_fitted" for name in frame.columns]
    residual_df = pd.DataFrame(residual, index=frame.index[p:], columns=residual_cols)
    fitted_df = pd.DataFrame(fitted, index=frame.index[p:], columns=fitted_cols)
    innovation_cov = np.cov(residual.T, ddof=1)
    innovation_cov = 0.5 * (innovation_cov + innovation_cov.T)

    return MultivariateVARCompanionFitExtendedModel(
        lags=p,
        names=list(frame.columns),
        coefs=coefs,
        innovation_cov=np.asarray(innovation_cov, dtype=float),
        residual=residual_df,
        fitted=fitted_df,
        state_frame=frame,
        companion_matrix=companion,
        spectral_radius=spectral_radius,
    )


def block_exogenous_diagnostics_final(
    fit: MultivariateVARCompanionFitExtendedModel,
) -> BlockExogenousVARXDiagnosticsFinal:
    names = list(fit.names)
    coefs = np.asarray(fit.coefs, dtype=float)
    drivers = names[1:]
    forbidden = coefs[:, 1:, 0]
    cov = np.asarray(fit.innovation_cov, dtype=float)
    diag = np.clip(np.diag(cov), 1e-14, None)
    corr = cov / np.sqrt(np.outer(diag, diag))
    corr = pd.DataFrame(np.clip(corr, -1.0, 1.0), index=names, columns=names)
    return BlockExogenousVARXDiagnosticsFinal(
        names=names,
        lags=int(fit.lags),
        spectral_radius=float(fit.spectral_radius),
        max_abs_forbidden_driver_from_spot=float(np.max(np.abs(forbidden))) if forbidden.size else 0.0,
        spot_from_driver_l2={name: float(np.sqrt(np.sum(coefs[:, 0, i] ** 2))) for i, name in enumerate(names[1:], start=1)},
        driver_from_spot_l2={name: float(np.sqrt(np.sum(coefs[:, i, 0] ** 2))) for i, name in enumerate(names[1:], start=1)},
        innovation_corr=corr,
    )


def build_block_exogenous_covariance_spec_final(
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

    state_frame = joint_state_frame_block_exogenous_final(workspace, active_covariates=active_covs)
    joint_var_fit = fit_block_exogenous_varx_final(state_frame)
    wishart = fit_wishart_full_covariance_model_extended_model(
        joint_var_fit.residual,
        target_name=f"{primary_covariate_name}_{len(active_covs)}d_block_exogenous_varx24_innovations",
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
    base_spec = workspace.base_workspace.model_specs["mle_notebook"]
    return CovariateDrivenStochasticCovarianceSpecification(
        name=f"{primary_covariate_name}_{len(active_covs)}d_block_exogenous_final",
        base_spec=base_spec,
        primary_covariate_name=str(primary_covariate_name),
        primary_covariate_label=workspace.covariate_variants[primary_covariate_name].label,
        active_covariates=active_covs,
        covariate_specs=covariate_specs,
        joint_var_fit=joint_var_fit,
        wishart=wishart,
        initial_history=joint_var_fit.state_frame.iloc[-p:].to_numpy(dtype=float),
        initial_spike_state=float(base_spec.initial_spike_state),
        intensity=workspace.intensity_by_covariate[primary_covariate_name],
    )


def _regularize_hourly_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index).sort_values().unique()
    if len(idx) <= 1:
        return idx
    step_hours = np.diff(idx.view("i8")) / 3.6e12
    step_hours = step_hours[step_hours > 0.0]
    if step_hours.size == 0:
        return idx
    modal_step = float(pd.Series(np.round(step_hours, 6)).mode().iloc[0])
    freq = "1h" if abs(modal_step - 1.0) < 1e-6 else f"{max(int(round(modal_step)), 1)}h"
    return pd.date_range(idx.min(), idx.max(), freq=freq)


def _proxy_row_to_matrix(proxy_row: pd.Series, names: list[str]) -> np.ndarray:
    d = len(names)
    mat = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(i, d):
            value = float(proxy_row[f"s[{names[i]},{names[j]}]"])
            mat[i, j] = value
            mat[j, i] = value
    return mat


def rebase_block_exogenous_spec_final(
    spec: CovariateDrivenStochasticCovarianceSpecification,
    workspace: ExtendedIntensityCalibrationWorkspace,
    *,
    index: pd.DatetimeIndex,
) -> tuple[CovariateDrivenStochasticCovarianceSpecification, pd.DatetimeIndex]:
    idx = _regularize_hourly_index(pd.DatetimeIndex(index))
    p = int(spec.joint_var_fit.lags)
    if len(idx) <= p:
        raise ValueError("The comparison index must contain more timestamps than the VAR lag order.")

    spot_variant = workspace.base_workspace.spot_variants["mle_notebook_full"]
    start_time = max(pd.Timestamp(idx[0]), pd.Timestamp(spec.wishart.hourly_proxy.index[0]))
    compare_index = idx[idx >= start_time]
    if len(compare_index) <= p:
        raise ValueError("The covariance-proxy overlap is too short for block-exogenous simulation.")
    compare_start = pd.Timestamp(compare_index[0])
    compare_index = pd.date_range(compare_start, pd.Timestamp(idx[-1]), freq="1h")

    state_hist = spec.joint_var_fit.state_frame.loc[:compare_start].dropna()
    spike_hist = pd.Series(spot_variant.spike_state).loc[:compare_start].dropna()
    if len(state_hist) < p:
        raise ValueError("Not enough lag history is available to rebase the block-exogenous simulation.")

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
        initial_history=state_hist.iloc[-p:].to_numpy(dtype=float),
        initial_spike_state=float(spike_hist.iloc[-1]) if len(spike_hist) else 0.0,
        intensity=spec.intensity,
    )
    return rebased, compare_index


def simulate_constant_vs_stochastic_block_exogenous_final(
    spec: CovariateDrivenStochasticCovarianceSpecification,
    workspace: ExtendedIntensityCalibrationWorkspace,
    *,
    index: pd.DatetimeIndex,
    num_paths: int,
    seed: Optional[int] = None,
) -> dict[str, dict]:
    rebased, compare_index = rebase_block_exogenous_spec_final(spec, workspace, index=index)
    baseline = simulate_intensity_paths_constant_covariance_extended_model(
        rebased,
        index=compare_index,
        num_paths=int(num_paths),
        seed=seed,
    )
    stochastic = simulate_intensity_paths_stochastic_covariance_extended_model(
        rebased,
        index=compare_index,
        num_paths=int(num_paths),
        seed=None if seed is None else int(seed) + 1,
    )
    return {"baseline": baseline, "stochastic": stochastic, "compare_index": compare_index}


def block_exogenous_diagnostics_table_final(
    spec: CovariateDrivenStochasticCovarianceSpecification,
) -> pd.DataFrame:
    fit = spec.joint_var_fit
    diag = block_exogenous_diagnostics_final(fit)
    rows: list[dict[str, float | str | int]] = [
        {
            "quantity": "spectral_radius",
            "value": float(diag.spectral_radius),
            "interpretation": "Companion spectral radius; stationarity requires value below one.",
        },
        {
            "quantity": "max_abs_forbidden_driver_from_spot",
            "value": float(diag.max_abs_forbidden_driver_from_spot),
            "interpretation": "Maximum absolute coefficient from lagged spot state into any physical driver equation.",
        },
    ]
    for name in diag.names[1:]:
        rows.append(
            {
                "quantity": f"l2_spot_from_{name}",
                "value": float(diag.spot_from_driver_l2[name]),
                "interpretation": "L2 norm of the allowed driver-to-price lag block.",
            }
        )
        rows.append(
            {
                "quantity": f"l2_{name}_from_spot",
                "value": float(diag.driver_from_spot_l2[name]),
                "interpretation": "L2 norm of the forbidden price-to-driver lag block; should be zero.",
            }
        )
    return pd.DataFrame(rows)


def block_exogenous_parameter_table_final(
    spec: CovariateDrivenStochasticCovarianceSpecification,
) -> pd.DataFrame:
    fit = spec.joint_var_fit
    rows: list[dict[str, float | str]] = []
    for lag in range(int(fit.lags)):
        coef = np.asarray(fit.coefs[lag], dtype=float)
        for i, target in enumerate(fit.names):
            for j, source in enumerate(fit.names):
                rows.append(
                    {
                        "parameter": f"phi_block_exog[{target},{source},lag{lag + 1}]",
                        "value": float(coef[i, j]),
                        "block": "block_exogenous_VARX",
                    }
                )
    rows.append({"parameter": "var_spectral_radius", "value": float(fit.spectral_radius), "block": "companion"})
    rows.append({"parameter": "active_dimension", "value": float(len(fit.names)), "block": "companion"})
    wishart = wishart_parameter_table_extended_model(spec.wishart)
    return pd.concat([pd.DataFrame(rows), wishart], ignore_index=True, sort=False)


__all__ = [
    "BlockExogenousVARXDiagnosticsFinal",
    "block_exogenous_diagnostics_final",
    "block_exogenous_diagnostics_table_final",
    "block_exogenous_parameter_table_final",
    "build_block_exogenous_covariance_spec_final",
    "fit_block_exogenous_varx_final",
    "joint_state_frame_block_exogenous_final",
    "rebase_block_exogenous_spec_final",
    "simulate_constant_vs_stochastic_block_exogenous_final",
]
