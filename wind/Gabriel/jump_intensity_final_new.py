from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from .intensity_ar import fit_ar24_exact_mle
    from .intensity_block_exogenous_final import (
        block_exogenous_diagnostics_table_final,
        fit_block_exogenous_varx_final,
    )
    from .intensity_data import logit_capacity_factor
    from .intensity_final import FinalIntensityCalibrationResult
    from .intensity_intensity import (
        KernelIntensityEstimate,
        TwoStateNegativeIntensityFit,
        estimate_intensity_kernel_paper,
        fit_two_state_negative_intensity,
    )
    from .intensity_intensity_extended_model import (
        CovariateIntensityCalibrationResult,
        build_empirical_state_transform,
    )
    from .intensity_model_stochastic_covariance_extended_model import (
        ActiveCovariateLatentSpecificationExtendedModel,
        CovariateDrivenStochasticCovarianceSpecification,
        simulate_intensity_paths_constant_covariance_extended_model,
        simulate_intensity_paths_stochastic_covariance_extended_model,
    )
    from .intensity_spikes import detect_spikes_paper, reconstruct_spike_path
    from .intensity_wishart_stochastic_covariance_extended_model import (
        WishartFullCovarianceCalibrationExtendedModel,
        fit_wishart_full_covariance_model_extended_model,
        wishart_parameter_table_extended_model,
    )
except ImportError:
    from intensity_ar import fit_ar24_exact_mle
    from intensity_block_exogenous_final import (
        block_exogenous_diagnostics_table_final,
        fit_block_exogenous_varx_final,
    )
    from intensity_data import logit_capacity_factor
    from intensity_final import FinalIntensityCalibrationResult
    from intensity_intensity import (
        KernelIntensityEstimate,
        TwoStateNegativeIntensityFit,
        estimate_intensity_kernel_paper,
        fit_two_state_negative_intensity,
    )
    from intensity_intensity_extended_model import (
        CovariateIntensityCalibrationResult,
        build_empirical_state_transform,
    )
    from intensity_model_stochastic_covariance_extended_model import (
        ActiveCovariateLatentSpecificationExtendedModel,
        CovariateDrivenStochasticCovarianceSpecification,
        simulate_intensity_paths_constant_covariance_extended_model,
        simulate_intensity_paths_stochastic_covariance_extended_model,
    )
    from intensity_spikes import detect_spikes_paper, reconstruct_spike_path
    from intensity_wishart_stochastic_covariance_extended_model import (
        WishartFullCovarianceCalibrationExtendedModel,
        fit_wishart_full_covariance_model_extended_model,
        wishart_parameter_table_extended_model,
    )


@dataclass(frozen=True)
class CorrectedJumpIntensityConfigNew:
    threshold_scale: float = 2.5
    low_price_quantile: float = 0.20
    high_price_quantile: float = 0.80
    wind_alignment: str = "same_hour"
    negative_bandwidth: float = 0.25
    positive_bandwidth: float = 0.30
    grid_low_quantile: float = 0.01
    grid_high_quantile: float = 0.995
    grid_points: int = 181


@dataclass
class CorrectedJumpIntensityCalibrationNew:
    config: CorrectedJumpIntensityConfigNew
    event_frame: pd.DataFrame
    corrected_jump_increment: pd.Series
    corrected_filtered_mask: pd.Series
    corrected_spike_state: pd.Series
    intensity: CovariateIntensityCalibrationResult
    positive_jump_sizes: np.ndarray
    negative_jump_sizes: np.ndarray
    event_summary: pd.DataFrame
    intensity_curves: pd.DataFrame


@dataclass
class JumpIntensityBlockExogenousWorkspaceNew:
    final_result: FinalIntensityCalibrationResult
    corrected_intensity: CorrectedJumpIntensityCalibrationNew
    state_frame: pd.DataFrame
    corrected_spot_state: pd.Series
    corrected_spot_ar_series: pd.Series
    corrected_spot_ar_fit: object
    specification: CovariateDrivenStochasticCovarianceSpecification


def classify_corrected_spikes_new(
    spot: pd.Series,
    wind_cf: pd.Series,
    *,
    config: CorrectedJumpIntensityConfigNew,
) -> pd.DataFrame:
    detection = detect_spikes_paper(spot, threshold_scale=float(config.threshold_scale))
    idx = pd.DatetimeIndex(detection.jump_mask.index)
    s = pd.Series(spot, copy=True).astype(float).reindex(idx)
    w = pd.Series(wind_cf, copy=True).astype(float).reindex(idx).interpolate(method="time").ffill().bfill()
    if config.wind_alignment == "lag1":
        driver = w.shift(1)
    elif config.wind_alignment == "same_hour":
        driver = w
    elif config.wind_alignment == "average_lag_same":
        driver = 0.5 * (w.shift(1) + w)
    else:
        raise ValueError(f"Unsupported wind_alignment: {config.wind_alignment}")

    jump_mask = detection.jump_mask.reindex(idx).fillna(False).astype(bool)
    jump_increment = detection.jump_increment.reindex(idx).fillna(0.0).astype(float)
    low_cut = float(s.quantile(float(config.low_price_quantile)))
    high_cut = float(s.quantile(float(config.high_price_quantile)))

    frame = pd.DataFrame(
        {
            "spot": s,
            "spot_before": s.shift(1),
            "spot_return": jump_increment,
            "wind_cf": w,
            "driver": driver,
            "jump_candidate": jump_mask,
            "all_negative_return_jump": jump_mask & (jump_increment < 0.0),
            "all_positive_return_jump": jump_mask & (jump_increment > 0.0),
            "corrected_negative_low_price_spike": jump_mask & (jump_increment < 0.0) & (s <= low_cut),
            "corrected_positive_high_price_spike": jump_mask & (jump_increment > 0.0) & (s >= high_cut),
        },
        index=idx,
    ).dropna(subset=["driver"])
    frame.attrs["low_price_cut"] = low_cut
    frame.attrs["high_price_cut"] = high_cut
    frame.attrs["threshold_by_year"] = detection.threshold_by_year
    return frame


def _corrected_filtered_mask_from_events(event_frame: pd.DataFrame, *, follow_up_drop: int = 10) -> pd.Series:
    event = (
        event_frame["corrected_negative_low_price_spike"].astype(bool)
        | event_frame["corrected_positive_high_price_spike"].astype(bool)
    )
    mask = pd.Series(False, index=event_frame.index, name="corrected_filtered_mask")
    positions = np.flatnonzero(event.to_numpy(dtype=bool))
    for pos in positions:
        hi = min(pos + int(follow_up_drop) + 1, len(mask))
        mask.iloc[pos:hi] = True
    return mask


def _kernel_curve_frame(kernel: KernelIntensityEstimate, *, side: str, label: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "side": side,
            "label": label,
            "wind_driver": np.asarray(kernel.grid, dtype=float),
            "intensity_year_inv": np.asarray(kernel.values, dtype=float),
            "bandwidth": float(kernel.bandwidth),
            "polynomial_degree": int(getattr(kernel, "polynomial_degree", 0)),
        }
    )


def calibrate_corrected_wind_jump_intensity_new(
    *,
    spot: pd.Series,
    wind_cf: pd.Series,
    spot_residual_for_rho: pd.Series,
    wind_residual_for_rho: pd.Series,
    beta: float,
    config: Optional[CorrectedJumpIntensityConfigNew] = None,
) -> CorrectedJumpIntensityCalibrationNew:
    cfg = config or CorrectedJumpIntensityConfigNew()
    event_frame = classify_corrected_spikes_new(spot, wind_cf, config=cfg)
    grid_low = max(0.0, float(event_frame["driver"].quantile(float(cfg.grid_low_quantile))))
    grid_high = min(0.99, float(event_frame["driver"].quantile(float(cfg.grid_high_quantile))))
    if not np.isfinite(grid_low) or not np.isfinite(grid_high) or grid_high <= grid_low:
        grid_low, grid_high = 0.0, 0.99
    grid = np.linspace(grid_low, grid_high, int(cfg.grid_points))

    neg_events = event_frame["corrected_negative_low_price_spike"].astype(float)
    pos_events = event_frame["corrected_positive_high_price_spike"].astype(float)
    neg_kernel = estimate_intensity_kernel_paper(
        covariate=event_frame["driver"],
        events=neg_events,
        grid=grid,
        h_min=0.0225,
        fixed_bandwidth=float(cfg.negative_bandwidth),
    )
    pos_kernel = estimate_intensity_kernel_paper(
        covariate=event_frame["driver"],
        events=pos_events,
        grid=grid,
        h_min=0.0089,
        fixed_bandwidth=float(cfg.positive_bandwidth),
    )
    neg_two_state = fit_two_state_negative_intensity(neg_kernel, monotonic="increasing")
    lambda_pos = float(pos_events.sum()) / max(len(event_frame) / 8760.0, 1e-12)

    common = pd.DatetimeIndex(spot_residual_for_rho.index).intersection(pd.DatetimeIndex(wind_residual_for_rho.index))
    rho = float(
        np.corrcoef(
            pd.Series(spot_residual_for_rho).loc[common].to_numpy(dtype=float),
            pd.Series(wind_residual_for_rho).loc[common].to_numpy(dtype=float),
        )[0, 1]
    )
    if not np.isfinite(rho):
        rho = 0.0

    corrected_jump_increment = pd.Series(0.0, index=event_frame.index, name="corrected_jump_increment")
    selected = event_frame["corrected_negative_low_price_spike"] | event_frame["corrected_positive_high_price_spike"]
    corrected_jump_increment.loc[selected] = event_frame.loc[selected, "spot_return"].to_numpy(dtype=float)
    corrected_filtered_mask = _corrected_filtered_mask_from_events(event_frame)
    corrected_spike_state = reconstruct_spike_path(corrected_jump_increment, beta=float(beta)).rename("corrected_spike_state")

    negative_jump_sizes = corrected_jump_increment.loc[corrected_jump_increment < 0.0].to_numpy(dtype=float)
    positive_jump_sizes = corrected_jump_increment.loc[corrected_jump_increment > 0.0].to_numpy(dtype=float)
    low_state = event_frame["driver"] <= float(neg_two_state.wp_threshold)
    high_state = event_frame["driver"] > float(neg_two_state.wp_threshold)
    event_summary = pd.DataFrame(
        [
            {
                "event_type": "negative_low_price",
                "count": int(neg_events.sum()),
                "intensity_low_year_inv": float(neg_two_state.lambda_neg_low),
                "intensity_high_year_inv": float(neg_two_state.lambda_neg_high),
                "threshold": float(neg_two_state.wp_threshold),
                "n_low_state_obs": int(low_state.sum()),
                "n_high_state_obs": int(high_state.sum()),
                "n_low_state_events": int((neg_events.astype(bool) & low_state).sum()),
                "n_high_state_events": int((neg_events.astype(bool) & high_state).sum()),
            },
            {
                "event_type": "positive_high_price",
                "count": int(pos_events.sum()),
                "constant_intensity_year_inv": float(lambda_pos),
                "threshold": np.nan,
                "n_low_state_obs": int(low_state.sum()),
                "n_high_state_obs": int(high_state.sum()),
                "n_low_state_events": int((pos_events.astype(bool) & low_state).sum()),
                "n_high_state_events": int((pos_events.astype(bool) & high_state).sum()),
            },
        ]
    )

    event_table = event_summary.copy()
    event_table["covariate"] = "wind_state"
    event_table["low_price_cut"] = float(event_frame.attrs["low_price_cut"])
    event_table["high_price_cut"] = float(event_frame.attrs["high_price_cut"])
    intensity = CovariateIntensityCalibrationResult(
        covariate_name="wind_state",
        covariate_label="Wind capacity factor",
        state_series=pd.Series(event_frame["driver"], copy=True).astype(float),
        driver_series=pd.Series(event_frame["driver"], copy=True).astype(float),
        driver_mode="raw",
        driver_label="Wind capacity factor corrected low/high price spike driver",
        negative_kernel=neg_kernel,
        positive_kernel=pos_kernel,
        negative_two_state=TwoStateNegativeIntensityFit(
            lambda_neg_low=float(neg_two_state.lambda_neg_low),
            lambda_neg_high=float(neg_two_state.lambda_neg_high),
            wp_threshold=float(neg_two_state.wp_threshold),
            loss=float(neg_two_state.loss),
        ),
        lambda_pos=float(lambda_pos),
        rho=float(rho),
        event_table=event_table,
        interval_low=float(grid.min()),
        interval_high=float(grid.max()),
        negative_estimator_type="corrected_low_price_paper_kernel_raw_wind_new",
        positive_estimator_type="corrected_high_price_paper_kernel_raw_wind_new",
        negative_event_count=int(neg_events.sum()),
        positive_event_count=int(pos_events.sum()),
        calibration_note=(
            "Corrected jump intensity: return-reversal candidates are split into low-price negative "
            "spikes and high-price positive spikes before estimating the wind-conditioned intensity."
        ),
    )
    curves = pd.concat(
        [
            _kernel_curve_frame(neg_kernel, side="negative_low_price", label="corrected_negative_low_price"),
            _kernel_curve_frame(pos_kernel, side="positive_high_price", label="corrected_positive_high_price"),
        ],
        ignore_index=True,
    )
    return CorrectedJumpIntensityCalibrationNew(
        config=cfg,
        event_frame=event_frame,
        corrected_jump_increment=corrected_jump_increment,
        corrected_filtered_mask=corrected_filtered_mask,
        corrected_spike_state=corrected_spike_state,
        intensity=intensity,
        positive_jump_sizes=positive_jump_sizes,
        negative_jump_sizes=negative_jump_sizes,
        event_summary=event_summary,
        intensity_curves=curves,
    )


def _spot_level_seasonality(final_result: FinalIntensityCalibrationResult, index: pd.DatetimeIndex) -> pd.Series:
    idx = pd.DatetimeIndex(index)
    if final_result.spot_seasonality is not None:
        season = final_result.spot_seasonality.level_seasonality.reindex(idx)
    else:
        season = final_result.base_workspace.spot_variants["mle_notebook_full"].seasonality.fitted.reindex(idx)
    season = season.interpolate(method="time").ffill().bfill()
    return pd.Series(season.to_numpy(dtype=float), index=idx, name="spot_level_seasonality_new")


def build_jump_intensity_block_exogenous_workspace_new(
    final_result: FinalIntensityCalibrationResult,
    *,
    intensity_config: Optional[CorrectedJumpIntensityConfigNew] = None,
    window_map: Optional[dict[str, int]] = None,
    stride_hours: int = 24,
    sim_paths: int = 128,
    random_seed: int = 20260507,
    max_norm: float = 0.95,
    b_convention: str = "simulator",
    eps: float = 1e-8,
) -> JumpIntensityBlockExogenousWorkspaceNew:
    base_workspace = final_result.base_workspace
    data = base_workspace.data
    spot = pd.Series(data.spot, copy=True).astype(float)
    wind_cf = pd.Series(data.wind_cf, copy=True).astype(float).reindex(spot.index).interpolate(method="time").ffill().bfill()
    spot_season = _spot_level_seasonality(final_result, spot.index)
    wind_variant = base_workspace.wind_variants["mle_notebook"]
    wind_state = pd.Series(wind_variant.seasonality.residual, name="wind_state").reindex(spot.index).dropna()
    common = pd.DatetimeIndex(spot.index).intersection(pd.DatetimeIndex(wind_state.index))
    spot = spot.loc[common]
    wind_cf = wind_cf.loc[common]
    spot_season = spot_season.loc[common]
    wind_state = wind_state.loc[common]

    preliminary_spot_state = pd.Series(
        base_workspace.spot_variants["mle_notebook_full"].continuous_series,
        name="spot_state_old",
    )
    corrected = calibrate_corrected_wind_jump_intensity_new(
        spot=spot,
        wind_cf=wind_cf,
        spot_residual_for_rho=preliminary_spot_state,
        wind_residual_for_rho=wind_variant.ar_fit.residual,
        beta=float(base_workspace.spike_detection.beta),
        config=intensity_config,
    )
    corrected_spike_state = corrected.corrected_spike_state.reindex(common).interpolate(method="time").ffill().bfill()
    corrected_spot_state = (spot - spot_season - corrected_spike_state).rename("spot_state")
    ar_mask = ~corrected.corrected_filtered_mask.reindex(common).fillna(False)
    corrected_spot_ar_series = corrected_spot_state.loc[ar_mask].dropna().rename("corrected_spot_ar_series")
    corrected_spot_ar_fit = fit_ar24_exact_mle(corrected_spot_ar_series)

    state_frame = pd.concat([corrected_spot_state, wind_state.rename("wind_state")], axis=1).dropna().sort_index()
    joint_var_fit = fit_block_exogenous_varx_final(state_frame)
    wishart = fit_wishart_full_covariance_model_extended_model(
        joint_var_fit.residual,
        target_name="wind_state_block_exogenous_corrected_jump_new",
        window_map=window_map,
        stride_hours=int(stride_hours),
        sim_paths=int(sim_paths),
        random_seed=int(random_seed),
        b_convention=b_convention,
        max_norm=float(max_norm),
        eps=float(eps),
    )

    wind_logit_series = pd.Series(logit_capacity_factor(wind_cf), index=wind_cf.index, name="wind_logit")
    wind_state_transform = build_empirical_state_transform(
        wind_logit_series,
        name="wind_logit_empirical_state_transform_new",
        lower_clip=0.01,
        upper_clip=0.99,
    )
    wind_covariate_spec = ActiveCovariateLatentSpecificationExtendedModel(
        name="wind_state",
        label="Wind capacity factor",
        transform="raw_logit",
        seasonality=wind_variant.seasonality,
        ar_fit=wind_variant.ar_fit,
        state_transform=wind_state_transform,
    )
    base_spec_old = base_workspace.model_specs["mle_notebook"]
    base_spec_new = replace(
        base_spec_old,
        name="jump_intensity_corrected_wind_block_exogenous_new",
        positive_jump_sizes=np.asarray(corrected.positive_jump_sizes, dtype=float),
        negative_jump_sizes=np.asarray(corrected.negative_jump_sizes, dtype=float),
        intensity=corrected.intensity,
        initial_spike_state=float(corrected.corrected_spike_state.iloc[-1]),
    )
    p = int(joint_var_fit.lags)
    spec = CovariateDrivenStochasticCovarianceSpecification(
        name="wind_state_block_exogenous_stochastic_covariance_corrected_jump_new",
        base_spec=base_spec_new,
        primary_covariate_name="wind_state",
        primary_covariate_label="Wind capacity factor",
        active_covariates=["wind_state"],
        covariate_specs={"wind_state": wind_covariate_spec},
        joint_var_fit=joint_var_fit,
        wishart=wishart,
        initial_history=state_frame.iloc[-p:].to_numpy(dtype=float),
        initial_spike_state=float(corrected.corrected_spike_state.iloc[-1]),
        intensity=corrected.intensity,
    )
    return JumpIntensityBlockExogenousWorkspaceNew(
        final_result=final_result,
        corrected_intensity=corrected,
        state_frame=state_frame,
        corrected_spot_state=corrected_spot_state,
        corrected_spot_ar_series=corrected_spot_ar_series,
        corrected_spot_ar_fit=corrected_spot_ar_fit,
        specification=spec,
    )


def _proxy_row_to_matrix_new(proxy_row: pd.Series, names: list[str]) -> np.ndarray:
    d = len(names)
    mat = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(i, d):
            value = float(proxy_row[f"s[{names[i]},{names[j]}]"])
            mat[i, j] = value
            mat[j, i] = value
    return mat


def rebase_jump_intensity_block_exogenous_spec_new(
    workspace: JumpIntensityBlockExogenousWorkspaceNew,
    *,
    index: pd.DatetimeIndex,
) -> tuple[CovariateDrivenStochasticCovarianceSpecification, pd.DatetimeIndex]:
    spec = workspace.specification
    idx = pd.DatetimeIndex(index).sort_values().unique()
    if len(idx) <= int(spec.joint_var_fit.lags):
        raise ValueError("The simulation index must contain more timestamps than the VAR lag order.")
    start_time = max(pd.Timestamp(idx[0]), pd.Timestamp(spec.wishart.hourly_proxy.index[0]))
    compare_index = idx[idx >= start_time]
    if len(compare_index) <= int(spec.joint_var_fit.lags):
        raise ValueError("The covariance-proxy overlap is too short for simulation.")
    compare_start = pd.Timestamp(compare_index[0])
    compare_index = pd.date_range(compare_start, pd.Timestamp(idx[-1]), freq="1h")

    state_hist = workspace.state_frame.loc[:compare_start].dropna()
    spike_hist = workspace.corrected_intensity.corrected_spike_state.loc[:compare_start].dropna()
    p = int(spec.joint_var_fit.lags)
    if len(state_hist) < p:
        raise ValueError("Not enough lag history is available to rebase the corrected jump model.")

    proxy_row = spec.wishart.hourly_proxy.loc[:compare_start].iloc[-1]
    sigma0_compare = _proxy_row_to_matrix_new(proxy_row, spec.wishart.variable_names)
    scale_matrix = np.asarray(spec.wishart.scale_matrix, dtype=float)
    inv_scale = np.linalg.inv(scale_matrix)
    sigma0_compare_standardized = inv_scale @ sigma0_compare @ inv_scale
    rebased = replace(
        spec,
        initial_history=state_hist.iloc[-p:].to_numpy(dtype=float),
        initial_spike_state=float(spike_hist.iloc[-1]) if len(spike_hist) else 0.0,
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
    )
    return rebased, compare_index


def simulate_jump_intensity_block_exogenous_new(
    workspace: JumpIntensityBlockExogenousWorkspaceNew,
    *,
    index: pd.DatetimeIndex,
    num_paths: int,
    seed: Optional[int] = None,
) -> dict[str, dict]:
    rebased, compare_index = rebase_jump_intensity_block_exogenous_spec_new(workspace, index=index)
    constant = simulate_intensity_paths_constant_covariance_extended_model(
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
    return {"constant": constant, "stochastic": stochastic, "compare_index": compare_index}


def jump_intensity_component_tables_new(
    workspace: JumpIntensityBlockExogenousWorkspaceNew,
    *,
    output_dir: Optional[str | Path] = None,
) -> dict[str, pd.DataFrame]:
    spec = workspace.specification
    tables = {
        "event_summary": workspace.corrected_intensity.event_summary.copy(),
        "intensity_curves": workspace.corrected_intensity.intensity_curves.copy(),
        "block_exogenous_diagnostics": block_exogenous_diagnostics_table_final(spec),
        "innovation_corr": pd.DataFrame(
            np.corrcoef(spec.joint_var_fit.residual.to_numpy(dtype=float), rowvar=False),
            index=spec.joint_var_fit.names,
            columns=spec.joint_var_fit.names,
        ),
        "wishart_parameters": wishart_parameter_table_extended_model(spec.wishart),
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(out / f"{name}.csv", index=name == "innovation_corr")
    return tables


__all__ = [
    "CorrectedJumpIntensityCalibrationNew",
    "CorrectedJumpIntensityConfigNew",
    "JumpIntensityBlockExogenousWorkspaceNew",
    "build_jump_intensity_block_exogenous_workspace_new",
    "calibrate_corrected_wind_jump_intensity_new",
    "classify_corrected_spikes_new",
    "jump_intensity_component_tables_new",
    "rebase_jump_intensity_block_exogenous_spec_new",
    "simulate_jump_intensity_block_exogenous_new",
]
