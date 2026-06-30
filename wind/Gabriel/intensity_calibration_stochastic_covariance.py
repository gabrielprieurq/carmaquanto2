from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd

try:
    from .intensity_calibration import IntensityCalibrationWorkspace
    from .intensity_model import IntensityModelSpecification
    from .intensity_model_stochastic_covariance import (
        IntensityStochasticCovarianceSpecification,
        VAR24CompanionFit,
        build_stochastic_covariance_spec,
        fit_var24_companion,
        simulate_intensity_paths_constant_covariance,
        simulate_intensity_paths_stochastic_covariance,
    )
    from .intensity_wishart_stochastic_covariance import (
        WishartFullCovarianceCalibration,
        fit_wishart_full_covariance_model,
        wishart_parameter_table,
    )
except ImportError:
    from intensity_calibration import IntensityCalibrationWorkspace
    from intensity_model import IntensityModelSpecification
    from intensity_model_stochastic_covariance import (
        IntensityStochasticCovarianceSpecification,
        VAR24CompanionFit,
        build_stochastic_covariance_spec,
        fit_var24_companion,
        simulate_intensity_paths_constant_covariance,
        simulate_intensity_paths_stochastic_covariance,
    )
    from intensity_wishart_stochastic_covariance import (
        WishartFullCovarianceCalibration,
        fit_wishart_full_covariance_model,
        wishart_parameter_table,
    )


def aligned_state_pair(
    workspace: IntensityCalibrationWorkspace,
    *,
    variant: str = "mle_notebook",
) -> pd.DataFrame:
    if variant == "paper":
        spot_variant = workspace.spot_variants["paper"]
        wind_variant = workspace.wind_variants["paper"]
    elif variant in {"mle_notebook", "harmonic_full"}:
        spot_variant = workspace.spot_variants["mle_notebook_full"]
        wind_variant = workspace.wind_variants["mle_notebook"]
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    common_index = pd.DatetimeIndex(spot_variant.continuous_series.index).intersection(
        pd.DatetimeIndex(wind_variant.seasonality.residual.index)
    )
    return pd.DataFrame(
        {
            "spot_state": pd.Series(spot_variant.continuous_series).loc[common_index],
            "wind_state": pd.Series(wind_variant.seasonality.residual).loc[common_index],
        }
    ).dropna()


def aligned_innovation_pair(
    workspace: IntensityCalibrationWorkspace,
    *,
    variant: str = "mle_notebook",
) -> pd.DataFrame:
    state_pair = aligned_state_pair(workspace, variant=variant)
    return fit_var24_companion(state_pair).residual


def select_base_model_spec(
    workspace: IntensityCalibrationWorkspace,
    *,
    variant: str = "mle_notebook",
) -> IntensityModelSpecification:
    if variant == "paper":
        return workspace.model_specs["paper"]
    if variant in {"mle_notebook", "harmonic_full"}:
        return workspace.model_specs["mle_notebook"]
    raise ValueError(f"Unsupported variant: {variant}")


@dataclass
class StochasticCovarianceExtensionWorkspace:
    base_workspace: IntensityCalibrationWorkspace
    base_variant: str
    state_pair: pd.DataFrame
    innovation_pair: pd.DataFrame
    companion_fit: VAR24CompanionFit
    reference_continuous_wishart: WishartFullCovarianceCalibration
    wishart: WishartFullCovarianceCalibration
    base_spec: IntensityModelSpecification
    stochastic_spec: IntensityStochasticCovarianceSpecification


def _select_variants_for_index(
    extension: StochasticCovarianceExtensionWorkspace,
) -> tuple[object, object]:
    if extension.base_variant == "paper":
        return (
            extension.base_workspace.spot_variants["paper"],
            extension.base_workspace.wind_variants["paper"],
        )
    return (
        extension.base_workspace.spot_variants["mle_notebook_full"],
        extension.base_workspace.wind_variants["mle_notebook"],
    )


def _build_rebased_spec_for_comparison(
    extension: StochasticCovarianceExtensionWorkspace,
    *,
    index: pd.DatetimeIndex,
) -> tuple[IntensityStochasticCovarianceSpecification, pd.DatetimeIndex]:
    idx = pd.DatetimeIndex(index)
    p = int(extension.companion_fit.lags)
    if len(idx) <= p:
        raise ValueError("The comparison index must contain more than 24 timestamps.")

    spot_variant, _ = _select_variants_for_index(extension)
    start_time = max(pd.Timestamp(idx[0]), pd.Timestamp(extension.wishart.hourly_proxy.index[0]))
    compare_index = idx[idx >= start_time]
    if len(compare_index) <= p:
        raise ValueError("The overlap between the requested index and the covariance-proxy support is too short.")

    compare_start = pd.Timestamp(compare_index[0])
    compare_index = pd.date_range(compare_start, pd.Timestamp(idx[-1]), freq="1h")
    state_hist = extension.state_pair.loc[:compare_start].dropna()
    spike_hist = pd.Series(spot_variant.spike_state).loc[:compare_start].dropna()
    if len(state_hist) < p:
        raise ValueError("Not enough lag history is available to rebase the comparison simulation.")

    initial_history = state_hist.iloc[-p:].to_numpy(dtype=float)
    initial_spike_state = float(spike_hist.iloc[-1]) if len(spike_hist) else 0.0

    proxy_row = extension.wishart.hourly_proxy.loc[:compare_start].iloc[-1]
    sigma0_compare = np.array(
        [
            [float(proxy_row["s11"]), float(proxy_row["s12"])],
            [float(proxy_row["s12"]), float(proxy_row["s22"])],
        ],
        dtype=float,
    )
    scale_matrix = np.asarray(extension.wishart.scale_matrix, dtype=float)
    inv_scale = np.linalg.inv(scale_matrix)
    sigma0_compare_standardized = inv_scale @ sigma0_compare @ inv_scale

    wishart_rebased = replace(
        extension.wishart,
        sigma0_forecast=sigma0_compare,
        sigma0_forecast_standardized=sigma0_compare_standardized,
    )
    stochastic_spec = build_stochastic_covariance_spec(
        extension.base_spec,
        extension.companion_fit,
        wishart_rebased,
        initial_history=initial_history,
        initial_spike_state=initial_spike_state,
        name=extension.stochastic_spec.name,
    )
    return stochastic_spec, compare_index


def run_stochastic_covariance_extension(
    workspace: IntensityCalibrationWorkspace,
    *,
    variant: str = "mle_notebook",
    window_map: Optional[dict[str, int]] = None,
    stride_hours: int = 24,
    sim_paths: int = 128,
    random_seed: int = 20260416,
    max_norm: float = 0.95,
    b_convention: str = "simulator",
    eps: float = 1e-8,
) -> StochasticCovarianceExtensionWorkspace:
    state_pair = aligned_state_pair(workspace, variant=variant)
    companion_fit = fit_var24_companion(state_pair)
    innovation_pair = companion_fit.residual

    reference_continuous_wishart = fit_wishart_full_covariance_model(
        state_pair.rename(columns={"spot_state": "spot_innovation", "wind_state": "wind_innovation"}),
        target_name=f"{variant}_continuous_pair",
        window_map=window_map,
        stride_hours=stride_hours,
        sim_paths=sim_paths,
        random_seed=random_seed,
        b_convention=b_convention,
        max_norm=max_norm,
        eps=eps,
    )
    wishart = fit_wishart_full_covariance_model(
        innovation_pair,
        target_name=f"{variant}_var24_innovations",
        window_map=window_map,
        stride_hours=stride_hours,
        sim_paths=sim_paths,
        random_seed=random_seed + 1000,
        b_convention=b_convention,
        max_norm=max_norm,
        eps=eps,
    )
    base_spec = select_base_model_spec(workspace, variant=variant)
    stochastic_spec = build_stochastic_covariance_spec(
        base_spec,
        companion_fit,
        wishart,
        name=f"{variant}_stochastic_covariance",
    )
    return StochasticCovarianceExtensionWorkspace(
        base_workspace=workspace,
        base_variant=str(variant),
        state_pair=state_pair,
        innovation_pair=innovation_pair,
        companion_fit=companion_fit,
        reference_continuous_wishart=reference_continuous_wishart,
        wishart=wishart,
        base_spec=base_spec,
        stochastic_spec=stochastic_spec,
    )


def simulate_constant_vs_stochastic_covariance(
    extension: StochasticCovarianceExtensionWorkspace,
    *,
    index: pd.DatetimeIndex,
    num_paths: int,
    seed: Optional[int] = None,
) -> dict[str, dict[str, np.ndarray]]:
    stochastic_spec, compare_index = _build_rebased_spec_for_comparison(
        extension,
        index=pd.DatetimeIndex(index),
    )
    baseline = simulate_intensity_paths_constant_covariance(
        stochastic_spec,
        index=compare_index,
        num_paths=num_paths,
        seed=seed,
    )
    stochastic = simulate_intensity_paths_stochastic_covariance(
        stochastic_spec,
        index=compare_index,
        num_paths=num_paths,
        seed=None if seed is None else int(seed) + 1,
    )
    return {"baseline": baseline, "stochastic": stochastic, "compare_index": compare_index}


def stochastic_covariance_summary_table(extension: StochasticCovarianceExtensionWorkspace) -> pd.DataFrame:
    summary = extension.wishart.summary_table.copy()
    summary["selected"] = summary["window"].astype(str).eq(extension.wishart.selected_window)
    return summary


def leverage_summary_table(extension: StochasticCovarianceExtensionWorkspace) -> pd.DataFrame:
    return extension.wishart.leverage_table.copy()


def stochastic_covariance_parameter_table(extension: StochasticCovarianceExtensionWorkspace) -> pd.DataFrame:
    out = wishart_parameter_table(extension.wishart)
    out = pd.concat(
        [
            out,
            pd.DataFrame(
                [
                    {"parameter": "var_spectral_radius", "value": float(extension.companion_fit.spectral_radius), "block": "companion"},
                    {"parameter": "var_resid_corr", "value": float(extension.innovation_pair.corr().iloc[0, 1]), "block": "companion"},
                    {"parameter": "state_corr", "value": float(extension.state_pair.corr().iloc[0, 1]), "block": "companion"},
                ]
            ),
        ],
        ignore_index=True,
    )
    return out


__all__ = [
    "StochasticCovarianceExtensionWorkspace",
    "aligned_innovation_pair",
    "aligned_state_pair",
    "leverage_summary_table",
    "run_stochastic_covariance_extension",
    "select_base_model_spec",
    "simulate_constant_vs_stochastic_covariance",
    "stochastic_covariance_parameter_table",
    "stochastic_covariance_summary_table",
]
