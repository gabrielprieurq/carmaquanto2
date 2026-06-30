from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLE_Wishart import WishartMLE, WishartMLEFit


def _symmetrize(x: np.ndarray) -> np.ndarray:
    return 0.5 * (x + np.swapaxes(x, -1, -2))


def _ensure_spd_batch(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    mats = _symmetrize(np.asarray(x, dtype=float))
    evals, evecs = np.linalg.eigh(mats)
    evals = np.clip(evals, eps, None)
    return _symmetrize((evecs * evals[..., None, :]) @ np.swapaxes(evecs, -1, -2))


def _sqrtm_spd_batch(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    mats = _ensure_spd_batch(x, eps=eps)
    evals, evecs = np.linalg.eigh(mats)
    sqrt_evals = np.sqrt(np.clip(evals, eps, None))
    return _symmetrize((evecs * sqrt_evals[..., None, :]) @ np.swapaxes(evecs, -1, -2))


def _matrix_to_rows(mats: np.ndarray, names: list[str], *, prefix: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    d = len(names)
    for i in range(d):
        for j in range(d):
            rows.append(
                {
                    "parameter": f"{prefix}[{names[i]},{names[j]}]",
                    "value": float(mats[i, j]),
                    "block": prefix,
                }
            )
    return rows


def build_full_covariance_proxy_extended_model(
    innovation_frame: pd.DataFrame,
    *,
    window_hours: int,
    stride_hours: int = 24,
    eps: float = 1e-8,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.DataFrame(innovation_frame, copy=True).astype(float).dropna().sort_index()
    names = [str(col).replace("_innovation", "") for col in frame.columns]
    d = len(names)
    if d < 2:
        raise ValueError("The Wishart covariance extension requires at least two innovation dimensions.")

    rows: list[dict[str, float]] = []
    indices: list[pd.Timestamp] = []
    values = frame.to_numpy(dtype=float)
    for end in range(window_hours - 1, len(frame), stride_hours):
        sample = values[end - window_hours + 1 : end + 1, :]
        cov = np.cov(sample.T, ddof=1)
        cov = _symmetrize(cov)
        diag = np.clip(np.diag(cov), eps, None)
        corr = cov / np.sqrt(np.outer(diag, diag))
        corr = np.clip(corr, -1.0, 1.0)
        row: dict[str, float] = {}
        for i in range(d):
            for j in range(i, d):
                row[f"s[{names[i]},{names[j]}]"] = float(cov[i, j])
        for i in range(d):
            for j in range(i + 1, d):
                row[f"corr[{names[i]},{names[j]}]"] = float(corr[i, j])
        rows.append(row)
        indices.append(frame.index[end])

    proxy = pd.DataFrame(rows, index=pd.DatetimeIndex(indices))
    sigma_path = np.stack(
        [
            np.array(
                [
                    [
                        proxy.iloc[k][f"s[{names[min(i, j)]},{names[max(i, j)]}]"] if i <= j
                        else proxy.iloc[k][f"s[{names[min(i, j)]},{names[max(i, j)]}]"]
                        for j in range(d)
                    ]
                    for i in range(d)
                ],
                dtype=float,
            )
            for k in range(len(proxy))
        ]
    )
    sigma_path = _ensure_spd_batch(sigma_path, eps=eps)
    t_grid_years = np.arange(len(proxy), dtype=float) * float(stride_hours) / 8760.0
    return proxy, sigma_path, t_grid_years


def simulate_wishart_innovation_system_extended_model(
    *,
    alpha: float,
    b: np.ndarray,
    A: np.ndarray,
    rho: np.ndarray,
    sigma0: np.ndarray,
    n_steps: int,
    dt_years: float,
    n_paths: int,
    seed: Optional[int] = None,
    eps: float = 1e-10,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    b = np.asarray(b, dtype=float)
    A = np.asarray(A, dtype=float)
    ata = A.T @ A
    rho = np.asarray(rho, dtype=float).reshape(-1)
    d = int(A.shape[0])
    if rho.size != d:
        raise ValueError(f"rho must have length {d}, got {rho.size}.")
    rho_norm = float(np.linalg.norm(rho))
    if rho_norm >= 1.0:
        rho = 0.95 * rho / max(rho_norm, 1e-12)
        rho_norm = float(np.linalg.norm(rho))
    bar_rho = float(np.sqrt(max(1.0 - rho_norm**2, 1e-12)))

    P = int(n_paths)
    Sigma = np.broadcast_to(np.asarray(sigma0, dtype=float), (P, d, d)).copy()
    Sigma = _ensure_spd_batch(Sigma, eps=eps)

    sigma_paths = np.zeros((P, int(n_steps) + 1, d, d), dtype=float)
    innovation_paths = np.zeros((P, int(n_steps), d), dtype=float)
    sigma_paths[:, 0] = Sigma

    sqrt_dt = float(np.sqrt(dt_years))
    dt_years = float(dt_years)
    for k in range(int(n_steps)):
        Sigma = _ensure_spd_batch(Sigma, eps=eps)
        sqrt_sigma = _sqrtm_spd_batch(Sigma, eps=eps)
        gaussian_mat = rng.normal(size=(P, d, d))
        gaussian_perp = rng.normal(size=(P, d))
        dW = gaussian_mat * sqrt_dt

        drift = float(alpha) * ata[None, :, :]
        drift = drift + np.einsum("ij,pjk->pik", b, Sigma) + np.einsum("pij,jk->pik", Sigma, b.T)
        diffusion = np.einsum("pij,pjk,kl->pil", sqrt_sigma, dW, A)
        diffusion = diffusion + np.einsum("ij,pkj,pkl->pil", A.T, dW, sqrt_sigma)
        Sigma_next = _ensure_spd_batch(Sigma + dt_years * drift + diffusion, eps=eps)

        shared_gaussian = bar_rho * gaussian_perp + np.einsum("pij,j->pi", gaussian_mat, rho)
        innovations = np.einsum("pij,pj->pi", sqrt_sigma, shared_gaussian)

        sigma_paths[:, k + 1] = Sigma_next
        innovation_paths[:, k] = innovations
        Sigma = Sigma_next

    return {
        "Sigma_paths": sigma_paths,
        "innovation_paths": innovation_paths,
    }


def _build_proxy_frame_from_sigma_paths(
    sigma_paths: np.ndarray,
    names: list[str],
    *,
    index: pd.DatetimeIndex,
    eps: float = 1e-10,
) -> pd.DataFrame:
    mats = np.asarray(sigma_paths, dtype=float)
    rows: list[dict[str, float]] = []
    d = len(names)
    for cov in mats:
        diag = np.clip(np.diag(cov), eps, None)
        corr = cov / np.sqrt(np.outer(diag, diag))
        corr = np.clip(corr, -1.0, 1.0)
        row: dict[str, float] = {}
        for i in range(d):
            for j in range(i, d):
                row[f"s[{names[i]},{names[j]}]"] = float(cov[i, j])
        for i in range(d):
            for j in range(i + 1, d):
                row[f"corr[{names[i]},{names[j]}]"] = float(corr[i, j])
        rows.append(row)
    return pd.DataFrame(rows, index=index)


def _evaluate_covariance_fit_extended_model(
    proxy: pd.DataFrame,
    sigma_paths: np.ndarray,
    names: list[str],
    *,
    eps: float = 1e-10,
) -> Dict[str, float]:
    sim_mean = np.mean(sigma_paths, axis=0)
    sim_proxy_mean = _build_proxy_frame_from_sigma_paths(
        sim_mean,
        names,
        index=proxy.index,
        eps=eps,
    )

    cov_rel_errors: list[float] = []
    corr_errors: list[float] = []
    out: Dict[str, float] = {}
    for i, ni in enumerate(names):
        emp = proxy[f"s[{ni},{ni}]"].to_numpy(dtype=float)
        sim = sim_proxy_mean[f"s[{ni},{ni}]"].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
        scale = max(float(np.mean(np.abs(emp))), eps)
        out[f"rmse_var_{ni}"] = rmse
        out[f"rmse_var_rel_{ni}"] = float(rmse / scale)
        cov_rel_errors.append(float(rmse / scale))
        for j in range(i + 1, len(names)):
            nj = names[j]
            emp_cov = proxy[f"s[{ni},{nj}]"].to_numpy(dtype=float)
            sim_cov = sim_proxy_mean[f"s[{ni},{nj}]"].to_numpy(dtype=float)
            scale_cov = max(float(np.mean(np.sqrt(np.maximum(proxy[f's[{ni},{ni}]'].to_numpy(dtype=float) * proxy[f's[{nj},{nj}]'].to_numpy(dtype=float), eps)))), eps)
            rmse_cov = float(np.sqrt(np.mean((emp_cov - sim_cov) ** 2)))
            out[f"rmse_cov_{ni}_{nj}"] = rmse_cov
            out[f"rmse_cov_rel_{ni}_{nj}"] = float(rmse_cov / scale_cov)
            cov_rel_errors.append(float(rmse_cov / scale_cov))

            emp_corr = proxy[f"corr[{ni},{nj}]"].to_numpy(dtype=float)
            sim_corr = sim_proxy_mean[f"corr[{ni},{nj}]"].to_numpy(dtype=float)
            rmse_corr = float(np.sqrt(np.mean((emp_corr - sim_corr) ** 2)))
            out[f"rmse_corr_{ni}_{nj}"] = rmse_corr
            corr_errors.append(rmse_corr)

    out["rmse_cov_rel_mean"] = float(np.mean(cov_rel_errors)) if cov_rel_errors else 0.0
    out["rmse_corr_mean"] = float(np.mean(corr_errors)) if corr_errors else 0.0
    out["selection_score"] = float(0.5 * out["rmse_cov_rel_mean"] + 0.5 * out["rmse_corr_mean"])
    return out


def estimate_leverage_vector_wishart_extended_model(
    innovation_frame: pd.DataFrame,
    hourly_proxy: pd.DataFrame,
    names: list[str],
    A: np.ndarray,
    *,
    max_norm: float = 0.95,
    eps: float = 1e-12,
) -> tuple[np.ndarray, pd.DataFrame]:
    innov = pd.DataFrame(innovation_frame, copy=True).astype(float).dropna().sort_index()
    d = len(names)
    diag_source_cols = [f"s[{name},{name}]" for name in names]
    proxy_diff = hourly_proxy[diag_source_cols].copy()
    proxy_diff = proxy_diff.rename(columns={f"s[{name},{name}]": f"var_{name}" for name in names})
    diag_cols = [f"var_{name}" for name in names]
    merged = innov.join(proxy_diff[diag_cols].diff(), how="inner").dropna()
    if len(merged) < 10:
        rho = np.zeros(d, dtype=float)
        rows = [{"quantity": f"rho[{name}]", "value": 0.0} for name in names]
        rows.append({"quantity": "rho_norm", "value": 0.0})
        return rho, pd.DataFrame(rows)

    v = np.zeros(d, dtype=float)
    rows: list[dict[str, float | str]] = []
    for i, name in enumerate(names):
        innov_col = f"{name}_innovation"
        var_col = f"var_{name}"
        mean_var = float(hourly_proxy[f"s[{name},{name}]"].reindex(merged.index).mean())
        cov_val = float(np.cov(merged[innov_col], merged[var_col], ddof=1)[0, 1])
        v[i] = cov_val / max(2.0 * mean_var, eps)
        rows.append({"quantity": f"cov({innov_col},d{var_col})", "value": cov_val})
        rows.append({"quantity": f"v[{name}]", "value": float(v[i])})

    rho = np.linalg.lstsq(np.asarray(A, dtype=float).T, v, rcond=None)[0]
    rho = np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(rho))
    if norm >= max_norm and norm > 0.0:
        rho = rho * (float(max_norm) / norm)
        norm = float(np.linalg.norm(rho))
    for i, name in enumerate(names):
        rows.append({"quantity": f"rho[{name}]", "value": float(rho[i])})
    rows.append({"quantity": "rho_norm", "value": norm})
    return rho.astype(float), pd.DataFrame(rows)


@dataclass
class WishartWindowCovarianceFitExtendedModel:
    window_label: str
    window_hours: int
    stride_hours: int
    proxy: pd.DataFrame
    proxy_standardized: pd.DataFrame
    sigma_path: np.ndarray
    sigma_path_standardized: np.ndarray
    t_grid_years: np.ndarray
    fit: WishartMLEFit
    sim_sigma_paths: np.ndarray
    sim_sigma_paths_standardized: np.ndarray
    summary: Dict[str, float]


@dataclass
class WishartFullCovarianceCalibrationExtendedModel:
    target_name: str
    variable_names: list[str]
    innovation_frame: pd.DataFrame
    innovation_frame_standardized: pd.DataFrame
    scale_matrix: np.ndarray
    candidate_fits: Dict[str, WishartWindowCovarianceFitExtendedModel]
    selected_window: str
    proxy: pd.DataFrame
    proxy_standardized: pd.DataFrame
    sigma_path: np.ndarray
    sigma_path_standardized: np.ndarray
    t_grid_years: np.ndarray
    fit: WishartMLEFit
    sigma0_fit: np.ndarray
    sigma0_fit_standardized: np.ndarray
    sigma0_forecast: np.ndarray
    sigma0_forecast_standardized: np.ndarray
    hourly_proxy: pd.DataFrame
    hourly_proxy_standardized: pd.DataFrame
    leverage: np.ndarray
    leverage_table: pd.DataFrame
    summary_table: pd.DataFrame


def fit_wishart_full_covariance_model_extended_model(
    innovation_frame: pd.DataFrame,
    *,
    target_name: str = "multivariate_innovations",
    window_map: Optional[Dict[str, int]] = None,
    stride_hours: int = 24,
    sim_paths: int = 128,
    random_seed: int = 20260416,
    b_convention: str = "simulator",
    max_norm: float = 0.95,
    eps: float = 1e-8,
) -> WishartFullCovarianceCalibrationExtendedModel:
    innov = pd.DataFrame(innovation_frame, copy=True).astype(float).dropna().sort_index()
    names = [str(col).replace("_innovation", "") for col in innov.columns]
    if window_map is None:
        window_map = {"7D": 24 * 7, "14D": 24 * 14, "30D": 24 * 30, "90D": 24 * 90}
    d = len(names)
    if d < 2:
        raise ValueError("Need at least two active innovation dimensions for Wishart calibration.")

    std = innov.std(ddof=1).replace(0.0, 1.0)
    scale_matrix = np.diag(std.to_numpy(dtype=float))
    inv_scale = np.linalg.inv(scale_matrix)
    innov_std = pd.DataFrame(
        innov.to_numpy(dtype=float) @ inv_scale.T,
        index=innov.index,
        columns=innov.columns,
    )

    wishart = WishartMLE(eps=eps)
    candidate_fits: Dict[str, WishartWindowCovarianceFitExtendedModel] = {}
    summary_rows: list[dict[str, float | str]] = []
    for i, (label, window_hours) in enumerate(window_map.items()):
        proxy_std, sigma_std, t_grid = build_full_covariance_proxy_extended_model(
            innov_std,
            window_hours=int(window_hours),
            stride_hours=int(stride_hours),
            eps=eps,
        )
        fit = wishart.fit_full_joint(
            sigma_std,
            T=float(t_grid[-1]) if len(t_grid) > 1 else float(stride_hours) / 8760.0,
            t_grid=t_grid,
            b_convention=b_convention,
        )
        wishart_sim_std = simulate_wishart_innovation_system_extended_model(
            alpha=float(fit.alpha),
            b=np.asarray(fit.b, dtype=float),
            A=np.asarray(fit.A, dtype=float),
            rho=np.zeros(d, dtype=float),
            sigma0=np.asarray(sigma_std[0], dtype=float),
            n_steps=len(sigma_std) - 1,
            dt_years=float(stride_hours) / 8760.0,
            n_paths=int(sim_paths),
            seed=int(random_seed) + i,
            eps=eps,
        )
        sim_std = wishart_sim_std["Sigma_paths"]
        sim_raw = scale_matrix[None, None, :, :] @ sim_std @ scale_matrix[None, None, :, :]
        sigma_raw = scale_matrix[None, :, :] @ sigma_std @ scale_matrix[None, :, :]
        proxy_raw = _build_proxy_frame_from_sigma_paths(sigma_raw, names, index=proxy_std.index, eps=eps)
        summary = _evaluate_covariance_fit_extended_model(proxy_raw, sim_raw, names, eps=eps)
        summary_rows.append(
            {
                "window": str(label),
                "window_hours": int(window_hours),
                **summary,
            }
        )
        candidate_fits[str(label)] = WishartWindowCovarianceFitExtendedModel(
            window_label=str(label),
            window_hours=int(window_hours),
            stride_hours=int(stride_hours),
            proxy=proxy_raw,
            proxy_standardized=proxy_std,
            sigma_path=sigma_raw,
            sigma_path_standardized=sigma_std,
            t_grid_years=t_grid,
            fit=fit,
            sim_sigma_paths=sim_raw,
            sim_sigma_paths_standardized=sim_std,
            summary=summary,
        )

    summary_table = pd.DataFrame(summary_rows).sort_values(
        by=["selection_score", "rmse_corr_mean", "rmse_cov_rel_mean", "window_hours"],
        ignore_index=True,
    )
    selected_window = str(summary_table.iloc[0]["window"])
    selected = candidate_fits[selected_window]
    hourly_proxy_std, hourly_sigma_std, hourly_t_grid = build_full_covariance_proxy_extended_model(
        innov_std,
        window_hours=int(selected.window_hours),
        stride_hours=1,
        eps=eps,
    )
    hourly_sigma = scale_matrix[None, :, :] @ hourly_sigma_std @ scale_matrix[None, :, :]
    hourly_proxy = _build_proxy_frame_from_sigma_paths(hourly_sigma, names, index=hourly_proxy_std.index, eps=eps)
    sigma0_fit_std = np.asarray(selected.sigma_path_standardized[0], dtype=float)
    sigma0_fit = np.asarray(selected.sigma_path[0], dtype=float)
    sigma0_forecast_std = np.asarray(hourly_sigma_std[0], dtype=float)
    sigma0_forecast = np.asarray(hourly_sigma[0], dtype=float)
    leverage, leverage_table = estimate_leverage_vector_wishart_extended_model(
        innov_std,
        hourly_proxy_std,
        names,
        np.asarray(selected.fit.A, dtype=float),
        max_norm=max_norm,
        eps=eps,
    )

    return WishartFullCovarianceCalibrationExtendedModel(
        target_name=str(target_name),
        variable_names=names,
        innovation_frame=innov,
        innovation_frame_standardized=innov_std,
        scale_matrix=scale_matrix,
        candidate_fits=candidate_fits,
        selected_window=selected_window,
        proxy=selected.proxy,
        proxy_standardized=selected.proxy_standardized,
        sigma_path=selected.sigma_path,
        sigma_path_standardized=selected.sigma_path_standardized,
        t_grid_years=selected.t_grid_years,
        fit=selected.fit,
        sigma0_fit=sigma0_fit,
        sigma0_fit_standardized=sigma0_fit_std,
        sigma0_forecast=sigma0_forecast,
        sigma0_forecast_standardized=sigma0_forecast_std,
        hourly_proxy=hourly_proxy,
        hourly_proxy_standardized=hourly_proxy_std,
        leverage=leverage,
        leverage_table=leverage_table,
        summary_table=summary_table,
    )


def wishart_parameter_table_extended_model(
    calibration: WishartFullCovarianceCalibrationExtendedModel,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = [
        {"parameter": "wishart_alpha", "value": float(calibration.fit.alpha), "block": "wishart"},
        {"parameter": "wishart_selected_window_hours", "value": float(calibration.summary_table.iloc[0]["window_hours"]), "block": "wishart"},
    ]
    rows.extend(_matrix_to_rows(np.asarray(calibration.fit.A, dtype=float), calibration.variable_names, prefix="wishart_A"))
    rows.extend(_matrix_to_rows(np.asarray(calibration.fit.b, dtype=float), calibration.variable_names, prefix="wishart_b"))
    rows.extend(_matrix_to_rows(np.asarray(calibration.scale_matrix, dtype=float), calibration.variable_names, prefix="scale_matrix"))
    rows.extend(_matrix_to_rows(np.asarray(calibration.fit.ata, dtype=float), calibration.variable_names, prefix="wishart_ata"))
    for i, name in enumerate(calibration.variable_names):
        rows.append({"parameter": f"wishart_rho[{name}]", "value": float(calibration.leverage[i]), "block": "leverage"})
    return pd.DataFrame(rows)


__all__ = [
    "WishartFullCovarianceCalibrationExtendedModel",
    "WishartWindowCovarianceFitExtendedModel",
    "build_full_covariance_proxy_extended_model",
    "fit_wishart_full_covariance_model_extended_model",
    "simulate_wishart_innovation_system_extended_model",
    "wishart_parameter_table_extended_model",
]
