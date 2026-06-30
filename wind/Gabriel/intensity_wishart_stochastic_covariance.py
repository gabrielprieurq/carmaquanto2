from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

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


def build_full_covariance_proxy(
    pair: pd.DataFrame,
    *,
    window_hours: int,
    stride_hours: int = 24,
    eps: float = 1e-8,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.DataFrame(pair, copy=True).astype(float)
    frame = frame.iloc[:, :2].dropna().sort_index()
    frame.columns = ["spot_innovation", "wind_innovation"]

    proxy = pd.DataFrame(
        {
            "s11": frame["spot_innovation"].rolling(window_hours, min_periods=window_hours).var(ddof=1),
            "s22": frame["wind_innovation"].rolling(window_hours, min_periods=window_hours).var(ddof=1),
            "s12": frame["spot_innovation"].rolling(window_hours, min_periods=window_hours).cov(frame["wind_innovation"]),
        }
    ).dropna()

    if stride_hours > 1:
        proxy = proxy.iloc[::stride_hours].copy()

    proxy["s11"] = proxy["s11"].clip(lower=eps)
    proxy["s22"] = proxy["s22"].clip(lower=eps)
    bound = 0.999 * np.sqrt(proxy["s11"] * proxy["s22"])
    proxy["s12"] = proxy["s12"].clip(lower=-bound, upper=bound)
    proxy["corr"] = proxy["s12"] / np.sqrt(proxy["s11"] * proxy["s22"])
    proxy["det"] = proxy["s11"] * proxy["s22"] - proxy["s12"] ** 2

    sigma_path = np.stack(
        [
            np.array([[row.s11, row.s12], [row.s12, row.s22]], dtype=float)
            for row in proxy.itertuples(index=False)
        ]
    )
    t_grid_years = np.arange(len(proxy), dtype=float) * float(stride_hours) / 8760.0
    return proxy, sigma_path, t_grid_years


def rolling_correlation_pair(
    pair: pd.DataFrame,
    *,
    window_hours: int,
    stride_hours: int = 1,
) -> pd.DataFrame:
    frame = pd.DataFrame(pair, copy=True).astype(float).dropna().sort_index()
    frame = frame.iloc[:, :2]
    frame.columns = ["x", "y"]
    corr = frame["x"].rolling(window_hours, min_periods=window_hours).corr(frame["y"]).dropna()
    out = pd.DataFrame({"corr": corr})
    if stride_hours > 1:
        out = out.iloc[::stride_hours].copy()
    return out


def simulate_wishart_innovation_system(
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
    rho_norm = float(np.linalg.norm(rho))
    if rho_norm >= 1.0:
        rho = 0.95 * rho / max(rho_norm, 1e-12)
        rho_norm = float(np.linalg.norm(rho))
    bar_rho = float(np.sqrt(max(1.0 - rho_norm**2, 1e-12)))

    P = int(n_paths)
    Sigma = np.broadcast_to(np.asarray(sigma0, dtype=float), (P, 2, 2)).copy()
    Sigma = _ensure_spd_batch(Sigma, eps=eps)

    sigma_paths = np.zeros((P, int(n_steps) + 1, 2, 2), dtype=float)
    innovation_paths = np.zeros((P, int(n_steps), 2), dtype=float)
    sigma_paths[:, 0] = Sigma

    dt_years = float(dt_years)
    sqrt_dt = float(np.sqrt(dt_years))
    for k in range(int(n_steps)):
        Sigma = _ensure_spd_batch(Sigma, eps=eps)
        sqrt_sigma = _sqrtm_spd_batch(Sigma, eps=eps)
        gaussian_mat = rng.normal(size=(P, 2, 2))
        gaussian_perp = rng.normal(size=(P, 2))
        dW = gaussian_mat * sqrt_dt

        drift = float(alpha) * ata[None, :, :]
        drift = drift + np.einsum("ij,pjk->pik", b, Sigma) + np.einsum("pij,jk->pik", Sigma, b.T)
        diffusion = np.einsum("pij,pjk,kl->pil", sqrt_sigma, dW, A)
        diffusion = diffusion + np.einsum("ij,pkj,pkl->pil", A.T, dW, sqrt_sigma)
        Sigma_next = Sigma + dt_years * drift + diffusion
        Sigma_next = _ensure_spd_batch(Sigma_next, eps=eps)

        shared_gaussian = bar_rho * gaussian_perp + np.einsum("pij,j->pi", gaussian_mat, rho)
        innovations = np.einsum("pij,pj->pi", sqrt_sigma, shared_gaussian)

        sigma_paths[:, k + 1] = Sigma_next
        innovation_paths[:, k] = innovations
        Sigma = Sigma_next

    corr_paths = sigma_paths[..., 0, 1] / np.sqrt(np.maximum(sigma_paths[..., 0, 0] * sigma_paths[..., 1, 1], eps))
    corr_paths = np.clip(corr_paths, -1.0, 1.0)
    return {
        "Sigma_paths": sigma_paths,
        "innovation_paths": innovation_paths,
        "corr_paths": corr_paths,
    }


def simulate_fitted_covariance_paths(
    wishart: WishartMLE,
    fit: WishartMLEFit,
    *,
    sigma0: np.ndarray,
    n_obs: int,
    step_hours: int,
    n_paths: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    T_years = (int(n_obs) - 1) * float(step_hours) / 8760.0
    sim = wishart.simulate_fonseca_paths(
        alpha=float(fit.alpha),
        A=np.asarray(fit.A, dtype=float),
        b=np.asarray(fit.b, dtype=float),
        rho=np.zeros(2, dtype=float),
        r=0.0,
        Sigma0=np.asarray(sigma0, dtype=float),
        y0=np.zeros(2, dtype=float),
        T=T_years,
        n_steps=int(n_obs) - 1,
        n_paths=int(n_paths),
        seed=int(seed),
        trace=True,
    )
    sigma_sim = np.asarray(sim["Sigma_paths"], dtype=float)
    corr_sim = sigma_sim[..., 0, 1] / np.sqrt(np.maximum(sigma_sim[..., 0, 0] * sigma_sim[..., 1, 1], 1e-10))
    corr_sim = np.clip(corr_sim, -1.0, 1.0)
    return sigma_sim, corr_sim, T_years


def _evaluate_covariance_fit(proxy: pd.DataFrame, sigma_paths: np.ndarray, eps: float = 1e-10) -> Dict[str, float]:
    emp_s11 = proxy["s11"].to_numpy(dtype=float)
    emp_s22 = proxy["s22"].to_numpy(dtype=float)
    emp_s12 = proxy["s12"].to_numpy(dtype=float)
    emp_corr = proxy["corr"].to_numpy(dtype=float)

    sim_s11 = sigma_paths[..., 0, 0]
    sim_s22 = sigma_paths[..., 1, 1]
    sim_s12 = sigma_paths[..., 0, 1]
    sim_corr = np.clip(sim_s12 / np.sqrt(np.maximum(sim_s11 * sim_s22, eps)), -1.0, 1.0)

    def _component_metrics(emp: np.ndarray, sim: np.ndarray, label: str) -> Dict[str, float]:
        sim_mean = np.mean(sim, axis=0)
        sim_q05 = np.quantile(sim, 0.05, axis=0)
        sim_q95 = np.quantile(sim, 0.95, axis=0)
        flat_sim = sim.reshape(-1)
        return {
            f"rmse_{label}": float(np.sqrt(np.mean((emp - sim_mean) ** 2))),
            f"mae_{label}": float(np.mean(np.abs(emp - sim_mean))),
            f"coverage90_{label}": float(np.mean((emp >= sim_q05) & (emp <= sim_q95))),
            f"ks_{label}": float(ks_2samp(emp, flat_sim).statistic),
            f"w1_{label}": float(wasserstein_distance(emp, flat_sim)),
            f"mean_emp_{label}": float(np.mean(emp)),
            f"mean_sim_{label}": float(np.mean(flat_sim)),
            f"std_emp_{label}": float(np.std(emp, ddof=1)),
            f"std_sim_{label}": float(np.std(flat_sim, ddof=1)),
        }

    out: Dict[str, float] = {}
    out.update(_component_metrics(emp_s11, sim_s11, "s11"))
    out.update(_component_metrics(emp_s22, sim_s22, "s22"))
    out.update(_component_metrics(emp_s12, sim_s12, "s12"))
    out.update(_component_metrics(emp_corr, sim_corr, "corr"))
    s11_scale = max(float(np.mean(emp_s11)), eps)
    s22_scale = max(float(np.mean(emp_s22)), eps)
    s12_scale = max(float(np.mean(np.sqrt(np.maximum(emp_s11 * emp_s22, eps)))), eps)
    out["rmse_s11_rel"] = float(out["rmse_s11"] / s11_scale)
    out["rmse_s22_rel"] = float(out["rmse_s22"] / s22_scale)
    out["rmse_s12_rel"] = float(out["rmse_s12"] / s12_scale)
    out["rmse_cov_rel_mean"] = float(
        np.mean([out["rmse_s11_rel"], out["rmse_s22_rel"], out["rmse_s12_rel"]])
    )
    out["rmse_total"] = float(
        np.sqrt(
            np.mean(
                [
                    out["rmse_s11"] ** 2,
                    out["rmse_s22"] ** 2,
                    out["rmse_s12"] ** 2,
                ]
            )
        )
    )
    # Correlation-aware dimensionless model-selection score.
    # This avoids selecting windows solely from the spot-variance scale.
    out["selection_score"] = float(0.5 * out["rmse_cov_rel_mean"] + 0.5 * out["rmse_corr"])
    return out


def estimate_leverage_vector_wishart(
    innovation_pair: pd.DataFrame,
    hourly_proxy: pd.DataFrame,
    A: np.ndarray,
    *,
    max_norm: float = 0.95,
) -> tuple[np.ndarray, pd.DataFrame]:
    innov = pd.DataFrame(innovation_pair, copy=True).astype(float).dropna().sort_index()
    innov = innov.iloc[:, :2]
    innov.columns = ["spot_innovation", "wind_innovation"]

    proxy = pd.DataFrame(hourly_proxy, copy=True)[["s11", "s22"]].astype(float).sort_index()
    merged = innov.join(proxy.diff().rename(columns={"s11": "ds11", "s22": "ds22"}), how="inner").dropna()

    if len(merged) < 10:
        rho = np.zeros(2, dtype=float)
        diagnostics = pd.DataFrame(
            [
                {"quantity": "raw_corr_spot_ds11", "value": 0.0},
                {"quantity": "raw_corr_wind_ds22", "value": 0.0},
                {"quantity": "v1", "value": 0.0},
                {"quantity": "v2", "value": 0.0},
                {"quantity": "rho1", "value": 0.0},
                {"quantity": "rho2", "value": 0.0},
                {"quantity": "rho_norm", "value": 0.0},
            ]
        )
        return rho, diagnostics

    mean_s11 = float(proxy["s11"].reindex(merged.index).mean())
    mean_s22 = float(proxy["s22"].reindex(merged.index).mean())
    cov_e1_ds11 = float(np.cov(merged["spot_innovation"], merged["ds11"], ddof=1)[0, 1])
    cov_e2_ds22 = float(np.cov(merged["wind_innovation"], merged["ds22"], ddof=1)[0, 1])
    v = np.array(
        [
            cov_e1_ds11 / max(2.0 * mean_s11, 1e-12),
            cov_e2_ds22 / max(2.0 * mean_s22, 1e-12),
        ],
        dtype=float,
    )
    rho = np.linalg.lstsq(np.asarray(A, dtype=float).T, v, rcond=None)[0]
    rho = np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(rho))
    if norm >= max_norm and norm > 0.0:
        rho = rho * (float(max_norm) / norm)
        norm = float(np.linalg.norm(rho))

    diagnostics = pd.DataFrame(
        [
            {"quantity": "raw_corr_spot_ds11", "value": float(np.corrcoef(merged["spot_innovation"], merged["ds11"])[0, 1])},
            {"quantity": "raw_corr_wind_ds22", "value": float(np.corrcoef(merged["wind_innovation"], merged["ds22"])[0, 1])},
            {"quantity": "cov_spot_ds11", "value": cov_e1_ds11},
            {"quantity": "cov_wind_ds22", "value": cov_e2_ds22},
            {"quantity": "v1", "value": float(v[0])},
            {"quantity": "v2", "value": float(v[1])},
            {"quantity": "rho1", "value": float(rho[0])},
            {"quantity": "rho2", "value": float(rho[1])},
            {"quantity": "rho_norm", "value": norm},
        ]
    )
    return rho.astype(float), diagnostics


@dataclass
class WishartWindowCovarianceFit:
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
class WishartFullCovarianceCalibration:
    target_name: str
    innovation_pair: pd.DataFrame
    innovation_pair_standardized: pd.DataFrame
    scale_matrix: np.ndarray
    candidate_fits: Dict[str, WishartWindowCovarianceFit]
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


def fit_wishart_full_covariance_model(
    innovation_pair: pd.DataFrame,
    *,
    target_name: str = "car24_innovations",
    window_map: Optional[Dict[str, int]] = None,
    stride_hours: int = 24,
    sim_paths: int = 128,
    random_seed: int = 20260416,
    b_convention: str = "simulator",
    max_norm: float = 0.95,
    eps: float = 1e-8,
) -> WishartFullCovarianceCalibration:
    window_map = {"7D": 24 * 7, "14D": 24 * 14, "30D": 24 * 30, "90D": 24 * 90} if window_map is None else dict(window_map)
    pair = pd.DataFrame(innovation_pair, copy=True).astype(float).dropna().sort_index()
    pair = pair.iloc[:, :2]
    pair.columns = ["spot_innovation", "wind_innovation"]
    scale_vec = pair.std(ddof=1).replace(0.0, np.nan).fillna(1.0).to_numpy(dtype=float)
    scale_matrix = np.diag(scale_vec)
    inv_scale = np.diag(1.0 / scale_vec)
    pair_std = pair.copy()
    pair_std["spot_innovation"] = pair_std["spot_innovation"] / scale_vec[0]
    pair_std["wind_innovation"] = pair_std["wind_innovation"] / scale_vec[1]

    wishart = WishartMLE(eps=eps)
    candidate_fits: Dict[str, WishartWindowCovarianceFit] = {}
    rows = []

    for idx, (label, window_hours) in enumerate(window_map.items()):
        proxy_std, sigma_emp_std, t_grid_years = build_full_covariance_proxy(
            pair_std,
            window_hours=int(window_hours),
            stride_hours=int(stride_hours),
            eps=eps,
        )
        fit = wishart.fit_full_joint(sigma_emp_std, t_grid=t_grid_years, b_convention=b_convention)
        sigma_sim_std, corr_sim_std, horizon_years = simulate_fitted_covariance_paths(
            wishart,
            fit,
            sigma0=np.asarray(sigma_emp_std[0], dtype=float),
            n_obs=len(proxy_std),
            step_hours=int(stride_hours),
            n_paths=int(sim_paths),
            seed=int(random_seed + idx),
        )
        sigma_emp = scale_matrix[None, :, :] @ sigma_emp_std @ scale_matrix[None, :, :]
        sigma_sim = scale_matrix[None, None, :, :] @ sigma_sim_std @ scale_matrix[None, None, :, :]
        proxy = proxy_std.copy()
        proxy["s11"] = proxy_std["s11"] * scale_vec[0] ** 2
        proxy["s22"] = proxy_std["s22"] * scale_vec[1] ** 2
        proxy["s12"] = proxy_std["s12"] * scale_vec[0] * scale_vec[1]
        proxy["corr"] = proxy_std["corr"]
        proxy["det"] = proxy["s11"] * proxy["s22"] - proxy["s12"] ** 2
        summary = _evaluate_covariance_fit(proxy, sigma_sim, eps=eps)
        summary.update(
            {
                "window": label,
                "window_hours": int(window_hours),
                "horizon_years": float(horizon_years),
                "alpha_hat": float(fit.alpha),
                "A11": float(fit.A[0, 0]),
                "A12": float(fit.A[0, 1]),
                "A21": float(fit.A[1, 0]),
                "A22": float(fit.A[1, 1]),
                "b11": float(fit.b[0, 0]),
                "b12": float(fit.b[0, 1]),
                "b21": float(fit.b[1, 0]),
                "b22": float(fit.b[1, 1]),
            }
        )
        rows.append(summary)
        candidate_fits[label] = WishartWindowCovarianceFit(
            window_label=label,
            window_hours=int(window_hours),
            stride_hours=int(stride_hours),
            proxy=proxy,
            proxy_standardized=proxy_std,
            sigma_path=sigma_emp,
            sigma_path_standardized=sigma_emp_std,
            t_grid_years=t_grid_years,
            fit=fit,
            sim_sigma_paths=sigma_sim,
            sim_sigma_paths_standardized=sigma_sim_std,
            summary=summary,
        )

    summary_table = pd.DataFrame(rows).sort_values(["selection_score", "rmse_corr", "rmse_cov_rel_mean", "rmse_total"])
    selected_window = str(summary_table.iloc[0]["window"])
    selected_fit = candidate_fits[selected_window]
    hourly_proxy_std, _, _ = build_full_covariance_proxy(
        pair_std,
        window_hours=int(selected_fit.window_hours),
        stride_hours=1,
        eps=eps,
    )
    hourly_proxy = hourly_proxy_std.copy()
    hourly_proxy["s11"] = hourly_proxy_std["s11"] * scale_vec[0] ** 2
    hourly_proxy["s22"] = hourly_proxy_std["s22"] * scale_vec[1] ** 2
    hourly_proxy["s12"] = hourly_proxy_std["s12"] * scale_vec[0] * scale_vec[1]
    hourly_proxy["corr"] = hourly_proxy_std["corr"]
    hourly_proxy["det"] = hourly_proxy["s11"] * hourly_proxy["s22"] - hourly_proxy["s12"] ** 2
    leverage, leverage_table = estimate_leverage_vector_wishart(
        pair_std,
        hourly_proxy_std,
        np.asarray(selected_fit.fit.A, dtype=float),
        max_norm=max_norm,
    )

    sigma0_forecast = np.array(
        [
            [hourly_proxy["s11"].iloc[-1], hourly_proxy["s12"].iloc[-1]],
            [hourly_proxy["s12"].iloc[-1], hourly_proxy["s22"].iloc[-1]],
        ],
        dtype=float,
    )
    sigma0_forecast = _ensure_spd_batch(sigma0_forecast[None, ...], eps=eps)[0]
    sigma0_forecast_standardized = _ensure_spd_batch((inv_scale @ sigma0_forecast @ inv_scale)[None, ...], eps=eps)[0]

    return WishartFullCovarianceCalibration(
        target_name=str(target_name),
        innovation_pair=pair,
        innovation_pair_standardized=pair_std,
        scale_matrix=scale_matrix,
        candidate_fits=candidate_fits,
        selected_window=selected_window,
        proxy=selected_fit.proxy,
        proxy_standardized=selected_fit.proxy_standardized,
        sigma_path=selected_fit.sigma_path,
        sigma_path_standardized=selected_fit.sigma_path_standardized,
        t_grid_years=selected_fit.t_grid_years,
        fit=selected_fit.fit,
        sigma0_fit=np.asarray(selected_fit.sigma_path[0], dtype=float),
        sigma0_fit_standardized=np.asarray(selected_fit.sigma_path_standardized[0], dtype=float),
        sigma0_forecast=sigma0_forecast,
        sigma0_forecast_standardized=sigma0_forecast_standardized,
        hourly_proxy=hourly_proxy,
        hourly_proxy_standardized=hourly_proxy_std,
        leverage=leverage,
        leverage_table=leverage_table,
        summary_table=summary_table.reset_index(drop=True),
    )


def wishart_parameter_table(calibration: WishartFullCovarianceCalibration) -> pd.DataFrame:
    fit = calibration.fit
    rows = [
        {"parameter": "alpha", "value": float(fit.alpha), "block": "wishart"},
        {"parameter": "A11", "value": float(fit.A[0, 0]), "block": "diffusion"},
        {"parameter": "A12", "value": float(fit.A[0, 1]), "block": "diffusion"},
        {"parameter": "A21", "value": float(fit.A[1, 0]), "block": "diffusion"},
        {"parameter": "A22", "value": float(fit.A[1, 1]), "block": "diffusion"},
        {"parameter": "b11", "value": float(fit.b[0, 0]), "block": "drift"},
        {"parameter": "b12", "value": float(fit.b[0, 1]), "block": "drift"},
        {"parameter": "b21", "value": float(fit.b[1, 0]), "block": "drift"},
        {"parameter": "b22", "value": float(fit.b[1, 1]), "block": "drift"},
        {"parameter": "rho1", "value": float(calibration.leverage[0]), "block": "leverage"},
        {"parameter": "rho2", "value": float(calibration.leverage[1]), "block": "leverage"},
        {"parameter": "sigma0_forecast_11", "value": float(calibration.sigma0_forecast[0, 0]), "block": "initial"},
        {"parameter": "sigma0_forecast_12", "value": float(calibration.sigma0_forecast[0, 1]), "block": "initial"},
        {"parameter": "sigma0_forecast_22", "value": float(calibration.sigma0_forecast[1, 1]), "block": "initial"},
        {"parameter": "selected_window_hours", "value": float(calibration.candidate_fits[calibration.selected_window].window_hours), "block": "selection"},
    ]
    return pd.DataFrame(rows)


__all__ = [
    "WishartFullCovarianceCalibration",
    "WishartWindowCovarianceFit",
    "build_full_covariance_proxy",
    "estimate_leverage_vector_wishart",
    "fit_wishart_full_covariance_model",
    "simulate_wishart_innovation_system",
    "wishart_parameter_table",
]
