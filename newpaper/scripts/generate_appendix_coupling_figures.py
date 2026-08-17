from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "newpaper" / "figures" / "appendix"
OUT.mkdir(parents=True, exist_ok=True)

PRICE_QMLE = ROOT / "germany" / "germany23+24+25" / "data" / "kalman" / "price_carma54_joint_qmle_result.json"
PRICE_PANEL = ROOT / "germany" / "germany23+24+25" / "data" / "seasonality" / "german_panel.csv"

BLUE = "#1f4e79"
BLACK = "#111111"
GRID = "#d9d9d9"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.45,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


FACTORS = [
    {
        "key": "temperature",
        "title": "Temperature",
        "factor_label": "temperature",
        "factor_symbol": "T",
        "qmle": ROOT / "temperature" / "data" / "carma" / "temperature_carma43_joint_qmle_result.json",
        "panel": ROOT / "temperature" / "data" / "carma" / "temperature_latent_panel.csv",
        "panel_column": "temperature_XtQ",
        "time_policy": "utc",
        "raw_xlabel": r"temperature residual $Y^T_t$",
        "state_xlabel": r"$b_T^\top R^T_t$",
        "out_name": "coupling_temperature_scatter",
    },
    {
        "key": "solar",
        "title": "Solar",
        "factor_label": "solar",
        "factor_symbol": "S",
        "qmle": ROOT
        / "solar"
        / "Intensity_Model_solar"
        / "data"
        / "carma"
        / "solar_carma43_joint_qmle_result.json",
        "panel": ROOT / "solar" / "Intensity_Model_solar" / "data" / "carma" / "solar_latent_panel.csv",
        "panel_column": "solar_XtQ",
        "time_policy": "berlin",
        "raw_xlabel": r"solar latent residual $Y^S_t$",
        "state_xlabel": r"$b_S^\top R^S_t$",
        "out_name": "coupling_solar_scatter",
    },
    {
        "key": "wind",
        "title": "Wind",
        "factor_label": "wind",
        "factor_symbol": "W",
        "qmle": ROOT / "wind" / "carma_coupling" / "data" / "carma" / "wind_carma43_joint_qmle_result.json",
        "panel": ROOT / "wind" / "carma_coupling" / "data" / "carma" / "wind_latent_panel.csv",
        "panel_column": "wind_XtQ",
        "time_policy": "utc",
        "raw_xlabel": r"wind logit residual $Y^W_t$",
        "state_xlabel": r"$b_W^\top R^W_t$",
        "out_name": "coupling_wind_scatter",
    },
]


def companion_from_ar(ar_coefficients: list[float]) -> np.ndarray:
    a = np.asarray(ar_coefficients, dtype=float)
    p = len(a)
    A = np.zeros((p, p), dtype=float)
    if p > 1:
        A[:-1, 1:] = np.eye(p - 1)
    A[-1, :] = -a[::-1]
    return A


def build_carma(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    A = companion_from_ar(obj["ar_coefficients"])
    b = np.asarray(obj["b_coefficients"], dtype=float)
    e = np.zeros(A.shape[0], dtype=float)
    e[-1] = 1.0
    sigma = float(np.sqrt(obj["nu2_qmle"]))
    return {"A": A, "b": b, "e": e, "sigma": sigma}


def van_loan(A: np.ndarray, G: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    M = np.block([[A, G @ G.T], [np.zeros((n, n)), -A.T]]) * float(dt)
    E = expm(M)
    F = E[:n, :n]
    Q = E[:n, n:] @ F.T
    return F, 0.5 * (Q + Q.T)


def stationary_covariance(A: np.ndarray, G: np.ndarray) -> np.ndarray:
    P = solve_continuous_lyapunov(A, -(G @ G.T))
    return 0.5 * (P + P.T)


def kf_scalar(y: np.ndarray, A: np.ndarray, G: np.ndarray, H: np.ndarray, dt: float = 1.0) -> dict:
    y = np.asarray(y, dtype=float)
    H = np.asarray(H, dtype=float).reshape(-1)
    p = A.shape[0]
    F, Q = van_loan(A, G, dt)
    x = np.zeros(p, dtype=float)
    P = stationary_covariance(A, G) + np.eye(p) * 1e-8
    xs = np.zeros((len(y), p), dtype=float)
    I = np.eye(p)

    for k, obs in enumerate(y):
        if k > 0:
            x = F @ x
            P = F @ P @ F.T + Q

        S = max(float(H @ P @ H) + 1e-8, 1e-14)
        v = float(obs - H @ x)
        K = (P @ H) / S
        x = x + K * v
        P = (I - np.outer(K, H)) @ P
        P = 0.5 * (P + P.T)
        xs[k] = x

    return {"F": F, "x": xs}


def berlin_naive_to_utc(index: pd.Index) -> pd.DatetimeIndex:
    timestamps = pd.Series(pd.to_datetime(index))
    localized = timestamps.dt.tz_localize("Europe/Berlin", ambiguous="NaT", nonexistent="NaT")
    return pd.DatetimeIndex(localized.dt.tz_convert("UTC"))


def load_price_panel() -> tuple[pd.DatetimeIndex, np.ndarray]:
    price = pd.read_csv(PRICE_PANEL, parse_dates=["datetime"])
    return pd.DatetimeIndex(pd.to_datetime(price["datetime"], utc=True)), price["log_price_resid"].to_numpy(float)


def load_factor_panel(cfg: dict) -> tuple[pd.DatetimeIndex, np.ndarray]:
    panel = pd.read_csv(cfg["panel"], index_col=0, parse_dates=True)
    if cfg["time_policy"] == "berlin":
        times = berlin_naive_to_utc(panel.index)
    else:
        times = pd.DatetimeIndex(pd.to_datetime(panel.index, utc=True))
    return times, panel[cfg["panel_column"]].to_numpy(float)


def state_residual_projection(
    factor_time: pd.DatetimeIndex,
    factor_y_raw: np.ndarray,
    factor_model: dict,
    price_time: pd.DatetimeIndex,
    price_y_raw: np.ndarray,
    price_model: dict,
) -> tuple[np.ndarray, np.ndarray]:
    factor_y = factor_y_raw - float(np.mean(factor_y_raw))
    price_y = price_y_raw - float(np.mean(price_y_raw))

    kf_f = kf_scalar(
        factor_y,
        factor_model["A"],
        factor_model["sigma"] * factor_model["e"][:, None],
        factor_model["b"],
    )
    kf_p = kf_scalar(
        price_y,
        price_model["A"],
        price_model["sigma"] * price_model["e"][:, None],
        price_model["b"],
    )

    R_f = kf_f["x"][1:] - (kf_f["F"] @ kf_f["x"][:-1].T).T
    R_p = kf_p["x"][1:] - (kf_p["F"] @ kf_p["x"][:-1].T).T

    factor_intervals = pd.DataFrame(
        {
            "datetime": factor_time[1:],
            "i_f": np.arange(len(R_f), dtype=int),
            "dt_f_h": np.asarray((factor_time[1:] - factor_time[:-1]) / pd.Timedelta(hours=1), dtype=float),
        }
    )
    price_intervals = pd.DataFrame(
        {
            "datetime": price_time[1:],
            "i_p": np.arange(len(R_p), dtype=int),
            "dt_p_h": np.asarray((price_time[1:] - price_time[:-1]) / pd.Timedelta(hours=1), dtype=float),
        }
    )
    factor_intervals = factor_intervals.dropna(subset=["datetime"])
    price_intervals = price_intervals.dropna(subset=["datetime"])
    factor_intervals = factor_intervals[np.isclose(factor_intervals["dt_f_h"], 1.0)]
    price_intervals = price_intervals[np.isclose(price_intervals["dt_p_h"], 1.0)]
    factor_intervals = factor_intervals.drop_duplicates("datetime", keep="first")
    price_intervals = price_intervals.drop_duplicates("datetime", keep="first")
    common = factor_intervals.merge(price_intervals, on="datetime", how="inner").sort_values("datetime")

    r_f_out = R_f[common["i_f"].to_numpy(int)] @ factor_model["b"]
    r_p_out = R_p[common["i_p"].to_numpy(int)] @ price_model["b"]
    return r_f_out, r_p_out


def raw_residual_common(
    factor_time: pd.DatetimeIndex,
    factor_y_raw: np.ndarray,
    price_time: pd.DatetimeIndex,
    price_y_raw: np.ndarray,
    factor_name: str,
) -> pd.DataFrame:
    factor_df = pd.DataFrame({"datetime": factor_time, factor_name: factor_y_raw}).dropna(subset=["datetime"])
    price_df = pd.DataFrame({"datetime": price_time, "price": price_y_raw}).dropna(subset=["datetime"])
    return (
        factor_df.drop_duplicates("datetime")
        .merge(price_df.drop_duplicates("datetime"), on="datetime", how="inner")
        .sort_values("datetime")
    )


def scatter_panel(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str, title: str, xlabel: str, ylabel: str) -> None:
    ax.scatter(x, y, s=6, alpha=0.18, color=color, edgecolors="none", rasterized=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def make_factor_figure(cfg: dict, price_model: dict, price_time: pd.DatetimeIndex, price_y_raw: np.ndarray) -> None:
    factor_model = build_carma(cfg["qmle"])
    factor_time, factor_y_raw = load_factor_panel(cfg)

    raw = raw_residual_common(factor_time, factor_y_raw, price_time, price_y_raw, cfg["key"])
    raw_corr = float(pearsonr(raw[cfg["key"]], raw["price"]).statistic)

    r_f_out, r_p_out = state_residual_projection(
        factor_time, factor_y_raw, factor_model, price_time, price_y_raw, price_model
    )
    state_corr = float(pearsonr(r_f_out, r_p_out).statistic)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.35))
    scatter_panel(
        axes[0],
        raw[cfg["key"]].to_numpy(float),
        raw["price"].to_numpy(float),
        BLACK,
        "Raw residual levels",
        cfg["raw_xlabel"],
        r"log-price residual $Y^P_t$",
    )
    scatter_panel(
        axes[1],
        r_f_out,
        r_p_out,
        BLUE,
        "Projected state residuals",
        cfg["state_xlabel"],
        r"$b_P^\top R^P_t$",
    )
    fig.tight_layout()
    fig.savefig(OUT / f"{cfg['out_name']}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{cfg['out_name']}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"{cfg['key']}: raw corr={raw_corr:.6f}, state corr={state_corr:.6f}")


def main() -> None:
    price_model = build_carma(PRICE_QMLE)
    price_time, price_y_raw = load_price_panel()
    for cfg in FACTORS:
        make_factor_figure(cfg, price_model, price_time, price_y_raw)


if __name__ == "__main__":
    main()
