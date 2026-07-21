from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.stats import norminvgauss


REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "germany" / "germany23+24+25" / "data"
KALMAN = DATA / "kalman"
OUTFIG = REPO / "newpaper" / "figures" / "calibration"
OUTFIG.mkdir(parents=True, exist_ok=True)

BLUE = "#1f4e79"
BLACK = "#111111"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 130,
    }
)


def psd_sqrt(M, tol=1e-12):
    M = 0.5 * (M + M.T)
    vals, vecs = np.linalg.eigh(M)
    if vals.min() < -tol:
        raise ValueError(f"matrix is not PSD: min eigenvalue={vals.min():.3e}")
    return vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None)))


def companion_from_ar(ar):
    ar = np.asarray(ar, dtype=float)
    p = len(ar)
    A = np.zeros((p, p))
    A[:-1, 1:] = np.eye(p - 1)
    A[-1, :] = -ar[::-1]
    return A


def discrete_noise_covariance(A, ep, delta=1.0):
    p = A.shape[0]
    M = np.zeros((2 * p, 2 * p))
    M[:p, :p] = A
    M[:p, p:] = np.outer(ep, ep)
    M[p:, p:] = -A.T
    E = expm(M * delta)
    F = E[:p, :p]
    Q = E[:p, p:] @ F.T
    return 0.5 * (Q + Q.T)


def simulate_gaussian_carma(A, b, driver_fit, n_steps, n_paths, seed=20260621):
    rng = np.random.default_rng(seed)
    p = A.shape[0]
    ep = np.zeros(p)
    ep[-1] = 1.0
    F = expm(A)
    Pi1 = solve_continuous_lyapunov(A, -np.outer(ep, ep))
    Q = discrete_noise_covariance(A, ep)
    pi_sqrt = psd_sqrt(Pi1)
    q_sqrt = psd_sqrt(Q)

    m_rate = float(driver_fit["m_rate"])
    nu2_rate = float(driver_fit["nu2_rate"])
    state_mean = m_rate * np.linalg.solve(-A, ep)
    drift_step = m_rate * np.linalg.solve(A, (F - np.eye(p)) @ ep)

    X = state_mean + np.sqrt(nu2_rate) * (
        rng.standard_normal((n_paths, p)) @ pi_sqrt.T
    )
    Y = np.empty((n_paths, n_steps))
    for t in range(n_steps):
        shock = np.sqrt(nu2_rate) * (rng.standard_normal((n_paths, p)) @ q_sqrt.T)
        X = X @ F.T + drift_step + shock
        Y[:, t] = X @ b
    return Y


def simulate_nig_carma_midpoint(
    A, b, nig_fit, n_steps, n_paths, burnin=5000, seed=20260622
):
    rng = np.random.default_rng(seed)
    p = A.shape[0]
    ep = np.zeros(p)
    ep[-1] = 1.0
    F = expm(A)
    shock_vec = expm(A * 0.5) @ ep
    mean_dL = float(nig_fit["mean"])
    state_mean = np.linalg.solve(np.eye(p) - F, shock_vec * mean_dL)

    X = np.tile(state_mean, (n_paths, 1))
    Y = np.empty((n_paths, n_steps))
    dL_all = norminvgauss.rvs(
        nig_fit["scipy_a"],
        nig_fit["scipy_b"],
        loc=nig_fit["scipy_loc"],
        scale=nig_fit["scipy_scale"],
        size=(n_paths, n_steps + burnin),
        random_state=rng,
    )

    out_i = 0
    for t in range(n_steps + burnin):
        X = X @ F.T + dL_all[:, t, None] * shock_vec
        if t >= burnin:
            Y[:, out_i] = X @ b
            out_i += 1
    return Y


def plot_distribution(ax, observed, gaussian_sim, nig_sim):
    observed = np.asarray(observed, dtype=float).ravel()
    gaussian_sim = np.asarray(gaussian_sim, dtype=float).ravel()
    nig_sim = np.asarray(nig_sim, dtype=float).ravel()
    lo = min(
        np.quantile(observed, 0.001),
        np.quantile(gaussian_sim, 0.001),
        np.quantile(nig_sim, 0.001),
    )
    hi = max(
        np.quantile(observed, 0.999),
        np.quantile(gaussian_sim, 0.999),
        np.quantile(nig_sim, 0.999),
    )
    bins = np.linspace(lo, hi, 130)

    ax.hist(
        gaussian_sim,
        bins=bins,
        density=True,
        color="#d8e1ea",
        edgecolor="none",
        alpha=0.95,
        label="Gaussian CARMA",
    )
    ax.hist(
        nig_sim,
        bins=bins,
        density=True,
        histtype="step",
        color=BLUE,
        lw=1.8,
        label="NIG CARMA",
    )
    ax.hist(
        observed,
        bins=bins,
        density=True,
        histtype="step",
        color=BLACK,
        lw=1.45,
        label="observed",
    )
    ax.set_xlim(lo, hi)
    ax.set_xlabel("German spot price (EUR/MWh)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def main():
    panel = pd.read_csv(
        DATA / "seasonality" / "german_panel.csv", index_col=0, parse_dates=True
    )
    with open(KALMAN / "price_carma54_joint_qmle_result.json", "r") as fh:
        qmle = json.load(fh)
    with open(KALMAN / "price_carma54_driver_fits.json", "r") as fh:
        driver = json.load(fh)

    panel_fit = panel.dropna(
        subset=["price_raw", "log_price", "log_price_seasonal", "log_price_resid"]
    ).copy()
    price_obs = panel_fit["price_raw"].to_numpy(float)
    log_price_obs = panel_fit["log_price"].to_numpy(float)
    log_price_seasonal = panel_fit["log_price_seasonal"].to_numpy(float)
    price_shift = float(np.median(np.exp(log_price_obs) - price_obs))

    ar = np.asarray(qmle["ar_coefficients"], dtype=float)
    b = np.asarray(qmle["b_coefficients"], dtype=float)
    A = companion_from_ar(ar)
    n_steps = len(price_obs)
    n_paths = 250

    gaussian_paths = simulate_gaussian_carma(
        A, b, driver["gaussian"], n_steps, n_paths
    )
    nig_paths = simulate_nig_carma_midpoint(A, b, driver["nig"], n_steps, n_paths)

    gaussian_price = np.exp(log_price_seasonal[None, :] + gaussian_paths) - price_shift
    nig_price = np.exp(log_price_seasonal[None, :] + nig_paths) - price_shift

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    plot_distribution(ax, price_obs, gaussian_price, nig_price)
    fig.tight_layout()

    fig_path = OUTFIG / "price_spot_final_price_distribution.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
