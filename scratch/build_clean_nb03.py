from pathlib import Path
import shutil

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "03mle.ipynb"
BACKUP_PATH = ROOT / "scratch" / "03mle_before_clean.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


if NB_PATH.exists() and not BACKUP_PATH.exists():
    shutil.copy2(NB_PATH, BACKUP_PATH)


nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "carma",
        "language": "python",
        "name": "carma",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}

nb.cells = [
    md(
        r"""
# CARMA(4,3) QMLE, recovered Levy driver, and driver-distribution diagnostics

This notebook keeps only the CARMA(4,3) workflow:

1. load the fixed CARMA(4,3) coefficients selected in `02order.ipynb`;
2. estimate the Levy drift and variance by Gaussian prediction-error QMLE;
3. recover hourly Levy driver increments `Delta L` from the smoothed CARMA state;
4. fit Gaussian and NIG laws to the recovered `Delta L`;
5. simulate the CARMA(4,3) driven by those fitted drivers and compare the simulated levels `Y_t` with the empirical deseasonalised log-price residuals.

Vocabulary used below:

- `Delta L`: increment of the latent Levy driver;
- `Delta Y`: hourly difference of the observed/simulated CARMA output;
- `Y`: deseasonalised log-price residual / CARMA output level.

The NIG law is fitted to `Delta L`, not to `Delta Y`.
"""
    ),
    code(
        r"""
%matplotlib inline

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize
from scipy.special import kve
from scipy.stats import norm, norminvgauss

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CWD = Path.cwd().resolve()
if (CWD / "data" / "seasonality" / "french_panel.csv").exists():
    CODE = CWD
elif (CWD.parent / "data" / "seasonality" / "french_panel.csv").exists():
    CODE = CWD.parent
else:
    raise FileNotFoundError("Run this notebook from the repo root or from notebooks/.")

DATA = CODE / "data"
OUT = DATA / "kalman"
FIG = CODE / "notebooks" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

pr = pd.read_csv(DATA / "seasonality" / "french_panel.csv", index_col=0)["log_price_resid"].to_numpy(float)
pr = pr[np.isfinite(pr)]
N = len(pr)
obs_delta_y = np.diff(pr)

print(f"{N:,} hourly deseasonalised log-price residuals")
print(f"Y empirical mean/std      = {pr.mean():.6e} / {pr.std(ddof=0):.6e}")
print(f"Delta Y empirical mean/std = {obs_delta_y.mean():.6e} / {obs_delta_y.std(ddof=0):.6e}")
"""
    ),
    md(
        r"""
## 1. Fixed CARMA(4,3) state-space model

The AR and MA coefficients are fixed by the previous ACF/order-selection step. This notebook estimates only the driver drift/scale and then studies the recovered driver distribution.
"""
    ),
    code(
        r"""
def build_companion_matrix(ar_coefficients):
    ar = np.asarray(ar_coefficients, dtype=float)
    p = len(ar)
    A = np.zeros((p, p))
    A[:-1, 1:] = np.eye(p - 1)
    A[-1, :] = -ar[::-1]

    expected = np.concatenate([[1.0], ar])
    actual = np.poly(np.linalg.eigvals(A)).real
    if not np.allclose(actual, expected, atol=1e-6):
        raise ValueError(f"Companion matrix check failed: {actual} vs {expected}")
    return A


def build_b_vector(b_coefficients, p):
    b = np.asarray(b_coefficients, dtype=float)
    q = len(b) - 1
    if q > p - 1:
        raise ValueError(f"q={q} must be <= p-1={p-1}")
    return np.concatenate([b, np.zeros(p - 1 - q)]) if q < p - 1 else b.copy()


def psd_sqrt(M, tol=1e-12):
    M = 0.5 * (M + M.T)
    vals, vecs = np.linalg.eigh(M)
    if vals.min() < -tol:
        raise ValueError(f"Matrix is not PSD; min eigenvalue={vals.min():.3e}")
    return vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None)))


def distribution_summary(x):
    x = np.asarray(x, dtype=float).ravel()
    s = pd.Series(x)
    return pd.Series({
        "mean": s.mean(),
        "std": s.std(ddof=0),
        "q01": s.quantile(0.01),
        "q05": s.quantile(0.05),
        "median": s.quantile(0.50),
        "q95": s.quantile(0.95),
        "q99": s.quantile(0.99),
        "skew": s.skew(),
        "excess_kurtosis": s.kurt(),
    })


def acf_1d(x, max_lag):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    return np.array([1.0] + [float(np.dot(x[:-k], x[k:]) / denom) for k in range(1, max_lag + 1)])


with open(OUT / "price_multiscale_carma_selected.json") as f:
    cfg = json.load(f)

ar_coefficients = np.array(cfg["ar_coefficients"], dtype=float)
b_coefficients = np.array(cfg["b_coefficients"], dtype=float)
roots = np.array(cfg["roots"], dtype=float)

p = len(ar_coefficients)
q = len(b_coefficients) - 1
Delta = 1.0

A = build_companion_matrix(ar_coefficients)
ep = np.zeros(p)
ep[-1] = 1.0
b_vec = build_b_vector(b_coefficients, p)
F = expm(A * Delta)
Pi1 = solve_continuous_lyapunov(A, -np.outer(ep, ep))
Q_base = Pi1 - F @ Pi1 @ F.T
Q_base = 0.5 * (Q_base + Q_base.T)
g_vec = np.linalg.solve(A, (F - np.eye(p)) @ ep)
dc_gain = float(b_coefficients[0] / ar_coefficients[-1])

if not np.all(roots[:, 0] < 0):
    raise ValueError("Non-causal CARMA roots detected.")

print(f"CARMA({p},{q})")
print(f"b_vec = {np.array2string(b_vec, precision=6)}")
print(f"max Re(root) = {roots[:, 0].max():.6e}")
print(f"Pi1 min eig  = {np.linalg.eigvalsh(Pi1).min():.3e}")
print(f"Qbase min eig = {np.linalg.eigvalsh(Q_base).min():.3e}")
print(f"DC gain b(0)/a(0) = {dc_gain:.6e}")
"""
    ),
    md(
        r"""
## 2. Gaussian prediction-error QMLE

The CARMA coefficients are fixed. QMLE estimates the Levy drift rate `m` and variance rate `nu2` using the Gaussian prediction-error likelihood.
"""
    ),
    code(
        r"""
def kalman_filter_fixed_params(A, b_vec, F, Q_base, Pi1, series):
    series = np.asarray(series, dtype=float)
    n = len(series)
    p = A.shape[0]

    U_hat = np.zeros(p)
    Pi = Pi1.copy()
    Psi = np.zeros((p, p))
    Omega = Pi.copy()

    residuals = np.empty(n)
    r_array = np.empty(n)
    Uhat_trace = np.empty((n, p))

    for i in range(n):
        Omega_b = Omega @ b_vec
        Delta_i = float(b_vec @ Omega_b)
        if Delta_i <= 0:
            raise RuntimeError(f"Delta[{i + 1}]={Delta_i:.6e} <= 0")

        Theta = F @ Omega_b
        resid = series[i] - float(b_vec @ U_hat)

        residuals[i] = resid
        r_array[i] = Delta_i
        Uhat_trace[i] = U_hat.copy()

        Pi_new = F @ Pi @ F.T + Q_base
        Psi_new = F @ Psi @ F.T + np.outer(Theta, Theta) / Delta_i
        Omega = Pi_new - Psi_new
        Omega = 0.5 * (Omega + Omega.T)
        Pi = Pi_new
        Psi = Psi_new
        U_hat = F @ U_hat + (Theta / Delta_i) * resid

    return residuals, r_array, Uhat_trace


def estimate_m_nu2(y, ar_coeffs, b_coeffs, Delta=1.0):
    y = np.asarray(y, dtype=float)
    n = len(y)
    p = len(ar_coeffs)

    A_loc = build_companion_matrix(ar_coeffs)
    ep_loc = np.zeros(p)
    ep_loc[-1] = 1.0
    b_loc = build_b_vector(b_coeffs, p)
    F_loc = expm(A_loc * Delta)
    Pi1_loc = solve_continuous_lyapunov(A_loc, -np.outer(ep_loc, ep_loc))
    Qb_loc = Pi1_loc - F_loc @ Pi1_loc @ F_loc.T
    Qb_loc = 0.5 * (Qb_loc + Qb_loc.T)

    dc_gain_loc = float(b_coeffs[0] / ar_coeffs[-1])

    e_raw, r, _ = kalman_filter_fixed_params(A_loc, b_loc, F_loc, Qb_loc, Pi1_loc, y)
    residuals_c, _, _ = kalman_filter_fixed_params(
        A_loc, b_loc, F_loc, Qb_loc, Pi1_loc, np.ones(n)
    )
    coef = dc_gain_loc * residuals_c

    m_hat = float(np.sum(e_raw * coef / r) / np.sum(coef ** 2 / r))
    residuals = e_raw - m_hat * coef
    nu2_hat = float(np.sum(residuals ** 2 / r) / n)
    loglik = -0.5 * (n * np.log(2.0 * np.pi * nu2_hat) + np.sum(np.log(r)) + n)

    return {
        "m_hat": m_hat,
        "nu2_hat": nu2_hat,
        "loglik": float(loglik),
        "r_array": r,
        "residuals": residuals,
        "e_raw": e_raw,
        "coef": coef,
    }


qmle = estimate_m_nu2(pr, ar_coefficients, b_coefficients, Delta=Delta)
m_hat = qmle["m_hat"]
nu2_hat = qmle["nu2_hat"]
r_array = qmle["r_array"]
eps_adj = qmle["residuals"]
eps_std = eps_adj / np.sqrt(nu2_hat * r_array)

qmle_json = {
    "m_qmle": float(m_hat),
    "nu2_qmle": float(nu2_hat),
    "loglik_qmle": float(qmle["loglik"]),
    "stationary_mean": float(m_hat * dc_gain),
    "carma_order": [int(p), int(q)],
    "source": "notebooks/03mle.ipynb",
}
(OUT / "price_carma_qmle_result.json").write_text(json.dumps(qmle_json, indent=2))

print(f"m_hat              = {m_hat:.6e}")
print(f"nu2_hat            = {nu2_hat:.6e}  sqrt={np.sqrt(nu2_hat):.6e}")
print(f"stationary mean Y  = {m_hat * dc_gain:.6e}")
print(f"loglik             = {qmle['loglik']:.3f}")
print(f"standardised innovations mean/std = {eps_std.mean():.4f} / {eps_std.std(ddof=0):.4f}")
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.hist(eps_std, bins=90, density=True, color="#4c78a8", alpha=0.35, label="standardised innovations")
x = np.linspace(np.quantile(eps_std, 0.001), np.quantile(eps_std, 0.999), 600)
ax.plot(x, norm.pdf(x), color="#111111", lw=1.2, label="N(0,1)")
ax.set_title("QMLE innovations")
ax.set_xlabel("standardised innovation")
ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=8)

ax = axes[1]
probs = np.linspace(0.01, 0.99, 99)
emp_q = np.quantile(eps_std, probs)
norm_q = norm.ppf(probs)
lim = max(abs(emp_q).max(), abs(norm_q).max()) * 1.05
ax.scatter(norm_q, emp_q, s=10, color="#4c78a8", alpha=0.7)
ax.plot([-lim, lim], [-lim, lim], color="#111111", lw=0.8)
ax.set_title("QQ vs Gaussian")
ax.set_xlabel("Gaussian quantiles")
ax.set_ylabel("empirical quantiles")

plt.tight_layout()
fig_path = FIG / "price_qmle_innovation_diagnostics.png"
plt.savefig(fig_path, dpi=130, bbox_inches="tight")
plt.show()
print(f"Saved: {fig_path.name}")
"""
    ),
    md(
        r"""
## 3. Recover hourly Levy driver increments

This keeps the previous recovery method: Kalman filtered states, RTS smoothing, then the Brockwell-Lindner modal recovery formula.
"""
    ),
    code(
        r"""
def kalman_filter_state_estimates(A, b_vec, F, Q_base, Pi1, series):
    series = np.asarray(series, dtype=float)
    n = len(series)
    p = A.shape[0]

    x_pred = np.zeros(p)
    P_pred = Pi1.copy()

    pred_states = np.empty((n, p))
    pred_covs = np.empty((n, p, p))
    filt_states = np.empty((n, p))
    filt_covs = np.empty((n, p, p))
    next_pred_states = np.empty((n, p))
    next_pred_covs = np.empty((n, p, p))
    residuals = np.empty(n)
    r_array = np.empty(n)

    for i in range(n):
        pred_states[i] = x_pred
        pred_covs[i] = P_pred

        P_b = P_pred @ b_vec
        Delta_i = float(b_vec @ P_b)
        if Delta_i <= 0:
            raise RuntimeError(f"Delta[{i + 1}]={Delta_i:.6e} <= 0")

        resid = series[i] - float(b_vec @ x_pred)
        gain = P_b / Delta_i
        x_filt = x_pred + gain * resid
        P_filt = P_pred - np.outer(P_b, P_b) / Delta_i
        P_filt = 0.5 * (P_filt + P_filt.T)

        x_next = F @ x_filt
        P_next = F @ P_filt @ F.T + Q_base
        P_next = 0.5 * (P_next + P_next.T)

        residuals[i] = resid
        r_array[i] = Delta_i
        filt_states[i] = x_filt
        filt_covs[i] = P_filt
        next_pred_states[i] = x_next
        next_pred_covs[i] = P_next

        x_pred = x_next
        P_pred = P_next

    return {
        "residuals": residuals,
        "r_array": r_array,
        "pred_states": pred_states,
        "pred_covs": pred_covs,
        "filt_states": filt_states,
        "filt_covs": filt_covs,
        "next_pred_states": next_pred_states,
        "next_pred_covs": next_pred_covs,
    }


def rts_smoother(F, filt_states, filt_covs, next_pred_states, next_pred_covs):
    n, p = filt_states.shape
    smooth_states = filt_states.copy()
    smooth_covs = filt_covs.copy()

    for i in range(n - 2, -1, -1):
        J = np.linalg.solve(next_pred_covs[i].T, (filt_covs[i] @ F.T).T).T
        smooth_states[i] = filt_states[i] + J @ (smooth_states[i + 1] - next_pred_states[i])
        smooth_covs[i] = filt_covs[i] + J @ (smooth_covs[i + 1] - next_pred_covs[i]) @ J.T
        smooth_covs[i] = 0.5 * (smooth_covs[i] + smooth_covs[i].T)

    return smooth_states, smooth_covs


def build_modal_matrix(A, roots_complex):
    roots_complex = np.asarray(roots_complex, dtype=complex)
    p = len(roots_complex)
    E = np.column_stack([lam ** np.arange(p) for lam in roots_complex])
    for r, lam in enumerate(roots_complex):
        err = np.linalg.norm(A @ E[:, r] - lam * E[:, r])
        if err >= 1e-8:
            raise ValueError(f"Eigenvector check failed for root {r}: err={err:.3e}")
    return E, np.linalg.inv(E)


def parse_roots(ar_coeffs, roots=None):
    if roots is None:
        return np.roots(np.concatenate([[1.0], np.asarray(ar_coeffs, dtype=float)]))
    return np.array([r[0] + 1j * r[1] if hasattr(r, "__len__") else complex(r) for r in roots])


def select_recovery_root(roots_complex, tol=1e-8):
    real_mask = np.abs(roots_complex.imag) < tol
    if not np.any(real_mask):
        raise ValueError("No real root available for modal recovery.")
    ridx = np.where(real_mask)[0]
    return int(ridx[np.argmin(np.abs(roots_complex[ridx].real))])


def levy_path_from_states(X_hat, ar_coeffs, b_coeffs, roots=None, Delta=1.0):
    X_hat = np.asarray(X_hat, dtype=float)
    roots_complex = parse_roots(ar_coeffs, roots)
    A_loc = build_companion_matrix(ar_coeffs)
    _, E_inv = build_modal_matrix(A_loc, roots_complex)

    idx_r = select_recovery_root(roots_complex)
    lam_r = float(roots_complex[idx_r].real)

    b_lam = float(sum(c * lam_r ** k for k, c in enumerate(b_coeffs)))
    xi_r = (E_inv @ X_hat.T)[idx_r]
    imag_frac = np.abs(xi_r.imag).max() / (np.abs(xi_r.real).max() + 1e-30)
    if imag_frac > 1e-6:
        raise ValueError(f"Imaginary residual too large in modal coordinate: {imag_frac:.2e}")

    Y_r = b_lam * xi_r.real
    a_desc = np.concatenate([[1.0], np.asarray(ar_coeffs, dtype=float)])
    a_prime = float(np.polyval(np.polyder(a_desc), lam_r))
    alpha_r = b_lam / a_prime

    times = np.arange(len(Y_r)) * Delta
    integ = cumulative_trapezoid(Y_r, times, initial=0.0)
    L_hat = (Y_r - Y_r[0] - lam_r * integ) / alpha_r

    return {
        "L_hat": L_hat,
        "increments": np.diff(L_hat),
        "Y_r": Y_r,
        "lambda_r": lam_r,
        "idx_r": idx_r,
        "alpha_r": float(alpha_r),
    }


def recover_levy_increments(y, ar_coeffs, b_coeffs, m_hat, nu2_hat, roots=None, Delta=1.0):
    y = np.asarray(y, dtype=float)
    p = len(ar_coeffs)
    A_loc = build_companion_matrix(ar_coeffs)
    ep_loc = np.zeros(p)
    ep_loc[-1] = 1.0
    b_loc = build_b_vector(b_coeffs, p)
    F_loc = expm(A_loc * Delta)
    Pi1_loc = solve_continuous_lyapunov(A_loc, -np.outer(ep_loc, ep_loc))
    Qb_loc = Pi1_loc - F_loc @ Pi1_loc @ F_loc.T
    Qb_loc = 0.5 * (Qb_loc + Qb_loc.T)
    dc = float(b_coeffs[0] / ar_coeffs[-1])

    W = y - m_hat * dc
    filt = kalman_filter_state_estimates(A_loc, b_loc, F_loc, Qb_loc, Pi1_loc, W)
    U_smooth, _ = rts_smoother(
        F_loc,
        filt["filt_states"],
        filt["filt_covs"],
        filt["next_pred_states"],
        filt["next_pred_covs"],
    )

    obs_resid = W - U_smooth @ b_loc
    shift = np.linalg.solve(-A_loc, ep_loc)
    X_hat = U_smooth + m_hat * shift

    out = levy_path_from_states(X_hat, ar_coeffs, b_coeffs, roots=roots, Delta=Delta)
    out["state_path"] = X_hat
    out["obs_resid_max"] = float(np.max(np.abs(obs_resid)))
    out["nu2_hat"] = float(nu2_hat)
    out["m_hat"] = float(m_hat)
    return out


levy_out = recover_levy_increments(
    pr,
    ar_coefficients,
    b_coefficients,
    m_hat=m_hat,
    nu2_hat=nu2_hat,
    roots=cfg["roots"],
    Delta=Delta,
)

L_hat = levy_out["L_hat"]
driver_delta_l = levy_out["increments"]

np.savez(
    OUT / "levy_increments_recovered.npz",
    L_hat=L_hat,
    increments=driver_delta_l,
    Y_r=levy_out["Y_r"],
    lambda_r=np.array([levy_out["lambda_r"]]),
    idx_r=np.array([levy_out["idx_r"]], dtype=int),
    alpha_r=np.array([levy_out["alpha_r"]]),
    nu2_hat=np.array([nu2_hat]),
    m_hat=np.array([m_hat]),
    obs_resid_max=np.array([levy_out["obs_resid_max"]]),
)

print(f"Selected recovery root lambda = {levy_out['lambda_r']:.6e}")
print(f"Recovered Delta L mean/std    = {driver_delta_l.mean():.6e} / {driver_delta_l.std(ddof=0):.6e}")
print(f"Recovered Delta L var / nu2   = {np.var(driver_delta_l) / nu2_hat:.6f}")
print(f"Max smoother observation residual = {levy_out['obs_resid_max']:.3e}")
"""
    ),
    md(
        r"""
## 4. Fit Gaussian and NIG laws to recovered `Delta L`

These fits are driver fits. They are not fits to `Delta Y`.
"""
    ),
    code(
        r"""
def log_nig_pdf_save03(x, mu, delta, alpha, beta):
    x = np.asarray(x, dtype=float)
    gamma = np.sqrt(alpha * alpha - beta * beta)
    xm = x - mu
    r = np.sqrt(delta * delta + xm * xm)
    return (
        np.log(alpha * delta / np.pi)
        + delta * gamma
        + beta * xm
        + np.log(kve(1, alpha * r))
        - alpha * r
        - np.log(r)
    )


def fit_nig_manual(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    mu_g = float(x.mean())
    sig_g = float(x.std(ddof=1))
    kurt_raw = float(np.mean((x - mu_g) ** 4) / sig_g ** 4)
    skew_raw = float(np.mean((x - mu_g) ** 3) / sig_g ** 3)

    alpha0 = np.sqrt(3.0 / (sig_g ** 2 * max(kurt_raw - 3.0, 0.01)))
    delta0 = sig_g ** 2 * alpha0
    beta0 = np.clip(
        skew_raw * alpha0 * np.sqrt(delta0 * alpha0) / 3.0,
        -0.9 * alpha0,
        0.9 * alpha0,
    )

    def pack(mu, delta, alpha, beta):
        return np.array([mu, np.log(delta), np.log(alpha), np.arctanh(beta / (0.99 * alpha))])

    def unpack(theta):
        mu = theta[0]
        delta = np.exp(theta[1])
        alpha = np.exp(theta[2])
        beta = np.tanh(theta[3]) * 0.99 * alpha
        return mu, delta, alpha, beta

    def nll(theta):
        mu, delta, alpha, beta = unpack(theta)
        ll = log_nig_pdf_save03(x, mu, delta, alpha, beta)
        if not np.all(np.isfinite(ll)):
            return 1e50
        return -float(ll.sum())

    theta0 = pack(mu_g, delta0, alpha0, beta0)
    best = None
    for beta_shift in np.linspace(-0.5, 0.5, 5):
        res = minimize(nll, theta0 + np.array([0.0, 0.0, 0.0, beta_shift]),
                       method="L-BFGS-B", options={"maxiter": 800, "ftol": 1e-10})
        if best is None or res.fun < best.fun:
            best = res

    mu, delta, alpha, beta = unpack(best.x)
    gamma = np.sqrt(alpha * alpha - beta * beta)
    return {
        "mu": float(mu),
        "delta": float(delta),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "scipy_a": float(alpha * delta),
        "scipy_b": float(beta * delta),
        "scipy_loc": float(mu),
        "scipy_scale": float(delta),
        "loglik": float(-best.fun),
        "mean": float(mu + delta * beta / gamma),
        "variance": float(delta * alpha * alpha / gamma ** 3),
        "std": float(np.sqrt(delta * alpha * alpha / gamma ** 3)),
        "skew": float(3 * beta / (alpha * np.sqrt(delta * gamma))),
        "excess_kurtosis": float(3 * (1 + 4 * beta * beta / (alpha * alpha)) / (delta * gamma)),
        "success": bool(best.success),
        "message": str(best.message),
        "N": int(len(x)),
    }


dL = np.asarray(driver_delta_l, dtype=float)
gaussian_driver = {
    "mean": float(dL.mean()),
    "variance": float(dL.var(ddof=0)),
    "std": float(dL.std(ddof=0)),
    "m_rate": float(dL.mean() / Delta),
    "nu2_rate": float(dL.var(ddof=0) / Delta),
    "N": int(len(dL)),
}
nig_driver = fit_nig_manual(dL)

driver_fits = {
    "gaussian": gaussian_driver,
    "nig": nig_driver,
    "source": "recovered hourly Levy increments from notebooks/03mle.ipynb",
}
(OUT / "price_carma43_driver_fits.json").write_text(json.dumps(driver_fits, indent=2))

print("Gaussian driver fit on Delta L")
print(f"  mean/std = {gaussian_driver['mean']:.6e} / {gaussian_driver['std']:.6e}")
print("NIG driver fit on Delta L")
print(f"  mu,delta,alpha,beta = {nig_driver['mu']:.6e}, {nig_driver['delta']:.6e}, {nig_driver['alpha']:.6e}, {nig_driver['beta']:.6e}")
print(f"  mean/std            = {nig_driver['mean']:.6e} / {nig_driver['std']:.6e}")
print(f"  skew/kurt           = {nig_driver['skew']:.4f} / {nig_driver['excess_kurtosis']:.4f}")
print(f"Saved: price_carma43_driver_fits.json")
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
probs = np.linspace(0.01, 0.99, 99)

ax = axes[0]
bins = np.linspace(np.quantile(dL, 0.001), np.quantile(dL, 0.999), 120)
x = np.linspace(bins[0], bins[-1], 700)
ax.hist(dL, bins=bins, density=True, histtype="step", color="#111111", lw=1.4, label="recovered Delta L")
ax.plot(x, norm.pdf(x, gaussian_driver["mean"], gaussian_driver["std"]),
        color="#4c78a8", lw=1.2, label="Gaussian fit")
ax.plot(x, norminvgauss.pdf(x, nig_driver["scipy_a"], nig_driver["scipy_b"],
                            loc=nig_driver["scipy_loc"], scale=nig_driver["scipy_scale"]),
        color="#c0392b", lw=1.2, label="NIG fit")
ax.set_title("Driver increment distribution")
ax.set_xlabel("Delta L")
ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=8)

ax = axes[1]
obs_q = np.quantile(dL, probs)
gauss_q = norm.ppf(probs, gaussian_driver["mean"], gaussian_driver["std"])
nig_q = norminvgauss.ppf(probs, nig_driver["scipy_a"], nig_driver["scipy_b"],
                         loc=nig_driver["scipy_loc"], scale=nig_driver["scipy_scale"])
lim = max(abs(obs_q).max(), abs(gauss_q).max(), abs(nig_q).max()) * 1.05
ax.scatter(gauss_q, obs_q, s=10, color="#4c78a8", alpha=0.65, label="Gaussian")
ax.scatter(nig_q, obs_q, s=10, color="#c0392b", alpha=0.65, label="NIG")
ax.plot([-lim, lim], [-lim, lim], color="#111111", lw=0.8)
ax.set_title("Driver QQ")
ax.set_xlabel("fitted quantiles")
ax.set_ylabel("recovered Delta L quantiles")
ax.legend(frameon=False, fontsize=8)

plt.tight_layout()
fig_path = FIG / "price_driver_deltaL_fits.png"
plt.savefig(fig_path, dpi=130, bbox_inches="tight")
plt.show()
print(f"Saved: {fig_path.name}")
"""
    ),
    md(
        r"""
## 5. Simulate CARMA(4,3) with fitted drivers

Gaussian driver simulation uses the exact sampled CARMA transition.

For NIG, the exact one-hour state shock is the stochastic integral

`int_0^1 exp(A(1-u)) e dL_u`.

The notebook uses a one-hour midpoint kernel for a simple no-substep path diagnostic. This avoids sub-hour stepping but should not be read as an exact NIG transition likelihood.
"""
    ),
    code(
        r"""
def simulate_gaussian_carma_exact(A, b_vec, F, Q_base, Pi1, ep, driver_fit,
                                  n_steps, n_paths, seed=20260621):
    rng = np.random.default_rng(seed)
    p = A.shape[0]
    m_rate = driver_fit["m_rate"]
    nu2_rate = driver_fit["nu2_rate"]

    state_mean = m_rate * np.linalg.solve(-A, ep)
    pi_sqrt = psd_sqrt(Pi1)
    q_sqrt = psd_sqrt(Q_base)
    g = np.linalg.solve(A, (F - np.eye(p)) @ ep)

    X = state_mean + np.sqrt(nu2_rate) * (rng.standard_normal((n_paths, p)) @ pi_sqrt.T)
    Y = np.empty((n_paths, n_steps))

    for t in range(n_steps):
        shocks = np.sqrt(nu2_rate) * (rng.standard_normal((n_paths, p)) @ q_sqrt.T)
        X = X @ F.T + m_rate * g + shocks
        Y[:, t] = X @ b_vec

    return Y


def simulate_nig_carma_midpoint(A, b_vec, F, ep, nig_fit,
                                n_steps, n_paths, burnin=5000, seed=20260622):
    rng = np.random.default_rng(seed)
    p = A.shape[0]
    shock_vec = expm(A * 0.5) @ ep
    mean_dL = nig_fit["mean"]
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
            Y[:, out_i] = X @ b_vec
            out_i += 1

    return Y, dL_all[:, burnin:]


N_PATHS = 250
gaussian_paths = simulate_gaussian_carma_exact(
    A, b_vec, F, Q_base, Pi1, ep, gaussian_driver,
    n_steps=N, n_paths=N_PATHS, seed=20260621,
)
nig_paths, nig_driver_draws = simulate_nig_carma_midpoint(
    A, b_vec, F, ep, nig_driver,
    n_steps=N, n_paths=N_PATHS, burnin=5000, seed=20260622,
)

gaussian_delta_y = np.diff(gaussian_paths, axis=1)
nig_delta_y = np.diff(nig_paths, axis=1)

print(f"Gaussian exact CARMA paths: {gaussian_paths.shape}")
print(f"NIG midpoint CARMA paths  : {nig_paths.shape}")
print(f"Gaussian Y mean/std       : {gaussian_paths.mean():.6e} / {gaussian_paths.std(ddof=0):.6e}")
print(f"NIG Y mean/std            : {nig_paths.mean():.6e} / {nig_paths.std(ddof=0):.6e}")
print(f"Observed Y mean/std       : {pr.mean():.6e} / {pr.std(ddof=0):.6e}")
"""
    ),
    code(
        r"""
summary = pd.DataFrame({
    "observed_Y": distribution_summary(pr),
    "gaussian_CARMA_Y": distribution_summary(gaussian_paths),
    "nig_CARMA_Y": distribution_summary(nig_paths),
    "observed_DeltaY": distribution_summary(obs_delta_y),
    "gaussian_CARMA_DeltaY": distribution_summary(gaussian_delta_y),
    "nig_CARMA_DeltaY": distribution_summary(nig_delta_y),
    "recovered_DeltaL": distribution_summary(dL),
    "nig_simulated_DeltaL": distribution_summary(nig_driver_draws),
}).T

summary_path = OUT / "price_carma43_distribution_summary.csv"
summary.to_csv(summary_path)
print(summary.to_string(float_format=lambda v: f"{v:.6e}"))
print(f"\nSaved: {summary_path}")
"""
    ),
    code(
        r"""
max_lag = 336
lags = np.arange(max_lag + 1)
acf_obs = acf_1d(pr, max_lag)
acf_gaussian = np.mean([acf_1d(path, max_lag) for path in gaussian_paths], axis=0)
acf_nig = np.mean([acf_1d(path, max_lag) for path in nig_paths], axis=0)

acf_diag_lags = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336]
acf_table = pd.DataFrame({
    "lag": acf_diag_lags,
    "observed": acf_obs[acf_diag_lags],
    "gaussian_CARMA": acf_gaussian[acf_diag_lags],
    "nig_CARMA": acf_nig[acf_diag_lags],
})
print(acf_table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
probs = np.linspace(0.01, 0.99, 99)

ax = axes[0, 0]
bins_y = np.linspace(
    min(np.quantile(pr, 0.001), np.quantile(gaussian_paths, 0.001), np.quantile(nig_paths, 0.001)),
    max(np.quantile(pr, 0.999), np.quantile(gaussian_paths, 0.999), np.quantile(nig_paths, 0.999)),
    120,
)
ax.hist(gaussian_paths.ravel(), bins=bins_y, density=True, color="#4c78a8", alpha=0.22, label="Gaussian driver")
ax.hist(nig_paths.ravel(), bins=bins_y, density=True, color="#c0392b", alpha=0.22, label="NIG driver")
ax.hist(pr, bins=bins_y, density=True, histtype="step", color="#111111", lw=1.4, label="observed")
ax.set_title("Level distribution: Y")
ax.set_xlabel("deseasonalised log-price residual")
ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=8)

ax = axes[0, 1]
obs_q = np.quantile(pr, probs)
gauss_q = np.quantile(gaussian_paths.ravel(), probs)
nig_q = np.quantile(nig_paths.ravel(), probs)
lim = max(abs(obs_q).max(), abs(gauss_q).max(), abs(nig_q).max()) * 1.05
ax.scatter(gauss_q, obs_q, s=10, color="#4c78a8", alpha=0.65, label="Gaussian driver")
ax.scatter(nig_q, obs_q, s=10, color="#c0392b", alpha=0.65, label="NIG driver")
ax.plot([-lim, lim], [-lim, lim], color="#111111", lw=0.8)
ax.set_title("Level QQ: observed vs simulated")
ax.set_xlabel("simulated Y quantiles")
ax.set_ylabel("observed Y quantiles")
ax.legend(frameon=False, fontsize=8)

ax = axes[1, 0]
ax.plot(lags, acf_obs, color="#111111", lw=1.3, label="observed")
ax.plot(lags, acf_gaussian, color="#4c78a8", lw=1.1, label="Gaussian driver")
ax.plot(lags, acf_nig, color="#c0392b", lw=1.1, label="NIG driver")
ax.set_title("Level autocorrelation")
ax.set_xlabel("lag (hours)")
ax.set_ylabel("ACF")
ax.legend(frameon=False, fontsize=8)

ax = axes[1, 1]
bins_dy = np.linspace(
    min(np.quantile(obs_delta_y, 0.001), np.quantile(gaussian_delta_y, 0.001), np.quantile(nig_delta_y, 0.001)),
    max(np.quantile(obs_delta_y, 0.999), np.quantile(gaussian_delta_y, 0.999), np.quantile(nig_delta_y, 0.999)),
    120,
)
ax.hist(gaussian_delta_y.ravel(), bins=bins_dy, density=True, color="#4c78a8", alpha=0.22, label="Gaussian driver")
ax.hist(nig_delta_y.ravel(), bins=bins_dy, density=True, color="#c0392b", alpha=0.22, label="NIG driver")
ax.hist(obs_delta_y, bins=bins_dy, density=True, histtype="step", color="#111111", lw=1.4, label="observed")
ax.set_title("Hourly differences: Delta Y")
ax.set_xlabel("Delta Y")
ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=8)

plt.tight_layout()
fig_path = FIG / "price_carma43_level_distribution_comparison.png"
plt.savefig(fig_path, dpi=130, bbox_inches="tight")
plt.show()
print(f"Saved: {fig_path.name}")
"""
    ),
    md(
        r"""
## 6. Key diagnostic identity

If the simulated levels have too much variance but their hourly differences look close, the likely reason is an autocorrelation compensation:

`Var(Delta Y) = 2 Var(Y) (1 - rho_1)`.
"""
    ),
    code(
        r"""
def level_delta_identity(name, y):
    y = np.asarray(y, dtype=float).ravel()
    dy = np.diff(y)
    var_y = float(np.var(y))
    var_dy = float(np.var(dy))
    rho1 = 1.0 - var_dy / (2.0 * var_y)
    return pd.Series({
        "std_Y": np.sqrt(var_y),
        "std_DeltaY": np.sqrt(var_dy),
        "rho1_implied": rho1,
        "varY_over_observed": var_y / np.var(pr),
        "varDeltaY_over_observed": var_dy / np.var(obs_delta_y),
    }, name=name)


identity = pd.DataFrame([
    level_delta_identity("observed", pr),
    level_delta_identity("gaussian_CARMA", gaussian_paths),
    level_delta_identity("nig_CARMA", nig_paths),
])

print(identity.to_string(float_format=lambda v: f"{v:.6e}"))
"""
    ),
]

nbf.write(nb, NB_PATH)
print(f"Wrote {NB_PATH}")
if BACKUP_PATH.exists():
    print(f"Backup: {BACKUP_PATH}")
