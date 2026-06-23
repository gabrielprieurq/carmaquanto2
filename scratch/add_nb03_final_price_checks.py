from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "03mle.ipynb"

nb = nbformat.read(NB_PATH, as_version=4)

# Make the first data-loading cell retain the full seasonality panel.
cell1 = nb.cells[1].source
old = '''pr = pd.read_csv(DATA / "seasonality" / "french_panel.csv", index_col=0)["log_price_resid"].to_numpy(float)
pr = pr[np.isfinite(pr)]
N = len(pr)
obs_delta_y = np.diff(pr)

print(f"{N:,} hourly deseasonalised log-price residuals")
print(f"Y empirical mean/std      = {pr.mean():.6e} / {pr.std(ddof=0):.6e}")
print(f"Delta Y empirical mean/std = {obs_delta_y.mean():.6e} / {obs_delta_y.std(ddof=0):.6e}")'''
new = '''panel = pd.read_csv(DATA / "seasonality" / "french_panel.csv", index_col=0, parse_dates=True)
panel = panel.dropna(subset=["price_raw", "log_price", "log_price_seasonal", "log_price_resid"])

price_obs = panel["price_raw"].to_numpy(float)
log_price_obs = panel["log_price"].to_numpy(float)
log_price_seasonal = panel["log_price_seasonal"].to_numpy(float)
pr = panel["log_price_resid"].to_numpy(float)

valid = (
    np.isfinite(price_obs)
    & np.isfinite(log_price_obs)
    & np.isfinite(log_price_seasonal)
    & np.isfinite(pr)
)
panel = panel.loc[valid]
price_obs = price_obs[valid]
log_price_obs = log_price_obs[valid]
log_price_seasonal = log_price_seasonal[valid]
pr = pr[valid]

N = len(pr)
obs_delta_y = np.diff(pr)
obs_delta_log_price = np.diff(log_price_obs)
obs_delta_price = np.diff(price_obs)
price_shift = float(np.median(np.exp(log_price_obs) - price_obs))
seasonality_identity_error = float(np.max(np.abs(log_price_obs - log_price_seasonal - pr)))

print(f"{N:,} hourly deseasonalised log-price residuals")
print(f"Y empirical mean/std       = {pr.mean():.6e} / {pr.std(ddof=0):.6e}")
print(f"Delta Y empirical mean/std = {obs_delta_y.mean():.6e} / {obs_delta_y.std(ddof=0):.6e}")
print(f"log-price shift inferred   = {price_shift:.6f}")
print(f"max |log_price - seasonal - resid| = {seasonality_identity_error:.3e}")'''
if old not in cell1:
    raise RuntimeError("Could not find the original residual-only data-loading block.")
nb.cells[1].source = cell1.replace(old, new)

# Remove previous version of this section if the script is rerun.
marker = "## 7. Final log-price and price checks"
nb.cells = [cell for cell in nb.cells if marker not in cell.source]

insert_at = None
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code" and "price_{model_tag}_level_distribution_comparison.png" in cell.source:
        insert_at = i + 1
        break

if insert_at is None:
    raise RuntimeError("Could not find the stochastic level comparison figure cell.")

for cell in nb.cells:
    if cell.cell_type == "markdown" and cell.source.startswith("## 7. Key diagnostic identity"):
        cell.source = cell.source.replace("## 7.", "## 8.", 1)

markdown = nbformat.v4.new_markdown_cell(
    """## 7. Final log-price and price checks

The final observed log-price is `log_price_seasonal + log_price_resid`. This section compares the empirical final series with the theoretical seasonal component plus simulated CARMA stochastic components. It then maps back to raw prices with `price = exp(log_price) - shift`.
"""
)

code = nbformat.v4.new_code_cell(
    r'''def final_distribution_summary(x):
    x = np.asarray(x, dtype=float).ravel()
    dx = np.diff(x)
    return pd.Series({
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=0)),
        "q01": float(np.quantile(x, 0.01)),
        "q05": float(np.quantile(x, 0.05)),
        "median": float(np.quantile(x, 0.50)),
        "q95": float(np.quantile(x, 0.95)),
        "q99": float(np.quantile(x, 0.99)),
        "skew": float(pd.Series(x).skew()),
        "excess_kurtosis": float(pd.Series(x).kurt()),
        "delta_std": float(np.std(dx, ddof=0)),
        "delta_q01": float(np.quantile(dx, 0.01)),
        "delta_q99": float(np.quantile(dx, 0.99)),
    })


def plot_final_series_checks(observed, gaussian_sim, nig_sim, observed_delta, gaussian_delta, nig_delta,
                             acf_max_lag, title_prefix, xlabel, delta_xlabel, fig_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    probs = np.linspace(0.01, 0.99, 99)

    ax = axes[0, 0]
    bins = np.linspace(
        min(np.quantile(observed, 0.001), np.quantile(gaussian_sim, 0.001), np.quantile(nig_sim, 0.001)),
        max(np.quantile(observed, 0.999), np.quantile(gaussian_sim, 0.999), np.quantile(nig_sim, 0.999)),
        120,
    )
    ax.hist(gaussian_sim.ravel(), bins=bins, density=True, color="#4c78a8", alpha=0.22, label="seasonal + Gaussian CARMA")
    ax.hist(nig_sim.ravel(), bins=bins, density=True, color="#c0392b", alpha=0.22, label="seasonal + NIG CARMA")
    ax.hist(observed, bins=bins, density=True, histtype="step", color="#111111", lw=1.4, label="observed")
    ax.set_title(f"{title_prefix}: distribution")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    obs_q = np.quantile(observed, probs)
    gauss_q = np.quantile(gaussian_sim.ravel(), probs)
    nig_q = np.quantile(nig_sim.ravel(), probs)
    lim_lo = min(obs_q.min(), gauss_q.min(), nig_q.min())
    lim_hi = max(obs_q.max(), gauss_q.max(), nig_q.max())
    pad = 0.05 * (lim_hi - lim_lo)
    ax.scatter(gauss_q, obs_q, s=10, color="#4c78a8", alpha=0.65, label="Gaussian driver")
    ax.scatter(nig_q, obs_q, s=10, color="#c0392b", alpha=0.65, label="NIG driver")
    ax.plot([lim_lo - pad, lim_hi + pad], [lim_lo - pad, lim_hi + pad], color="#111111", lw=0.8)
    ax.set_title(f"{title_prefix}: QQ observed vs simulated")
    ax.set_xlabel("simulated quantiles")
    ax.set_ylabel("observed quantiles")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    lag_grid = np.arange(acf_max_lag + 1)
    acf_obs_final = acf_1d(observed, acf_max_lag)
    acf_gauss_final = np.mean([acf_1d(path, acf_max_lag) for path in gaussian_sim], axis=0)
    acf_nig_final = np.mean([acf_1d(path, acf_max_lag) for path in nig_sim], axis=0)
    ax.plot(lag_grid, acf_obs_final, color="#111111", lw=1.3, label="observed")
    ax.plot(lag_grid, acf_gauss_final, color="#4c78a8", lw=1.1, label="Gaussian driver")
    ax.plot(lag_grid, acf_nig_final, color="#c0392b", lw=1.1, label="NIG driver")
    ax.set_title(f"{title_prefix}: ACF")
    ax.set_xlabel("lag (hours)")
    ax.set_ylabel("ACF")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    bins_delta = np.linspace(
        min(np.quantile(observed_delta, 0.001), np.quantile(gaussian_delta, 0.001), np.quantile(nig_delta, 0.001)),
        max(np.quantile(observed_delta, 0.999), np.quantile(gaussian_delta, 0.999), np.quantile(nig_delta, 0.999)),
        120,
    )
    ax.hist(gaussian_delta.ravel(), bins=bins_delta, density=True, color="#4c78a8", alpha=0.22, label="Gaussian driver")
    ax.hist(nig_delta.ravel(), bins=bins_delta, density=True, color="#c0392b", alpha=0.22, label="NIG driver")
    ax.hist(observed_delta, bins=bins_delta, density=True, histtype="step", color="#111111", lw=1.4, label="observed")
    ax.set_title(f"{title_prefix}: hourly differences")
    ax.set_xlabel(delta_xlabel)
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fig_path.name}")


if gaussian_paths.shape[1] != N or nig_paths.shape[1] != N:
    raise ValueError(
        f"Simulated paths length mismatch: gaussian={gaussian_paths.shape[1]}, "
        f"nig={nig_paths.shape[1]}, observed={N}"
    )

gaussian_log_price_paths = log_price_seasonal[None, :] + gaussian_paths
nig_log_price_paths = log_price_seasonal[None, :] + nig_paths

gaussian_price_paths = np.exp(gaussian_log_price_paths) - price_shift
nig_price_paths = np.exp(nig_log_price_paths) - price_shift

gaussian_delta_log_price = np.diff(gaussian_log_price_paths, axis=1)
nig_delta_log_price = np.diff(nig_log_price_paths, axis=1)
gaussian_delta_price = np.diff(gaussian_price_paths, axis=1)
nig_delta_price = np.diff(nig_price_paths, axis=1)

final_summary = pd.DataFrame({
    "observed_log_price": final_distribution_summary(log_price_obs),
    "gaussian_log_price": final_distribution_summary(gaussian_log_price_paths),
    "nig_log_price": final_distribution_summary(nig_log_price_paths),
    "observed_price": final_distribution_summary(price_obs),
    "gaussian_price": final_distribution_summary(gaussian_price_paths),
    "nig_price": final_distribution_summary(nig_price_paths),
}).T

final_summary_path = OUT / f"price_{model_tag}_final_logprice_price_summary.csv"
final_summary.to_csv(final_summary_path)
print(final_summary.to_string(float_format=lambda v: f"{v:.6e}"))
print(f"\nSaved: {final_summary_path}")

plot_final_series_checks(
    observed=log_price_obs,
    gaussian_sim=gaussian_log_price_paths,
    nig_sim=nig_log_price_paths,
    observed_delta=obs_delta_log_price,
    gaussian_delta=gaussian_delta_log_price,
    nig_delta=nig_delta_log_price,
    acf_max_lag=336,
    title_prefix="Final log-price",
    xlabel="log(price + shift)",
    delta_xlabel="Delta log(price + shift)",
    fig_path=FIG / f"price_{model_tag}_final_logprice_comparison.png",
)

plot_final_series_checks(
    observed=price_obs,
    gaussian_sim=gaussian_price_paths,
    nig_sim=nig_price_paths,
    observed_delta=obs_delta_price,
    gaussian_delta=gaussian_delta_price,
    nig_delta=nig_delta_price,
    acf_max_lag=336,
    title_prefix="Final raw price",
    xlabel="EUR/MWh",
    delta_xlabel="Delta EUR/MWh",
    fig_path=FIG / f"price_{model_tag}_final_price_comparison.png",
)
'''
)

nb.cells.insert(insert_at, markdown)
nb.cells.insert(insert_at + 1, code)

for cell in nb.cells:
    if "outputs" in cell:
        cell["outputs"] = []
    if "execution_count" in cell:
        cell["execution_count"] = None

nbformat.write(nb, NB_PATH)
print(f"Updated {NB_PATH}")
