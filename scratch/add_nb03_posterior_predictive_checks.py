from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "03mle.ipynb"

nb = nbformat.read(NB_PATH, as_version=4)

marker = "## 6. Posterior predictive checks"
nb.cells = [cell for cell in nb.cells if marker not in cell.source]

insert_at = None
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code" and "acf_table = pd.DataFrame" in cell.source:
        insert_at = i + 1
        break

if insert_at is None:
    raise RuntimeError("Could not find the ACF diagnostic cell.")

for cell in nb.cells:
    if cell.cell_type == "markdown" and cell.source.startswith("## 6. Key diagnostic identity"):
        cell.source = cell.source.replace("## 6.", "## 7.", 1)

markdown = nbformat.v4.new_markdown_cell(
    """## 6. Posterior predictive checks

These checks compare the realised empirical path with the distribution of statistics obtained from simulated CARMA paths. The key question is whether the observed statistic is typical under the model, not only whether two histograms look similar.
"""
)

code = nbformat.v4.new_code_cell(
    r'''PPC_LAGS = [1, 6, 24, 72, 168, 336]


def acf_selected_lags(x, lags):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    vals = []
    for lag in lags:
        if lag <= 0:
            vals.append(1.0)
        elif lag >= len(x) or denom <= 0.0:
            vals.append(np.nan)
        else:
            vals.append(float(np.dot(x[:-lag], x[lag:]) / denom))
    return np.array(vals, dtype=float)


def skew_and_excess_kurtosis(x):
    x = np.asarray(x, dtype=float)
    z = x - x.mean()
    sd = z.std(ddof=0)
    if sd <= 0.0:
        return np.nan, np.nan
    z = z / sd
    return float(np.mean(z ** 3)), float(np.mean(z ** 4) - 3.0)


def path_ppc_stats(y, lags=PPC_LAGS):
    y = np.asarray(y, dtype=float)
    dy = np.diff(y)
    skew_y, kurt_y = skew_and_excess_kurtosis(y)
    _, kurt_dy = skew_and_excess_kurtosis(dy)

    out = {
        "Y_std": float(np.std(y, ddof=0)),
        "Y_skew": skew_y,
        "Y_excess_kurtosis": kurt_y,
        "Y_q01": float(np.quantile(y, 0.01)),
        "Y_q05": float(np.quantile(y, 0.05)),
        "Y_q95": float(np.quantile(y, 0.95)),
        "Y_q99": float(np.quantile(y, 0.99)),
        "DeltaY_std": float(np.std(dy, ddof=0)),
        "DeltaY_excess_kurtosis": kurt_dy,
        "DeltaY_q01": float(np.quantile(dy, 0.01)),
        "DeltaY_q99": float(np.quantile(dy, 0.99)),
    }

    acf_y = acf_selected_lags(y, lags)
    acf_dy = acf_selected_lags(dy, lags)
    for lag, val in zip(lags, acf_y):
        out[f"ACF_Y_{lag}h"] = float(val)
    for lag, val in zip(lags, acf_dy):
        out[f"ACF_DeltaY_{lag}h"] = float(val)
    return pd.Series(out)


def simulated_ppc_stats(paths, lags=PPC_LAGS):
    rows = [path_ppc_stats(path, lags=lags) for path in np.asarray(paths)]
    return pd.DataFrame(rows)


def posterior_predictive_table(observed_stats, simulated_stats, model):
    rows = []
    for metric, observed in observed_stats.items():
        sims = simulated_stats[metric].dropna().to_numpy(float)
        q025, q50, q975 = np.quantile(sims, [0.025, 0.50, 0.975])
        obs_percentile = float(np.mean(sims <= observed))
        two_sided_p = float(min(1.0, 2.0 * min(obs_percentile, 1.0 - obs_percentile)))
        rows.append({
            "model": model,
            "metric": metric,
            "observed": float(observed),
            "sim_mean": float(np.mean(sims)),
            "sim_sd": float(np.std(sims, ddof=1)),
            "sim_q025": float(q025),
            "sim_q50": float(q50),
            "sim_q975": float(q975),
            "obs_percentile": obs_percentile,
            "two_sided_p": two_sided_p,
            "inside_95": bool(q025 <= observed <= q975),
            "n_paths": int(len(sims)),
        })
    return pd.DataFrame(rows)


observed_ppc = path_ppc_stats(pr)
gaussian_ppc_paths = simulated_ppc_stats(gaussian_paths)
nig_ppc_paths = simulated_ppc_stats(nig_paths)

ppc_table = pd.concat([
    posterior_predictive_table(observed_ppc, gaussian_ppc_paths, "Gaussian CARMA"),
    posterior_predictive_table(observed_ppc, nig_ppc_paths, "NIG CARMA"),
], ignore_index=True)

ppc_path = OUT / f"price_{model_tag}_posterior_predictive_checks.csv"
ppc_table.to_csv(ppc_path, index=False)

display_cols = [
    "metric",
    "observed",
    "sim_mean",
    "sim_q025",
    "sim_q975",
    "obs_percentile",
    "two_sided_p",
    "inside_95",
]

print("NIG CARMA posterior predictive checks")
print(
    ppc_table.loc[ppc_table["model"] == "NIG CARMA", display_cols]
    .to_string(index=False, float_format=lambda v: f"{v:.6e}")
)

print("\nMetrics outside the 95% posterior predictive envelope")
outliers = ppc_table.loc[~ppc_table["inside_95"], ["model"] + display_cols]
if len(outliers) == 0:
    print("None")
else:
    print(outliers.to_string(index=False, float_format=lambda v: f"{v:.6e}"))

print(f"\nSaved: {ppc_path}")
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
