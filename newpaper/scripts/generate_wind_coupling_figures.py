from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[2]
WIND_COUPLING = ROOT / "wind" / "carma_coupling" / "data" / "coupling"
WIND_CARMA = ROOT / "wind" / "carma_coupling" / "data" / "carma"
PRICE_SEAS = ROOT / "germany" / "germany23+24+25" / "data" / "seasonality"
OUT = ROOT / "newpaper" / "figures" / "coupling"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#1f4e79"
BLACK = "#111111"
GRID = "#d9d9d9"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.55,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def savefig(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")


with open(WIND_COUPLING / "wind_price_coupling_result.json", "r", encoding="utf-8") as fh:
    result = json.load(fh)

fit_table = pd.read_csv(WIND_COUPLING / "price_idiosyncratic_driver_fit_comparison.csv")
diagnostics = pd.read_csv(WIND_COUPLING / "wind_price_lambda_diagnostics.csv").iloc[0]
aligned = np.load(WIND_COUPLING / "wind_price_aligned_drivers.npz", allow_pickle=True)

lambda_hat = float(result["lambda_hat"])
dW_wind = aligned["dW_wind"].astype(float)
dW_price_marginal = aligned["dW_price_marginal"].astype(float)
dL_price_idio = aligned["dL_price_idio"].astype(float)
lambda_wind_component = aligned["lambda_wind_component"].astype(float)

gaussian_row = fit_table.loc[fit_table["model"].eq("Gaussian")].iloc[0]
nig_row = fit_table.loc[fit_table["model"].eq("NIG")].iloc[0]

# Raw residual scatter.
wind_panel = pd.read_csv(WIND_CARMA / "wind_latent_panel.csv", index_col=0, parse_dates=True)
wind_df = pd.DataFrame(
    {
        "datetime": pd.to_datetime(wind_panel.index, utc=True),
        "wind_residual": wind_panel["wind_XtQ"].to_numpy(float),
    }
).dropna()

price_panel = pd.read_csv(PRICE_SEAS / "german_panel.csv", parse_dates=["datetime"])
price_df = pd.DataFrame(
    {
        "datetime": pd.to_datetime(price_panel["datetime"], utc=True),
        "price_residual": price_panel["log_price_resid"].to_numpy(float),
    }
).dropna()

raw = (
    wind_df.drop_duplicates("datetime")
    .merge(price_df.drop_duplicates("datetime"), on="datetime", how="inner")
    .sort_values("datetime")
)
raw_corr_test = pearsonr(raw["wind_residual"], raw["price_residual"])

fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.scatter(
    raw["wind_residual"],
    raw["price_residual"],
    s=7,
    alpha=0.20,
    color=BLUE,
    edgecolors="none",
    rasterized=True,
)
ax.axhline(0.0, color=BLACK, lw=0.7, alpha=0.45)
ax.axvline(0.0, color=BLACK, lw=0.7, alpha=0.45)
ax.set_title(f"Raw residual levels, correlation = {raw_corr_test.statistic:.4f}")
ax.set_xlabel(r"wind logit residual $Y^W_t$")
ax.set_ylabel(r"log-price residual $Y^P_t$")
fig.tight_layout()
savefig(fig, "wind_price_raw_residual_scatter")
plt.close(fig)

corr_before_test = pearsonr(dW_price_marginal, dW_wind)
corr_after_test = pearsonr(dL_price_idio, dW_wind)

summary = {
    "raw_residual_corr": float(raw_corr_test.statistic),
    "raw_residual_pvalue": float(raw_corr_test.pvalue),
    "lambda_hat": lambda_hat,
    "state_residual_output_corr": float(diagnostics["state_residual_output_corr"]),
    "corr_before": float(corr_before_test.statistic),
    "corr_before_pvalue": float(corr_before_test.pvalue),
    "corr_after": float(corr_after_test.statistic),
    "corr_after_pvalue": float(corr_after_test.pvalue),
    "std_gaussian_idio": float(gaussian_row["std"]),
    "std_nig_idio": float(nig_row["std"]),
    "std_lambda_wind": float(np.std(lambda_wind_component, ddof=1)),
    "mean_lambda_wind": float(np.mean(lambda_wind_component)),
    "n_common": int(result["n_common_hourly_intervals"]),
}
pd.Series(summary).to_csv(OUT / "wind_price_coupling_summary.csv", header=["value"])
print(pd.Series(summary).to_string())
