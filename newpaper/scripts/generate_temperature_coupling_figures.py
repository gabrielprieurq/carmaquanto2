from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[2]
TEMP_COUPLING = ROOT / "temperature" / "data" / "coupling"
TEMP_CARMA = ROOT / "temperature" / "data" / "carma"
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


with open(TEMP_COUPLING / "temperature_price_coupling_result.json", "r", encoding="utf-8") as fh:
    result = json.load(fh)

fit_table = pd.read_csv(TEMP_COUPLING / "price_idiosyncratic_driver_fit_comparison.csv")
aligned = np.load(TEMP_COUPLING / "temperature_price_aligned_drivers.npz")

lambda_hat = float(result["lambda_hat"])
dW_temperature = aligned["dW_temperature"].astype(float)
dW_price_marginal = aligned["dW_price_marginal"].astype(float)
dL_price_idio = aligned["dL_price_idio"].astype(float)
lambda_temperature_component = aligned["lambda_temperature_component"].astype(float)

gaussian_row = fit_table.loc[fit_table["model"].eq("Gaussian")].iloc[0]
nig_row = fit_table.loc[fit_table["model"].eq("NIG")].iloc[0]

# Raw residual scatter.
temperature_panel = pd.read_csv(TEMP_CARMA / "temperature_latent_panel.csv", index_col=0, parse_dates=True)
temperature_df = pd.DataFrame(
    {
        "datetime": pd.to_datetime(temperature_panel.index, utc=True),
        "temperature_residual": temperature_panel["temperature_XtQ"].to_numpy(float),
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
    temperature_df.drop_duplicates("datetime")
    .merge(price_df.drop_duplicates("datetime"), on="datetime", how="inner")
    .sort_values("datetime")
)
raw_corr = float(pearsonr(raw["temperature_residual"], raw["price_residual"]).statistic)

fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.scatter(
    raw["temperature_residual"],
    raw["price_residual"],
    s=7,
    alpha=0.20,
    color=BLUE,
    edgecolors="none",
    rasterized=True,
)
ax.axhline(0.0, color=BLACK, lw=0.7, alpha=0.45)
ax.axvline(0.0, color=BLACK, lw=0.7, alpha=0.45)
ax.set_title(f"Raw residual levels, correlation = {raw_corr:.4f}")
ax.set_xlabel(r"temperature residual $Y^T_t$")
ax.set_ylabel(r"log-price residual $Y^P_t$")
fig.tight_layout()
savefig(fig, "temperature_price_raw_residual_scatter")
plt.close(fig)

corr_before_test = pearsonr(dW_price_marginal, dW_temperature)
corr_after_test = pearsonr(dL_price_idio, dW_temperature)

summary = {
    "raw_residual_corr": raw_corr,
    "lambda_hat": lambda_hat,
    "corr_before": float(corr_before_test.statistic),
    "corr_before_pvalue": float(corr_before_test.pvalue),
    "corr_after": float(corr_after_test.statistic),
    "corr_after_pvalue": float(corr_after_test.pvalue),
    "std_gaussian_idio": float(gaussian_row["std"]),
    "std_nig_idio": float(nig_row["std"]),
    "std_lambda_temperature": float(np.std(lambda_temperature_component, ddof=1)),
    "mean_lambda_temperature": float(np.mean(lambda_temperature_component)),
    "n_common": int(result["n_common_hourly_intervals"]),
}
pd.Series(summary).to_csv(OUT / "temperature_price_coupling_summary.csv", header=["value"])
print(pd.Series(summary).to_string())
