from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "newpaper" / "figures" / "appendix"
OUT.mkdir(parents=True, exist_ok=True)

BLACK = "black"
BLUE = "steelblue"
WEEK_START = pd.Timestamp("2025-06-01", tz="UTC")
WEEK_END = pd.Timestamp("2025-06-08", tz="UTC")
F2Y_START = pd.Timestamp("2025-01-01", tz="UTC")
F2Y_END = pd.Timestamp("2025-07-01", tz="UTC")

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig, stem):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def format_dates(ax, interval=1):
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xlabel("")


def legend_right(ax):
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0)


def load_time_indexed_csv(path, index_col="datetime"):
    df = pd.read_csv(path, parse_dates=[index_col]).set_index(index_col).sort_index()
    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def price_profile_class(index):
    dow = index.dayofweek
    month = index.month
    classes = np.zeros(len(index), dtype=int)
    for i, (d, m) in enumerate(zip(dow, month)):
        if d <= 4:
            classes[i] = m
        elif d == 5:
            if m in (1, 2, 12):
                classes[i] = 13
            elif m in (3, 4, 5):
                classes[i] = 14
            elif m in (6, 7, 8):
                classes[i] = 15
            else:
                classes[i] = 16
        else:
            if m in (1, 2, 12):
                classes[i] = 17
            elif m in (3, 4, 5):
                classes[i] = 18
            elif m in (6, 7, 8):
                classes[i] = 19
            else:
                classes[i] = 20
    return classes


def fit_price_f2d(log_price):
    day_mean = log_price.resample("D").mean()
    day_mean_per_hour = day_mean.reindex(log_price.index.floor("D")).to_numpy(float)
    f2d_obs = pd.Series(log_price.to_numpy(float) / day_mean_per_hour, index=log_price.index)

    index = log_price.index
    classes = price_profile_class(index)
    hour_dummies = pd.get_dummies(index.hour, drop_first=True).astype(float)
    x_base = np.column_stack([np.ones(len(index)), hour_dummies.to_numpy()])
    f2d_hat = np.zeros(len(index), dtype=float)

    for class_id in np.unique(classes):
        mask = classes == class_id
        x_class = x_base[mask]
        y_class = log_price.iloc[mask].to_numpy(float)
        coeffs, *_ = np.linalg.lstsq(x_class, y_class, rcond=None)
        predicted = x_class @ coeffs
        mean_shape = predicted.mean()
        f2d_hat[mask] = predicted / mean_shape if mean_shape != 0 else 1.0

    return f2d_obs.rename("F2D observed"), pd.Series(f2d_hat, index=index, name="F2D fitted")


def fit_price_f2y(log_price):
    daily = log_price.resample("D").mean().dropna()
    yearly_mean = daily.groupby(daily.index.year).transform("mean")
    f2y_obs = (daily / yearly_mean).rename("F2Y observed")

    x = pd.DataFrame(index=f2y_obs.index)
    x["const"] = 1.0
    for d, label in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]):
        x[label] = (x.index.dayofweek == d).astype(float)
    for m in range(2, 13):
        if m != 8:
            x[f"Month_{m}"] = (x.index.month == m).astype(float)
    x["Aug_early"] = ((x.index.month == 8) & (x.index.day <= 15)).astype(float)
    x["Aug_late"] = ((x.index.month == 8) & (x.index.day > 15)).astype(float)

    coeffs, *_ = np.linalg.lstsq(x.to_numpy(float), f2y_obs.to_numpy(float), rcond=None)
    f2y_hat = pd.Series(x.to_numpy(float) @ coeffs, index=f2y_obs.index, name="F2Y fitted")
    return f2y_obs, f2y_hat


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_price_figures(panel):
    week = panel.loc[(panel.index >= WEEK_START) & (panel.index < WEEK_END)].copy()
    log_price = panel["log_price"].replace([np.inf, -np.inf], np.nan).dropna()
    f2d_obs, f2d_hat = fit_price_f2d(log_price)
    f2y_obs, f2y_hat = fit_price_f2y(log_price)

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 4.8), sharex=True)
    axes[0].plot(week.index, week["price_raw"], color=BLACK, linewidth=0.9)
    axes[0].set_title("Raw German day-ahead price")
    axes[0].set_ylabel("EUR/MWh")
    axes[1].plot(week.index, week["log_price"], color=BLUE, linewidth=0.9)
    axes[1].set_title("Shifted log-price")
    axes[1].set_ylabel("log(EUR/MWh + 1000)")
    format_dates(axes[1], interval=1)
    save(fig, "price_raw_and_shifted_log_week")

    f2d_week = pd.concat([f2d_obs, f2d_hat], axis=1).loc[WEEK_START:WEEK_END - pd.Timedelta(hours=1)]
    fig, ax = plt.subplots(figsize=(8.4, 2.8))
    ax.plot(f2d_week.index, f2d_week["F2D observed"], color=BLACK, linewidth=0.85, label="Observed F2D")
    ax.plot(f2d_week.index, f2d_week["F2D fitted"], color=BLUE, linewidth=1.1, label="Fitted F2D")
    ax.set_title("Intraday price factor")
    ax.set_ylabel("Relative factor")
    legend_right(ax)
    format_dates(ax, interval=1)
    save(fig, "price_f2d_week")

    f2y_window = pd.concat([f2y_obs, f2y_hat], axis=1).loc[F2Y_START:F2Y_END - pd.Timedelta(days=1)]
    fig, ax = plt.subplots(figsize=(8.4, 2.8))
    ax.plot(f2y_window.index, f2y_window["F2Y observed"], color=BLACK, linewidth=0.85, label="Observed F2Y")
    ax.plot(f2y_window.index, f2y_window["F2Y fitted"], color=BLUE, linewidth=1.1, label="Fitted F2Y")
    ax.set_title("Daily price factor")
    ax.set_ylabel("Relative factor")
    legend_right(ax)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlabel("")
    save(fig, "price_f2y_first_semester")

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 4.8), sharex=True)
    axes[0].plot(week.index, week["log_price"], color=BLACK, linewidth=0.85, label="Observed log-price")
    axes[0].plot(week.index, week["log_price_seasonal"], color=BLUE, linewidth=1.1, label="Seasonal fit")
    axes[0].set_title("Final price seasonal fit")
    axes[0].set_ylabel("log(EUR/MWh + 1000)")
    legend_right(axes[0])
    axes[1].plot(week.index, week["log_price_resid"], color=BLACK, linewidth=0.85)
    axes[1].set_title("Deseasonalized price residual")
    axes[1].set_ylabel("Residual")
    format_dates(axes[1], interval=1)
    save(fig, "price_final_fit_and_residual_week")

    q_low, q_high = pd.concat([panel["log_price"], panel["log_price_seasonal"]]).quantile([0.001, 0.999])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.hist(panel["log_price"], bins=90, range=(q_low, q_high), density=True, histtype="step", linewidth=1.1, color=BLACK, label="Observed log-price")
    ax.hist(panel["log_price_seasonal"], bins=90, range=(q_low, q_high), density=True, histtype="step", linewidth=1.3, color=BLUE, label="Seasonal component")
    ax.set_title("Price seasonal distribution")
    ax.set_xlabel("log(EUR/MWh + 1000)")
    ax.set_ylabel("Density")
    legend_right(ax)
    save(fig, "price_log_distribution")


def generate_temperature_figures(panel):
    week = panel.loc[(panel.index >= WEEK_START) & (panel.index < WEEK_END)].copy()
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 4.8), sharex=True)
    axes[0].plot(week.index, week["temp_raw"], color=BLACK, linewidth=0.9, label="Observed temperature")
    axes[0].plot(week.index, week["temp_seasonal"], color=BLUE, linewidth=1.1, label="Seasonal fit")
    axes[0].set_title("Temperature seasonal fit")
    axes[0].set_ylabel("Degrees Celsius")
    legend_right(axes[0])
    axes[1].plot(week.index, week["temp_resid"], color=BLACK, linewidth=0.9)
    axes[1].set_title("Deseasonalized temperature residual")
    axes[1].set_ylabel("Degrees Celsius")
    format_dates(axes[1], interval=1)
    save(fig, "temperature_fit_and_residual_week")

    q_low, q_high = pd.concat([panel["temp_raw"], panel["temp_seasonal"]]).quantile([0.005, 0.995])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.hist(panel["temp_raw"], bins=80, range=(q_low, q_high), density=True, histtype="step", linewidth=1.1, color=BLACK, label="Observed")
    ax.hist(panel["temp_seasonal"], bins=80, range=(q_low, q_high), density=True, histtype="step", linewidth=1.3, color=BLUE, label="Seasonal component")
    ax.set_title("Temperature seasonal distribution")
    ax.set_xlabel("Degrees Celsius")
    ax.set_ylabel("Density")
    legend_right(ax)
    save(fig, "temperature_distribution")


def generate_solar_figures(solar):
    solar = solar.copy()
    alpha, beta = 1e-4, 0.9998
    solar["solar_seasonality_cf_scale"] = (
        solar["solar_clear_sky"] * (1.0 - alpha - beta * sigmoid(solar["solar_latent_seasonality"]))
    ).clip(0.0, 1.0)
    week = solar.loc[(solar.index >= WEEK_START) & (solar.index < WEEK_END)].copy()

    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    ax.plot(week.index, week["solar_cf"], color=BLACK, linewidth=0.85, label="Observed capacity factor")
    ax.plot(week.index, week["solar_clear_sky"], color=BLUE, linewidth=1.05, label="Clear-sky proxy")
    ax.set_title("Solar production and clear-sky proxy")
    ax.set_ylabel("Capacity factor")
    legend_right(ax)
    format_dates(ax, interval=1)
    save(fig, "solar_clear_sky_week")

    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    ax.plot(week.index, week["solar_latent_Y"], color=BLACK, linewidth=0.85, label="Latent logit coordinate")
    ax.plot(week.index, week["solar_latent_seasonality"], color=BLUE, linewidth=1.1, label="Seasonal fit")
    ax.set_title("Solar seasonal fit in latent logit coordinates")
    ax.set_ylabel("Logit")
    legend_right(ax)
    format_dates(ax, interval=1)
    save(fig, "solar_logit_fit_week")

    fig, ax = plt.subplots(figsize=(8.4, 2.8))
    ax.plot(week.index, week["solar_XtQ"], color=BLACK, linewidth=0.85)
    ax.set_title("Deseasonalized solar latent residual")
    ax.set_ylabel("Residual")
    format_dates(ax, interval=1)
    save(fig, "solar_logit_residual_week")

    latent_raw = solar["solar_latent_Y"].replace([np.inf, -np.inf], np.nan).dropna()
    latent_seas = solar["solar_latent_seasonality"].replace([np.inf, -np.inf], np.nan).dropna()
    q_low, q_high = pd.concat([latent_raw, latent_seas]).quantile([0.005, 0.995])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.hist(latent_raw, bins=90, range=(q_low, q_high), density=True, histtype="step", linewidth=1.1, color=BLACK, label="Latent coordinate")
    ax.hist(latent_seas, bins=90, range=(q_low, q_high), density=True, histtype="step", linewidth=1.3, color=BLUE, label="Latent seasonality")
    ax.set_title("Solar latent distribution")
    ax.set_xlabel("Latent logit coordinate")
    ax.set_ylabel("Density")
    legend_right(ax)
    save(fig, "solar_latent_distribution")


def generate_wind_figures(wind):
    wind = wind.copy()
    week = wind.loc[(wind.index >= WEEK_START) & (wind.index < WEEK_END)].copy()

    fig, ax = plt.subplots(figsize=(8.4, 2.8))
    ax.plot(week.index, week["wind_cf"], color=BLACK, linewidth=0.9)
    ax.set_title("Wind capacity factor")
    ax.set_ylabel("Capacity factor")
    format_dates(ax, interval=1)
    save(fig, "wind_capacity_factor_week")

    fig, ax = plt.subplots(figsize=(8.4, 2.8))
    ax.plot(week.index, week["wind_logit"], color=BLACK, linewidth=0.85, label="Logit coordinate")
    ax.plot(week.index, week["wind_logit_seasonality"], color=BLUE, linewidth=1.05, label="Logit seasonality")
    ax.set_title("Wind seasonal fit in latent logit coordinates")
    ax.set_ylabel("Logit")
    legend_right(ax)
    format_dates(ax, interval=1)
    save(fig, "wind_logit_fit_week")

    fig, ax = plt.subplots(figsize=(8.4, 2.8))
    ax.plot(week.index, week["wind_XtQ"], color=BLACK, linewidth=0.85)
    ax.set_title("Deseasonalized wind latent residual")
    ax.set_ylabel("Residual")
    format_dates(ax, interval=1)
    save(fig, "wind_logit_residual_week")

    latent_raw = wind["wind_logit"].replace([np.inf, -np.inf], np.nan).dropna()
    latent_seas = wind["wind_logit_seasonality"].replace([np.inf, -np.inf], np.nan).dropna()
    q_low, q_high = pd.concat([latent_raw, latent_seas]).quantile([0.005, 0.995])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.hist(latent_raw, bins=90, range=(q_low, q_high), density=True, histtype="step", linewidth=1.1, color=BLACK, label="Logit coordinate")
    ax.hist(latent_seas, bins=90, range=(q_low, q_high), density=True, histtype="step", linewidth=1.3, color=BLUE, label="Logit seasonality")
    ax.set_title("Wind latent distribution")
    ax.set_xlabel("Latent logit coordinate")
    ax.set_ylabel("Density")
    legend_right(ax)
    save(fig, "wind_latent_distribution")


def main():
    german = load_time_indexed_csv(ROOT / "germany" / "germany23+24+25" / "data" / "seasonality" / "german_panel.csv")
    solar = pd.read_csv(
        ROOT / "solar" / "Intensity_Model_solar" / "data" / "carma" / "solar_latent_panel.csv",
        index_col=0,
        parse_dates=True,
    ).sort_index()
    solar.index = pd.DatetimeIndex(solar.index).tz_localize("UTC")
    wind = load_time_indexed_csv(ROOT / "wind" / "carma_coupling" / "data" / "carma" / "wind_latent_panel.csv")

    generate_price_figures(german)
    generate_temperature_figures(german)
    generate_solar_figures(solar)
    generate_wind_figures(wind)
    print(f"Appendix deseasonalization figures written to {OUT}")


if __name__ == "__main__":
    main()
