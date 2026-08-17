from __future__ import annotations

import json
import math
import re
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "newpaper" / "sections" / "appendix_calibration.tex"


FACTORS = [
    {
        "key": "price",
        "title": "German Spot Prices",
        "caption_name": "German spot price",
        "order": "CARMA(5,4)",
        "init_json": ROOT
        / "germany"
        / "germany23+24+25"
        / "data"
        / "kalman"
        / "price_multiscale_carma_selected.json",
        "qmle_json": ROOT
        / "germany"
        / "germany23+24+25"
        / "data"
        / "kalman"
        / "price_carma54_joint_qmle_result.json",
        "fits_json": ROOT
        / "germany"
        / "germany23+24+25"
        / "data"
        / "kalman"
        / "price_carma54_driver_fits.json",
        "notebook": ROOT / "germany" / "germany23+24+25" / "03mle.ipynb",
        "figure": "calibration/price_spot_distribution_checks.pdf",
        "figure_width": "0.82\\textwidth",
        "figure_caption": (
            "Distribution checks for the German price marginal model at the "
            "residual factor, shifted log-price, and EUR/MWh price levels."
        ),
    },
    {
        "key": "temperature",
        "title": "Temperature",
        "caption_name": "temperature",
        "order": "CARMA(4,3)",
        "init_json": ROOT
        / "temperature"
        / "data"
        / "carma"
        / "temperature_multiscale_carma43_selected.json",
        "qmle_json": ROOT
        / "temperature"
        / "data"
        / "carma"
        / "temperature_carma43_joint_qmle_result.json",
        "fits_json": ROOT
        / "temperature"
        / "data"
        / "carma"
        / "temperature_carma43_driver_fits.json",
        "notebook": ROOT / "temperature" / "02mle_temperature.ipynb",
        "figure": "calibration/temperature_distribution_checks.pdf",
        "figure_width": "0.97\\textwidth",
        "figure_caption": (
            "Distribution checks for the German temperature marginal model at "
            "the deseasonalized residual, hourly residual-increment, and "
            "physical temperature levels."
        ),
    },
    {
        "key": "solar",
        "title": "Solar Capacity Factor",
        "caption_name": "solar capacity factor",
        "order": "CARMA(4,3)",
        "init_json": ROOT
        / "solar"
        / "Intensity_Model_solar"
        / "data"
        / "carma"
        / "solar_multiscale_carma43_selected.json",
        "qmle_json": ROOT
        / "solar"
        / "Intensity_Model_solar"
        / "data"
        / "carma"
        / "solar_carma43_joint_qmle_result.json",
        "fits_json": ROOT
        / "solar"
        / "Intensity_Model_solar"
        / "data"
        / "carma"
        / "solar_carma43_driver_fits.json",
        "notebook": ROOT / "solar" / "Intensity_Model_solar" / "02mle_solar.ipynb",
        "figure": "calibration/solar_distribution_checks.pdf",
        "figure_width": "0.97\\textwidth",
        "figure_caption": (
            "Distribution checks for the German solar marginal model at the "
            "deseasonalized latent residual, hourly residual-increment, and "
            "physical capacity-factor levels."
        ),
    },
    {
        "key": "wind",
        "title": "Wind Capacity Factor",
        "caption_name": "wind capacity factor",
        "order": "CARMA(4,3)",
        "init_json": ROOT
        / "wind"
        / "carma_coupling"
        / "data"
        / "carma"
        / "wind_multiscale_carma43_selected.json",
        "qmle_json": ROOT
        / "wind"
        / "carma_coupling"
        / "data"
        / "carma"
        / "wind_carma43_joint_qmle_result.json",
        "fits_json": ROOT
        / "wind"
        / "carma_coupling"
        / "data"
        / "carma"
        / "wind_carma43_driver_fits.json",
        "notebook": ROOT / "wind" / "carma_coupling" / "02mle_wind.ipynb",
        "figure": "calibration/wind_distribution_checks.pdf",
        "figure_width": "0.97\\textwidth",
        "figure_caption": (
            "Distribution checks for the German wind marginal model at the "
            "deseasonalized logit residual, hourly residual-increment, and "
            "physical capacity-factor levels."
        ),
    },
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def initial_exact_qmle(path: Path) -> tuple[float, float, float]:
    nb = nbformat.read(path, as_version=4)
    chunks: list[str] = []
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            if "text" in output:
                chunks.append("".join(output["text"]))
            if "data" in output and "text/plain" in output["data"]:
                chunks.append("".join(output["data"]["text/plain"]))
    text = "\n".join(chunks)
    match = re.search(
        r"Initial exact QMLE.*?loglik\s*=\s*([+\-0-9.eE]+)"
        r".*?m,\s*nu2\s*=\s*([+\-0-9.eE]+),\s*([+\-0-9.eE]+)",
        text,
        flags=re.S | re.I,
    )
    if match is None:
        raise RuntimeError(f"Could not find initial exact QMLE output in {path}")
    return tuple(float(x) for x in match.groups())


def fixed_sig(x: float, sig: int = 3) -> str:
    if not math.isfinite(x):
        return f"{x:g}"
    if x == 0:
        return "0." + "0" * (sig - 1)
    exponent = math.floor(math.log10(abs(x)))
    decimals = max(sig - exponent - 1, 0)
    rounded = f"{x:.{decimals}f}"
    # Rounding can move 9.99 to 10.0, changing the base-10 exponent.
    new_exponent = math.floor(math.log10(abs(float(rounded))))
    if new_exponent != exponent:
        decimals = max(sig - new_exponent - 1, 0)
        rounded = f"{x:.{decimals}f}"
    return rounded


def tex_number(x: float, sig: int = 3) -> str:
    if not math.isfinite(x):
        return f"{x:g}"
    if x == 0:
        return fixed_sig(x, sig)
    ax = abs(x)
    if ax < 1e-3 or ax >= 1e4:
        exponent = math.floor(math.log10(ax))
        mantissa = x / (10**exponent)
        mantissa_s = f"{mantissa:.{sig - 1}f}"
        if abs(float(mantissa_s)) >= 10:
            mantissa = mantissa / 10
            exponent += 1
            mantissa_s = f"{mantissa:.{sig - 1}f}"
        return rf"{mantissa_s}{{\times}}10^{{{exponent}}}"
    return fixed_sig(x, sig)


def tex_math_number(x: float, sig: int = 3) -> str:
    return rf"\({tex_number(x, sig)}\)"


def tex_loglik(x: float) -> str:
    return rf"\({x:.0f}\)"


def split_roots(roots: list[list[float]]) -> tuple[list[complex], complex | None]:
    values = [complex(r[0], r[1]) for r in roots]
    real_roots = sorted(
        [z for z in values if abs(z.imag) < 1e-10],
        key=lambda z: math.log(2) / (-z.real),
    )
    complex_roots = sorted(
        [z for z in values if z.imag > 1e-10],
        key=lambda z: abs(z.imag),
    )
    return real_roots, complex_roots[0] if complex_roots else None


def root_rows(init_roots: list[list[float]], qmle_roots: list[list[float]]) -> list[tuple[str, str, str]]:
    init_real, init_complex = split_roots(init_roots)
    qmle_real, qmle_complex = split_roots(qmle_roots)
    rows: list[tuple[str, str, str]] = []
    for idx, (z0, z1) in enumerate(zip(init_real, qmle_real), start=1):
        rows.append(
            (
                rf"\(\lambda_{idx}\)",
                tex_math_number(z0.real),
                tex_math_number(z1.real),
            )
        )
    if init_complex is not None and qmle_complex is not None:
        rows.append(
            (
                r"\(\lambda_o\)",
                rf"\({tex_number(init_complex.real)}\pm{tex_number(abs(init_complex.imag))}i\)",
                rf"\({tex_number(qmle_complex.real)}\pm{tex_number(abs(qmle_complex.imag))}i\)",
            )
        )
    return rows


def value_rows(symbol: str, init_values: list[float], qmle_values: list[float], start: int) -> list[tuple[str, str, str]]:
    rows = []
    for idx, (v0, v1) in enumerate(zip(init_values, qmle_values), start=start):
        rows.append((rf"\({symbol}_{idx}\)", tex_math_number(v0), tex_math_number(v1)))
    return rows


def table_rows(cfg: dict) -> tuple[list[tuple[str, str, str]], dict]:
    init = load_json(cfg["init_json"])
    qmle = load_json(cfg["qmle_json"])
    init_loglik, init_m, init_nu2 = initial_exact_qmle(cfg["notebook"])
    rows = [
        ("Log-likelihood", tex_loglik(init_loglik), tex_loglik(qmle["loglik_qmle"])),
        (r"Levy drift \(m\)", tex_math_number(init_m), tex_math_number(qmle["m_qmle"])),
        (r"Levy variance \(\nu^2\)", tex_math_number(init_nu2), tex_math_number(qmle["nu2_qmle"])),
    ]
    rows.extend(root_rows(init["roots"], qmle["ar_roots"]))
    rows.extend(value_rows("a", init["ar_coefficients"], qmle["ar_coefficients"], 1))
    rows.extend(value_rows("b", init["b_coefficients"], qmle["b_coefficients"], 0))
    return rows, load_json(cfg["fits_json"])["nig"]


def write_carma_table(cfg: dict, rows: list[tuple[str, str, str]]) -> str:
    label = f"tab:app-{cfg['key']}-carma-params"
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        rf"  \caption{{Full {cfg['caption_name']} {cfg['order']} calibration parameters.}}",
        rf"  \label{{{label}}}",
        r"  \small",
        r"  \begin{tabular}{@{}lll@{}}",
        r"    \toprule",
        r"    Quantity & ACF initialization & QMLE \\",
        r"    \midrule",
    ]
    section_breaks = {"\\lambda_1", "a_1", "b_0"}
    for name, init_value, qmle_value in rows:
        if any(marker in name for marker in section_breaks):
            lines.append(r"    \midrule")
        lines.append(f"    {name} & {init_value} & {qmle_value} \\\\")
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def write_nig_table(cfg: dict, nig: dict) -> str:
    label = f"tab:app-{cfg['key']}-nig-params"
    rows = [
        (r"\(\alpha\)", nig["alpha"]),
        (r"\(\beta\)", nig["beta"]),
        (r"\(\delta\)", nig["delta"]),
        (r"\(\mu\)", nig["mu"]),
        (r"\(\gamma\)", nig["gamma"]),
        ("Mean", nig["mean"]),
        ("Standard deviation", nig["std"]),
        ("Skewness", nig["skew"]),
        ("Excess kurtosis", nig["excess_kurtosis"]),
        ("Log-likelihood", nig["loglik"]),
    ]
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        rf"  \caption{{NIG fit for recovered hourly Levy increments in the {cfg['caption_name']} model.}}",
        rf"  \label{{{label}}}",
        r"  \small",
        r"  \begin{tabular}{@{}lr@{}}",
        r"    \toprule",
        r"    Quantity & Value \\",
        r"    \midrule",
    ]
    for name, value in rows:
        formatted = tex_loglik(float(value)) if name == "Log-likelihood" else tex_math_number(float(value))
        lines.append(f"    {name} & {formatted} \\\\")
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def write_distribution_figure(cfg: dict) -> str:
    return "\n".join(
        [
            r"\begin{figure}[H]",
            r"  \centering",
            rf"  \includegraphics[width={cfg['figure_width']}]{{{cfg['figure']}}}",
            rf"  \caption{{{cfg['figure_caption']}}}",
            rf"  \label{{fig:app-{cfg['key']}-distribution-checks}}",
            r"\end{figure}",
        ]
    )


def main() -> None:
    parts = [
        r"\section{Marginal Calibration Diagnostics}",
        r"\label{app:marginal-calibration}",
    ]
    for idx, cfg in enumerate(FACTORS):
        rows, nig = table_rows(cfg)
        if idx > 0:
            parts.append(r"\clearpage")
        parts.extend(
            [
                "",
                rf"\subsection{{{cfg['title']}}}",
                rf"\label{{app:{cfg['key']}-calibration}}",
                write_carma_table(cfg, rows),
                "",
                write_nig_table(cfg, nig),
                "",
                write_distribution_figure(cfg),
                "",
                r"\FloatBarrier",
            ]
        )
    OUT.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
