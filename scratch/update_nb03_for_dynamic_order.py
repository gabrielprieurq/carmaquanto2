from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "03mle.ipynb"

nb = nbformat.read(NB_PATH, as_version=4)

nb.cells[0].source = """# CARMA QMLE, recovered Levy driver, and driver-distribution diagnostics

This notebook uses the CARMA order and coefficients exported by `02order.ipynb`.

1. load the fixed CARMA coefficients selected in `02order.ipynb`;
2. estimate the Levy drift and variance by Gaussian prediction-error QMLE;
3. recover hourly Levy driver increments `Delta L` from the smoothed CARMA state;
4. fit Gaussian and NIG laws to the recovered `Delta L`;
5. simulate the selected CARMA model driven by those fitted drivers and compare the simulated levels `Y_t` with the empirical deseasonalised log-price residuals.

Vocabulary used below:

- `Delta L`: increment of the latent Levy driver;
- `Delta Y`: hourly difference of the observed/simulated CARMA output;
- `Y`: deseasonalised log-price residual / CARMA output level.

The NIG law is fitted to `Delta L`, not to `Delta Y`.
"""

nb.cells[2].source = """## 1. Fixed CARMA state-space model

The AR and MA coefficients are fixed by the previous ACF/order-selection step. This notebook estimates only the driver drift/scale and then studies the recovered driver distribution.
"""

nb.cells[12].source = """## 5. Simulate selected CARMA model with fitted drivers

Gaussian driver simulation uses the exact sampled CARMA transition.

For NIG, the exact one-hour state shock is the stochastic integral

`int_0^1 exp(A(1-u)) e dL_u`.

The notebook uses one NIG driver increment per hour and a midpoint kernel for a simple diagnostic. It does not simulate sub-hour paths and should not be read as an exact NIG transition likelihood.
"""

cell3 = nb.cells[3].source
old = """p = len(ar_coefficients)
q = len(b_coefficients) - 1
Delta = 1.0
"""
new = """p = len(ar_coefficients)
q = len(b_coefficients) - 1
Delta = 1.0
model_name = f"CARMA({p},{q})"
model_tag = f"carma{p}{q}"
"""
if old not in cell3:
    raise RuntimeError("Could not find model order block in cell 3")
cell3 = cell3.replace(old, new)
cell3 = cell3.replace('print(f"CARMA({p},{q})")', 'print(model_name)')
nb.cells[3].source = cell3

cell5 = nb.cells[5].source
old = '(OUT / "price_carma_qmle_result.json").write_text(json.dumps(qmle_json, indent=2))'
new = """qmle_path = OUT / f"price_{model_tag}_qmle_result.json"
qmle_path.write_text(json.dumps(qmle_json, indent=2))
(OUT / "price_carma_qmle_result.json").write_text(json.dumps(qmle_json, indent=2))"""
if old not in cell5:
    raise RuntimeError("Could not find QMLE write in cell 5")
cell5 = cell5.replace(old, new)
cell5 = cell5.replace(
    'print(f"standardised innovations mean/std = {eps_std.mean():.4f} / {eps_std.std(ddof=0):.4f}")',
    'print(f"standardised innovations mean/std = {eps_std.mean():.4f} / {eps_std.std(ddof=0):.4f}")\n'
    'print(f"Saved: {qmle_path.name} and price_carma_qmle_result.json")',
)
nb.cells[5].source = cell5

cell6 = nb.cells[6].source.replace(
    'fig_path = FIG / "price_qmle_innovation_diagnostics.png"',
    'fig_path = FIG / f"price_{model_tag}_qmle_innovation_diagnostics.png"',
)
nb.cells[6].source = cell6

cell8 = nb.cells[8].source
cell8 = cell8.replace(
    'np.savez(\n    OUT / "levy_increments_recovered.npz",',
    'levy_npz_path = OUT / f"price_{model_tag}_levy_increments_recovered.npz"\nnp.savez(\n    levy_npz_path,',
)
cell8 = cell8.replace(
    'print(f"Selected recovery root lambda = {levy_out[\'lambda_r\']:.6e}")',
    'recovery_half_life_h = np.log(2.0) / (-levy_out["lambda_r"])\n'
    'print(f"Selected recovery root lambda = {levy_out[\'lambda_r\']:.6e}")\n'
    'print(f"Selected recovery root half-life = {recovery_half_life_h:.3f} h = {recovery_half_life_h / 24.0:.3f} d")',
)
cell8 = cell8.replace(
    'print(f"Max smoother observation residual = {levy_out[\'obs_resid_max\']:.3e}")',
    'print(f"Max smoother observation residual = {levy_out[\'obs_resid_max\']:.3e}")\n'
    'print(f"Saved: {levy_npz_path.name}")',
)
nb.cells[8].source = cell8

cell10 = nb.cells[10].source
cell10 = cell10.replace(
    '(OUT / "price_carma43_driver_fits.json").write_text(json.dumps(driver_fits, indent=2))',
    'driver_fits_path = OUT / f"price_{model_tag}_driver_fits.json"\n'
    'driver_fits_path.write_text(json.dumps(driver_fits, indent=2))',
)
cell10 = cell10.replace(
    'print(f"Saved: price_carma43_driver_fits.json")',
    'print(f"Saved: {driver_fits_path.name}")',
)
nb.cells[10].source = cell10

cell11 = nb.cells[11].source.replace(
    'fig_path = FIG / "price_driver_deltaL_fits.png"',
    'fig_path = FIG / f"price_{model_tag}_driver_deltaL_fits.png"',
)
nb.cells[11].source = cell11

cell14 = nb.cells[14].source.replace(
    'summary_path = OUT / "price_carma43_distribution_summary.csv"',
    'summary_path = OUT / f"price_{model_tag}_distribution_summary.csv"',
)
nb.cells[14].source = cell14

cell16 = nb.cells[16].source.replace(
    'fig_path = FIG / "price_carma43_level_distribution_comparison.png"',
    'fig_path = FIG / f"price_{model_tag}_level_distribution_comparison.png"',
)
nb.cells[16].source = cell16

for cell in nb.cells:
    if "outputs" in cell:
        cell["outputs"] = []
    if "execution_count" in cell:
        cell["execution_count"] = None

nbformat.write(nb, NB_PATH)
print(f"Updated {NB_PATH}")
