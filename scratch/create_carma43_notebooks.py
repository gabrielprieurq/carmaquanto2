import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "solar" / "Intensity_Model_solar"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def src(cell):
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def put(cell, text):
    cell["source"] = text


def save(path, nb):
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        else:
            cell.pop("execution_count", None)
            cell.pop("outputs", None)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def need(text, old, new):
    if old not in text:
        raise RuntimeError("Expected source fragment not found.")
    return text.replace(old, new, 1)


def build_order():
    nb = deepcopy(load(CODE / "02order_solar.ipynb"))
    c = nb["cells"]
    put(c[0], "# Solar CARMA(4,3) order and ACF-based initialisation\n\nTwo real AR roots plus one fixed 24-hour complex-conjugate pair. Outputs are\nversioned for CARMA(4,3) and do not overwrite the CARMA(5,4) calibration.\n")
    put(c[1], src(c[1]).replace("02order_solar.ipynb", "04order_solar.ipynb"))
    t = src(c[2])
    t = need(t, 'Kr, Kc = 3, 1\nm = Kr + Kc\nn_hl = Kr + Kc\ncomponent_labels = np.array(["real_fast", "real_mid", "real_slow", "osc_daily"])',
                'Kr, Kc = 2, 1\nm = Kr + Kc\nn_hl = Kr + Kc\ncomponent_labels = np.array(["real_fast", "real_mid", "osc_daily"])')
    t = need(t, '    "lower_h": [2.0, 5.0, 10*24.0, 12.0],\n    "upper_h": [24.0, 7 * 24.0, 365.0 * 24.0, 180.0 * 24.0],',
                '    "lower_h": [2.0, 10.0, 12.0],\n    "upper_h": [24.0, 7 * 24.0, 180.0 * 24.0],')
    old_starts = '''start_hl = np.array([
    [2.0, 24.0, 30.0 * 24.0, 72.0],
    [4.0, 36.0, 60.0 * 24.0, 96.0],
    [1.0, 12.0, 120.0 * 24.0, 48.0],
    [6.0, 48.0, 180.0 * 24.0, 120.0],
    [3.0, 30.0, 240.0 * 24.0, 96.0],
    [8.0, 72.0, 365.0 * 24.0, 96.0],
    [10.0, 120.0, 90.0 * 24.0, 168.0],
], dtype=float)'''
    new_starts = '''start_hl = np.array([
    [2.0, 24.0, 72.0],
    [4.0, 36.0, 96.0],
    [1.0, 12.0, 48.0],
    [6.0, 48.0, 120.0],
    [3.0, 72.0, 96.0],
    [8.0, 120.0, 96.0],
    [10.0, 160.0, 168.0],
], dtype=float)'''
    t = need(t, old_starts, new_starts).replace("[real fast, real mid, real slow, osc daily]", "[real fast, real mid, osc daily]")
    put(c[2], t)
    t = src(c[3]).replace("# Spectral factorisation CARMA(5,4) + save", "# Spectral factorisation CARMA(4,3) + versioned export")
    for old, new in [
        ("solar_multiscale_carma_selected.json", "solar_multiscale_carma43_selected.json"),
        ("solar_carma_b_init.json", "solar_carma43_b_init.json"),
        ("solar_multiscale_carma_selected_components.csv", "solar_multiscale_carma43_selected_components.csv"),
        ("Solar XtQ. Kr=3 Kc=1 period=24h fixed;", "Solar XtQ. Kr=2 Kc=1 period=24h fixed;"),
    ]:
        t = t.replace(old, new)
    put(c[3], t)
    c = c[:4]
    nb["cells"] = c
    return nb


THETA = r'''def theta_from_initial_roots():
    ar_hl = np.sort(half_life_from_kappa(-ar_real_init))
    ma_hl = float(half_life_from_kappa(-ma_real_init[0]))
    ma_c_hl = float(half_life_from_kappa(-ma_complex_init[0].real))
    ma_omega = float(abs(ma_complex_init[0].imag))
    return np.r_[np.log(ar_hl), np.log(ma_hl), np.log(ma_c_hl), ma_omega]


def unpack_theta(theta):
    ar_fast_hl, ar_mid_hl = np.exp(theta[:2])
    ma_hl, ma_c_hl, ma_omega = float(np.exp(theta[2])), float(np.exp(theta[3])), float(theta[4])
    ar_kappa = kappa_from_half_life([ar_fast_hl, ar_mid_hl])
    ma_kappa, ma_c_kappa = float(kappa_from_half_life(ma_hl)), float(kappa_from_half_life(ma_c_hl))
    ar_roots = np.array([-ar_kappa[0], -ar_kappa[1],
                         complex(-AR_DAILY_KAPPA_FIXED, omega_ar_fixed),
                         complex(-AR_DAILY_KAPPA_FIXED, -omega_ar_fixed)], dtype=complex)
    ma_roots = np.array([-ma_kappa, complex(-ma_c_kappa, ma_omega),
                         complex(-ma_c_kappa, -ma_omega)], dtype=complex)
    return ar_roots, ma_roots, coeffs_from_roots(ar_roots), b_coeffs_from_roots(ma_roots)


'''


def build_mle():
    nb = deepcopy(load(CODE / "03mle_solar.ipynb"))
    c = nb["cells"]
    put(c[0], "# Solar CARMA(4,3) QMLE, Levy recovery, and physical solar diagnostics\n\nThis notebook consumes the CARMA(4,3) initialisation from 04order_solar and runs\nthe same exact QMLE, Levy recovery, NIG calibration, simulations, and PPCs.\n")
    c.pop(1)
    put(c[1], src(c[1]).replace("03mle_solar.ipynb", "04mle_solar.ipynb"))
    put(c[2], '''# Discrete-time ARMA(4,3) diagnostic
import statsmodels.api as sm
y_arma = np.asarray(pr, dtype=float)
y_arma = y_arma[np.isfinite(y_arma)] - np.nanmean(y_arma)
arma43 = sm.tsa.SARIMAX(y_arma, order=(4, 0, 3), trend="n", enforce_stationarity=True, enforce_invertibility=True)
arma43_res = arma43.fit(method="lbfgs", maxiter=1000, disp=False)
print(f"ARMA(4,3): loglik={arma43_res.llf:.3f}; AIC={arma43_res.aic:.3f}; BIC={arma43_res.bic:.3f}")
display(pd.DataFrame({"param": arma43_res.param_names, "value": arma43_res.params}))
''')
    t = src(c[4])
    t = t.replace('"solar_multiscale_carma_selected.json"', '"solar_multiscale_carma43_selected.json"')
    t = t.replace('if p != 5 or q != 4:\n    raise ValueError(f"This notebook expects the solar CARMA(5,4) topology from 02order_solar; got CARMA({p},{q}).")',
                  'if p != 4 or q != 3:\n    raise ValueError(f"This notebook expects CARMA(4,3) from 04order_solar; got CARMA({p},{q}).")')
    t = t.replace('len(ar_real_init) != 3', 'len(ar_real_init) != 2').replace("Expected three real AR roots", "Expected two real AR roots")
    t = t.replace('len(ma_real_init) != 2', 'len(ma_real_init) != 1').replace("Expected two real MA roots", "Expected one real MA root")
    put(c[4], t)
    t = src(c[6])
    a, b = t.index("def theta_from_initial_roots():"), t.index("def acf_model_from_state")
    t = t[:a] + THETA + t[b:]
    old = '''AR_FAST_BOUNDS_H = (2, 24)
AR_MID_BOUNDS_H = (5.0, 7*24.0)
AR_SLOW_BOUNDS_H = (3*24.0, 365.0 * 24.0)

MA_REAL_BOUNDS_H = (0.3, 720.0)
MA_COMPLEX_HL_BOUNDS_H = (0.3, 720.0)
MA_COMPLEX_OMEGA_BOUNDS = (1e-4, np.pi - 1e-4)

theta0 = theta_from_initial_roots()

lower = np.r_[
    np.log(AR_FAST_BOUNDS_H[0]),
    np.log(AR_MID_BOUNDS_H[0]),
    np.log(AR_SLOW_BOUNDS_H[0]),
    [np.log(MA_REAL_BOUNDS_H[0])] * 2,
    np.log(MA_COMPLEX_HL_BOUNDS_H[0]),
    MA_COMPLEX_OMEGA_BOUNDS[0],
]

upper = np.r_[
    np.log(AR_FAST_BOUNDS_H[1]),
    np.log(AR_MID_BOUNDS_H[1]),
    np.log(AR_SLOW_BOUNDS_H[1]),
    [np.log(MA_REAL_BOUNDS_H[1])] * 2,
    np.log(MA_COMPLEX_HL_BOUNDS_H[1]),
    MA_COMPLEX_OMEGA_BOUNDS[1],
]'''
    new = '''AR_FAST_BOUNDS_H = (2.0, 24.0)
AR_MID_BOUNDS_H = (5.0, 7.0 * 24.0)
MA_REAL_BOUNDS_H = (0.3, 720.0)
MA_COMPLEX_HL_BOUNDS_H = (0.3, 720.0)
MA_COMPLEX_OMEGA_BOUNDS = (1e-4, np.pi - 1e-4)

theta0 = theta_from_initial_roots()
lower = np.r_[np.log(AR_FAST_BOUNDS_H[0]), np.log(AR_MID_BOUNDS_H[0]), np.log(MA_REAL_BOUNDS_H[0]), np.log(MA_COMPLEX_HL_BOUNDS_H[0]), MA_COMPLEX_OMEGA_BOUNDS[0]]
upper = np.r_[np.log(AR_FAST_BOUNDS_H[1]), np.log(AR_MID_BOUNDS_H[1]), np.log(MA_REAL_BOUNDS_H[1]), np.log(MA_COMPLEX_HL_BOUNDS_H[1]), MA_COMPLEX_OMEGA_BOUNDS[1]]'''
    t = need(t, old, new).replace('print(f"  slow  = {np.exp(theta0[2]):.3f} h")\n', '')
    t = t.replace('start[:6] += rng.normal(0.0, SD_LOG_HL, size=6)\n        start[6] += rng.normal(0.0, SD_OMEGA_MA)',
                  'start[:4] += rng.normal(0.0, SD_LOG_HL, size=4)\n        start[4] += rng.normal(0.0, SD_OMEGA_MA)')
    t = t.replace('"ar_complex_half_life_fixed_from_02"', '"ar_complex_half_life_fixed_from_04"')
    t = t.replace('"source": "solar/Intensity_Model_solar/03mle_solar.ipynb"', '"source": "solar/Intensity_Model_solar/04mle_solar.ipynb"')
    t = t.replace('AR daily root fixed from notebook 02', 'daily AR pair fixed from 04order')
    t = t.replace('(OUT / "solar_carma_qmle_result.json").write_text(json.dumps(qmle_json, indent=2))\n', '')
    put(c[6], t)
    t = src(c[7]).replace('["AR real 1", "AR real 2", "AR real 3", "AR daily"]', '["AR real fast", "AR real mid", "AR daily"]')
    put(c[7], t)
    put(c[8], '''# CARMA(4,3) QMLE report
print(f"CARMA({p},{q}): objective={opt['objective']:.12f}; loglik={qmle['loglik']:.6f}; ACF MSE={opt['acf_mse']:.12f}")
print(f"m={m_hat:.12e}; nu2={nu2_hat:.12e}; stationary mean={m_hat * dc_gain:.12e}")
print(f"AR real bounds (h): fast={AR_FAST_BOUNDS_H}, mid={AR_MID_BOUNDS_H}; daily pair is fixed from 04order.")
for family, rr in (("AR", roots_complex), ("MA", ma_roots_complex)):
    print(family, [complex(z) for z in rr])
''')
    put(c[23], src(c[23]).replace('zip(["fast", "mid", "slow"], ar_real_hl)', 'zip(["fast", "mid"], ar_real_hl)'))
    return nb


save(CODE / "04order_solar.ipynb", build_order())
save(CODE / "04mle_solar.ipynb", build_mle())
print("Created CARMA(4,3) notebooks.")

