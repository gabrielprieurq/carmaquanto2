from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "03mle.ipynb"

nb = nbformat.read(NB_PATH, as_version=4)

cell3 = nb.cells[3].source
anchor = """def psd_sqrt(M, tol=1e-12):
    M = 0.5 * (M + M.T)
    vals, vecs = np.linalg.eigh(M)
    if vals.min() < -tol:
        raise ValueError(f"Matrix is not PSD; min eigenvalue={vals.min():.3e}")
    return vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None)))


def distribution_summary(x):
"""
insert = """def psd_sqrt(M, tol=1e-12):
    M = 0.5 * (M + M.T)
    vals, vecs = np.linalg.eigh(M)
    if vals.min() < -tol:
        raise ValueError(f"Matrix is not PSD; min eigenvalue={vals.min():.3e}")
    return vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None)))


def discrete_noise_covariance(A, ep, Delta=1.0):
    # Van Loan block exponential for int_0^Delta exp(Au) e e' exp(A'u) du.
    A = np.asarray(A, dtype=float)
    ep = np.asarray(ep, dtype=float)
    p = A.shape[0]
    M = np.zeros((2 * p, 2 * p))
    M[:p, :p] = A
    M[:p, p:] = np.outer(ep, ep)
    M[p:, p:] = -A.T
    E = expm(M * Delta)
    F_block = E[:p, :p]
    Q = E[:p, p:] @ F_block.T
    return 0.5 * (Q + Q.T)


def distribution_summary(x):
"""
if anchor not in cell3:
    raise RuntimeError("Could not insert discrete_noise_covariance in cell 3")
cell3 = cell3.replace(anchor, insert)
cell3 = cell3.replace(
    """Q_base = Pi1 - F @ Pi1 @ F.T
Q_base = 0.5 * (Q_base + Q_base.T)""",
    "Q_base = discrete_noise_covariance(A, ep, Delta)",
)
nb.cells[3].source = cell3

for idx in [5, 8]:
    src = nb.cells[idx].source
    src = src.replace(
        """Qb_loc = Pi1_loc - F_loc @ Pi1_loc @ F_loc.T
    Qb_loc = 0.5 * (Qb_loc + Qb_loc.T)""",
        "Qb_loc = discrete_noise_covariance(A_loc, ep_loc, Delta)",
    )
    nb.cells[idx].source = src

for cell in nb.cells:
    if "outputs" in cell:
        cell["outputs"] = []
    if "execution_count" in cell:
        cell["execution_count"] = None

nbformat.write(nb, NB_PATH)
print(f"Updated {NB_PATH}")
