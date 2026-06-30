from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import numpy as np
from scipy.linalg import solve_sylvester

_TRAPEZOID = getattr(np, "trapezoid", None)
if _TRAPEZOID is None:
    _TRAPEZOID = np.trapz

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

Fonseca_model = None


Array = np.ndarray


@dataclass
class WishartPanelStatistics:
    """Discrete approximations of the path statistics used by the Wishart MLE."""

    R: Array
    U: float
    Q: float
    Z: float
    x0_sum: Array
    xT_sum: Array
    n_paths: int
    n_steps: int
    T: float
    dt: float


@dataclass
class WishartMLEFit:
    """Container for calibration outputs."""

    alpha: Optional[float]
    b: Array
    A: Optional[Array] = None
    ata: Optional[Array] = None
    transformed_b: Optional[Array] = None
    statistics: Optional[WishartPanelStatistics] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    method: str = ""


class WishartMLE:
    r"""
    Maximum-likelihood calibration utilities for the Wishart covariance process

        dX_t = (\alpha A^\top A + b X_t + X_t b^\top) dt
             + \sqrt{X_t} dW_t A + A^\top dW_t^\top \sqrt{X_t}.

    The implementation follows the observable-path likelihood formulas in
    Alfonsi, Kebaier and Rey (2016), but replaces the continuous-time
    integrals/quadratic covariations by discrete approximations on the
    simulated time grid returned by `Fonseca_model.gen(...)`.

    Two estimation layers are provided.

    1. Identity-diffusion case (`A = I`) with symmetric drift `b = b^\top`:
       the paper gives explicit estimators based on

           R_T = \int_0^T X_s ds,
           U_T = \int_0^T Tr[X_s^{-1}] ds,
           Q_T = U_T^{-1},
           Z_T = log det(X_T) - log det(X_0).

       For one path, the closed-form estimator is

           \hat b_T = L^{-1}_{R_T, T^2 Q_T}
                      (X_T - X_0 - T(Q_T Z_T + 1 + d) I_d),

           \hat \alpha_T = 1 + d + Q_T Z_T - 2 T Q_T Tr[\hat b_T],

       where

           L_{X,a}(Y) = YX + XY - 2 a Tr[Y] I_d.

       The implementation also supports a panel of independent paths by summing
       the log-likelihood contributions across paths. In that case the same
       formulas hold with aggregated sufficient statistics and `T` replaced by
       `n_paths * T` in the scalar drift terms.

    2. Full Wishart calibration:
       first estimate `A^\top A` from realized quadratic covariations, then use
       the congruence transform from the paper

           Y_t = (A^\top)^{-1} X_t A^{-1},

       so that the transformed process has identity diffusion matrix.
       The transformed symmetric drift is estimated by the explicit formulas
       above and mapped back to the original coordinates.

       The module exposes two `b`-transformation conventions:

       - `simulator`: the pathwise congruence induced by the local
         `Fonseca_model` parameterization,
       - `paper`: the congruence-law convention stated in the paper.

       The default is `simulator`, because the user asked to calibrate against
       the exact dynamics implemented in this repository.

    This matches the setup used in the paper's numerical section and in the
    local `Fonseca_model` simulation workflow.
    """

    def __init__(self, eps: float = 1e-10):
        self.eps = float(eps)

    # ------------------------------------------------------------------
    # Simulation helper matching the user's current Monte Carlo workflow
    # ------------------------------------------------------------------
    @staticmethod
    def simulate_fonseca_paths(
        *,
        alpha: float,
        A: Array,
        b: Array,
        rho: Array,
        r: float,
        Sigma0: Array,
        y0: Array,
        T: float,
        n_steps: int,
        n_paths: int,
        comb: str = "1",
        trace: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[str, Array]:
        try:
            from application import Fonseca_model as fonseca_model_cls
        except ImportError as exc:
            raise ImportError(
                "Fonseca_model is unavailable in this standalone bundle. "
                "The Gabriel notebooks use the direct Wishart innovation simulator instead."
            ) from exc
        if seed is not None:
            np.random.seed(seed)

        gen = fonseca_model_cls(r=r, rho=np.asarray(rho, dtype=float), coef=alpha, b=np.asarray(b, dtype=float), a=np.asarray(A, dtype=float))
        xt, yt = gen.gen(
            x=np.asarray(Sigma0, dtype=float),
            y=np.asarray(y0, dtype=float),
            T=float(T),
            N=int(n_steps),
            num=int(n_paths),
            comb=comb,
            trace=trace,
        )
        return {
            "Sigma_paths": xt,
            "X_paths": yt,
            "S_paths": np.exp(yt),
            "t_grid": np.linspace(0.0, float(T), int(n_steps) + 1),
        }

    # ------------------------------------------------------------------
    # Generic linear-algebra helpers
    # ------------------------------------------------------------------
    def _symmetrize(self, x: Array) -> Array:
        return 0.5 * (x + np.swapaxes(x, -1, -2))

    def _as_panel(self, sigma_paths: Array) -> Array:
        x = np.asarray(sigma_paths, dtype=float)
        if x.ndim == 3:
            x = x[None, ...]
        if x.ndim != 4:
            raise ValueError("sigma_paths must have shape (N+1,d,d) or (P,N+1,d,d).")
        if x.shape[-1] != x.shape[-2]:
            raise ValueError("The last two dimensions must form square covariance matrices.")
        return self._symmetrize(x)

    def _time_grid(self, paths: Array, *, T: Optional[float], t_grid: Optional[Array]) -> tuple[Array, float]:
        n_steps = paths.shape[1] - 1
        if t_grid is None:
            if T is None:
                raise ValueError("Either T or t_grid must be provided.")
            dt = float(T) / n_steps
            return np.linspace(0.0, float(T), n_steps + 1), dt
        grid = np.asarray(t_grid, dtype=float)
        if grid.shape != (n_steps + 1,):
            raise ValueError("t_grid has incompatible shape.")
        dt = float(np.mean(np.diff(grid)))
        return grid, dt

    def _regularize_spd(self, x: Array) -> Array:
        d = x.shape[-1]
        return self._symmetrize(x) + self.eps * np.eye(d)

    def _safe_logdet(self, x: Array) -> Array:
        sign, logdet = np.linalg.slogdet(self._regularize_spd(x))
        if np.any(sign <= 0.0):
            raise np.linalg.LinAlgError("Encountered a non-positive definite matrix while computing log-det.")
        return logdet

    def _safe_inv(self, x: Array) -> Array:
        return np.linalg.inv(self._regularize_spd(x))

    def _solve_Lx(self, x: Array, c: Array) -> Array:
        y = solve_sylvester(x, x, c)
        return self._symmetrize(y)

    def _solve_Lx_a(self, x: Array, a: float, c: Array) -> Array:
        r_inv = self._safe_inv(x)
        s_c = self._solve_Lx(x, c)
        denom = 1.0 - float(a) * np.trace(r_inv)
        if abs(denom) < self.eps:
            raise np.linalg.LinAlgError("The operator L_{X,a} is nearly singular for the supplied data.")
        trace_y = np.trace(s_c) / denom
        y = s_c + float(a) * trace_y * r_inv
        return self._symmetrize(y)

    # ------------------------------------------------------------------
    # Observable path statistics
    # ------------------------------------------------------------------
    def compute_panel_statistics(
        self,
        sigma_paths: Array,
        *,
        T: Optional[float] = None,
        t_grid: Optional[Array] = None,
    ) -> WishartPanelStatistics:
        paths = self._as_panel(sigma_paths)
        grid, dt = self._time_grid(paths, T=T, t_grid=t_grid)
        T_total = float(grid[-1] - grid[0])

        R_each = _TRAPEZOID(paths, x=grid, axis=1)
        x0 = paths[:, 0]
        xT = paths[:, -1]

        inv_paths = self._safe_inv(paths)
        trace_inv = np.trace(inv_paths, axis1=-2, axis2=-1)
        U_each = _TRAPEZOID(trace_inv, x=grid, axis=1)
        U = float(U_each.sum())
        if U <= 0.0:
            raise ValueError("The integral of Tr[X_t^{-1}] must be positive.")

        Z = float(np.sum(self._safe_logdet(xT) - self._safe_logdet(x0)))

        return WishartPanelStatistics(
            R=np.sum(R_each, axis=0),
            U=U,
            Q=1.0 / U,
            Z=Z,
            x0_sum=np.sum(x0, axis=0),
            xT_sum=np.sum(xT, axis=0),
            n_paths=paths.shape[0],
            n_steps=paths.shape[1] - 1,
            T=T_total,
            dt=dt,
        )

    # ------------------------------------------------------------------
    # Diffusion-matrix estimation from realized covariations
    # ------------------------------------------------------------------
    def estimate_ata(
        self,
        sigma_paths: Array,
        *,
        T: Optional[float] = None,
        t_grid: Optional[Array] = None,
        project_spd: bool = True,
    ) -> Array:
        paths = self._as_panel(sigma_paths)
        grid, _ = self._time_grid(paths, T=T, t_grid=t_grid)
        d = paths.shape[-1]

        ints = _TRAPEZOID(paths, x=grid, axis=1).sum(axis=0)
        dX = np.diff(paths, axis=1)

        ata = np.zeros((d, d), dtype=float)
        for i in range(d):
            denom = max(float(ints[i, i]), self.eps)
            qv = float(np.sum(dX[:, :, i, i] ** 2))
            ata[i, i] = 0.25 * qv / denom

        for i in range(d):
            for j in range(i + 1, d):
                denom_i = max(float(ints[i, i]), self.eps)
                denom_j = max(float(ints[j, j]), self.eps)
                int_ij = float(ints[i, j])

                qcov_i = float(np.sum(dX[:, :, i, j] * dX[:, :, i, i]))
                qcov_j = float(np.sum(dX[:, :, i, j] * dX[:, :, j, j]))

                gij_i = (0.5 * qcov_i - ata[i, i] * int_ij) / denom_i
                gij_j = (0.5 * qcov_j - ata[j, j] * int_ij) / denom_j
                ata[i, j] = ata[j, i] = 0.5 * (gij_i + gij_j)

        ata = self._symmetrize(ata)
        if not project_spd:
            return ata

        evals, evecs = np.linalg.eigh(ata)
        evals = np.maximum(evals, self.eps)
        return self._symmetrize(evecs @ np.diag(evals) @ evecs.T)

    def ata_to_cholesky_factor(self, ata: Array) -> Array:
        return np.linalg.cholesky(self._regularize_spd(ata)).T

    def transform_to_identity_diffusion(self, sigma_paths: Array, A: Array) -> Array:
        paths = self._as_panel(sigma_paths)
        A = np.asarray(A, dtype=float)
        A_inv = np.linalg.inv(A)
        A_inv_T = A_inv.T
        transformed = A_inv_T[None, None, :, :] @ paths @ A_inv[None, None, :, :]
        return self._symmetrize(transformed)

    def transform_b_to_identity_coordinates(self, b: Array, A: Array, *, convention: str = "simulator") -> Array:
        b = np.asarray(b, dtype=float)
        A = np.asarray(A, dtype=float)
        A_inv = np.linalg.inv(A)
        A_inv_T = A_inv.T

        if convention == "simulator":
            return A_inv_T @ b @ A.T
        if convention == "paper":
            return A_inv_T @ b @ A_inv
        raise ValueError("convention must be either 'simulator' or 'paper'.")

    def recover_original_b(self, transformed_b: Array, A: Array, *, convention: str = "simulator") -> Array:
        transformed_b = np.asarray(transformed_b, dtype=float)
        A = np.asarray(A, dtype=float)

        if convention == "simulator":
            return self._symmetrize(A.T @ transformed_b @ np.linalg.inv(A.T))
        if convention == "paper":
            return self._symmetrize(A.T @ transformed_b @ A)
        raise ValueError("convention must be either 'simulator' or 'paper'.")

    # ------------------------------------------------------------------
    # Closed-form symmetric-drift estimators
    # ------------------------------------------------------------------
    def fit_identity_joint(
        self,
        sigma_paths: Array,
        *,
        T: Optional[float] = None,
        t_grid: Optional[Array] = None,
    ) -> WishartMLEFit:
        stats = self.compute_panel_statistics(sigma_paths, T=T, t_grid=t_grid)
        d = stats.R.shape[0]
        eye = np.eye(d)

        scalar_horizon = stats.n_paths * stats.T
        rhs = stats.xT_sum - stats.x0_sum - scalar_horizon * (stats.Q * stats.Z + 1.0 + d) * eye
        b_hat = self._solve_Lx_a(stats.R, scalar_horizon**2 * stats.Q, rhs)
        alpha_hat = 1.0 + d + stats.Q * stats.Z - 2.0 * scalar_horizon * stats.Q * np.trace(b_hat)

        diagnostics = {
            "trace_b": float(np.trace(b_hat)),
            "Q": float(stats.Q),
            "Z": float(stats.Z),
            "scalar_horizon": float(scalar_horizon),
        }
        return WishartMLEFit(
            alpha=float(alpha_hat),
            b=self._symmetrize(b_hat),
            statistics=stats,
            diagnostics=diagnostics,
            method="identity_joint_closed_form",
        )

    def fit_identity_known_alpha(
        self,
        sigma_paths: Array,
        *,
        alpha: float,
        T: Optional[float] = None,
        t_grid: Optional[Array] = None,
    ) -> WishartMLEFit:
        stats = self.compute_panel_statistics(sigma_paths, T=T, t_grid=t_grid)
        d = stats.R.shape[0]
        eye = np.eye(d)
        rhs = stats.xT_sum - stats.x0_sum - float(alpha) * stats.n_paths * stats.T * eye
        b_hat = self._solve_Lx(stats.R, rhs)

        return WishartMLEFit(
            alpha=float(alpha),
            b=self._symmetrize(b_hat),
            statistics=stats,
            diagnostics={"scalar_horizon": float(stats.n_paths * stats.T)},
            method="identity_known_alpha_closed_form",
        )

    # ------------------------------------------------------------------
    # Full-parameter calibration via A^T A and Wishart normalization
    # ------------------------------------------------------------------
    def fit_full_joint(
        self,
        sigma_paths: Array,
        *,
        T: Optional[float] = None,
        t_grid: Optional[Array] = None,
        b_convention: str = "simulator",
    ) -> WishartMLEFit:
        ata_hat = self.estimate_ata(sigma_paths, T=T, t_grid=t_grid)
        A_hat = self.ata_to_cholesky_factor(ata_hat)
        transformed = self.transform_to_identity_diffusion(sigma_paths, A_hat)
        fit_tilde = self.fit_identity_joint(transformed, T=T, t_grid=t_grid)

        b_hat = self.recover_original_b(fit_tilde.b, A_hat, convention=b_convention)
        diagnostics = dict(fit_tilde.diagnostics)
        diagnostics["transformed_symmetry_error"] = float(np.linalg.norm(fit_tilde.b - fit_tilde.b.T))
        diagnostics["b_convention"] = b_convention

        return WishartMLEFit(
            alpha=fit_tilde.alpha,
            b=b_hat,
            A=A_hat,
            ata=ata_hat,
            transformed_b=fit_tilde.b,
            statistics=fit_tilde.statistics,
            diagnostics=diagnostics,
            method="full_joint_via_ata_and_identity_transform",
        )

    def fit_full_known_alpha(
        self,
        sigma_paths: Array,
        *,
        alpha: float,
        T: Optional[float] = None,
        t_grid: Optional[Array] = None,
        b_convention: str = "simulator",
    ) -> WishartMLEFit:
        ata_hat = self.estimate_ata(sigma_paths, T=T, t_grid=t_grid)
        A_hat = self.ata_to_cholesky_factor(ata_hat)
        transformed = self.transform_to_identity_diffusion(sigma_paths, A_hat)
        fit_tilde = self.fit_identity_known_alpha(transformed, alpha=alpha, T=T, t_grid=t_grid)
        b_hat = self.recover_original_b(fit_tilde.b, A_hat, convention=b_convention)

        return WishartMLEFit(
            alpha=float(alpha),
            b=b_hat,
            A=A_hat,
            ata=ata_hat,
            transformed_b=fit_tilde.b,
            statistics=fit_tilde.statistics,
            diagnostics=dict(fit_tilde.diagnostics),
            method="full_known_alpha_via_ata_and_identity_transform",
        )

    # ------------------------------------------------------------------
    # Per-path estimators for Monte Carlo histograms
    # ------------------------------------------------------------------
    def fit_many_identity_joint(
        self,
        sigma_paths: Array,
        *,
        T: Optional[float] = None,
        t_grid: Optional[Array] = None,
    ) -> list[WishartMLEFit]:
        panel = self._as_panel(sigma_paths)
        return [self.fit_identity_joint(panel[p], T=T, t_grid=t_grid) for p in range(panel.shape[0])]

    def fit_many_identity_known_alpha(
        self,
        sigma_paths: Array,
        *,
        alpha: float,
        T: Optional[float] = None,
        t_grid: Optional[Array] = None,
    ) -> list[WishartMLEFit]:
        panel = self._as_panel(sigma_paths)
        return [self.fit_identity_known_alpha(panel[p], alpha=alpha, T=T, t_grid=t_grid) for p in range(panel.shape[0])]

    def fit_many_full_joint(
        self,
        sigma_paths: Array,
        *,
        T: Optional[float] = None,
        t_grid: Optional[Array] = None,
        b_convention: str = "simulator",
    ) -> list[WishartMLEFit]:
        panel = self._as_panel(sigma_paths)
        return [self.fit_full_joint(panel[p], T=T, t_grid=t_grid, b_convention=b_convention) for p in range(panel.shape[0])]

    # ------------------------------------------------------------------
    # Diagnostics used by the notebook experiments
    # ------------------------------------------------------------------
    def stationary_mean(self, *, alpha: float, b: Array, ata: Array) -> Array:
        b = np.asarray(b, dtype=float)
        ata = np.asarray(ata, dtype=float)
        rhs = -float(alpha) * ata
        mean = solve_sylvester(b, b.T, rhs)
        return self._symmetrize(mean)

    def drift_matrix(self, x: Array, *, alpha: float, b: Array, ata: Array) -> Array:
        x = np.asarray(x, dtype=float)
        b = np.asarray(b, dtype=float)
        ata = np.asarray(ata, dtype=float)
        return float(alpha) * ata + b @ x + x @ b.T

    def drift_residual_summary(
        self,
        sigma_paths: Array,
        *,
        alpha: float,
        b: Array,
        ata: Array,
        T: Optional[float] = None,
        t_grid: Optional[Array] = None,
    ) -> Dict[str, float]:
        paths = self._as_panel(sigma_paths)
        grid, dt = self._time_grid(paths, T=T, t_grid=t_grid)
        _ = grid

        x_left = paths[:, :-1]
        dx = np.diff(paths, axis=1)

        drift = np.empty_like(x_left)
        for i in range(paths.shape[0]):
            for k in range(paths.shape[1] - 1):
                drift[i, k] = self.drift_matrix(x_left[i, k], alpha=alpha, b=b, ata=ata)

        residuals = dx - dt * drift
        frob = np.sqrt(np.sum(residuals**2, axis=(-2, -1)))
        return {
            "rmse_frobenius": float(np.sqrt(np.mean(frob**2))),
            "mean_frobenius": float(np.mean(frob)),
            "median_frobenius": float(np.median(frob)),
        }


__all__ = [
    "WishartMLE",
    "WishartMLEFit",
    "WishartPanelStatistics",
]
