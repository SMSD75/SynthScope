"""CuRe stage 2: the interference-suppressed source-response subspace (arXiv 2609.14316, Sec. 3.2, Eq. 5-11).

The official release has no training code; this follows the paper's equations. For trajectories T in {RF, RR, FF}
(real->generated, real->real, generated->generated) with features z_{i,a} of the mixtures on the grid
A = {0, .2, .4, .6, .8, 1}:

    b_i^T  = sum_a (a - a_bar)(z_{i,a} - z_bar_i) / sum_a (a - a_bar)^2      (Eq. 6, the per-trajectory slope)
    C_F    = E_i[b_i^RF b_i^RF^T]                                           (Eq. 7)
    C_M    = E_i[b_i^RR b_i^RR^T] + E_i[b_i^FF b_i^FF^T]                     (Eq. 8)
    C_S    = Cov_i(z_bar_i^RF)                                              (Eq. 9)

each divided by its trace, then the generalised eigenproblem

    C_F w = g (lambda_s C_S + lambda_m C_M + eps I) w                       (Eq. 10)

whose k = 128 leading eigenvectors are W [1024, 128] (Eq. 11); mu is the mean RF trajectory feature (Sec. 3.3).

Unspecified in the paper (documented assumptions, see the README "Training recipe"): lambda_s, lambda_m, eps (defaults
below); the scale of the eigenvectors (here B-orthonormal, W^T B W = I, the convention of LAPACK's / scipy's generalised
symmetric solver; the released W is not unit-norm, its columns have norms 6.8-8.8); C_S uses the 1/N covariance (the trace
normalisation removes the factor). Everything is computed in float64.
"""
from __future__ import annotations

import torch

GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_LAMBDA_S = 1.0
DEFAULT_LAMBDA_M = 1.0
DEFAULT_EPS = 1e-4


def slope_weights(grid=GRID) -> torch.Tensor:
    """``c_a = (a - a_bar) / sum (a - a_bar)^2`` so that Eq. 6 is ``b_i = sum_a c_a z_{i,a}`` (the ``z_bar`` term vanishes
    because ``sum_a (a - a_bar) = 0``)."""
    a = torch.tensor(grid, dtype=torch.float64)
    centered = a - a.mean()
    return centered / (centered ** 2).sum()


def trajectory_summary(z: torch.Tensor, grid=GRID) -> tuple[torch.Tensor, torch.Tensor]:
    """``z`` [N, len(grid), D] trajectory features -> (slopes b [N, D] (Eq. 6), means z_bar [N, D]), float64."""
    z = z.double()
    c = slope_weights(grid).to(z.device)
    return torch.einsum("a,nad->nd", c, z), z.mean(dim=1)


def second_moment(b: torch.Tensor) -> torch.Tensor:
    """``E_i[b_i b_i^T]`` (uncentred), float64."""
    b = b.double()
    return b.T @ b / b.shape[0]


def covariance(x: torch.Tensor) -> torch.Tensor:
    """``Cov_i(x_i)`` with the 1/N normalisation, float64."""
    x = x.double()
    x = x - x.mean(dim=0, keepdim=True)
    return x.T @ x / x.shape[0]


def trace_normalised(c: torch.Tensor) -> torch.Tensor:
    return c / torch.trace(c)


def statistics(b_rf, b_rr, b_ff, zbar_rf) -> dict:
    """Eq. 7-9, each normalised by its trace, plus mu (the mean RF trajectory feature = the mean of the z_bar^RF)."""
    return {
        "C_F": trace_normalised(second_moment(b_rf)),
        "C_M": trace_normalised(second_moment(b_rr) + second_moment(b_ff)),
        "C_S": trace_normalised(covariance(zbar_rf)),
        "mu": zbar_rf.double().mean(dim=0),
    }


def solve(c_f, c_m, c_s, k=128, lambda_s=DEFAULT_LAMBDA_S, lambda_m=DEFAULT_LAMBDA_M, eps=DEFAULT_EPS):
    """Eq. 10-11: the ``k`` largest generalised eigenpairs of ``C_F w = g B w``, ``B = lambda_s C_S + lambda_m C_M + eps I``.
    Cholesky-whitened symmetric problem: ``B = L L^T``, ``M = L^-1 C_F L^-T = V G V^T``, ``w = L^-T v`` (so
    ``W^T B W = I``). Returns ``(W [D, k], gammas [k])``, eigenvalues in descending order, float64."""
    c_f, c_m, c_s = c_f.double(), c_m.double(), c_s.double()
    d = c_f.shape[0]
    b = lambda_s * c_s + lambda_m * c_m + eps * torch.eye(d, dtype=torch.float64, device=c_f.device)
    b = (b + b.T) / 2
    chol = torch.linalg.cholesky(b)
    eye = torch.eye(d, dtype=torch.float64, device=c_f.device)
    l_inv = torch.linalg.solve_triangular(chol, eye, upper=False)
    m = l_inv @ ((c_f + c_f.T) / 2) @ l_inv.T
    gammas, v = torch.linalg.eigh((m + m.T) / 2)               # ascending
    order = torch.argsort(gammas, descending=True)[:k]
    w = l_inv.T @ v[:, order]
    return w, gammas[order]


def subspace_overlap(w_a: torch.Tensor, w_b: torch.Tensor) -> float:
    """Mean squared cosine of the principal angles between span(w_a) and span(w_b) (1 = same subspace; two random
    k-dim subspaces of R^D give about k / D)."""
    qa, _ = torch.linalg.qr(w_a.double())
    qb, _ = torch.linalg.qr(w_b.double())
    s = torch.linalg.svdvals(qa.T @ qb)
    return float((s ** 2).mean())
