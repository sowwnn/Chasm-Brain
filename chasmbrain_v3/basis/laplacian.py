"""Graph Laplacian eigenmodes (spec §4). Saved to basis/laplacian_subj{N}.npz."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .nested import NestedBasis


def laplacian(A: sp.csr_matrix, normalized: bool = True) -> sp.csr_matrix:
    """L = I - D^{-1/2} A D^{-1/2} (normalized) or L = D - A (unnormalized)."""
    deg = np.asarray(A.sum(axis=1)).ravel()
    n = A.shape[0]
    if not normalized:
        return sp.diags(deg) - A
    d_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return sp.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt


def eigenmodes(L: sp.csr_matrix, K: int):
    """K smallest eigenpairs. Uses shift-invert (sigma=0) for speed and stability on large sparse L,
    which is equivalent to which='SM' but avoids scipy's well-known poor convergence for that mode.
    Returns (lambdas [K] ascending, U [n_voxels, K]) with U columns L2-normalised."""
    n = L.shape[0]
    K = min(K, n - 1)
    lambdas, U = spla.eigsh(L.tocsc(), k=K, sigma=0, which="LM")
    order = np.argsort(lambdas)
    lambdas, U = lambdas[order], U[:, order]
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    return lambdas.astype(np.float64), U.astype(np.float64)


def eigenmodes_per_component(A: sp.csr_matrix, labels: np.ndarray, K: int):
    """Eigenmodes computed separately per connected component (policy='per_component', spec §4),
    with K split proportional to component size (at least 1 mode per non-trivial component) and
    modes from all components merged, re-sorted by eigenvalue, and re-embedded into n_voxels-wide
    vectors (zero outside the component). Returns (lambdas [<=K], U [n_voxels, <=K])."""
    n = A.shape[0]
    comps = np.unique(labels)
    sizes = np.array([np.sum(labels == c) for c in comps])
    budgets = np.maximum(1, np.round(K * sizes / sizes.sum()).astype(int))
    budgets = np.minimum(budgets, sizes - 1)   # eigsh needs k < n for that component
    budgets = np.maximum(budgets, 0)

    all_lambdas, all_U = [], []
    for c, budget in zip(comps, budgets):
        if budget <= 0:
            continue
        idx = np.where(labels == c)[0]
        Ac = A[idx][:, idx]
        Lc = laplacian(Ac, normalized=True)
        lam, Uc = eigenmodes(Lc, int(budget))
        Uf = np.zeros((n, Uc.shape[1]))
        Uf[idx] = Uc
        all_lambdas.append(lam)
        all_U.append(Uf)
    lambdas = np.concatenate(all_lambdas)
    U = np.concatenate(all_U, axis=1)
    order = np.argsort(lambdas)[:K]
    return lambdas[order], U[:, order]


class LaplacianBasis(NestedBasis):
    """k(t) = round(t*K); project = U_t^T v; lift = U_t c. U orthonormal, so nesting holds automatically."""

    def __init__(self, U: np.ndarray, lambdas: np.ndarray):
        self.U, self.lambdas = U, lambdas
        self.K = U.shape[1]

    def k(self, t: float) -> int:
        return max(1, round(t * self.K)) if t > 0 else 0

    def project(self, v: np.ndarray, t: float) -> np.ndarray:
        return self.U[:, : self.k(t)].T @ v

    def lift(self, c: np.ndarray, t: float) -> np.ndarray:
        kt = self.k(t)
        return self.U[:, :kt] @ c[:kt]

    def band(self, t_lo: float, t_hi: float) -> np.ndarray:
        return np.arange(self.k(t_lo), self.k(t_hi))
