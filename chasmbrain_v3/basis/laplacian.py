"""Graph Laplacian eigenmodes (spec §4). Saved to basis/laplacian_subj{N}.npz."""
import numpy as np
import scipy.sparse as sp
from .nested import NestedBasis


def laplacian(A: sp.csr_matrix, normalized: bool = True) -> sp.csr_matrix:
    """L = I - D^{-1/2} A D^{-1/2}."""
    raise NotImplementedError


def eigenmodes(L: sp.csr_matrix, K: int):
    """K smallest eigenpairs via scipy.sparse.linalg.eigsh(L, k=K, which='SM'). Returns (lambdas [K], U [n_voxels, K])."""
    raise NotImplementedError


class LaplacianBasis(NestedBasis):
    """k(t) = round(t*K); project = U_t^T v; lift = U_t c. U orthonormal, so nesting holds automatically."""

    def __init__(self, U: np.ndarray, lambdas: np.ndarray):
        self.U, self.lambdas = U, lambdas
