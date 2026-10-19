"""Voxel adjacency graph from the mask (spec §4)."""
import numpy as np
import scipy.sparse as sp


def build_voxel_graph(mask: np.ndarray, coords: np.ndarray, connectivity: int = 26) -> sp.csr_matrix:
    """Edges between neighbouring voxels inside the mask. Weight 1, or exp(-d^2/sigma^2)."""
    raise NotImplementedError


def connected_components(A: sp.csr_matrix) -> np.ndarray:
    """Component label per voxel (hemispheres, small islands)."""
    raise NotImplementedError


def handle_components(A: sp.csr_matrix, labels: np.ndarray, policy: str = "per_component"):
    """policy: per_component (default; eigenmodes per piece, modes split by size) | bridge (virtual midline edges)."""
    raise NotImplementedError
