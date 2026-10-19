"""Voxel adjacency graph from the mask (spec §4)."""
import numpy as np
import scipy.sparse as sp

# 26-connectivity offsets (all non-zero combinations of {-1,0,1}^3); 6-connectivity uses only the
# axis-aligned ones (|offset| has a single non-zero entry).
_OFFSETS_26 = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
_OFFSETS_6 = [o for o in _OFFSETS_26 if sum(c != 0 for c in o) == 1]


def build_voxel_graph(mask: np.ndarray, coords: np.ndarray, connectivity: int = 26, sigma: float = None) -> sp.csr_matrix:
    """Edges between neighbouring voxels inside the mask.
    mask: bool [X, Y, Z]. coords: [n_voxels, 3] int, in the same voxel order as the data (row i of
    coords is voxel i). Weight 1, or exp(-d^2/sigma^2) if sigma is given (sigma in voxel units)."""
    if connectivity not in (6, 26):
        raise ValueError("connectivity must be 6 or 26")
    offsets = _OFFSETS_6 if connectivity == 6 else _OFFSETS_26
    idx_of = -np.ones(mask.shape, dtype=np.int64)
    idx_of[tuple(coords.T)] = np.arange(len(coords))
    shape = np.array(mask.shape)

    rows, cols = [], []
    for dx, dy, dz in offsets:
        nb = coords + np.array([dx, dy, dz])
        inside = np.all((nb >= 0) & (nb < shape), axis=1)
        nb_idx = np.full(len(coords), -1, dtype=np.int64)
        nb_idx[inside] = idx_of[tuple(nb[inside].T)]
        valid = nb_idx >= 0
        rows.append(np.where(valid)[0])
        cols.append(nb_idx[valid])
    rows, cols = np.concatenate(rows), np.concatenate(cols)

    if sigma is None:
        data = np.ones(len(rows), dtype=np.float64)
    else:
        d2 = ((coords[rows] - coords[cols]) ** 2).sum(axis=1)
        data = np.exp(-d2 / sigma**2)

    n = len(coords)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    return A.maximum(A.T)   # symmetrize (each edge is found from both endpoints, so this is a no-op if consistent)


def connected_components(A: sp.csr_matrix) -> np.ndarray:
    """Component label per voxel (hemispheres, small islands)."""
    n_comp, labels = sp.csgraph.connected_components(A, directed=False)
    return labels


def handle_components(A: sp.csr_matrix, labels: np.ndarray, policy: str = "per_component", coords: np.ndarray = None):
    """policy='per_component' (default): return (A, labels) unchanged; eigenmodes.eigenmodes is called
    once per component elsewhere, with the mode budget split proportional to component size (spec §4).
    policy='bridge': connect every pair of components with a virtual edge between their nearest voxels
    (by coordinate distance), so a single eigendecomposition covers the whole mask. Requires coords."""
    if policy == "per_component":
        return A, labels
    if policy != "bridge":
        raise ValueError("policy must be 'per_component' or 'bridge'")
    if coords is None:
        raise ValueError("policy='bridge' requires coords")
    comps = np.unique(labels)
    if len(comps) < 2:
        return A, labels
    A = A.tolil()
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            idx_i, idx_j = np.where(labels == comps[i])[0], np.where(labels == comps[j])[0]
            d2 = ((coords[idx_i, None, :] - coords[idx_j, None, :, :].transpose(1, 0, 2)) ** 2).sum(-1)
            a, b = np.unravel_index(np.argmin(d2), d2.shape)
            u, v = idx_i[a], idx_j[b]
            A[u, v] = A[v, u] = 1.0
    merged_labels = np.zeros_like(labels)   # all one component now
    return A.tocsr(), merged_labels
