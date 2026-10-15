"""Client similarity graph, low-frequency basis, Nystrom extension and ridge recovery of missing client updates."""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


def pairwise_distances(updates: np.ndarray) -> np.ndarray:
    updates = np.asarray(updates, dtype=np.float64)
    sq = np.sum(updates ** 2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (updates @ updates.T)
    np.maximum(d2, 0.0, out=d2)
    dist = np.sqrt(d2)
    np.fill_diagonal(dist, 0.0)
    return dist


def similarity_matrix(dist: np.ndarray):
    n = dist.shape[0]
    dbar = float(dist.sum() / (n * n))
    if dbar <= 0:
        return np.ones_like(dist), 0.0
    P = np.exp(-dist / dbar)
    np.fill_diagonal(P, 1.0)
    return P, dbar


def _orthonormal_basis(U: np.ndarray, degrees: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(np.sqrt(degrees)[:, None] * U)
    return np.ascontiguousarray(Q)


@dataclass
class GraphBasis:
    vertices: np.ndarray
    Q: np.ndarray
    eigenvalues: np.ndarray
    K: int
    P: Optional[np.ndarray] = field(repr=False, default=None)
    degrees: Optional[np.ndarray] = field(repr=False, default=None)
    U: Optional[np.ndarray] = field(repr=False, default=None)
    dbar: float = 0.0

    def subspace_residual(self, X: np.ndarray) -> float:
        X = np.asarray(X, dtype=np.float64)
        denom = float(np.sum(X ** 2))
        if denom <= 0:
            return 0.0
        resid = X - self.Q @ (self.Q.T @ X)
        return float(np.sum(resid ** 2) / denom)

    def extend(self, new_ids: Sequence[int], new_updates: np.ndarray,
               neighbour_ids: Sequence[int], neighbour_updates: np.ndarray) -> "GraphBasis":
        if self.U is None or self.degrees is None or self.dbar <= 0:
            raise ValueError("basis cannot be extended")
        new_ids = [int(i) for i in new_ids]
        if not new_ids:
            return self
        known = set(int(v) for v in self.vertices)
        if any(i in known for i in new_ids):
            raise ValueError("client already on the graph")
        nbr_local, nbr_rows = [], []
        for j, row in zip([int(i) for i in neighbour_ids], np.asarray(neighbour_updates, float)):
            if j in known:
                nbr_local.append(self._lookup[j])
                nbr_rows.append(row)
        if not nbr_local:
            raise ValueError("no sampled client is on the graph")
        nbr_local = np.asarray(nbr_local, dtype=int)
        nbr_rows = np.stack(nbr_rows, axis=0)

        new_updates = np.atleast_2d(np.asarray(new_updates, dtype=np.float64))
        scale = self.Q.shape[0] / float(len(nbr_local))
        lam = np.asarray(self.eigenvalues[:self.K], dtype=np.float64)
        lam = np.where(np.abs(lam) < 1e-12, 1e-12, lam)

        sq_n = np.sum(new_updates ** 2, axis=1)
        sq_j = np.sum(nbr_rows ** 2, axis=1)
        d2 = sq_n[:, None] + sq_j[None, :] - 2.0 * (new_updates @ nbr_rows.T)
        np.maximum(d2, 0.0, out=d2)
        W = np.exp(-np.sqrt(d2) / self.dbar)
        deg_new = np.maximum(scale * W.sum(axis=1), 1e-12)
        S_rows = scale * W / np.sqrt(deg_new[:, None] * self.degrees[nbr_local][None, :])
        U_new = (S_rows @ self.U[nbr_local, :]) / lam[None, :]

        U_ext = np.vstack([self.U, U_new])
        deg_ext = np.concatenate([self.degrees, deg_new])
        return GraphBasis(
            vertices=np.concatenate([self.vertices, np.asarray(new_ids, dtype=int)]),
            Q=_orthonormal_basis(U_ext, deg_ext), eigenvalues=self.eigenvalues, K=self.K,
            degrees=deg_ext, U=U_ext, dbar=self.dbar)

    def drop(self, ids: Sequence[int]) -> "GraphBasis":
        if self.U is None or self.degrees is None:
            raise ValueError("basis cannot be reduced")
        gone = {int(i) for i in ids} & {int(v) for v in self.vertices}
        if not gone:
            return self
        keep = np.array([i for i, v in enumerate(self.vertices) if int(v) not in gone], dtype=int)
        if keep.size <= self.K:
            raise ValueError("too few vertices left")
        U_k, deg_k = self.U[keep], self.degrees[keep]
        return GraphBasis(vertices=self.vertices[keep], Q=_orthonormal_basis(U_k, deg_k),
                          eigenvalues=self.eigenvalues, K=self.K, degrees=deg_k, U=U_k, dbar=self.dbar)

    def local_index(self, global_ids: Sequence[int]) -> np.ndarray:
        return np.array([self._lookup[int(g)] for g in global_ids], dtype=int)

    @property
    def _lookup(self) -> dict:
        if getattr(self, "_lut", None) is None:
            object.__setattr__(self, "_lut", {int(v): i for i, v in enumerate(self.vertices)})
        return self._lut


def build_graph_basis(updates: np.ndarray, vertices: Sequence[int], K: int) -> GraphBasis:
    updates = np.asarray(updates, dtype=np.float64)
    K = int(min(K, updates.shape[0]))
    P, dbar = similarity_matrix(pairwise_distances(updates))
    degrees = np.maximum(P.sum(axis=0), 1e-12)
    d_inv_sqrt = 1.0 / np.sqrt(degrees)
    S = P * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    S = 0.5 * (S + S.T)
    eigvals, U = np.linalg.eigh(S)
    order = np.argsort(eigvals)[::-1]
    eigvals, U = eigvals[order], U[:, order]
    U_K = U[:, :K]
    return GraphBasis(vertices=np.asarray(list(vertices), dtype=int), Q=_orthonormal_basis(U_K, degrees),
                      eigenvalues=eigvals, K=K, P=P, degrees=degrees, U=np.ascontiguousarray(U_K), dbar=dbar)


@dataclass
class RecoveryResult:
    coeffs: np.ndarray
    sigma_min: float
    v_tau: float


def ridge_recovery(basis: GraphBasis, observed_local: np.ndarray, observed_updates: np.ndarray,
                   weights_local: np.ndarray, missing_local: np.ndarray, ridge_lambda: float) -> RecoveryResult:
    observed_local = np.asarray(observed_local, dtype=int)
    missing_local = np.asarray(missing_local, dtype=int)
    Y = np.asarray(observed_updates, dtype=np.float64)
    B = basis.Q[observed_local, :]
    svals = np.linalg.svd(B, compute_uv=False)
    B_ridge = np.linalg.solve(B.T @ B + ridge_lambda * np.eye(B.shape[1]), B.T)
    p = np.asarray(weights_local, dtype=np.float64)
    jbar_p = np.zeros(basis.Q.shape[0], dtype=np.float64)
    jbar_p[missing_local] = p[missing_local]
    tau = p[observed_local] + B_ridge.T @ (basis.Q.T @ jbar_p)
    return RecoveryResult(coeffs=B_ridge @ Y, sigma_min=float(svals.min()) if svals.size else 0.0,
                          v_tau=float(np.sum(tau ** 2)))
