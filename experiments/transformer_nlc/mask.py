"""NumPy physical attention masks with True meaning allowed.

Eq. 8's product form is completed at m=0 using Fig. 4's central red region.
This intentionally differs from a literal Algorithm 1 m!=0 implementation.
The unresolved Fig. 4b density discrepancy is retained in the fixed spec.
"""

import math

import numpy as np


def _geometry(t: int, b: int, k: int) -> tuple[int, int]:
    if any(type(value) is not int or value < 1 for value in (t, b, k)):
        raise ValueError("t, b and k must be positive integers")
    if k != 9:
        raise ValueError("only fixed k=9 mask coordinates are defined")
    ell = 2 * t - k + 1
    if ell < 0 or ell % 2:
        raise ValueError("ell must be nonnegative and even")
    return ell, b + ell


def individual_mask(t: int, rho: float = 2.6, *, k: int = 9) -> np.ndarray:
    """Return inclusive coordinates [-ell/2, ell/2] as a bool square."""
    ell, _ = _geometry(t, 1, k)
    if isinstance(rho, bool) or not isinstance(rho, (int, float)):
        raise ValueError("rho must be finite and nonnegative")
    if not math.isfinite(rho) or rho < 0:
        raise ValueError("rho must be finite and nonnegative")
    q = ell // 2
    coordinates = np.arange(-q, q + 1, dtype=np.int64)
    return np.abs(coordinates[:, None] * coordinates[None, :]) <= rho * q


def block_mask(t: int, b: int, rho: float | None = 2.6, *, k: int = 9) -> np.ndarray:
    """Union b shifted individual squares; rho=None gives all-allowed dense."""
    ell, n = _geometry(t, b, k)
    if rho is None:
        return np.ones((n, n), dtype=bool)
    local = individual_mask(t, rho, k=k)
    allowed = np.zeros((n, n), dtype=bool)
    for i in range(b):
        allowed[i:i + ell + 1, i:i + ell + 1] |= local
    validate_mask(allowed)
    return allowed


def validate_mask(allowed: np.ndarray) -> None:
    if allowed.dtype != np.bool_ or allowed.ndim != 2 or allowed.shape[0] != allowed.shape[1]:
        raise ValueError("allowed mask must be a square bool array")
    if allowed.shape[0] == 0 or not np.all(allowed.any(axis=-1)):
        raise ValueError("attention mask must permit at least one key in every row")


def additive_mask(allowed: np.ndarray, *, dtype=np.float32) -> np.ndarray:
    """Convert True/False to 0/-inf, suitable for addition before softmax."""
    allowed = np.asarray(allowed)
    validate_mask(allowed)
    if not np.issubdtype(np.dtype(dtype), np.floating):
        raise ValueError("additive masks require a floating dtype")
    return np.where(allowed, 0.0, -np.inf).astype(dtype)
