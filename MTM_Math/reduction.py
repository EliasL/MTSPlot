"""Vectorized 2D Lagrange and unit-shear plastic reduction."""

import numpy as np


def lagrange_reduction_components(C11, C22, C12, loops=1000):
    a, b, c = (np.array(x, dtype=float, copy=True) for x in (C11, C22, C12))
    if a.shape != b.shape or a.shape != c.shape:
        raise ValueError("Metric components must have equal shapes")
    if not np.all(np.isfinite([a, b, c])) or np.any(a <= 0) or np.any(a*b-c*c <= 0):
        raise ValueError("Expected finite positive-definite metrics")
    M = np.broadcast_to(np.eye(2), a.shape + (2, 2)).copy()
    flip = np.array([[1, 0], [0, -1]])
    swap = np.array([[0, 1], [1, 0]])
    shear = np.array([[1, -1], [0, 1]])
    for _ in range(loops):
        negative = c < 0
        c[negative] *= -1
        M[negative] = M[negative] @ flip
        exchange = b < a
        a[exchange], b[exchange] = b[exchange].copy(), a[exchange].copy()
        M[exchange] = M[exchange] @ swap
        shift = 2*c > a
        if not np.any(negative | exchange | shift):
            return a, b, c, M
        b[shift] += a[shift] - 2*c[shift]
        c[shift] -= a[shift]
        M[shift] = M[shift] @ shear
    raise RuntimeError(f"Lagrange reduction did not converge in {loops} iterations")


def lagrange_reduction(C, loops=1000):
    C = np.asarray(C, dtype=float)
    if C.shape[-2:] != (2, 2) or not np.allclose(C, C.swapaxes(-1, -2)):
        raise ValueError("Expected symmetric (..., 2, 2) matrices")
    a, b, c, M = lagrange_reduction_components(C[..., 0, 0], C[..., 1, 1], C[..., 0, 1], loops)
    result = C.copy()
    result[..., 0, 0], result[..., 1, 1] = a, b
    result[..., 0, 1] = result[..., 1, 0] = c
    return result, M


def plastic_reduction_components(C11, C22, C12, loops=1000, compute_M=False):
    """Reduce broadcastable metric components into the elastic domain.

    Ported from SimulationScripts_PhD/MTMath/reduction.py: apply one integer
    shear at a time, using horizontal shears when the diagonals tie. No flips
    or swaps are applied. Inputs are copied. Invalid metrics and failure to
    converge raise errors rather than returning an unreduced result.

    With compute_M=True, return M such that C_reduced = M.T @ C @ M.
    Otherwise M is a small (2, 2) zero placeholder, as in the original API.
    """
    a, b, c = (v.copy() for v in np.broadcast_arrays(
        *[np.asarray(v, dtype=float) for v in (C11, C22, C12)]))
    if not np.all(np.isfinite([a, b, c])) or np.any(a <= 0) or np.any(a*b-c*c <= 0):
        raise ValueError("Expected finite positive-definite metrics")
    M = np.broadcast_to(np.eye(2), a.shape + (2, 2)).copy() if compute_M else np.zeros((2, 2))
    for iteration in range(loops + 1):
        active = np.abs(c) > .5 * np.minimum(a, b)
        if not np.any(active):
            return (*[float(v) if a.ndim == 0 else v for v in (a, b, c)], M)
        if iteration == loops:
            break
        horizontal = active & (a <= b)
        vertical = active & ~horizontal
        m = np.where(c > 0, -1.0, 1.0)
        b[horizontal] += 2*m[horizontal]*c[horizontal] + a[horizontal]
        c[horizontal] += m[horizontal]*a[horizontal]
        a[vertical] += 2*m[vertical]*c[vertical] + b[vertical]
        c[vertical] += m[vertical]*b[vertical]
        if compute_M:
            for mask, i, j in ((horizontal, 0, 1), (vertical, 1, 0)):
                shear = np.broadcast_to(np.eye(2), (np.count_nonzero(mask), 2, 2)).copy()
                shear[:, i, j] = m[mask]
                M[mask] = M[mask] @ shear
    raise RuntimeError(f"Plastic reduction did not converge in {loops} iterations")


def plastic_reduction(C, loops=1000, compute_M=False):
    """Return a plastic-reduced copy of symmetric (..., 2, 2) metrics and M.

    The transform M is computed only when compute_M=True; see the component API.
    """
    C = np.asarray(C, dtype=float)
    if C.shape[-2:] != (2, 2) or not np.allclose(C, C.swapaxes(-1, -2)):
        raise ValueError("Expected symmetric (..., 2, 2) matrices")
    a, b, c, M = plastic_reduction_components(C[..., 0, 0], C[..., 1, 1], C[..., 0, 1],
                                             loops=loops, compute_M=compute_M)
    result = C.copy()
    result[..., 0, 0], result[..., 1, 1] = a, b
    result[..., 0, 1] = result[..., 1, 0] = c
    return result, M
