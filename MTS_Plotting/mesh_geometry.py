"""Periodic images and deformation coordinates for native MTS2D meshes."""

import numpy as np


def reference_cell(config):
    width, height = config["cols"], config["rows"]
    if config["energyFunction"] == "contiTriangular":
        return np.array([[width, height / 2], [0, np.sqrt(3) * height / 2]])
    return np.diag([float(width), float(height)])


def periodic_basis(points, reference_ids, area):
    """Find primitive translations from coincident reference nodes in ghost images."""
    ids = np.asarray(reference_ids).reshape(-1)
    if ids.shape != (len(points),) or not np.all(np.isfinite(ids) & (ids == np.floor(ids))):
        raise ValueError("Invalid periodic reference node IDs.")
    _, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
    offsets = points - points[first[inverse]]
    # Native positions are printed with finite precision. Keep one original
    # vector from each set of numerically equivalent translations.
    tolerance = 1e-9 * max(1., np.max(np.abs(points)))
    _, unique = np.unique(np.rint(offsets / tolerance), axis=0, return_index=True)
    offsets = offsets[unique]
    lengths = np.linalg.norm(offsets, axis=1)
    offsets = offsets[np.argsort(lengths)]
    offsets = offsets[np.linalg.norm(offsets, axis=1) > tolerance]
    for i, a in enumerate(offsets):
        rest = offsets[i + 1:]
        determinants = a[0] * rest[:, 1] - a[1] * rest[:, 0]
        matches = np.flatnonzero(np.isclose(np.abs(determinants), area, rtol=1e-6, atol=tolerance))
        for match in matches:
            basis = np.column_stack((a, rest[match]))
            coordinates = np.linalg.solve(basis, offsets.T).T
            if np.allclose(coordinates, np.rint(coordinates), rtol=0, atol=1e-5):
                return reduce_basis(basis)
    raise ValueError("Could not verify both periodic translations from native ghost nodes.")


def reduce_basis(basis):
    """Remove integer skew without changing the lattice of translations."""
    a, b = basis.T.copy()
    for _ in range(100):
        if b @ b < a @ a:
            a, b = b, a
        multiple = np.rint((a @ b) / (a @ a))
        if not multiple:
            return np.column_stack((a, b))
        b -= multiple * a
    raise ValueError("Periodic basis reduction did not converge.")


def visible_periodic_images(polygons, basis, bounds):
    """Return visible translated polygons (or points) and their source indices."""
    lo, hi = bounds
    inverse = np.linalg.inv(basis)
    centres = polygons.mean(axis=1) @ inverse.T
    polygons = polygons - (np.floor(centres) @ basis.T)[:, None, :]
    lower, upper = polygons.min(axis=1), polygons.max(axis=1)
    # Any candidate translation lies in this Cartesian box. Transform all
    # four corners to obtain conservative integer ranges in the lattice.
    shift_lo, shift_hi = lo - upper.max(axis=0), hi - lower.min(axis=0)
    corners = np.array([[x, y] for x in (shift_lo[0], shift_hi[0]) for y in (shift_lo[1], shift_hi[1])])
    lattice = corners @ inverse.T
    start = np.ceil(lattice.min(axis=0) - 1e-10).astype(int)
    end = np.floor(lattice.max(axis=0) + 1e-10).astype(int)
    visible, indices = [], []
    for i in range(start[0], end[0] + 1):
        for j in range(start[1], end[1] + 1):
            shift = basis @ (i, j)
            inside = np.flatnonzero(np.all((upper + shift >= lo) & (lower + shift <= hi), axis=1))
            if len(inside):
                visible.append(polygons[inside] + shift)
                indices.append(inside)
    if not visible:
        raise ValueError("No periodic triangles intersect the viewport.")
    return np.concatenate(visible), np.concatenate(indices)


def integer_shears(matrix):
    """Signed horizontal and vertical counts of the square-model reduction."""
    metric = matrix.swapaxes(-1, -2) @ matrix
    a, b, c = metric[:, 0, 0].copy(), metric[:, 0, 1].copy(), metric[:, 1, 1].copy()
    if np.any(a <= 0) or np.any(a * c <= b * b):
        raise ValueError("Integer shear requires a positive-definite deformation metric.")
    counts = np.zeros((len(matrix), 2), dtype=int)
    for _ in range(1000):
        active = np.abs(b) > 0.5 * np.minimum(a, c)
        if not active.any():
            return counts
        horizontal = active & (a < c)
        vertical = active & ~horizontal
        step = -b[active] / np.minimum(a, c)[active]
        n = np.zeros(len(a), dtype=int)
        n[active] = np.copysign(np.floor(np.abs(step) + 0.5), step).astype(int)
        counts[horizontal, 0] -= n[horizontal]
        counts[vertical, 1] -= n[vertical]
        a[vertical] += 2 * n[vertical] * b[vertical] + n[vertical] ** 2 * c[vertical]
        c[horizontal] += 2 * n[horizontal] * b[horizontal] + n[horizontal] ** 2 * a[horizontal]
        b += n * np.where(horizontal, a, c)
    raise ValueError("Integer shear decomposition did not converge.")


def poincare_coordinates(matrix):
    metric = matrix.swapaxes(-1, -2) @ matrix
    determinant = np.linalg.det(matrix)
    if np.any(determinant <= 0):
        raise ValueError("The Poincaré disk requires orientation-preserving deformation.")
    a, b, c = metric[:, 0, 0], metric[:, 0, 1], metric[:, 1, 1]
    return np.column_stack((a - c, 2 * b)) / (a + c + 2 * determinant)[:, None]
