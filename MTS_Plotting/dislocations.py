"""Burgers circuits using native elastic maps or a Delaunay mesh of atom positions."""

from collections import defaultdict
from itertools import product

import numpy as np

from .mesh_geometry import visible_periodic_images


DEFAULT_BOND_TOLERANCE = .25


def circuit_charges(edge_ids, directions, hops, reliable, incidence):
    """Sum trusted lattice hops around connected regions of rejected bonds."""
    neighbours = defaultdict(list)
    invalid = ~reliable[edge_ids]
    for flat_index in np.flatnonzero(invalid):
        face, side = divmod(flat_index, 3)
        neighbours[edge_ids[face, side]].append(face)
    adjacency = defaultdict(set)
    for faces in neighbours.values():
        for face in faces:
            adjacency[face].update(faces)
    exterior = np.any(incidence[edge_ids] == 1, axis=1)
    signed = directions[..., None] * hops[edge_ids] * ~invalid[..., None]
    sums = signed.sum(axis=1)
    visited = set()
    candidates = []
    for face in sorted(set(adjacency) | set(np.flatnonzero(np.any(sums, axis=1)))):
        if face in visited:
            continue
        region, pending = [], [face]
        while pending:
            member = pending.pop()
            if member in visited:
                continue
            visited.add(member)
            region.append(member)
            pending.extend(adjacency[member] - visited)
        charge = sums[region].sum(axis=0)
        if np.any(charge) and not np.any(exterior[region]):
            candidates.append((region, charge))
    return candidates


def delaunay_dislocations(points, cells, lattice, *, periodic=None, tiled=False, tolerance=DEFAULT_BOND_TOLERANCE):
    """Use current node geometry, with native cells defining the sample outline."""
    from matplotlib.tri import Triangulation

    if not .10 <= tolerance <= .40:
        raise ValueError("Choose a Delaunay bond tolerance between 0.10 and 0.40 lattice spacings.")

    def inside_sample(positions):
        finder = Triangulation(*points.T, triangles=cells.copy()).get_trifinder()
        return finder(*positions.T) >= 0

    if periodic is None:
        triangles = Triangulation(*points.T).triangles
        triangles = triangles[inside_sample(points[triangles].mean(axis=1))]
        return dislocation_candidates(points, triangles, None, lattice, geometry_tolerance=tolerance)

    basis, reference = periodic
    images, triangles, identities = periodic_delaunay(points, basis, reference)
    locations, burgers = dislocation_candidates(images, triangles, None, lattice, periodic=(basis, identities),
                                                geometry_tolerance=tolerance)
    if len(locations) and not tiled:
        locations, sources = visible_periodic_images(locations[:, None, :], basis, (points.min(axis=0), points.max(axis=0)))
        locations = locations[:, 0]
        inside = inside_sample(locations)
        locations, burgers = locations[inside], burgers[sources[inside]]
    return locations, burgers


def periodic_delaunay(points, basis, reference):
    """Triangulate a periodic halo and retain one copy of each face by circumcentre."""
    from matplotlib.tri import Triangulation

    inverse = np.linalg.inv(basis)
    first = np.unique(reference, return_index=True)[1]
    fractional = points[first] @ inverse.T
    fractional -= np.floor(fractional)
    spacing = np.sqrt(abs(np.linalg.det(basis)) / len(first))
    # Break square-lattice ties consistently across copies of the same atom.
    perturbation = np.random.default_rng(0).uniform(-1e-8, 1e-8, fractional.shape) * spacing
    for halo in (3, 6, 12):
        padding = halo * spacing * np.linalg.norm(inverse, axis=1)
        extent = np.ceil(padding).astype(int)
        copies, identities = [], []
        for shift in product(*(range(-n, n + 1) for n in extent)):
            shifted = fractional + shift
            keep = np.flatnonzero(np.all((shifted >= -padding) & (shifted <= 1 + padding), axis=1))
            copies.append(shifted[keep] @ basis.T)
            identities.append(keep)
        images, ids = np.concatenate(copies), np.concatenate(identities)
        search = images + perturbation[ids]
        cells = Triangulation(*search.T).triangles
        start, end, other = search[cells].transpose(1, 0, 2)
        sides = np.stack((end - start, other - start), axis=1)
        offset = np.linalg.solve(sides, (.5 * np.sum(sides ** 2, axis=2))[..., None])[..., 0]
        centre = (start + offset) @ inverse.T
        centre = np.where(np.abs(centre - np.rint(centre)) < 1e-10, np.rint(centre), centre)
        keep = np.all((centre >= 0) & (centre < 1), axis=1)
        radius = np.linalg.norm(offset[keep], axis=1)[:, None] * np.linalg.norm(inverse, axis=1)
        if np.count_nonzero(keep) == 2 * len(first) and np.all(
                (centre[keep] - radius >= -padding) & (centre[keep] + radius <= 1 + padding)):
            return images, cells[keep], ids
    raise ValueError("Could not close the periodic Delaunay triangulation.")


def lattice_hops(offsets, basis, lattice):
    coordinates = np.linalg.solve(basis, offsets[..., None])[..., 0]
    hops = np.rint(coordinates)
    error = np.linalg.norm((coordinates - hops) @ lattice.T, axis=-1)
    length = np.linalg.norm(hops @ lattice.T, axis=-1)
    return hops, np.where((length > 0) & (length <= 2 + 1e-8), error, np.inf)


def geometry_hops(offsets, faces, endpoints, lattice, atoms, tolerance):
    """Fit a common lattice gauge, then independently refine it at each atom."""
    sample = offsets[np.linspace(0, len(offsets) - 1, min(len(offsets), 2048), dtype=int)]
    faces = faces[np.linspace(0, len(faces) - 1, min(len(faces), 32), dtype=int)]
    triangular = bool(lattice[0, 1])
    order = np.argsort(np.linalg.norm(faces, axis=2), axis=1)
    candidates = np.take_along_axis(faces, order[:, :2, None], axis=1).transpose(0, 2, 1)
    acute = (candidates[:, :, 0] * candidates[:, :, 1]).sum(axis=1) >= 0
    if triangular:
        candidates[~acute, :, 1] *= -1
    # Choose a positive orientation while preserving the primitive cell.
    reverse = np.linalg.det(candidates) < 0
    if triangular:
        candidates[reverse] = candidates[reverse, :, ::-1]
    else:
        candidates[reverse, :, 1] *= -1
    candidates = candidates[np.linalg.det(candidates) > 1e-10 * abs(np.linalg.det(lattice))]
    if not len(candidates):
        raise ValueError("Could not fit a lattice to the Delaunay triangles.")
    basis = min(candidates, key=lambda candidate:
                np.mean(np.minimum(lattice_hops(sample, candidate, lattice)[1], tolerance) ** 2))
    for _ in range(2):
        hops, error = lattice_hops(sample, basis, lattice)
        trusted = error <= tolerance
        fitted, _, rank, _ = np.linalg.lstsq(hops[trusted], sample[trusted], rcond=None)
        if rank == 2 and np.linalg.det(fitted) > 0:
            basis = fitted.T

    hops, error = lattice_hops(offsets, basis, lattice)
    good = error <= .8 * tolerance
    known = hops[good]
    normal = np.zeros((atoms, 2, 2))
    rhs = np.zeros_like(normal)
    count = np.zeros(atoms, dtype=int)
    for vertex in endpoints[good].T:
        np.add.at(normal, vertex, known[:, :, None] * known[:, None, :])
        np.add.at(rhs, vertex, known[:, :, None] * offsets[good, None, :])
        np.add.at(count, vertex, 1)
    local = np.broadcast_to(basis, normal.shape).copy()
    fit = (count >= 3) & (np.linalg.det(normal) > 0)
    refined = np.linalg.solve(normal[fit], rhs[fit]).transpose(0, 2, 1)
    valid = np.linalg.det(refined) > .25 * np.linalg.det(basis)
    local[np.flatnonzero(fit)[valid]] = refined[valid]
    start, end = endpoints.T
    left, left_error = lattice_hops(offsets, local[start], lattice)
    right, right_error = lattice_hops(offsets, local[end], lattice)
    reliable = (left_error <= tolerance) & (right_error <= tolerance) & np.all(left == right, axis=1)
    return left.astype(int), reliable, basis


def dislocation_candidates(points, cells, elastic, lattice, *, periodic=None, geometry_tolerance=DEFAULT_BOND_TOLERANCE):
    """Return locations and Burgers vectors for closed, nonzero lattice circuits.

    Two estimates of each bond must agree within the lattice spacing tolerance.
    Estimates use either the adjacent elastic maps or the atom neighbourhoods.
    Circuits enclose ambiguous bonds and exclude free boundaries.
    """
    if not len(cells):
        return np.empty((0, 2)), np.empty((0, 2))
    cells = np.array(cells, copy=True)
    triangles = points[cells]
    a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    clockwise = a[:, 0] * b[:, 1] < a[:, 1] * b[:, 0]
    cells[clockwise] = cells[clockwise][:, [0, 2, 1]]
    ends = np.roll(cells, -1, axis=1)
    ids = np.arange(len(points))
    images = np.zeros((len(points), 2), dtype=int)
    if periodic is not None:
        basis, reference = periodic
        _, first, ids = np.unique(np.asarray(reference).reshape(-1), return_index=True, return_inverse=True)
        inverse = np.linalg.inv(basis)
        shifts = (points - points[first[ids]]) @ inverse.T
        images = np.rint(shifts).astype(int)
        if not np.allclose(shifts, images, rtol=0, atol=1e-5):
            raise ValueError("Periodic node images disagree with the mesh lattice.")
    shifts = images[ends] - images[cells]
    same = ids[cells] == ids[ends]
    if np.any(same & ~np.any(shifts, axis=-1)):
        raise ValueError("Dislocation analysis requires distinct atoms on every mesh bond.")
    forward_image = (shifts[..., 0] > 0) | ((shifts[..., 0] == 0) & (shifts[..., 1] > 0))
    directions = np.where((ids[cells] < ids[ends]) | (same & forward_image), 1, -1)
    shifts *= directions[..., None]
    keys = np.concatenate((np.minimum(ids[cells], ids[ends])[..., None],
                           np.maximum(ids[cells], ids[ends])[..., None], shifts), axis=-1)
    _, first, edge_ids, incidence = np.unique(keys.reshape(-1, 4), axis=0,
                                             return_index=True, return_inverse=True, return_counts=True)
    if np.any(incidence > 2):
        raise ValueError("Dislocation analysis requires a manifold mesh.")
    if periodic is not None and elastic is None and np.any(incidence != 2):
        raise ValueError("Periodic Delaunay edges must have two incident faces.")
    orientation = np.zeros(len(first), dtype=int)
    np.add.at(orientation, edge_ids, directions.ravel())
    if np.any((incidence == 2) & (orientation != 0)):
        raise ValueError("Dislocation analysis requires consistently oriented mesh faces.")
    offsets = (points[ends] - points[cells]) * directions[..., None]
    if elastic is None:
        endpoints = keys.reshape(-1, 4)[first, :2]
        hops, reliable, burgers_basis = geometry_hops(offsets.reshape(-1, 2)[first], offsets, endpoints, lattice,
                                                      ids.max() + 1, geometry_tolerance)
    else:
        coordinates = np.linalg.solve(elastic @ lattice, offsets.swapaxes(1, 2)).swapaxes(1, 2).reshape(-1, 2)
        integer = np.rint(coordinates).astype(int)
        agreed = np.all(integer == integer[first][edge_ids], axis=1)
        close = np.linalg.norm((coordinates - integer) @ lattice.T, axis=1) <= .25
        reliable = np.ones(len(first), dtype=bool)
        np.logical_and.at(reliable, edge_ids, agreed & close)
        hops, burgers_basis = integer[first], lattice
    centres = points[cells].mean(axis=1)
    locations, burgers = [], []
    for region, charge in circuit_charges(edge_ids.reshape(-1, 3), directions, hops, reliable, incidence):
        positions = centres[region]
        if periodic is not None:
            shifts = (positions - positions[0]) @ inverse.T
            positions = positions - np.rint(shifts) @ basis.T
        locations.append(positions.mean(axis=0))
        burgers.append(burgers_basis @ charge)
    return np.asarray(locations).reshape(-1, 2), np.asarray(burgers).reshape(-1, 2)


def dislocation_segments(locations, burgers, scale):
    unit = burgers / np.linalg.norm(burgers, axis=1)[:, None]
    # Counterclockwise circuits use +z. The stem points along b × z, toward the extra half-plane.
    normal = np.column_stack((unit[:, 1], -unit[:, 0]))
    return np.concatenate((np.stack((locations - scale * unit / 2, locations + scale * unit / 2), axis=1),
                           np.stack((locations, locations + scale * normal), axis=1)))
