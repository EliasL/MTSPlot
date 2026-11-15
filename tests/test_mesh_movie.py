"""Numerical checks for mesh animation."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from MTS_Plotting.dislocations import circuit_charges, delaunay_dislocations, dislocation_candidates, dislocation_segments
from MTS_Plotting.mesh_geometry import periodic_basis, reduce_basis, visible_periodic_images
from MTS_Plotting.mesh_movie import display_limit_sources, frame_schedule, node_classes, opacity_values, render
from MTS_Plotting.render_capacity import worker_count


class MeshMovieTests(unittest.TestCase):
    def test_first_frame_preview_uses_last_vtu_energy_range_by_default(self):
        import meshio
        from MTS_Management.config import Config

        with TemporaryDirectory() as temporary:
            config = Config()
            folder = Path(temporary) / config.folder_name()
            (folder / "data").mkdir(parents=True)
            config.write(folder / "config.conf")
            (folder / "manifest.json").write_text(json.dumps({
                "settings": config.as_dict(), "config_hash": config.config_hash,
            }))
            (folder / "macroData.csv").write_text(
                "load_step,load,minX,minY,maxX,maxY,max_energy\n"
                "0,0,0,0,1,1,9\n"
                "1,1,0,0,1,1,2\n")
            points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
            cells = [("triangle", np.array([[0, 1, 2]]))]
            for step, energy in enumerate((9., 2.)):
                mesh = meshio.Mesh(points, cells, cell_data={"energy_field": [np.array([energy])]})
                mesh.write(folder / "data" / f"mesh_load={step}_nrM=0_.{step}.vtu")
            spec = dict(kind="energy", matrix="T", tiling="cell", playback="saved", fps=1,
                        seconds_per_load=1, minimum_seconds=1, preview=True, preview_frame="first",
                        show_title=False, dpi=100)
            with patch("matplotlib.image.imsave"):
                latest = render(folder, folder / "preview.png", spec)
                full = render(folder, folder / "preview.png", spec | {"range_from_last_frame": False})
            self.assertEqual(latest["energy_range"], (0., 2.))
            self.assertEqual(full["energy_range"], (0., 9.))

    def test_display_limits_follow_selected_saved_frames(self):
        frames = [{"step": 2, "load": .001}, {"step": 4, "load": .003}]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv = root / "macroData.csv"
            csv.write_text(
                "load_step,load,minX,minY,maxX,maxY,max_energy\n"
                "2,0.001,-1,0,5,3,0.4\n"
                "3,0.002,-9,-9,20,20,90\n"
                "4,0.003,-2,1,4,7,0.7\n")
            limits = {"kind": "energy", "tiled": False, "limit": None, "lower_limit": 0}
            bounds, maximum, scans = display_limit_sources(root, frames, **limits)
            np.testing.assert_allclose(bounds, [[-2, 0], [5, 7]])
            self.assertEqual(maximum, .7)
            self.assertFalse(any(scans))
            # Files from an interrupted or resumed run may not share a step/load history.
            frames[1]["load"] = .004
            frames.append({"step": 6, "load": .005})
            bounds, maximum, scans = display_limit_sources(root, frames, **limits)
            np.testing.assert_allclose(bounds, [[-1, 0], [5, 3]])
            self.assertEqual(maximum, .4)
            self.assertFalse(any(scans))

    def test_opacity_interpolates_and_extends_the_endpoint_values(self):
        positions = np.array([0., .25, .5, .75, 1.])
        np.testing.assert_allclose(opacity_values(positions, 1, [[.25, .1], [.75, .9]]),
                                   [.1, .1, .5, .9, .9])
        np.testing.assert_allclose(opacity_values(positions, 1, [[.4, .3]]), .3)

    def test_recovers_an_edge_dislocation_in_both_lattices(self):
        from matplotlib.tri import Triangulation

        grid = np.array([(i + .25, j + .5) for i in range(-12, 13) for j in range(-12, 13)])
        for lattice in (np.eye(2), np.array([[1, .5], [0, np.sqrt(3) / 2]])):
            ideal = grid @ lattice.T
            points = ideal.copy()
            # Add one horizontal winding using the phase of a Volterra defect.
            for _ in range(20):
                points[:, 0] = ideal[:, 0] + np.arctan2(points[:, 1], points[:, 0]) / (2 * np.pi)
            cells = Triangulation(*points.T).triangles
            x, y = points[cells].mean(axis=1).T
            radius = 2 * np.pi * (x*x + y*y)
            inverse_elastic = np.broadcast_to(np.eye(2), (len(cells), 2, 2)).copy()
            inverse_elastic[:, 0, 0] += y / radius
            inverse_elastic[:, 0, 1] -= x / radius
            centres, burgers = dislocation_candidates(points, cells, np.linalg.inv(inverse_elastic), lattice)
            np.testing.assert_array_equal(burgers, [[-1, 0]])
            self.assertLess(np.linalg.norm(centres[0]), .5)
            # The rows bracketing the core have one extra atom above it.
            near = np.abs(points[:, 0]) < 3.9
            above = np.count_nonzero(near & (grid[:, 1] == .5))
            below = np.count_nonzero(near & (grid[:, 1] == -.5))
            self.assertEqual(above - below, 1)
            _, stem = dislocation_segments(centres, burgers, 1.)
            self.assertGreater(stem[1, 1], stem[0, 1])

            angle = np.pi / 6
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            deformation = rotation @ [[1, 1.4], [0, 1]]
            centres, burgers = delaunay_dislocations(points @ deformation.T, cells, lattice)
            np.testing.assert_allclose(burgers, [-rotation[:, 0]], atol=.03)
            self.assertLess(np.linalg.norm(centres[0]), .6)

    def test_delaunay_tracks_opposite_dislocations_across_a_periodic_seam(self):
        size = 25
        grid = np.array([(i + .25, j + .5) for i in range(size) for j in range(size)])
        points = grid.copy()
        # A ratio of periodic sine functions supplies opposite phase windings.
        left, right = -.35 + 11.4j, 4.8 + 11.4j
        for _ in range(20):
            z = points[:, 0] + 1j * points[:, 1]
            phase = np.angle(np.sin(np.pi * (z - left) / size) / np.sin(np.pi * (z - right) / size))
            points[:, 0] = grid[:, 0] + phase / (2 * np.pi)
        shear = np.array([[1, 2.38], [0, 1]])
        basis = reduce_basis(size * shear)
        points = points @ shear.T
        # Include an exported ghost of the first atom.
        points = np.vstack((points, points[0] + basis[:, 1]))
        ids = np.r_[np.arange(len(grid)), 0]
        centres, burgers = delaunay_dislocations(points, None, np.eye(2), periodic=(basis, ids), tiled=True)
        order = np.argsort(burgers[:, 0])
        centres = centres[order] @ np.linalg.inv(shear).T % size
        np.testing.assert_allclose(centres, [[left.real % size, left.imag], [right.real, right.imag]], atol=.35)
        np.testing.assert_allclose(burgers[order], [[-1, 0], [1, 0]], atol=.03)

    def test_burgers_circuit_excludes_surface_regions_and_affine_strain(self):
        edges = np.array([[0, 1, 2], [0, 3, 4]])
        signs = np.array([[1, 1, 1], [-1, 1, 1]])
        hops = np.array([[0, 0], [1, 0], [0, 1], [1, 0], [-1, -1]])
        reliable = [False, True, True, True, True]
        reliable = np.asarray(reliable)
        [(region, charge)] = circuit_charges(edges, signs, hops, reliable, np.full(5, 2))
        self.assertEqual(set(region), {0, 1})
        np.testing.assert_array_equal(charge, [1, 0])
        self.assertEqual(circuit_charges(edges, signs, hops, reliable, np.array([2, 1, 2, 2, 2])), [])

        affine = np.array([[1.1, 2.4], [.1, .9]])
        grid = np.array([(x, y) for y in range(4) for x in range(4)])
        cells = [[i, i + 1, i + 4] for i in range(11) if i % 4 != 3]
        cells += [[i + 1, i + 5, i + 4] for i in range(11) if i % 4 != 3]
        for lattice in (np.eye(2), np.array([[1, .5], [0, np.sqrt(3) / 2]])):
            points = grid @ lattice.T @ affine.T
            ids = ((grid[:, 0] % 3) + 3 * (grid[:, 1] % 3))[:, None]
            positions, burgers = dislocation_candidates(
                points, cells, np.broadcast_to(affine, (len(cells), 2, 2)), lattice,
                periodic=(3 * affine @ lattice, ids))
            self.assertEqual(len(positions), 0)
            self.assertEqual(len(burgers), 0)

    def test_automatic_workers_reserve_memory_and_one_cpu(self):
        self.assertEqual(worker_count(100, 310, 16), 2)
        self.assertEqual(worker_count(100, 20_000, 4), 3)
        self.assertEqual(worker_count(100, 0, 8), 1)
        self.assertEqual(worker_count(100, 20_000, 1), 1)

    def test_moving_free_nodes_keep_their_free_role(self):
        roles = node_classes({"fixed": [1, 0, 1],
                              "displacement": [[0, 0, 0], [.4, 0, 0], [0, -.2, 0]]})
        self.assertEqual(roles.tolist(), ["Fixed", "Free", "Displaced"])

    def test_tiling_covers_a_fixed_viewport_under_large_shear(self):
        # A two-triangle periodic cell. The four points are images of the
        # same node, as in the native export for this one-node fixture.
        for shear in (0., 32.4, -32.4):
            with self.subTest(shear=shear):
                points = np.array([[0, 0], [2, 0], [shear, 1], [shear + 2, 1]])
                triangles = points[[[0, 1, 2], [1, 3, 2]]]
                basis = periodic_basis(points, [0, 0, 0, 0], 2.)
                polygons, identities = visible_periodic_images(triangles, basis, [[0, 0], [2, 1]])
                edges = np.roll(polygons, -1, axis=1) - polygons
                for x in np.linspace(.013, 1.973, 23):
                    for y in np.linspace(.021, .981, 19):
                        delta = [x, y] - polygons
                        cross = edges[..., 0] * delta[..., 1] - edges[..., 1] * delta[..., 0]
                        self.assertEqual(np.count_nonzero(np.all(cross >= -1e-12, axis=1)), 1)
                self.assertEqual(set(identities), {0, 1})

    def test_playback_counts_the_reversal_and_enforces_minimum_duration(self):
        frames = [{"load": value} for value in (0., .5, 0., .25)]
        options = dict(playback="load", fps=10, seconds_per_load=4, minimum_seconds=2.25)
        schedule = frame_schedule(frames, **options)
        self.assertEqual(len(schedule), 50)
        self.assertEqual(list(dict.fromkeys(schedule)), [0, 1, 2, 3])
        short = frame_schedule(frames[:1], **options)
        self.assertEqual(len(short), 23)
        self.assertFalse(short.any())


if __name__ == "__main__":
    unittest.main()
