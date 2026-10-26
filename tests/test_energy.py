import unittest
import numpy as np

from MTM_Math.energy import ContiEnergy, PieceWiseQuadratic, rotation, stretch_from_C
from MTM_Math.reduction import lagrange_reduction, plastic_reduction, plastic_reduction_components


class EnergyTests(unittest.TestCase):
    def test_plastic_reduction_preserves_metric_and_unit_shear_choice(self):
        gamma = np.array([4.25, -3.5, 2.75])
        F = np.broadcast_to(np.eye(2), (3, 2, 2)).copy()
        F[:, 0, 1] = gamma
        C = np.concatenate((F.swapaxes(-1, -2) @ F, F @ F.swapaxes(-1, -2)))
        original = C.copy()
        reduced, M = plastic_reduction(C, compute_M=True)
        np.testing.assert_allclose(reduced, M.swapaxes(-1, -2) @ C @ M)
        np.testing.assert_allclose(np.linalg.det(M), 1)
        np.testing.assert_allclose(np.linalg.det(reduced), np.linalg.det(C))
        self.assertTrue(np.all(np.abs(reduced[:, 0, 1]) <=
                               .5*np.minimum(reduced[:, 0, 0], reduced[:, 1, 1])))
        np.testing.assert_array_equal(C, original)
        a, b, c, _ = plastic_reduction_components(1, 1+gamma**2, gamma)
        np.testing.assert_allclose(a, reduced[:3, 0, 0])
        np.testing.assert_allclose(b, reduced[:3, 1, 1])
        np.testing.assert_allclose(c, reduced[:3, 0, 1])
        _, tie_M = plastic_reduction([[2, 1.25], [1.25, 2]], loops=1, compute_M=True)
        np.testing.assert_array_equal(tie_M, [[1, -1], [0, 1]])
        with self.assertRaises(RuntimeError):
            plastic_reduction(C, loops=1)
        with self.assertRaises(ValueError):
            plastic_reduction([[1, 2], [2, 1]])

    def test_gradient_tangent_and_rotation(self):
        F = np.array([[1.03, .12], [.02, .98]])
        h = 1e-6
        P = ContiEnergy.P_from_F(F)
        A = ContiEnergy.elasticity_tensor(F, eulerian=False)
        for k, l in np.ndindex(2, 2):
            perturbation = np.zeros((2, 2))
            perturbation[k, l] = h
            derivative = (ContiEnergy.energy_from_F(F+perturbation)-
                          ContiEnergy.energy_from_F(F-perturbation))/(2*h)
            self.assertAlmostEqual(derivative, P[k, l], places=7)
            derivative_P = (ContiEnergy.P_from_F(F+perturbation)-
                            ContiEnergy.P_from_F(F-perturbation))/(2*h)
            np.testing.assert_allclose(derivative_P, A[:, :, k, l], atol=2e-7)
        R = rotation(.43)
        np.testing.assert_allclose(ContiEnergy.cauchy_from_F(R@F),
                                   R@ContiEnergy.cauchy_from_F(F)@R.T, atol=1e-10)

    def test_reduction_and_returned_metric(self):
        F = np.array([[1., 2.1], [0., 1.]])
        C = F.T@F
        before = C.copy()
        reduced, M = lagrange_reduction(C)
        np.testing.assert_allclose(C, before)
        np.testing.assert_allclose(reduced, M.T@C@M)
        energy, returned = ContiEnergy.energy_from_F(F, returnReducedC=True)
        np.testing.assert_allclose(returned, reduced)
        self.assertAlmostEqual(energy, ContiEnergy.energy_from_simple_shear(.1), places=12)
        self.assertAlmostEqual(energy, ContiEnergy.energy_from_simple_shear(2.1), places=12)
        with self.assertRaises(RuntimeError):
            lagrange_reduction(C, loops=1)

    def test_noise_reference_subclasses_and_invalid_inputs(self):
        raw = ContiEnergy.energy_from_reduced_C(np.eye(2), noise=1.2, zeroReference=False)
        self.assertAlmostEqual(ContiEnergy.energy_from_F(np.eye(2), noise=1.2),
                               raw-ContiEnergy.ground_state_energy())
        # Initializing one energy must not contaminate another subclass's caches.
        ContiEnergy.numeric_potential()
        F = np.array([[1., .1], [0., 1.]])
        self.assertAlmostEqual(PieceWiseQuadratic.energy_from_F(F), .00505, places=12)
        self.assertAlmostEqual(float(ContiEnergy.moduli_at_F(np.eye(2)).mu), 6.075757575757576)
        for F in (np.zeros((2, 2)), np.array([[-1., 0], [0, 1.]]), np.full((2, 2), np.nan)):
            with self.assertRaises(ValueError):
                ContiEnergy.cauchy_from_F(F)
        with self.assertRaises(ValueError):
            stretch_from_C(np.diag([-1., 1.]))


if __name__ == "__main__":
    unittest.main()
