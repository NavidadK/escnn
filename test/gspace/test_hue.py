import unittest
from unittest import TestCase

import numpy as np

from escnn.gspaces import Hue2D, hueOnR2


class TestHueGSpace(TestCase):

    def test_build(self):
        gs = hueOnR2(4, maximum_frequency=5)
        self.assertIsInstance(gs, Hue2D)
        self.assertEqual(gs.dimensionality, 2)
        self.assertEqual(gs.fibergroup.order(), 4)

    def test_basespace_action_is_identity(self):
        gs = hueOnR2(6, maximum_frequency=5)

        for g in gs.fibergroup.testing_elements():
            A = gs.basespace_action(g)
            self.assertTrue(np.allclose(A, np.eye(2), atol=1e-7))

    def test_phase_representation_matches_rotation_matrix(self):
        H = 8
        gs = hueOnR2(H, maximum_frequency=5)
        rho = gs.phase_repr

        for k in range(H):
            g = gs.fibergroup.element(k)
            alpha = 2.0 * np.pi * k / H
            expected = np.array([
                [np.cos(alpha), -np.sin(alpha)],
                [np.sin(alpha), np.cos(alpha)],
            ])
            self.assertTrue(np.allclose(rho(g), expected, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
