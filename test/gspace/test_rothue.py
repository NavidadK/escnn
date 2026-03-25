import unittest
from unittest import TestCase

import numpy as np

from escnn.gspaces import rotHueOnR2
from escnn.group import DirectProductGroup


class TestRotHueGSpace(TestCase):

    def test_build(self):
        gs = rotHueOnR2(4, 6, maximum_frequency=5)
        self.assertEqual(gs.dimensionality, 2)
        self.assertEqual(gs.rotations_order, 4)
        self.assertEqual(gs.hue_order, 6)
        self.assertIsNotNone(gs.basespace_action)

    def test_hue_generator_keeps_coordinates(self):
        gs = rotHueOnR2(4, 6, maximum_frequency=5)
        G = gs.fibergroup
        self.assertIsInstance(G, DirectProductGroup)

        hue_gen = G.inclusion2(G.G2.element(1))
        A = gs.basespace_action(hue_gen)
        self.assertTrue(np.allclose(A, np.eye(2), atol=1e-7))

    def test_rotation_generator_is_orthogonal(self):
        gs = rotHueOnR2(8, 8, maximum_frequency=5)
        G = gs.fibergroup
        self.assertIsInstance(G, DirectProductGroup)

        rot_gen = G.inclusion1(G.G1.element(1))
        A = gs.basespace_action(rot_gen)

        self.assertTrue(np.allclose(A.T @ A, np.eye(2), atol=1e-7))
        self.assertTrue(np.isclose(np.linalg.det(A), 1.0, atol=1e-7))


if __name__ == "__main__":
    unittest.main()

