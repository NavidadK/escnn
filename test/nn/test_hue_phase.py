import unittest
from unittest import TestCase

import torch

from escnn import gspaces
from escnn import nn as enn


def _hue_shift_hsv(x: torch.Tensor, k: int, H: int) -> torch.Tensor:
    y = x.clone()
    y[:, 0:1] = torch.remainder(y[:, 0:1] + (float(k) / float(H)), 1.0)
    return y


class TestHuePhaseModules(TestCase):

    def test_encoder_equivariance(self):
        torch.manual_seed(0)

        H = 5
        gs = gspaces.hueOnR2(H)
        encoder = enn.HSVHuePhaseEncoder(gs)

        x = torch.rand(2, 3, 11, 13)
        k = 2
        g = gs.fibergroup.element(k)
        x_shift = _hue_shift_hsv(x, k, H)

        a = encoder(x_shift).tensor
        b = encoder(x).transform(g).tensor
        self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=1e-6))

        encoded = encoder(x)
        encoded_shift = encoded.transform(g).tensor
        self.assertTrue(torch.allclose(encoded_shift[:, 2:3], encoded.tensor[:, 2:3], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(encoded_shift[:, 3:4], encoded.tensor[:, 3:4], atol=1e-6, rtol=1e-6))

    def test_lift_group_conv_equivariance(self):
        torch.manual_seed(0)

        H = 5
        model = enn.HuePhaseLiftGroupConv2D(
            H=H,
            hidden_fields=3,
            out_fields=2,
            kernel_size=5,
            padding=2,
            bias=False,
        )
        model.eval()

        x = torch.rand(2, 3, 17, 19)
        k = 3
        g = model.gspace.fibergroup.element(k)
        x_shift = _hue_shift_hsv(x, k, H)

        y1 = model(x_shift).tensor
        y2 = model(x).transform(g).tensor
        self.assertTrue(torch.allclose(y1, y2, atol=2e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
