from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn

import escnn.nn as enn
from escnn.gspaces.hue import Hue2D, hueOnR2

from ..field_type import FieldType
from ..geometric_tensor import GeometricTensor
from .conv import R2Conv


__all__ = [
    "HSVHuePhaseEncoder",
    "HuePhaseLiftGroupConv2D",
]


class HSVHuePhaseEncoder(nn.Module):
    r"""
    Encode HSV images into a hue-phase field:
    :math:`[h, s, v] \mapsto [\cos(2\pi h), \sin(2\pi h), s, v]`.
    """

    def __init__(self, gspace: Hue2D):
        super().__init__()

        if not isinstance(gspace, Hue2D):
            raise TypeError("HSVHuePhaseEncoder expects a Hue2D gspace.")

        phase_repr = gspace.phase_repr
        self.gspace = gspace
        self.out_type = FieldType(
            gspace,
            [gspace.phase_repr, gspace.trivial_repr, gspace.trivial_repr],
        )

    def forward(self, hsv: torch.Tensor) -> GeometricTensor:
        if hsv.ndim != 4:
            raise ValueError(f"Expected HSV tensor of shape [B,3,H,W], got ndim={hsv.ndim}.")
        if hsv.shape[1] != 3:
            raise ValueError(f"Expected HSV tensor with 3 channels, got {hsv.shape[1]}.")

        hue = hsv[:, 0:1] # torch.remainder(hsv[:, 0:1], 1.0)
        sat = hsv[:, 1:2]
        val = hsv[:, 2:3]

        theta = (2.0 * np.pi) * hue
        phase = torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)
        encoded = torch.cat([phase, sat, val], dim=1)

        return GeometricTensor(encoded, self.out_type)


class HuePhaseLiftGroupConv2D(nn.Module):
    r"""
    Minimal hue-equivariant model:
    HSV -> hue-phase encoder -> lifting R2Conv -> group R2Conv.
    """

    def __init__(
        self,
        H: int,
        hidden_fields: int = 8,
        out_fields: int = 8,
        kernel_size: int = 3,
        padding: Optional[int] = None,
        bias: bool = False,
        maximum_frequency: int = 6,
        gspace: Optional[Hue2D] = None,
    ):
        super().__init__()

        if gspace is None:
            gspace = hueOnR2(H, maximum_frequency=maximum_frequency)
        elif not isinstance(gspace, Hue2D):
            raise TypeError("gspace must be a Hue2D instance.")

        if hidden_fields <= 0 or out_fields <= 0:
            raise ValueError("hidden_fields and out_fields must be positive.")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        if padding is None:
            padding = kernel_size // 2

        self.gspace = gspace
        self.encoder = HSVHuePhaseEncoder(gspace)
        print(f'Size: {self.gspace.fibergroup.order}')

        self.in_type = self.encoder.out_type
        self.hidden_type = FieldType(gspace, [gspace.regular_repr] * hidden_fields)
        self.out_type = FieldType(gspace, [gspace.regular_repr] * out_fields)

        self.lift_conv = R2Conv(
            self.in_type,
            self.hidden_type,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.group_conv = R2Conv(
            self.hidden_type,
            self.out_type,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.gpool = enn.GroupPooling(self.group_conv.out_type, mode="avg")

    def forward_from_encoded(self, x: GeometricTensor) -> GeometricTensor:
        if x.type != self.in_type:
            raise TypeError("Input GeometricTensor type does not match hue-phase input type.")
        x = self.lift_conv(x)
        x = self.group_conv(x)
        return x

    def forward(self, hsv: torch.Tensor) -> tuple[GeometricTensor, GeometricTensor, GeometricTensor]:
        x_tr = self.encoder(hsv)
        z = self.lift_conv(x_tr)
        y = self.group_conv(z)
        g = self.gpool(y)
        return x_tr, z, y, g
