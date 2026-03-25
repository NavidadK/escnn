# from __future__ import annotations

# from escnn import gspaces
# from escnn import kernels

# from .utils import rotate_array_2d

# from escnn.group import *

# import numpy as np

# from typing import Tuple, Union, Callable, List

# __all__ = [
#     "Hue2D",
#     "HueOnR2",
# ]

# class Hue2D(gspaces.GSpace2D):
#     r"""
#     G-space on :math:`R^2` with fiber group :math:`C_N`:
#     discrete hue shifts (:math:`C_H`).

#     The hue factor acts trivially on base-space coordinates and only in feature space.
#     """

#     def __init__(
#         self,
#         N: int,
#         maximum_frequency: int = 6,
#         _supergroup: Group = None,
#         _sg_id=None,
#         _super_action: Representation = None,
#     ):
#         assert isinstance(N, int) and N > 0

#         self.N = N

#         cg = cyclic_group(N)

from __future__ import annotations

from escnn import gspaces
from escnn import kernels

from escnn.group import *

from typing import Tuple, Callable, List


__all__ = [
    "Hue2D",
    "hueOnR2",
]


class Hue2D(gspaces.GSpace2D):
    r"""
    G-space on :math:`R^2` with fiber group :math:`C_H` acting only on feature channels.
    The base-space action is the identity for all hue elements, i.e. no spatial transform.
    """

    def __init__(self, H: int, maximum_frequency: int = 6, _fibergroup: Group = None):
        assert isinstance(H, int) and H > 0
        assert isinstance(maximum_frequency, int) and maximum_frequency >= 0

        self.H = H
        self.maximum_frequency = maximum_frequency

        if _fibergroup is None:
            fibergroup = cyclic_group(H)
        else:
            assert isinstance(_fibergroup, CyclicGroup)
            assert _fibergroup.order() == H
            fibergroup = _fibergroup

        base_action_name = "__hue_on_r2_base_action__"
        if base_action_name in fibergroup.representations:
            self._base_action = fibergroup.representations[base_action_name]
        else:
            self._base_action = directsum(
                [fibergroup.trivial_representation, fibergroup.trivial_representation],
                name=base_action_name,
            )

        name = f"{fibergroup}_on_R2[hueOnR2(H={H})]"
        gspaces.GSpace.__init__(self, fibergroup, 2, name)

    def restrict(self, id) -> Tuple[gspaces.GSpace, Callable, Callable]:
        subgroup, inclusion, restriction = self.fibergroup.subgroup(id)
        subspace = Hue2D(subgroup.order(), maximum_frequency=self.maximum_frequency, _fibergroup=subgroup)
        return subspace, inclusion, restriction

    @property
    def phase_repr(self) -> Representation:
        r"""
        2D hue phase representation used for :math:`[\cos\theta, \sin\theta]`.
        """
        phase = self.fibergroup.irrep(1)
        if phase.size != 2:
            raise ValueError("Hue phase representation requires H > 2 so that irrep(1) is 2-dimensional.")
        return phase

    def _basis_generator(
        self,
        in_repr: Representation,
        out_repr: Representation,
        rings: List[float],
        sigma: List[float],
        **kwargs,
    ) -> kernels.KernelBasis:
        maximum_frequency = kwargs.get("maximum_frequency", None)
        basis_filter = kwargs.get("filter", None)

        return kernels.kernels_HueCN_act_R2(
            in_repr,
            out_repr,
            hue_order=self.H,
            radii=rings,
            sigma=sigma,
            maximum_frequency=maximum_frequency,
            filter=basis_filter,
        )

    @property
    def basespace_action(self) -> Representation:
        return self._base_action

    def __eq__(self, other):
        if isinstance(other, Hue2D):
            return self.H == other.H and self.fibergroup == other.fibergroup
        return False

    def __hash__(self):
        return hash((self.H, self.fibergroup))


def hueOnR2(H: int, maximum_frequency: int = 6) -> Hue2D:
    r"""
    Describes hue-only cyclic symmetry :math:`C_H` for planar images, with no spatial action.

    Args:
        H (int): number of discrete hue shifts.
        maximum_frequency (int): maximum frequency used by kernel bases.
    """
    assert isinstance(H, int) and H > 0
    assert isinstance(maximum_frequency, int) and maximum_frequency >= 0

    return Hue2D(H, maximum_frequency=maximum_frequency)
