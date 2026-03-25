from __future__ import annotations

from escnn import gspaces
from escnn import kernels

from escnn.group import *

from typing import Tuple, Callable, List


__all__ = [
    "RotHue2D",
    "rotHueOnR2",
]


class RotHue2D(gspaces.GSpace2D):
    r"""
    G-space on :math:`R^2` with fiber group :math:`C_N \times C_H`:
    discrete planar rotations (:math:`C_N`) and discrete hue shifts (:math:`C_H`).

    The hue factor acts trivially on base-space coordinates and only in feature space.
    """

    def __init__(
        self,
        N: int,
        H: int,
        maximum_frequency: int = 6,
        _supergroup: Group = None,
        _sg_id=None,
        _super_action: Representation = None,
    ):
        assert isinstance(N, int) and N > 0
        assert isinstance(H, int) and H > 0

        self.N = N
        self.H = H

        if _supergroup is None:
            o2 = o2_group(maximum_frequency=maximum_frequency)
            ch = cyclic_group(H)
            _supergroup = direct_product(o2, ch)

            # Standard 2D action from the O(2) factor, trivial hue factor.
            _super_action = _supergroup.irrep(
                o2.irrep(1, 1).id,
                ch.trivial_representation.id,
                0,
            )

            sg_id = ((None, N), ch.subgroup_self_id)
            _sg_id = _supergroup._process_subgroup_id(sg_id)
        else:
            assert _sg_id is not None
            assert _super_action is not None

        fibergroup, inclusion, restriction = _supergroup.subgroup(_sg_id)

        self._supergroup = _supergroup
        self._super_action = _super_action
        self._sg_id = _sg_id
        self._inclusion = inclusion
        self._restriction = restriction
        self._base_action = _super_action.restrict(_sg_id)

        name = f"{fibergroup}_on_R2[rotHueOnR2(N={N}, H={H})]"
        gspaces.GSpace.__init__(self, fibergroup, 2, name)

    def restrict(self, id) -> Tuple[gspaces.GSpace, Callable, Callable]:
        sg_id = self._supergroup._combine_subgroups(self._sg_id, id)
        sg, inclusion, restriction = self.fibergroup.subgroup(id)

        subspace = RotHue2D(
            self.N,
            self.H,
            _supergroup=self._supergroup,
            _sg_id=sg_id,
            _super_action=self._super_action,
        )

        assert subspace.fibergroup == sg
        return subspace, inclusion, restriction

    @property
    def rotations_order(self):
        first_id = self._sg_id[0]
        if isinstance(first_id, tuple) and len(first_id) > 1:
            return first_id[1]
        return None

    @property
    def flips_order(self):
        first_id = self._sg_id[0]
        if isinstance(first_id, tuple) and len(first_id) > 0:
            return 1 if first_id[0] is not None else 0
        return 0

    @property
    def hue_order(self):
        second_id = self._sg_id[1]
        if isinstance(second_id, int):
            return second_id
        subgroup = self._supergroup.G2.subgroup(second_id)[0]
        return subgroup.order()

    def _basis_generator(
        self,
        in_repr: Representation,
        out_repr: Representation,
        rings: List[float],
        sigma: List[float],
        **kwargs,
    ) -> kernels.KernelBasis:
        maximum_frequency = kwargs.get("maximum_frequency", None)

        return kernels.kernels_O2xCN_subgroup_act_R2(
            in_repr,
            out_repr,
            sg_id=self._sg_id,
            hue_order=self.H,
            radii=rings,
            sigma=sigma,
            maximum_frequency=maximum_frequency,
            axis=0.0,
            adjoint=None,
            filter=None,
        )

    @property
    def basespace_action(self) -> Representation:
        return self._base_action
    
    @property
    def phase_repr(self) -> Representation:
        r"""
        2D hue phase representation used for :math:`[\cos\theta, \sin\theta]`.
        """
        phase = self.fibergroup.irrep(1)
        if phase.size != 2:
            raise ValueError("Hue phase representation requires H > 2 so that irrep(1) is 2-dimensional.")
        return phase

    def __eq__(self, other):
        if isinstance(other, RotHue2D):
            return (
                self.N == other.N
                and self.H == other.H
                and self._sg_id == other._sg_id
                and self.fibergroup == other.fibergroup
            )
        return False

    def __hash__(self):
        return hash((self.N, self.H, self._sg_id, self.fibergroup))


def rotHueOnR2(N: int, H: int = None, maximum_frequency: int = 6) -> RotHue2D:
    r"""
    Describes the joint symmetry :math:`C_N \times C_H` on planar inputs.

    Args:
        N (int): number of discrete spatial rotations.
        H (int, optional): number of discrete hue shifts. Defaults to ``N``.
        maximum_frequency (int): maximum angular frequency used by kernel bases.
    """
    assert isinstance(N, int) and N > 0
    if H is None:
        H = N
    assert isinstance(H, int) and H > 0

    return RotHue2D(N, H, maximum_frequency=maximum_frequency)

