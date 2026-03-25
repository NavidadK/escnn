from escnn.kernels.basis import KernelBasis, AdjointBasis
from escnn.kernels.steerable_basis import SteerableKernelBasis
from escnn.kernels.wignereckart_solver import RestrictedWignerEckartBasis

from escnn.kernels.polar_basis import GaussianRadialProfile, circular_harmonics
from escnn.kernels.steerable_filters_basis import SteerableFiltersBasis

from escnn.group import *

import numpy as np
import torch

from typing import List, Union, Callable, Dict, Tuple, Iterable

__all__ = [
    "CircularShellsBasisO2XCN",
    "kernels_O2xCN_subgroup_act_R2",
    "kernels_HueCN_act_R2",
    "kernels_CNxCN_act_R2",
]


def _rot_freq_from_o2_id(o2_id: Tuple[int, int]) -> int:
    # O(2) irrep id is (flip_frequency, rotation_frequency)
    if isinstance(o2_id, tuple) and len(o2_id) == 2:
        return int(o2_id[1])
    return 0


class CircularShellsBasisO2XCN(SteerableFiltersBasis):

    def __init__(
        self,
        L: int,
        radial: GaussianRadialProfile,
        hue_order: int,
        filter: Callable[[Dict], bool] = None,
        axis: float = np.pi / 2,
    ):
        r"""
        Tensor-product basis over :math:`R^2` with angular harmonics from :math:`O(2)`
        and an additional trivial action on the hue cyclic factor :math:`C_H`.

        The steerable basis group is the direct product :math:`O(2)\times C_H`.
        Only irreps with trivial hue frequency are used in the scalar angular basis.
        """

        self.L = L
        self.hue_order = hue_order
        self.axis = axis

        assert isinstance(radial, GaussianRadialProfile)
        assert isinstance(hue_order, int) and hue_order > 0

        self._angular_dim = 2 * L + 1
        self._num_inv_spaces = 0

        o2 = o2_group(L)
        cn = cyclic_group(hue_order)
        G = direct_product(o2, cn)

        self._o2 = o2
        self._cn = cn
        self._hue_trivial_id = cn.trivial_representation.id

        def j_id_from_freq(j: int) -> Tuple:
            o2_id = (int(j > 0), j)
            return (o2_id, self._hue_trivial_id, 0)

        self._j_id_from_freq = j_id_from_freq

        if filter is not None:
            _filter = torch.zeros(self._angular_dim * len(radial), dtype=torch.bool)
            js = []
            _idx_map = []
            _steerable_idx_map = []

            i = 0
            steerable_i = 0
            for j in range(self.L + 1):
                j_id = self._j_id_from_freq(j)
                irrep_j = G.irrep(*j_id)
                dim = irrep_j.size

                attr_irrep = {
                    "irrep:" + k: v
                    for k, v in irrep_j.attributes.items()
                }
                # Keep compatibility with standard bandlimiting filters used in R2Conv
                attr_irrep["irrep:frequency"] = _rot_freq_from_o2_id(j_id[0])
                attr_irrep["j"] = j_id

                multiplicity = 0
                for attr_r in radial:
                    attr = {}
                    attr.update(attr_r)
                    attr.update(attr_irrep)

                    if filter(attr):
                        multiplicity += 1
                        _filter[i:i + dim] = 1
                        _idx_map += list(range(i, i + dim))
                        _steerable_idx_map.append(steerable_i)

                    i += dim
                    steerable_i += 1

                js.append((j_id, multiplicity))
                self._num_inv_spaces += multiplicity

            self._idx_map = np.array(_idx_map)
            self._steerable_idx_map = np.array(_steerable_idx_map)
        else:
            _filter = None
            self._idx_map = None
            self._steerable_idx_map = None
            js = [
                (self._j_id_from_freq(j), len(radial))
                for j in range(L + 1)
            ]
            self._num_inv_spaces = len(radial) * (L + 1)

        # Base-space action: standard O(2) action with trivial hue factor.
        o2_std = o2.standard_representation()
        action = G.irrep(o2_std.id, self._hue_trivial_id, 0)
        action = change_basis(
            action,
            o2_std(o2.element((0, axis), "radians")),
            name=f"StandardAction|axis=[{axis}]",
        )

        super(CircularShellsBasisO2XCN, self).__init__(G, action, js)
        self.radial = radial

        if _filter is None:
            self._filter = None
        else:
            self.register_buffer("_filter", _filter)

    def sample(self, points: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        assert len(points.shape) == 2
        assert points.shape[1] == self.dimensionality, (points.shape, self.dimensionality)

        S = points.shape[0]
        radii = torch.norm(points, dim=1, keepdim=True)

        non_origin_mask = (radii > 1e-9).reshape(-1)
        sphere = points[non_origin_mask, :] / radii[non_origin_mask, :]

        if out is None:
            out = torch.empty(S, self.dim, 1, 1, device=points.device, dtype=points.dtype)
        assert out.shape == (S, self.dim, 1, 1)

        radial = self.radial.sample(radii)
        assert radial.shape[-2:] == (1, 1)
        radial = radial[..., 0, 0]

        circular = torch.empty(S, self._angular_dim, device=points.device, dtype=points.dtype)
        circular[non_origin_mask, :] = circular_harmonics(sphere, self.L, phase=self.axis)
        circular[~non_origin_mask, :1] = 1.0
        circular[~non_origin_mask, 1:] = 0.0

        tensor_product = torch.einsum("pa,pb->pab", radial, circular)
        n_radii = len(self.radial)

        if self._filter is None:
            tmp_out = out
        else:
            tmp_out = torch.empty(
                S,
                self._angular_dim * n_radii,
                1,
                1,
                device=points.device,
                dtype=points.dtype,
            )

        for j in range(self.L + 1):
            dim = 2 if j > 0 else 1
            last = 2 * j + 1
            first = last - dim
            tmp_out[:, first * n_radii:last * n_radii, 0, 0].view(
                S, n_radii, dim
            )[:] = tensor_product[:, :, first:last]

        if self._filter is not None:
            out[:] = tmp_out[:, self._filter, ...]

        return out

    def _irrep_attrs(self, j_id: Tuple) -> Dict:
        irrep_j = self.group.irrep(*j_id)
        attrs = {
            "irrep:" + k: v
            for k, v in irrep_j.attributes.items()
        }
        attrs["irrep:frequency"] = _rot_freq_from_o2_id(j_id[0])
        return attrs

    def steerable_attrs_iter(self):
        idx = 0
        i = 0
        radial_attrs = list(self.radial)

        for j in range(self.L + 1):
            dim = 2 if j > 0 else 1
            j_id = self._j_id_from_freq(j)
            attr_irrep = self._irrep_attrs(j_id)

            for radial_idx, attr_r in enumerate(radial_attrs):
                if self._filter is None or (self._filter[i:i + dim] == 1).all():
                    assert attr_r["idx"] == radial_idx

                    attr = {}
                    attr.update(attr_irrep)
                    attr.update(attr_r)
                    attr["idx"] = idx
                    attr["radial_idx"] = radial_idx
                    attr["j"] = j_id
                    attr["shape"] = (1, 1)

                    yield attr
                    idx += 1
                i += dim

    def steerable_attrs(self, idx):
        assert idx < self._num_inv_spaces, (idx, self._num_inv_spaces)

        if self._steerable_idx_map is None:
            _idx = idx
        else:
            _idx = self._steerable_idx_map[idx]

        j, radial_idx = divmod(_idx, len(self.radial))
        j_id = self._j_id_from_freq(j)
        attr_irrep = self._irrep_attrs(j_id)
        attr_r = self.radial[radial_idx]

        attr = {}
        attr.update(attr_irrep)
        attr.update(attr_r)
        attr["idx"] = idx
        attr["radial_idx"] = radial_idx
        attr["j"] = j_id
        attr["shape"] = (1, 1)
        return attr

    def steerable_attrs_j_iter(self, j: Tuple) -> Iterable:
        j_id = j
        o2_id = j_id[0]
        _, freq = o2_id
        expected_flag = int(freq > 0)
        if o2_id[0] != expected_flag:
            return

        idx = 0
        for _freq in range(freq):
            idx += self.multiplicity(self._j_id_from_freq(_freq))

        dim = 2 if freq > 0 else 1
        i = 0

        attr_irrep = self._irrep_attrs(j_id)
        radial_attrs = list(self.radial)

        for radial_idx, attr_r in enumerate(radial_attrs):
            if self._filter is None or (self._filter[i:i + dim] == 1).all():
                attr = {}
                attr.update(attr_irrep)
                attr.update(attr_r)
                attr["idx"] = idx
                attr["radial_idx"] = radial_idx
                attr["j"] = j_id
                attr["shape"] = (1, 1)
                yield attr
                idx += 1
            i += dim

    def steerable_attrs_j(self, j: Tuple, idx) -> Dict:
        assert idx < self.multiplicity(j), (idx, self.multiplicity(j))

        o2_id = j[0]
        freq = o2_id[1]
        for _freq in range(freq):
            idx += self.multiplicity(self._j_id_from_freq(_freq))

        if self._steerable_idx_map is None:
            _idx = idx
        else:
            _idx = self._steerable_idx_map[idx]

        _j, radial_idx = divmod(_idx, len(self.radial))
        j_id = self._j_id_from_freq(_j)
        attr_irrep = self._irrep_attrs(j_id)
        attr_r = self.radial[radial_idx]

        attr = {}
        attr.update(attr_irrep)
        attr.update(attr_r)
        attr["idx"] = idx
        attr["radial_idx"] = radial_idx
        attr["j"] = j_id
        attr["shape"] = (1, 1)
        return attr

    def __getitem__(self, idx):
        assert idx < self.dim, (idx, self.dim)
        if self._idx_map is None:
            _idx = idx
        else:
            _idx = self._idx_map[idx]

        block_start = 0
        freq = None
        for j in range(self.L + 1):
            dim = 2 if j > 0 else 1
            block = dim * len(self.radial)
            if block_start + block > _idx:
                freq = j
                local = _idx - block_start
                radial_idx, m = divmod(local, dim)
                break
            block_start += block

        assert freq is not None
        j_id = self._j_id_from_freq(freq)
        attr_irrep = self._irrep_attrs(j_id)
        attr_r = self.radial[radial_idx]

        attr = {}
        attr.update(attr_irrep)
        attr.update(attr_r)
        attr["idx"] = idx
        attr["radial_idx"] = radial_idx
        attr["j"] = j_id
        attr["m"] = m
        attr["shape"] = (1, 1)
        return attr

    def __iter__(self):
        idx = 0
        i = 0
        radial_attrs = list(self.radial)

        for j in range(self.L + 1):
            dim = 2 if j > 0 else 1
            j_id = self._j_id_from_freq(j)
            attr_irrep = self._irrep_attrs(j_id)

            for radial_idx, attr_r in enumerate(radial_attrs):
                for m in range(dim):
                    if self._filter is None or self._filter[i] == 1:
                        attr = {}
                        attr.update(attr_irrep)
                        attr.update(attr_r)
                        attr["idx"] = idx
                        attr["radial_idx"] = radial_idx
                        attr["j"] = j_id
                        attr["m"] = m
                        attr["shape"] = (1, 1)
                        yield attr
                        idx += 1
                    i += 1

    def __eq__(self, other):
        if not isinstance(other, CircularShellsBasisO2XCN):
            return False
        return (
            self.radial == other.radial
            and self.L == other.L
            and self.hue_order == other.hue_order
            and self._filter == other._filter
        )

    def __hash__(self):
        return self.L + 1000 * self.hue_order + hash(self.radial) + hash(self._filter)


def kernels_O2xCN_subgroup_act_R2(
    in_repr: Representation,
    out_repr: Representation,
    sg_id,
    hue_order: int,
    radii: List[float],
    sigma: Union[List[float], float],
    maximum_frequency: int = None,
    axis: float = np.pi / 2.0,
    adjoint: np.ndarray = None,
    filter: Callable[[Dict], bool] = None,
) -> KernelBasis:
    r"""
    Build a steerable kernel basis for subgroups of :math:`O(2)\times C_H` acting on :math:`R^2`,
    where :math:`C_H` acts trivially on base-space coordinates and non-trivially on feature channels.
    """

    assert isinstance(hue_order, int) and hue_order > 0

    if maximum_frequency is None:
        max_in = max(_rot_freq_from_o2_id(irr[0]) for irr in in_repr.irreps)
        max_out = max(_rot_freq_from_o2_id(irr[0]) for irr in out_repr.irreps)
        maximum_frequency = max_in + max_out

    o2 = o2_group(maximum_frequency)
    cn = cyclic_group(hue_order)
    super_group = direct_product(o2, cn)
    group, _, _ = super_group.subgroup(sg_id)

    assert in_repr.group == group
    assert out_repr.group == group

    radial_profile = GaussianRadialProfile(radii, sigma)
    basis = SteerableKernelBasis(
        CircularShellsBasisO2XCN(
            maximum_frequency,
            radial_profile,
            hue_order=hue_order,
            filter=filter,
            axis=axis,
        ),
        in_repr,
        out_repr,
        RestrictedWignerEckartBasis,
        sg_id=sg_id,
    )

    if adjoint is not None and not np.allclose(adjoint, np.eye(2)):
        assert adjoint.shape == (2, 2)
        basis = AdjointBasis(basis, adjoint)

    return basis


def kernels_HueCN_act_R2(
    in_repr: Representation,
    out_repr: Representation,
    hue_order: int,
    radii: List[float],
    sigma: Union[List[float], float],
    maximum_frequency: int = None,
    filter: Callable[[Dict], bool] = None,
) -> KernelBasis:
    r"""
    Build a steerable kernel basis for hue-only equivariance :math:`C_H` on :math:`R^2`.

    The hue group acts trivially on base-space coordinates and non-trivially on feature channels.
    """

    assert isinstance(hue_order, int) and hue_order > 0
    assert in_repr.group == out_repr.group
    assert isinstance(in_repr.group, CyclicGroup)
    assert in_repr.group.order() == hue_order

    # Restrict O(2) to C1 (trivial spatial action) and keep full hue subgroup C_H.
    sg_id = ((None, 1), hue_order)
    return kernels_O2xCN_subgroup_act_R2(
        in_repr,
        out_repr,
        sg_id=sg_id,
        hue_order=hue_order,
        radii=radii,
        sigma=sigma,
        maximum_frequency=maximum_frequency,
        axis=np.pi / 2.0,
        adjoint=None,
        filter=filter,
    )


def kernels_CNxCN_act_R2(
    in_repr: Representation,
    out_repr: Representation,
    N: int,
    hue_order: int,
    radii: List[float],
    sigma: Union[List[float], float],
    maximum_frequency: int = None,
    filter: Callable[[Dict], bool] = None,
) -> KernelBasis:
    r"""
    Convenience wrapper for the subgroup :math:`C_N \times C_H < O(2)\times C_H`.
    """
    sg_id = ((None, N), hue_order)
    return kernels_O2xCN_subgroup_act_R2(
        in_repr,
        out_repr,
        sg_id=sg_id,
        hue_order=hue_order,
        radii=radii,
        sigma=sigma,
        maximum_frequency=maximum_frequency,
        axis=np.pi / 2.0,
        adjoint=None,
        filter=filter,
    )
