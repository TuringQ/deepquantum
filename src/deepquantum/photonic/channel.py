"""
Photonic quantum channels
"""

from typing import Any, List, Optional, Union

import numpy as np
import torch

import deepquantum.photonic as dqp
from .gate import BeamSplitterSingle
from .operation import Channel


class PhotonLoss(Channel):
    r"""Photon loss channel on single mode.

    This channel couples the target mode $\hat{a}$ to the vacuum mode $\hat{b}$ using
    following transformation:

    .. math::

       \hat{a}_{\text{out}} = \sqrt{T}\hat{a}_{\text{in}} + \sqrt{1-T}\hat{b}_{\text{vac}}

    Args:
        inputs (Any, optional): The parameter of the channel. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 1,
        wires: Union[int, List[int], None] = None,
        cutoff: Optional[int] = None,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name='PhotonLoss', nmode=nmode, wires=wires, cutoff=cutoff)
        self.requires_grad = requires_grad
        self.gate = BeamSplitterSingle(inputs=inputs, nmode=self.nmode + 1, wires=self.wires + [self.nmode],
                                       cutoff=cutoff, den_mat=True, convention='h', requires_grad=requires_grad, noise=False)
        self.npara = 1

    @property
    def theta(self):
        return self.gate.theta

    @property
    def t(self):
        """Transmittance."""
        return torch.cos(self.theta / 2) ** 2

    def update_matrix_state(self) -> torch.Tensor:
        """Update the local transformation matrix acting on Fock state tensors."""
        return self.get_matrix_state(self.theta)

    def get_matrix_state(self, theta: Any) -> torch.Tensor:
        """Get the local Kraus matrix acting on Fock state density matrix.

        See https://arxiv.org/pdf/1012.4266 Eq.(2.4)
        """
        matrix = self.gate.get_matrix_state(self.gate.get_matrix(theta))
        return matrix[..., 0].permute([1, 0, 2])

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        self.gate.init_para(inputs)

    def op_gaussian(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Perform a forward pass for Gaussian states.

        See https://arxiv.org/pdf/quant-ph/0503237 Eq.(4.19), Eq.(4.20)
        """
        cov, mean = x
        wires = self.wires + [wire + self.nmode for wire in self.wires]
        identity = self.theta.new_ones(2 * self.nmode).diag()
        g_t_sqrt = self.theta.new_ones(2 * self.nmode).diag()
        g_t = self.theta.new_ones(2 * self.nmode).diag()
        sigma_inf = self.theta.new_zeros(2 * self.nmode).diag()
        t_sqrt = self.theta.new_ones(2).diag() * torch.cos(self.theta / 2)
        t = self.theta.new_ones(2).diag() * torch.cos(self.theta / 2) ** 2
        sigma_h = self.theta.new_ones(2).diag() * dqp.hbar / (4 * dqp.kappa ** 2)
        g_t_sqrt[np.ix_(wires, wires)] = t_sqrt
        g_t[np.ix_(wires, wires)] = t
        sigma_inf[np.ix_(wires, wires)] = sigma_h
        cov = g_t_sqrt @ cov @ g_t_sqrt + (identity - g_t) @ sigma_inf
        mean = g_t_sqrt @ mean
        return [cov, mean]

    def extra_repr(self) -> str:
        return f'wires={self.wires}, transmittance={self.t.item()}'
