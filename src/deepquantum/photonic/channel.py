"""
Photonic quantum channels
"""

from typing import Any, List, Optional, Tuple, Union

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
        """Update the local Kraus matrices acting on Fock state density matrices."""
        return self.get_matrix_state(self.theta)

    def get_matrix_state(self, theta: Any) -> torch.Tensor:
        """Get the local Kraus matrices acting on Fock state density matrices.

        See https://arxiv.org/pdf/1012.4266 Eq.(2.4)
        """
        matrix = self.gate.get_matrix_state(self.gate.get_matrix(theta))
        return matrix[..., 0].permute([1, 0, 2])

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        self.gate.init_para(inputs)

    def update_transform_xy(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the local transformation matrices X and Y acting on Gaussian states.

        See https://arxiv.org/pdf/quant-ph/0503237 Eq.(4.19), Eq.(4.20)
        """
        g_t_sqrt = self.theta.new_ones(2).diag() * torch.cos(self.theta / 2)
        g_t = self.theta.new_ones(2).diag() * torch.cos(self.theta / 2) ** 2
        identity = self.theta.new_ones(2).diag()
        sigma_h = self.theta.new_ones(2).diag() * dqp.hbar / (4 * dqp.kappa ** 2)
        matrix_x = g_t_sqrt
        matrix_y = (identity - g_t) @ sigma_h
        self.matrix_x = matrix_x.detach()
        self.matrix_y = matrix_y.detach()
        return matrix_x, matrix_y

    def extra_repr(self) -> str:
        return f'wires={self.wires}, transmittance={self.t.item()}'
