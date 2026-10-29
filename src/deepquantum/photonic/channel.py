"""Photonic quantum channels"""

from collections.abc import Sequence
from math import log
from typing import Any

import torch
from torch import nn

import deepquantum.photonic as dqp

from .gate import BeamSplitterSingle
from .operation import Channel


def _loss_to_theta(inputs: Any, convention: str) -> torch.Tensor | None:
    """Convert a loss angle, transmittance or dB value to the internal angle."""
    while isinstance(inputs, list):
        inputs = inputs[0]
    if inputs is None:
        return None
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs, dtype=torch.float)
    if convention == 'theta':
        return inputs
    elif convention == 't':
        return torch.arccos(inputs**0.5) * 2
    else:
        t = 10 ** (-inputs / 10)
        return torch.arccos(t**0.5) * 2


class OPA(Channel):
    r"""Single-mode optical parametric amplifier with uniform distributed vacuum loss.

    Gain, loss and pump phase are constant along the device, with an undepleted
    classical pump. The complete channel is ``mean -> X @ mean`` and
    ``cov -> X @ cov @ X.T + Y``. Gaussian and Bosonic states are supported.

    With ``r = g L``, ``T = cos(loss_theta / 2)**2`` and ``a = -log(T) = 2 alpha L``,

    .. math::

        X = R(\theta/2)\operatorname{diag}(e^{-r-a/2}, e^{r-a/2})R(\theta/2)^T,

        Y = a v_0 R(\theta/2)\operatorname{diag}
            (\operatorname{exprel}(-2r-a),\operatorname{exprel}(2r-a))R(\theta/2)^T,

    where :math:`v_0=\hbar/(4\kappa^2)`, :math:`R(\phi)` is the counterclockwise
    quadrature rotation, and ``exprel(z) = expm1(z) / z`` continued to 1 at zero.
    The squeezing convention matches :class:`~deepquantum.photonic.gate.Squeezing`:
    positive ``r`` and ``theta=0`` squeeze x and amplify p.
    See https://arxiv.org/abs/2606.02524v1, Eq. (16) and Methods, Internal Loss.

    The three internal parameters are ``r``, ``theta`` and ``loss_theta``.
    Loss uses the same angle-to-transmittance map as :class:`PhotonLoss`; here the
    angle only sets attenuation and adds no phase rotation. Complete loss gives
    ``X=0, Y=v_0 I``. Encoding consumes all three inputs under the chosen convention.
    Trainable parameters always use the internal angle, keeping ``T`` in [0, 1].
    With ``convention='gain_db'``, inputs are ``[gain_db, theta, loss_db]``:
    ``gain_db`` is the coherent seed's net power gain along the amplified axis,
    including distributed loss, and ``r = (gain_db + loss_db) * log(10) / 20``.
    This gain excludes spontaneous background and external input/output losses.

    Args:
        inputs: The parameters ``[r, theta, loss]``, or ``[gain_db, theta, loss_db]``
            under ``convention='gain_db'``. Random if ``None``. Default: ``None``
        nmode: The number of modes that the quantum operation acts on. Default: 1
        wires: The single target mode. Default: ``None``
        cutoff: The Fock space truncation. Default: ``None``
        convention: The input convention: ``'theta'`` for the loss angle,
            ``'t'`` for pump-off power transmittance, or ``'db'`` for power loss in dB.
            These retain ``r`` as the first input. ``'gain_db'`` uses net power gain
            and pump-off power loss in dB as the first and third inputs.
            Default: ``'theta'``
        requires_grad: Whether all three parameters are ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """

    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 1,
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        convention: str = 'theta',
        requires_grad: bool = False,
    ) -> None:
        super().__init__(name='OPA', nmode=nmode, wires=wires, cutoff=cutoff)
        assert len(self.wires) == 1, 'OPA must act on one mode.'
        assert convention in ('theta', 't', 'db', 'gain_db'), 'Invalid convention'
        self.convention = convention
        self.requires_grad = requires_grad
        self.npara = 3
        if inputs is not None:
            assert len(inputs) == 3, 'OPA requires [r, theta, loss].'
            loss = torch.as_tensor(inputs[2])
            if loss.numel() != 1 or loss.is_complex() or not torch.isfinite(loss).all():
                raise ValueError('Loss must be a finite real scalar.')
            if convention == 't' and ((loss < 0) | (loss > 1)).any():
                raise ValueError('Transmittance must be in [0, 1].')
            if convention in ('db', 'gain_db') and (loss < 0).any():
                raise ValueError('Loss in dB must be nonnegative.')
        self.init_para(inputs)

    @property
    def t(self) -> torch.Tensor:
        """Pump-off power transmittance."""
        angle = self.loss_theta.remainder(2 * torch.pi)
        return torch.where(angle == torch.pi, torch.zeros_like(angle), torch.cos(angle / 2).square())

    @property
    def db(self) -> torch.Tensor:
        """Pump-off power loss in dB."""
        return -10 * torch.log10(self.t)

    def inputs_to_tensor(self, inputs: Any = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert inputs to r, theta and the internal loss angle."""
        if inputs is None:
            return torch.rand(1)[0], torch.rand(1)[0] * 2 * torch.pi, torch.rand(1)[0] * 2 * torch.pi
        r, theta, loss = inputs
        if not isinstance(r, torch.Tensor):
            r = torch.tensor(r, dtype=torch.float)
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, dtype=torch.float)
        if self.convention == 'gain_db':
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, dtype=torch.float)
            r = (r + loss) * (log(10) / 20)
        return r, theta, _loss_to_theta(loss, self.convention)

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        values = self.inputs_to_tensor(inputs)
        for name, value in zip(('r', 'theta', 'loss_theta'), values, strict=True):
            if self.requires_grad:
                setattr(self, name, nn.Parameter(value))
            else:
                self.register_buffer(name, value)

    def _apply(self, fn: Any, recurse: bool = True) -> 'OPA':
        """Preserve the complete-loss endpoint when converting dtype."""
        dtype = self.loss_theta.dtype
        full_loss = self.t == 0
        super()._apply(fn, recurse)
        if self.loss_theta.dtype != dtype:
            # Casting float32 pi to float64 must still represent exactly zero transmittance.
            with torch.no_grad():
                self.loss_theta.copy_(torch.where(full_loss.to(self.loss_theta.device), torch.pi, self.loss_theta))
        return self

    @staticmethod
    def _exprel(x: torch.Tensor) -> torch.Tensor:
        """Evaluate exprel(x) = expm1(x) / x, continued smoothly to 1 at zero."""
        small = x.abs() < 0.05
        denominator = torch.where(small, torch.ones_like(x), x)
        series_x = torch.where(small, x, torch.zeros_like(x))
        # The Taylor branch avoids cancellation in gradients near zero.
        series = 1 / 120 + series_x * (1 / 720 + series_x * (1 / 5040 + series_x / 40320))
        series = 1 + series_x * (1 / 2 + series_x * (1 / 6 + series_x * (1 / 24 + series_x * series)))
        # Factor expm1(x) so it cannot overflow while expm1(x) / x is still finite.
        regular = (torch.expm1(x / 2) / denominator) * (torch.exp(x / 2) + 1)
        return torch.where(small, series, regular)

    def update_transform_xy(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the complete local channel matrices in x, p order."""
        r, theta = self.r.reshape(()), self.theta.reshape(())
        t = self.t.to(r).reshape(())
        full_loss = t == 0
        # Evaluate the complete-loss limit separately, avoiding log(0) and inf * 0.
        attenuation = -torch.log(torch.where(full_loss, torch.ones_like(t), t))
        r = torch.where(full_loss, torch.zeros_like(r), r)
        cos, sin = torch.cos(theta / 2), torch.sin(theta / 2)
        squeezed = torch.stack([cos.square(), cos * sin, cos * sin, sin.square()]).reshape(2, 2)
        amplified = torch.stack([sin.square(), -cos * sin, -cos * sin, cos.square()]).reshape(2, 2)
        rate_s, rate_a = -2 * r - attenuation, 2 * r - attenuation
        matrix_x = torch.exp(rate_s / 2) * squeezed + torch.exp(rate_a / 2) * amplified
        matrix_y = attenuation * (self._exprel(rate_s) * squeezed + self._exprel(rate_a) * amplified)
        matrix_x = torch.where(full_loss, torch.zeros_like(matrix_x), matrix_x)
        matrix_y = torch.where(full_loss, torch.eye(2, dtype=r.dtype, device=r.device), matrix_y)
        matrix_y = matrix_y * dqp.hbar / (4 * dqp.kappa**2)
        self.matrix_x, self.matrix_y = matrix_x.detach(), matrix_y.detach()
        return matrix_x, matrix_y

    def update_matrix_state(self) -> torch.Tensor:
        """Fock Kraus matrices are not implemented for OPA yet."""
        raise NotImplementedError('OPA supports Gaussian and Bosonic states only; Fock support is not implemented.')

    def extra_repr(self) -> str:
        return f'wires={self.wires}, r={self.r.item()}, theta={self.theta.item()}, transmittance={self.t.item()}'


class PhotonLoss(Channel):
    r"""Photon loss channel on single mode.

    This channel couples the target mode $\hat{a}$ to the vacuum mode $\hat{b}$ using
    following transformation:

    .. math::

       \hat{a}_{\text{out}} = \sqrt{T}\hat{a}_{\text{in}} + \sqrt{1-T}\hat{b}_{\text{vac}}

    Args:
        inputs: The parameter of the channel. Default: ``None``
        nmode: The number of modes that the quantum operation acts on. Default: 1
        wires: The indices of the modes that the quantum operation acts on. Default: ``None``
        cutoff: The Fock space truncation. Default: ``None``
        convention: The convention of the input parameter, including ``'theta'``, ``'t'`` and ``'db'``.
            Default: ``'theta'``
        requires_grad: Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """

    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 1,
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        convention: str = 'theta',
        requires_grad: bool = False,
    ) -> None:
        super().__init__(name='PhotonLoss', nmode=nmode, wires=wires, cutoff=cutoff)
        assert convention in ('theta', 't', 'db'), 'Invalid convention'
        self.convention = convention
        self.requires_grad = requires_grad
        self.gate = BeamSplitterSingle(
            inputs=self._inputs_to_theta(inputs),
            nmode=self.nmode + 1,
            wires=self.wires + [self.nmode],
            cutoff=cutoff,
            den_mat=True,
            convention='h',
            requires_grad=requires_grad,
        )
        self.npara = 1

    @property
    def theta(self):
        return self.gate.theta

    @property
    def t(self):
        """Transmittance."""
        return torch.cos(self.theta / 2) ** 2

    @property
    def db(self):
        """Photon loss in dB."""
        return -10 * torch.log10(self.t)

    @staticmethod
    def _apply_unitary(
        unitary: torch.Tensor | None, matrix: torch.Tensor, wires: Sequence[int], nmode: int
    ) -> torch.Tensor:
        """Extend a Fock basis unitary with one vacuum environment mode per loss."""
        if unitary is None:
            unitary = torch.eye(nmode, dtype=matrix.dtype, device=matrix.device)
        dtype = torch.promote_types(unitary.dtype, matrix.dtype)
        unitary, matrix = unitary.to(device=matrix.device, dtype=dtype), matrix.to(dtype)
        for wire in wires:
            unitary = torch.block_diag(unitary, unitary.new_ones(1, 1))
            rows = torch.tensor([wire, unitary.shape[-1] - 1], device=unitary.device)
            columns = torch.arange(unitary.shape[-1], device=unitary.device)
            unitary = unitary.index_put((rows[:, None], columns), matrix @ unitary[rows])
        return unitary

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
        self.gate.init_para(self._inputs_to_theta(inputs))

    def _inputs_to_theta(self, inputs: Any = None) -> torch.Tensor | None:
        """Convert inputs under the chosen convention to the internal loss theta."""
        return _loss_to_theta(inputs, self.convention)

    def update_transform_xy(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the local transformation matrices X and Y acting on Gaussian states.

        See https://arxiv.org/pdf/quant-ph/0503237 Eq.(4.19), Eq.(4.20)
        """
        g_t_sqrt = self.theta.new_ones(2).diag() * torch.cos(self.theta / 2)
        g_t = self.theta.new_ones(2).diag() * torch.cos(self.theta / 2) ** 2
        identity = self.theta.new_ones(2).diag()
        sigma_h = self.theta.new_ones(2).diag() * dqp.hbar / (4 * dqp.kappa**2)
        matrix_x = g_t_sqrt
        matrix_y = (identity - g_t) @ sigma_h
        self.matrix_x = matrix_x.detach()
        self.matrix_y = matrix_y.detach()
        return matrix_x, matrix_y

    def extra_repr(self) -> str:
        return f'wires={self.wires}, transmittance={self.t.item()}'
