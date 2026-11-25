"""Photonic quantum channels"""

from collections.abc import Sequence
from math import log
from typing import Any

import torch
from torch import nn

import deepquantum.photonic as dqp

from ..qmath import evolve_state
from .bargmann import BargmannPolyExp, _gaussian_bargmann
from .gate import BeamSplitterSingle, Displacement
from .operation import Channel
from .qmath import _times_i


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


def _vacuum_loss_coefficients(control: torch.Tensor, *, is_power: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the signed transmitted field and lost power in the supplied coordinate."""
    if is_power:
        return control.sqrt(), 1 - control
    half_angle = control / 2
    return half_angle.cos(), half_angle.sin().square()


def _vacuum_loss_xy(field: torch.Tensor, loss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Local attenuation and vacuum-noise matrices used by both loss and coherent displacement."""
    identity = torch.eye(2, dtype=field.dtype, device=field.device)
    return field * identity, loss * identity * dqp.hbar / (4 * dqp.kappa**2)


class _CalibratedLossChannel(Channel):
    """Keep fixed and encoded loss in its physical coordinate and optimized loss in its signed angle."""

    def __getattr__(self, name: str) -> Any:
        value = super().__getattr__(name)
        if name == 'loss_theta' and value is None:
            return _loss_to_theta(self.loss_control, self.convention)
        return value

    def _set_controls(self, names: tuple[str, ...], values: Sequence[torch.Tensor], raw_loss: Any) -> None:
        """Register the active controls, leaving an angle slot for hardware overrides."""
        for name, value in zip(names, values, strict=True):
            if self.requires_grad:
                setattr(self, name, nn.Parameter(value))
            elif name == 'loss_theta' and self.convention in ('t', 'db', 'gain_db'):
                if raw_loss is None:
                    power = (value / 2).cos().square()
                    raw_loss = power if self.convention == 't' else -10 * power.log10()
                self.register_buffer('loss_control', torch.as_tensor(raw_loss, dtype=value.dtype, device=value.device))
                self.register_buffer(name, None)
            else:
                self.register_buffer(name, value)

    @property
    def _native_loss(self) -> torch.Tensor | None:
        """Read the physical calibration unless a signed angle is active."""
        if self._buffers.get('loss_theta') is None:
            return self._buffers.get('loss_control')
        return None

    @property
    def t(self) -> torch.Tensor:
        """Signal power transmittance."""
        if (loss := self._native_loss) is not None:
            return loss if self.convention == 't' else 10 ** (-loss / 10)
        angle = self.loss_theta.remainder(2 * torch.pi)
        return torch.where(angle == torch.pi, 0.0, (angle / 2).cos().square())

    def _apply(self, fn: Any, recurse: bool = True) -> '_CalibratedLossChannel':
        """Preserve an angular complete-loss setting across dtype conversion."""
        if self._native_loss is not None:
            return super()._apply(fn, recurse)
        dtype = self.loss_theta.dtype
        full_loss = self.loss_theta == torch.pi
        super()._apply(fn, recurse)
        if self.loss_theta.dtype != dtype:
            with torch.no_grad():
                self.loss_theta.copy_(torch.where(full_loss.to(self.loss_theta.device), torch.pi, self.loss_theta))
        return self


class DisplacementChannel(_CalibratedLossChannel):
    r"""Displacement by mixing the signal with a coherent auxiliary mode.

    The auxiliary mode has coherent amplitude :math:`\beta=Ae^{i\phi}` and mean photon number
    :math:`|\beta|^2`, where :math:`A` and :math:`\phi` are ``amplitude`` and ``phase`` (in radians).
    For the mixing angle :math:`\theta` (``loss_theta``), the beam splitter acts as

    .. math::

        \hat a_{\mathrm{out}} = \cos(\theta/2)\hat a_{\mathrm{in}} + \sin(\theta/2)\hat b_{\mathrm{in}}.

    Tracing out the auxiliary output gives attenuation followed by displacement
    :math:`\alpha=\beta\sin(\theta/2)`. The signal power transmittance is :math:`t=\cos^2(\theta/2)`.
    A transmittance input selects :math:`\theta\in[0,\pi]`. An angle input preserves the field sign;
    a negative amplitude corresponds to a phase shift of :math:`\pi`.
    See https://arxiv.org/pdf/1009.1844v2 Eq.(8), Eq.(18).

    Args:
        inputs: The parameters of the channel (``amplitude``, ``phase`` and ``t``).
            The third input is interpreted by ``convention``. Default: ``None``
        nmode: The number of modes that the quantum operation acts on. Default: 1
        wires: The indices of the modes that the quantum operation acts on. Default: ``None``
        cutoff: The Fock space truncation. Default: ``None``
        convention: The convention of the third input, ``'t'`` for signal power transmittance or
            ``'loss_theta'`` for the internal coupling angle in radians. Default: ``'t'``
        requires_grad: Whether the parameters are ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """

    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 1,
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        convention: str = 't',
        requires_grad: bool = False,
    ) -> None:
        super().__init__(name='DisplacementChannel', nmode=nmode, wires=wires, cutoff=cutoff)
        assert len(self.wires) == 1, 'DisplacementChannel must act on one mode.'
        assert convention in ('t', 'loss_theta'), 'Invalid coupling convention'
        self.convention = convention
        self.requires_grad = requires_grad
        self.npara = 3
        if inputs is not None:
            if len(inputs) != 3:
                raise ValueError('Expected [amplitude, phase, t].')
            for value in inputs:
                value = torch.as_tensor(value)
                if value.numel() != 1 or value.is_complex() or not torch.isfinite(value).all():
                    raise ValueError('DisplacementChannel controls must be finite real scalars.')
            t = torch.as_tensor(inputs[2])
            if convention == 't' and ((t < 0) | (t > 1)).any():
                raise ValueError('Signal power retention t must be in [0, 1].')
        self.init_para(inputs)

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        if inputs is None:
            values = [torch.rand(()), torch.rand(()) * 2 * torch.pi, torch.rand(()) * torch.pi]
        else:
            values = [
                v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.get_default_dtype()) for v in inputs
            ]
            values = [v if v.is_floating_point() else v.to(torch.get_default_dtype()) for v in values]
            if self.convention == 't':
                values[2] = _loss_to_theta(values[2], 't')
        self._set_controls(('amplitude', 'phase', 'loss_theta'), values, None if inputs is None else inputs[2])

    @property
    def coherent_amplitude(self) -> torch.Tensor:
        """Complex auxiliary amplitude, whose squared magnitude is its mean photon number."""
        return self.amplitude * torch.exp(_times_i(self.phase))

    @property
    def _drive(self) -> torch.Tensor:
        """Signed displacement radius; a dark auxiliary contributes zero at every transmittance."""
        if (power := self._native_loss) is not None:
            dark = (power == 1) & (self.amplitude == 0)
            lost = torch.where(dark, 1.0, 1 - power)
            return self.amplitude * torch.where(dark, 0.0, lost.sqrt())
        return self.amplitude * (self.loss_theta / 2).sin()

    @property
    def displacement_amplitude(self) -> torch.Tensor:
        """Complex displacement amplitude, including the beam-splitter coupling sign."""
        return self._drive * torch.exp(_times_i(self.phase))

    def update_matrix_state(self) -> torch.Tensor:
        """Update the local Kraus matrices acting on Fock state density matrices."""
        loss = PhotonLoss(self.loss_theta, cutoff=self.cutoff)
        displacement = Displacement(cutoff=self.cutoff).get_matrix_state(self._drive, self.phase)
        return displacement @ loss.update_matrix_state()

    def _loss_coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        power = self._native_loss
        return _vacuum_loss_coefficients(self.loss_theta if power is None else power, is_power=power is not None)

    def update_transform_xy(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the local transformation matrices X and Y acting on Gaussian states."""
        return _vacuum_loss_xy(*self._loss_coefficients())

    def update_transform_xyd(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update the local Gaussian channel matrices and displacement vector."""
        matrix_x, matrix_y = self.update_transform_xy()
        alpha = self.displacement_amplitude.reshape(())
        displacement = torch.stack([alpha.real, alpha.imag]).reshape(2, 1) * dqp.hbar**0.5 / dqp.kappa
        return matrix_x, matrix_y, displacement

    def op_den_mat(self, x: torch.Tensor) -> torch.Tensor:
        """Apply reduced vacuum loss followed by the coherent auxiliary displacement."""
        field, lost = self._loss_coefficients()
        x = PhotonLoss._apply_density(x, self.nmode, self.wires[0], self.cutoff, field, lost)
        drive = Displacement([self._drive, self.phase], self.nmode, self.wires, self.cutoff, den_mat=True)
        return drive(x)

    def extra_repr(self) -> str:
        return (
            f'wires={self.wires}, amplitude={self.amplitude.item()}, '
            f'phase={self.phase.item()}, loss_theta={self.loss_theta.item()}'
        )


class _CheckCalibration(torch.autograd.Function):
    """Validate every physical setting, including all rows of a vmapped control."""

    @staticmethod
    def forward(valid, message):
        if not bool(valid.all()):
            raise ValueError(message)
        return valid

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def vmap(info, in_dims, valid, message):
        return _CheckCalibration.apply(valid, message), in_dims[0]


class OPA(_CalibratedLossChannel):
    r"""Single-mode optical parametric amplifier with uniform distributed vacuum loss.

    Gain, loss and pump phase are constant along the device, with an undepleted classical pump.
    Let :math:`r=gL` be the integrated gain and :math:`T=\cos^2(\theta_\ell/2)` the pump-off power
    transmittance, with :math:`\theta_\ell` stored in ``loss_theta``. Writing :math:`a=-\ln T`, the channel is

    .. math::

        X = R(\theta/2)\operatorname{diag}(e^{-r-a/2},e^{r-a/2})R(\theta/2)^T,

        Y = a v_0 R(\theta/2)\operatorname{diag}
            (\operatorname{exprel}(-2r-a),\operatorname{exprel}(2r-a))R(\theta/2)^T,

    where :math:`v_0=\hbar/(4\kappa^2)`, :math:`R` is a quadrature rotation and
    :math:`\operatorname{exprel}(z)=(e^z-1)/z`, with :math:`\operatorname{exprel}(0)=1`.
    Positive :math:`r` and :math:`\theta=0` squeeze :math:`x` and amplify :math:`p`.
    The internal parameters are ``r``, ``theta`` and ``loss_theta``.
    See https://arxiv.org/abs/2606.02524v1 Eq.(16) and Methods, Internal Loss.

    Gaussian and Bosonic states use :math:`X` and :math:`Y` directly. The Bargmann backend
    contracts the channel kernel with the kernel of the density operator. For a Fock density matrix,
    the projected channel superoperator has four local axes of length ``cutoff``. Output weight
    above the cutoff is discarded without renormalization.

    ``HardwareModel.opa().submodes(relative_r=...)`` sets fixed orthogonal gain
    modes with :math:`r_k=r q_k`, where :math:`q_k` is ``relative_r[k]``. The same distributed loss acts on
    every basis mode, including zero-gain modes. This assumes mode-independent
    attenuation and a fixed gain basis along the device.

    Args:
        inputs: The parameters of the channel (``r``, ``theta`` and loss).
            With ``convention='gain_db'``, use ``[gain_db, theta, loss_db]``, where gain is the net
            coherent power gain along the amplified axis. Gain :math:`G_{\rm dB}` and pump-off loss
            :math:`L_{\rm dB}` give :math:`r=(G_{\rm dB}+L_{\rm dB})\ln(10)/20`.
            The amplified-axis calibration requires :math:`G_{\rm dB}\ge -L_{\rm dB}`.
            Default: ``None``
        nmode: The number of modes that the quantum operation acts on. Default: 1
        wires: The indices of the modes that the quantum operation acts on. Default: ``None``
        cutoff: The Fock space truncation. Default: ``None``
        convention: The loss convention: ``'theta'`` for the loss angle, ``'t'`` for pump-off power
            transmittance, ``'db'`` for power loss in dB, or ``'gain_db'`` for gain and loss in dB.
            Default: ``'theta'``
        requires_grad: Whether the parameters are ``nn.Parameter`` or ``buffer``.
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
            if r.is_complex() or loss.is_complex():
                raise ValueError('OPA calibration requires real gain_db and loss_db.')
            _CheckCalibration.apply(
                torch.isfinite(r) & torch.isfinite(loss) & (loss >= 0) & (r + loss >= 0),
                'OPA calibration requires finite gain_db and loss_db, loss_db >= 0 and gain_db + loss_db >= 0.',
            )
            r = (r + loss) * (log(10) / 20)
        return r, theta, _loss_to_theta(loss, self.convention)

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        values = self.inputs_to_tensor(inputs)
        self._set_controls(('r', 'theta', 'loss_theta'), values, None if inputs is None else inputs[2])

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
        """Update the local transformation matrices X and Y acting on Gaussian states."""
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

    def op_den_mat(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass for density matrices."""
        wires = self.wires + [wire + self.nmode for wire in self.wires]
        dim = self.cutoff ** len(wires)
        matrix = self.get_superoperator().reshape(dim, dim)
        return evolve_state(x, matrix, 2 * self.nmode, wires, self.cutoff)

    def extra_repr(self) -> str:
        return f'wires={self.wires}, r={self.r.item()}, theta={self.theta.item()}, transmittance={self.t.item()}'


class _PowerLossCoupler(BeamSplitterSingle):
    """Expose the beam-splitter angle while evolving from a supplied power transmittance."""

    def init_para(self, inputs: Any = None, *, convention: str = 'theta') -> None:
        """Keep the supplied physical control, or the random initial angle until data is encoded."""
        name = 'angle' if inputs is None or convention == 'theta' else convention
        self._set_control(name, self.inputs_to_tensor(inputs))
        self.update_matrix()
        self.update_transform_xp()

    def _set_control(self, name: str, value: torch.Tensor) -> None:
        for coordinate in ('angle', 't', 'db'):
            self.register_buffer(coordinate, value if coordinate == name else None)

    @property
    def power(self) -> torch.Tensor | None:
        if self.angle is not None:
            return None
        power = self.t if self.db is None else 10 ** (-self.db / 10)
        # Encoded inputs outside the physical domain propagate NaN, including under vmap.
        return torch.where((power >= 0) & (power <= 1), power, float('nan'))

    @property
    def theta(self) -> torch.Tensor:
        return self.angle if self.angle is not None else 2 * self.power.sqrt().acos()

    def _load_from_state_dict(self, state_dict, prefix, *args):
        # A freshly constructed encoder holds its initial angle until its first physical input.
        for name in ('angle', 't', 'db'):
            if prefix + name in state_dict and getattr(self, name) is None:
                self._set_control(name, torch.empty_like(next(self.buffers())))
                break
        super()._load_from_state_dict(state_dict, prefix, *args)
        self.update_matrix()
        self.update_transform_xp()


class PhotonLoss(Channel):
    r"""Photon loss channel on single mode.

    This channel couples the target mode $\hat{a}$ to the vacuum mode $\hat{b}$ using
    following transformation:

    .. math::

       \hat{a}_{\text{out}} = \cos(\theta/2)\hat{a}_{\text{in}} + \sin(\theta/2)\hat{b}_{\text{vac}}

    The power transmittance is :math:`T=\cos^2(\theta/2)`. Fixed and encoded power or dB inputs
    act directly on the reduced state. A trainable coupler uses the angle as its optimizer parameter.

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
        theta = _loss_to_theta(inputs, convention)
        if convention == 't' and theta is not None:
            _CheckCalibration.apply(
                torch.isfinite(theta) & (not theta.is_complex()),
                'Loss transmittance must be real, finite and lie in [0, 1]',
            )
        coupler = _PowerLossCoupler if convention != 'theta' and not requires_grad else BeamSplitterSingle
        self.gate = coupler(
            inputs=theta,
            nmode=self.nmode + 1,
            wires=self.wires + [self.nmode],
            cutoff=cutoff,
            den_mat=True,
            convention='h',
            requires_grad=requires_grad,
        )
        if isinstance(self.gate, _PowerLossCoupler) and inputs is not None:
            self.gate._set_control(convention, self.gate.inputs_to_tensor(inputs))
        self.npara = 1

    @property
    def theta(self):
        return self.gate.theta

    @property
    def t(self):
        """Transmittance."""
        if (power := getattr(self.gate, 'power', None)) is not None:
            return power
        return (self.theta / 2).cos().square()

    def _loss_coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        power = getattr(self.gate, 'power', None)
        return _vacuum_loss_coefficients(self.theta if power is None else power, is_power=power is not None)

    @property
    def db(self):
        """Photon loss in dB."""
        return -10 * torch.log10(self.t)

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        if isinstance(self.gate, _PowerLossCoupler):
            self.gate.init_para(inputs, convention=self.convention)
        else:
            self.gate.init_para(_loss_to_theta(inputs, self.convention))

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

    def get_matrix_state(self, theta: Any) -> torch.Tensor:
        """Get the local Kraus matrices acting on Fock state density matrices.

        See https://arxiv.org/pdf/1012.4266 Eq.(2.4)
        """
        matrix = self.gate.get_matrix(theta)
        levels = torch.arange(1, self.cutoff, dtype=matrix.real.dtype, device=matrix.device)
        columns = [matrix.new_ones(1)]
        # Add one input photon to |n-1, 0>. Only |n-lost, lost> can be occupied.
        for n in range(1, self.cutoff):
            factor = (levels[:n] / n).sqrt()
            transmitted = columns[-1] * matrix[0, 0] * factor.flip((0,))
            reflected = columns[-1] * matrix[1, 0] * factor
            columns.append(torch.nn.functional.pad(transmitted, (0, 1)) + torch.nn.functional.pad(reflected, (1, 0)))
        photons, lost = torch.tril_indices(self.cutoff, self.cutoff, device=matrix.device)
        return matrix.new_zeros((self.cutoff,) * 3).index_put((lost, photons - lost, photons), torch.cat(columns))

    def update_matrix_state(self) -> torch.Tensor:
        """Update the local Kraus matrices acting on Fock state density matrices."""
        return self.get_matrix_state(self.theta)

    def op_den_mat(self, x: torch.Tensor) -> torch.Tensor:
        """Apply vacuum loss, summing over the same lost photons on ket and bra."""
        return self._apply_density(x, self.nmode, self.wires[0], self.cutoff, *self._loss_coefficients())

    @staticmethod
    def _apply_density(
        x: torch.Tensor, nmode: int, wire: int, cutoff: int, field: torch.Tensor, loss: torch.Tensor
    ) -> torch.Tensor:
        """Apply the reduced vacuum-loss kernel to one wire."""
        # Each lost photon contributes 1 - T to the density matrix. Taking its
        # square root separately in each Kraus amplitude is singular at T = 1.
        axes = (wire + 1, wire + nmode + 1)
        rho = x.movedim(axes, (-2, -1))
        levels = torch.arange(cutoff, device=field.device)
        weight = field.reshape(()).pow(levels[:, None] + levels)
        result = weight * rho
        for lost in range(1, cutoff):
            factor = (levels[lost:].to(field) / lost).sqrt()
            weight = weight[:-1, :-1] * factor[:, None] * factor * loss
            term = rho[..., lost:, lost:] * weight
            result = result + torch.nn.functional.pad(term, (0, lost, 0, lost))
        return result.movedim((-2, -1), axes)

    def update_transform_xy(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the local transformation matrices X and Y acting on Gaussian states.

        See https://arxiv.org/pdf/quant-ph/0503237 Eq.(4.19), Eq.(4.20)
        """
        matrix_x, matrix_y = _vacuum_loss_xy(*self._loss_coefficients())
        self.matrix_x = matrix_x.detach()
        self.matrix_y = matrix_y.detach()
        return matrix_x, matrix_y

    def op_cv(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Attenuate the target quadratures and add their independent vacuum noise."""
        cov, mean = x[:2]
        local_x, local_y = self.update_transform_xy()
        mask = (torch.arange(self.nmode, device=cov.device) == self.wires[0]).repeat(2)
        scale = torch.where(mask, local_x[0, 0], 1.0)
        cov = scale[:, None] * cov * scale
        cov.diagonal(dim1=-2, dim2=-1).add_(mask * local_y[0, 0])
        return [cov, scale[:, None].to(mean.dtype) * mean, *x[2:]]

    def get_bargmann(self) -> BargmannPolyExp:
        r"""Get the Bargmann representation of the channel.

        With field transmittance :math:`t=\cos(\theta/2)` and power transmittance :math:`T=t^2`, the kernel is

        .. math::

            K(z,w,u,v) = \exp\!\left[t\,(zu + wv) + (1-T)\,uv\right].

        Variables are ordered as ket output, bra output, ket input, bra input.
        """
        if len(self.wires) != 1:
            raise NotImplementedError('PhotonLoss acts on a single mode')
        amplitude, loss = self._loss_coefficients()
        amplitude = amplitude + 0j
        zero = torch.zeros_like(amplitude)
        mat_a = torch.stack(
            [
                zero,
                zero,
                amplitude,
                zero,
                zero,
                zero,
                zero,
                amplitude,
                amplitude,
                zero,
                zero,
                loss,
                zero,
                amplitude,
                loss,
                zero,
            ]
        ).reshape(4, 4)
        return _gaussian_bargmann(mat_a, mat_a.new_zeros(4), mat_a.new_ones(()))

    def extra_repr(self) -> str:
        return f'wires={self.wires}, transmittance={self.t.item()}'
