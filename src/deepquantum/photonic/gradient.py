"""Derivatives with respect to optical power at zero transmitted or reflected field.

For p=s² and f'(0)=0, the one-sided slope is f''(0)/2. For a final scalar
objective built with a shared coupling angle, transmittance_grad evaluates
that limit directly. For native power inputs, _Field and _Readout pass the
curvature of a built-in readout through the ordinary backward call.
"""

from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from typing import Any

import torch
from torch.autograd.function import FunctionCtx

_Fields = list[tuple[torch.Tensor, torch.Tensor]]
_recording: ContextVar[_Fields | None] = ContextVar('loss_fields', default=None)


def _coupling_fields(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cos(theta/2) and sin(theta/2), with the exact field zero and angle derivatives at pi."""
    full_loss = theta == torch.pi
    half = torch.where(full_loss, torch.pi - theta, theta) / 2
    cos, sin = half.cos(), half.sin()
    return torch.where(full_loss, sin, cos), torch.where(full_loss, cos, sin)


def _field_power_slope(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Right derivative in p=s² from the first two derivatives in the signed field s."""
    singular = torch.copysign(torch.full_like(first, torch.inf), first)
    # Projection at finite Fock cutoff can leave a genuine linear term at
    # roundoff. Its sign cannot be inferred from the quadratic coefficient.
    unresolved = (first != 0) & (first.abs() <= 16 * torch.finfo(first.dtype).eps * second.abs())
    return torch.where(first == 0, second / 2, torch.where(unresolved | first.isnan(), torch.nan, singular))


def transmittance_grad(observable: torch.Tensor, loss_theta: torch.Tensor) -> torch.Tensor:
    r"""Differentiate a real scalar observable with respect to the power of a signed loss coupler.

    Build the observable with the same differentiable coupling angle at every use of the
    shared control, using ``loss`` or ``d_channel(..., convention='loss_theta')``.
    The physical branch is :math:`T=\cos^2(\theta/2)` with :math:`0\leq\theta\leq\pi`.
    For a smooth function of the angle with zero first derivative, the one-sided power
    slope is twice its angle curvature at complete loss and minus twice at unit transmission.
    A resolved linear field term gives a signed infinity. A term at roundoff gives NaN.
    Distributed OPA noise at nonzero pump has a logarithmic full-loss limit; differentiate
    its native power input instead.

    Args:
        observable: Real scalar objective, including any operations applied after the circuit readout
        loss_theta: Real scalar coupling angle in radians, with ``requires_grad=True``

    Returns:
        The derivative with respect to power, on the dtype and device of the angle.

    Examples:
        >>> import deepquantum.photonic as dqp
        >>> angle = torch.tensor(torch.pi, dtype=torch.double, requires_grad=True)
        >>> circuit = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
        >>> circuit.d(0, 0.5, 0.0)
        >>> circuit.loss(0, encode=True)
        >>> circuit.loss(0, encode=True)
        >>> _ = circuit.double()(angle.expand(2))
        >>> dqp.transmittance_grad(circuit.quadrature_mean(0).sum(), angle).item()
        1.0
    """
    if observable.numel() != 1 or not observable.is_floating_point():
        raise ValueError('observable must be a real floating-point scalar')
    if loss_theta.numel() != 1 or not loss_theta.is_floating_point() or not loss_theta.requires_grad:
        raise ValueError('loss_theta must be a real floating-point scalar with requires_grad=True')
    if not bool((loss_theta >= 0) & (loss_theta <= torch.pi)):
        raise ValueError('loss_theta must be in [0, pi]')
    zero = torch.zeros_like(loss_theta)
    if not observable.requires_grad:
        return zero
    endpoint = bool((loss_theta == 0) | (loss_theta == torch.pi))
    (first,) = torch.autograd.grad(observable, loss_theta, create_graph=endpoint, retain_graph=True, allow_unused=True)
    if first is None:
        return zero
    if not endpoint:
        return -2 * first / loss_theta.detach().sin()
    second = None
    if first.requires_grad:
        (second,) = torch.autograd.grad(first, loss_theta, retain_graph=True, allow_unused=True)
    # The vanishing signed field is cos(theta/2) at pi and sin(theta/2) at zero.
    sign = 1 if bool(loss_theta == torch.pi) else -1
    return sign * _field_power_slope(-sign * 2 * first.detach(), 4 * (zero if second is None else second))


class _Field(torch.autograd.Function):
    """Use s=sqrt(p) inside the simulation and collect the curvature of the final readout at s=0."""

    @staticmethod
    def forward(power: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return power.sqrt(), torch.zeros_like(power)

    @staticmethod
    def setup_context(ctx: FunctionCtx, inputs: tuple, output: tuple) -> None:
        (power,) = inputs
        field, curvature = output
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(field)
        ctx.save_for_forward(field)
        fields = _recording.get()
        if fields is not None and power.requires_grad and bool((power == 0).any()):
            fields.append((field, curvature))

    @staticmethod
    def vmap(info: Any, in_dims: tuple, power: torch.Tensor) -> tuple:
        return _Field.apply(power), (in_dims[0], in_dims[0])

    @staticmethod
    def jvp(ctx: FunctionCtx, tangent: torch.Tensor) -> tuple:
        (field,) = ctx.saved_tensors
        return tangent / (2 * field), torch.zeros_like(tangent)

    @staticmethod
    def backward(ctx: FunctionCtx, first: torch.Tensor | None, second: torch.Tensor | None) -> torch.Tensor:
        (field,) = ctx.saved_tensors
        first = torch.zeros_like(field) if first is None else first
        second = torch.zeros_like(field) if second is None else second
        zero = field == 0
        if torch.is_grad_enabled() and first.requires_grad and bool(zero.any()):
            # create_graph=True also retains the curvature of a user's objective
            # composed *after* a public readout, e.g. the square of a field mean.
            indices = zero.flatten().nonzero().flatten()
            diagonal = []
            for i in indices.unbind():
                (row,) = torch.autograd.grad(
                    first.flatten()[i], field, create_graph=True, retain_graph=True, allow_unused=True
                )
                diagonal.append(first.new_zeros(()) if row is None else row.flatten()[i])
            second = second.flatten().index_copy(0, indices, torch.stack(diagonal)).view_as(field)
        regular = first / (2 * torch.where(zero, 1.0, field))
        return torch.where(zero, _field_power_slope(first, second), regular)


def _loss_field(power: torch.Tensor) -> torch.Tensor:
    if not torch.is_grad_enabled():
        return power.sqrt()
    return _Field.apply(power)[0]


class _Readout(torch.autograd.Function):
    """Supply the signed-field curvature of the contracted observable in its own backward pass."""

    @staticmethod
    def forward(value: torch.Tensor, fields: tuple, *curvatures: torch.Tensor) -> torch.Tensor:
        return value.view_as(value)

    @staticmethod
    def setup_context(ctx: FunctionCtx, inputs: tuple, output: torch.Tensor) -> None:
        value, fields, *_ = inputs
        ctx.save_for_backward(value, *fields)

    @staticmethod
    def jvp(ctx: FunctionCtx, tangent: torch.Tensor, *unused: Any) -> torch.Tensor:
        return tangent.view_as(tangent)

    @staticmethod
    def backward(ctx: FunctionCtx, cotangent: torch.Tensor) -> tuple:
        value, *fields = ctx.saved_tensors
        if torch.is_grad_enabled():
            return cotangent, None, *(None for _ in fields)
        with torch.enable_grad():
            firsts = torch.autograd.grad(
                value, fields, cotangent, create_graph=True, retain_graph=True, allow_unused=True
            )
            seconds = []
            for first, field in zip(firsts, fields, strict=True):
                second = None
                if first is not None and first.requires_grad:
                    (second,) = torch.autograd.grad(first.sum(), field, retain_graph=True, allow_unused=True)
                seconds.append(second)
        return cotangent, None, *seconds


def _attach(value: Any, fields: _Fields) -> Any:
    if not fields or not torch.is_grad_enabled():
        return value
    if isinstance(value, torch.Tensor) and value.requires_grad:
        roots, curvatures = zip(*fields, strict=True)
        return _Readout.apply(value, roots, *curvatures)
    if isinstance(value, tuple):
        return tuple(_attach(item, fields) for item in value)
    if isinstance(value, list):
        return [_attach(item, fields) for item in value]
    if isinstance(value, dict):
        # Probability tables are also stored as circuit.state.
        for key, probability in value.items():
            value[key] = _attach(probability, fields)
    return value


def _record_loss_fields(forward: Callable) -> Callable:
    @wraps(forward)
    def run(circuit: Any, *args: Any, **kwargs: Any) -> Any:
        from .state import BargmannState

        parent = _recording.get()
        fields = []
        token = _recording.set(fields)
        try:
            value = forward(circuit, *args, **kwargs)
        finally:
            _recording.reset(token)
        if isinstance(value, BargmannState):
            inherited = getattr(value, '_loss_fields', ())
            if inherited:
                fields = list({id(pair[0]): pair for pair in (*inherited, *fields)}.values())
            value._loss_fields = fields
        circuit._loss_fields = fields
        if parent is not None:
            parent.extend(fields)
            return value
        return _attach(value, fields)

    return run


def _loss_readout(readout: Callable) -> Callable:
    @wraps(readout)
    def read(owner: Any, *args: Any, **kwargs: Any) -> Any:
        fields = getattr(owner, '_loss_fields', None)
        if not fields or _recording.get() is not None or not torch.is_grad_enabled():
            return readout(owner, *args, **kwargs)
        # Nested readouts contribute once, at the outermost observable.
        token = _recording.set([])
        try:
            value = readout(owner, *args, **kwargs)
        finally:
            _recording.reset(token)
        return _attach(value, fields)

    return read


def _physical_probability(value: torch.Tensor) -> torch.Tensor:
    """Clamp negative roundoff in a probability, retaining the derivative of the analytic expression."""
    probability = value.real
    return probability - probability.detach().clamp_max(0)


class _OPAFullLossNoise(torch.autograd.Function):
    """Vacuum noise and its one-sided power derivative at complete distributed loss."""

    generate_vmap_rule = True

    @staticmethod
    def forward(power: torch.Tensor, pump: torch.Tensor, contrast: torch.Tensor) -> torch.Tensor:
        vacuum = torch.eye(2, dtype=power.dtype, device=power.device)
        return vacuum + torch.zeros_like(power) + torch.zeros_like(pump) + torch.zeros_like(contrast)

    @staticmethod
    def setup_context(ctx: FunctionCtx, inputs: tuple, output: torch.Tensor) -> None:
        _, pump, contrast = inputs
        ctx.save_for_backward(pump, contrast)
        ctx.save_for_forward(pump, contrast)

    @staticmethod
    def jvp(ctx: FunctionCtx, d_power: torch.Tensor, d_pump: torch.Tensor, d_contrast: torch.Tensor) -> torch.Tensor:
        pump, contrast = ctx.saved_tensors
        identity = torch.eye(2, dtype=contrast.dtype, device=contrast.device)
        leading = torch.where(contrast == 0, identity, pump.sign() * contrast) * d_power
        vertical = torch.where(leading == 0, 0.0, torch.copysign(torch.full_like(leading, torch.inf), leading))
        return torch.where(pump == 0, -identity * d_power, vertical)

    @staticmethod
    def backward(ctx: FunctionCtx, grad: torch.Tensor) -> tuple:
        pump, contrast = ctx.saved_tensors
        trace = grad.diagonal().sum()
        components = grad * contrast
        axis = components.sum()
        # With a=-log(T), Y=I+2r(Pa-Ps)/a+(2r)²I/a²+... .
        # Either inverse-log term dominates every positive power of T. For
        # zero pump the exact noise is (1-T)I.
        resolved = axis.abs() > 4 * torch.finfo(grad.dtype).eps * components.abs().sum()
        leading = torch.where(resolved, pump.sign() * axis, trace)
        vertical = torch.where(leading == 0, 0.0, torch.copysign(torch.full_like(leading, torch.inf), leading))
        return torch.where(pump == 0, -trace, vertical), None, None
