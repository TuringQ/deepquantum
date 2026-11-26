"""One-sided power derivatives at a zero transmitted or reflected field.

For p=s², a readout f(s) with f'(0)=0 has df/dp=f''(0)/2. The ordinary
reverse pass through sqrt only has f'(0). Each zero field therefore has a
second, zero-valued output whose cotangent carries the readout curvature.
The scope follows one circuit execution and its returned state.
"""

from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from typing import Any

import torch
from torch.autograd.function import FunctionCtx

_Fields = list[tuple[torch.Tensor, torch.Tensor]]
_recording: ContextVar[_Fields | None] = ContextVar('loss_fields', default=None)


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
        singular = torch.copysign(torch.full_like(first, torch.inf), first)
        # f(s) = f(0) + a*s + b*s² + ... . Only a zero linear term has
        # df/dp = b. Projecting a displaced Fock state can leave a genuine
        # linear term at roundoff. Its sign cannot be inferred from b.
        unresolved = (first != 0) & (first.abs() <= 16 * torch.finfo(field.dtype).eps * second.abs())
        limit = torch.where(first == 0, second / 2, torch.where(unresolved, torch.nan, singular))
        return torch.where(zero, limit, regular)


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
