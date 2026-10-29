r"""Bargmann kernels of DeepQuantum photonic gates and channels.

A ``k``-mode unitary kernel has variables ``(z_out_1..z_out_k, w_in_1..w_in_k)`` and satisfies
:math:`K(z,w)=\sum_{m,n}U_{mn}z^mw^n/\sqrt{m!n!}`, where :math:`U_{mn}` is exactly the Fock matrix
returned by the gate's ``get_matrix_state()`` (including the global phase). A ``k``-mode channel
kernel has variables ``(z_out, w_out, u_in, v_in)`` (ket output, bra output, ket input, bra input),
each of length ``k``.

The formulas were checked element by element against ``get_matrix_state()`` (max error 2e-15).
See ``Bargmann/claude_docs/02_bargmann_backend_implementation_plan.md`` Section 4.5.
"""

import torch

from ..channel import PhotonLoss
from ..gate import Barrier, BeamSplitter, Displacement, PhaseShift, Squeezing, Squeezing2, UAnyGate
from ..operation import Operation
from .ansatz import PolyExpAnsatz


def _complex(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_complex() else x + 0j


def _kernel(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> PolyExpAnsatz:
    nvar = a.shape[-1]
    dtype = torch.promote_types(torch.promote_types(a.dtype, b.dtype), c.dtype)
    return PolyExpAnsatz(
        a.to(dtype).reshape(1, 1, nvar, nvar), b.to(dtype).reshape(1, 1, nvar), c.to(dtype).reshape(1, 1), nvar
    )


def passive_kernel(u: torch.Tensor) -> PolyExpAnsatz:
    r"""Passive linear-optical gate with :math:`Ua^\dagger U^\dagger=u^Ta^\dagger`: :math:`K=\exp(z^Tu\,w)`."""
    u = _complex(u)
    k = u.shape[-1]
    zero = u.new_zeros(k, k)
    a = torch.cat([torch.cat([zero, u], -1), torch.cat([u.mT, zero], -1)], -2)
    return _kernel(a, u.new_zeros(2 * k), u.new_ones(()))


def displacement_kernel(r: torch.Tensor, theta: torch.Tensor) -> PolyExpAnsatz:
    r"""Displacement :math:`D(\alpha)` with :math:`\alpha=re^{i\theta}`.

    .. math::

        K=e^{-|\alpha|^2/2}\exp(\alpha z-\bar\alpha w+zw).
    """
    alpha = r * torch.exp(1j * theta)
    zero, one = torch.zeros_like(alpha), torch.ones_like(alpha)
    a = torch.stack([zero, one, one, zero]).reshape(2, 2)
    b = torch.stack([alpha, -alpha.conj()])
    # |alpha|^2 written smoothly so that the gradient at alpha = 0 is well defined.
    return _kernel(a, b, torch.exp(-(alpha.real**2 + alpha.imag**2) / 2))


def squeezing_kernel(r: torch.Tensor, theta: torch.Tensor) -> PolyExpAnsatz:
    r"""Single-mode squeezing in DeepQuantum's convention.

    .. math::

        K=\frac{1}{\sqrt{\cosh r}}\exp\left[-\frac{e^{i\theta}\tanh r}{2}z^2
        +\operatorname{sech}r\,zw+\frac{e^{-i\theta}\tanh r}{2}w^2\right].
    """
    phase = torch.exp(1j * theta)
    tanh = _complex(torch.tanh(r))
    sech = _complex(1 / torch.cosh(r))
    a = torch.stack([-phase * tanh, sech, sech, tanh / phase]).reshape(2, 2)
    return _kernel(a, a.new_zeros(2), torch.cosh(r) ** -0.5)


def two_mode_squeezing_kernel(r: torch.Tensor, theta: torch.Tensor) -> PolyExpAnsatz:
    r"""Two-mode squeezing in DeepQuantum's convention, variables ``(z1, z2, w1, w2)``.

    .. math::

        K=\operatorname{sech}r\exp\left[e^{i\theta}\tanh r\,z_1z_2
        +\operatorname{sech}r(z_1w_1+z_2w_2)-e^{-i\theta}\tanh r\,w_1w_2\right].
    """
    phase = torch.exp(1j * theta)
    tanh = _complex(torch.tanh(r))
    sech = _complex(1 / torch.cosh(r))
    zero = torch.zeros_like(sech)
    zz = phase * tanh
    ww = -tanh / phase
    a = torch.stack(
        [zero, zz, sech, zero, zz, zero, zero, sech, sech, zero, zero, ww, zero, sech, ww, zero]
    ).reshape(4, 4)
    return _kernel(a, a.new_zeros(4), sech)


def loss_kernel(transmittance: torch.Tensor) -> PolyExpAnsatz:
    r"""Pure-loss channel with transmittance :math:`T`, variables ``(z_out, w_out, u_in, v_in)``.

    .. math::

        K=\exp\left[\sqrt T\,zu+\sqrt T\,wv+(1-T)\,uv\right].
    """
    t = _complex(transmittance)
    sqrt_t = _complex(torch.sqrt(transmittance))
    zero = torch.zeros_like(t)
    a = torch.stack(
        [
            zero, zero, sqrt_t, zero,
            zero, zero, zero, sqrt_t,
            sqrt_t, zero, zero, 1 - t,
            zero, sqrt_t, 1 - t, zero,
        ]
    ).reshape(4, 4)  # fmt: skip
    return _kernel(a, a.new_zeros(4), a.new_ones(()))


def op_to_kernel(op: Operation) -> tuple[PolyExpAnsatz, str] | None:
    """Return the Bargmann kernel of a photonic operation and its kind (``'unitary'`` or ``'channel'``).

    Parameters are read from the live gate tensors (not the detached caches), once per call,
    so that gradients flow and parameter noise is sampled consistently. Returns ``None`` for
    operations without effect such as ``Barrier``.

    Raises:
        NotImplementedError: If the operation is not supported by the Bargmann backend.
    """
    if isinstance(op, Barrier):
        return None
    if isinstance(op, PhotonLoss):
        if len(op.wires) != 1:
            raise NotImplementedError('PhotonLoss acts on a single mode')
        return loss_kernel(op.t), 'channel'
    if isinstance(op, (PhaseShift, BeamSplitter, UAnyGate)):
        u = op.update_matrix()
        if u.shape[-2:] != (len(op.wires), len(op.wires)):
            raise NotImplementedError(f'{op.name} does not provide a mode unitary matrix')
        return passive_kernel(u), 'unitary'
    if isinstance(op, (Squeezing, Squeezing2, Displacement)):
        r, theta = op.inputs_to_tensor([op.r, op.theta])
        if isinstance(op, Squeezing):
            return squeezing_kernel(r, theta), 'unitary'
        if isinstance(op, Squeezing2):
            return two_mode_squeezing_kernel(r, theta), 'unitary'
        return displacement_kernel(r, theta), 'unitary'
    raise NotImplementedError(f'{op.name} is not supported by the Bargmann backend')
