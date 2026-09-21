"""Differentiable Gaussian Bargmann kernels and PNR conditional Fock coefficients.

Density kernels use all ket variables followed by all bra variables. Quadratures
use DeepQuantum's xxpp ordering and current hbar/kappa units. No full multimode
Fock state or environment purification is constructed.
"""

import math
from functools import lru_cache

import torch
from torch.nn import functional

import deepquantum.photonic as dqp

from .qmath import quadrature_to_ladder

# Bounds include recurrence indices and intermediates, not only the output.
_MAX_WORK_BYTES = 256 * 1024**2


def gaussian_to_bargmann(cov: torch.Tensor, mean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert Gaussian moments to the density kernel ``c exp(z.T A z / 2 + b.T z)``.

    Args:
        cov: Real covariance tensor of shape ``(batch, 2M, 2M)`` in xxpp order.
        mean: Real mean tensor of shape ``(batch, 2M, 1)`` in the same units.

    Returns:
        Complex symmetric ``A`` and complex ``b`` with shapes ``(batch, 2M, 2M)``
        and ``(batch, 2M)``, and real vacuum probabilities ``c`` of shape ``(batch,)``.
        The first M variables label ket indices; the last M label bra indices.
    """
    if cov.ndim != 3 or cov.shape[-1] != cov.shape[-2] or cov.shape[-1] % 2 or cov.shape[-1] == 0:
        raise ValueError('cov must have shape (batch, 2M, 2M), with M > 0')
    if mean.ndim != 3 or mean.shape[-2:] != (cov.shape[-1], 1):
        raise ValueError('mean must have shape (batch, 2M, 1), matching cov')
    if cov.dtype not in (torch.float32, torch.float64) or mean.dtype != cov.dtype or mean.device != cov.device:
        raise ValueError('cov and mean must share a device and float32 or float64 dtype')
    if not bool(torch.isfinite(cov).all() & torch.isfinite(mean).all()):
        raise ValueError('Gaussian moments must be finite')
    try:
        batch = torch.broadcast_shapes(cov.shape[:1], mean.shape[:1])[0]
    except RuntimeError as exc:
        raise ValueError('cov and mean batch dimensions must be broadcastable') from exc
    cov = cov.expand(batch, -1, -1)
    mean = mean.expand(batch, -1, -1)
    identity = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
    # Q = W Gamma W^dagger with unitary W. Solve the real Gamma system:
    # this avoids a complex LU factorization (unsupported by PyTorch MPS).
    scale = 2 * dqp.kappa**2 / dqp.hbar
    gamma = scale * cov + identity / 2
    inverse = quadrature_to_ladder(torch.linalg.solve(gamma, identity.expand_as(gamma))) / scale
    beta = quadrature_to_ladder(mean)
    solved_mean = (inverse @ beta).squeeze(-1)
    nmode = cov.shape[-1] // 2
    exchange = identity.roll(nmode, dims=0).to(inverse.dtype)
    a = (identity - inverse) @ exchange
    a = (a + a.mT) / 2
    log_c = -(torch.linalg.slogdet(gamma)[1] + (beta.squeeze(-1).conj() * solved_mean).sum(-1).real) / 2
    return a, solved_mean, log_c.exp()


def _check_work(bounds: tuple[int, ...], batch: int, element_size: int, polynomial_size: int = 1) -> None:
    size = math.prod(bounds)
    # Conservative working-set estimate; autograd/framework overhead may add more.
    estimate = size * (80 * max(len(bounds), 1) + 4 * batch * element_size * polynomial_size)
    if estimate > _MAX_WORK_BYTES:
        raise MemoryError('Bargmann recurrence exceeds the working-set limit; reduce cutoff or herald photon numbers')


@lru_cache(maxsize=4)
def _recurrence_plan(bounds: tuple[int, ...]) -> tuple:
    """Cache only discrete CPU indices and constant weights, never an autograd graph."""
    ranges = [torch.arange(bound, device='cpu') for bound in bounds]
    states = torch.cartesian_prod(*ranges).reshape(-1, len(bounds))
    strides = torch.tensor([math.prod(bounds[i + 1 :]) for i in range(len(bounds))], device='cpu')
    totals = states.sum(-1)
    layers = []
    for degree in range(1, int(totals.max()) + 1):
        ids = torch.where(totals == degree)[0]
        selected = states[ids]
        mode = (selected > 0).long().argmax(-1)
        row = torch.arange(len(ids), device='cpu')
        previous = selected.clone()
        previous[row, mode] -= 1
        previous_ids = (previous * strides).sum(-1)
        valid = previous > 0
        lower = (previous_ids[:, None] - strides).clamp_min(0)
        number = selected[row, mode].double()
        weights = (previous / number[:, None]).sqrt() * valid
        layers.append((ids, mode, previous_ids, lower, weights, number.rsqrt()))
    return tuple(layers)


def _device_layers(bounds: tuple[int, ...], a: torch.Tensor):
    for layer in _recurrence_plan(bounds):
        yield tuple(t.to(device=a.device, dtype=a.real.dtype if t.is_floating_point() else t.dtype) for t in layer)


def _fock_coefficients(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, bounds: tuple[int, ...]) -> torch.Tensor:
    """Evaluate normalized Taylor coefficients on a rectangular index set."""
    batch = a.shape[0]
    if not bounds:
        return c.to(a.dtype)
    _check_work(bounds, batch, a.element_size())
    values = torch.cat((c.to(a.dtype)[:, None], a.new_zeros(batch, math.prod(bounds) - 1)), dim=-1)
    for ids, mode, previous, lower, weights, inverse_sqrt in _device_layers(bounds, a):
        terms = (a[:, mode, :] * weights * values[:, lower]).sum(-1)
        terms = terms + b[:, mode] * inverse_sqrt * values[:, previous]
        values = values.index_copy(1, ids, terms)
    return values.reshape(batch, *bounds)


def _single_mode_density(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, wires: tuple[int, ...], herald: tuple[int, ...], cutoff: int
) -> torch.Tensor:
    """Extract one remaining mode via a finite polynomial times a two-variable Gaussian."""
    nmode = a.shape[-1] // 2
    keep = next(i for i in range(nmode) if i not in wires)
    measured = list(wires) + [i + nmode for i in wires]
    output = [keep, keep + nmode]
    photons = herald * 2
    degree = sum(photons)
    bounds = tuple(h + 1 for h in photons)
    batch = a.shape[0]
    _check_work(bounds, batch, a.element_size(), (degree + 1) ** 2)
    _check_work((cutoff, cutoff), batch, a.element_size())
    km = a[:, measured][:, :, measured]
    cross = a[:, measured][:, :, output]
    bm = b[:, measured]
    poly = a.new_ones(batch, 1, 1, 1)
    poly = functional.pad(poly, (0, degree, 0, degree))
    prev_ids = torch.zeros(1, dtype=torch.long, device=a.device)
    old_ids, old_poly = prev_ids, poly
    layers = _device_layers(bounds, a) if bounds else ()
    for ids, mode, previous, lower, weights, inverse_sqrt in layers:
        before = poly[:, torch.searchsorted(prev_ids, previous)]
        linear_z = functional.pad(before[:, :, :-1, :], (0, 0, 1, 0))
        linear_w = functional.pad(before[:, :, :, :-1], (1, 0, 0, 0))
        current = (
            cross[:, mode, 0, None, None] * linear_z
            + cross[:, mode, 1, None, None] * linear_w
            + bm[:, mode, None, None] * before
        ) * inverse_sqrt[None, :, None, None]
        for j in range(len(photons)):
            positions = torch.searchsorted(old_ids, lower[:, j].contiguous()).clamp(max=len(old_ids) - 1)
            current = current + (km[:, mode, j] * weights[:, j])[..., None, None] * old_poly[:, positions]
        old_ids, old_poly, prev_ids, poly = prev_ids, poly, ids, current
    polynomial = poly[:, -1]
    base = _fock_coefficients(a[:, output][:, :, output], b[:, output], c, (cutoff, cutoff))
    factorial = torch.lgamma(torch.arange(cutoff, dtype=a.real.dtype, device=a.device) + 1)
    result = a.new_zeros(batch, cutoff, cutoff)
    for u in range(min(degree + 1, cutoff)):
        for v in range(min(degree - u + 1, cutoff)):
            # Do not skip terms based on a currently zero trainable displacement.
            factors = torch.exp(
                (factorial[u:] - factorial[: cutoff - u])[:, None] / 2
                + (factorial[v:] - factorial[: cutoff - v])[None, :] / 2
            )
            term = polynomial[:, u, v, None, None] * base[:, : cutoff - u, : cutoff - v] * factors
            result = result + functional.pad(term, (v, 0, u, 0))
    return result


def conditional_fock(
    cov: torch.Tensor,
    mean: torch.Tensor,
    wires: tuple[int, ...],
    herald: tuple[int, ...],
    cutoff: int,
    den_mat: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an unnormalized PNR conditional tensor and its marginal event probability.

    Args:
        cov: Batched xxpp covariances, shape ``(batch, 2M, 2M)``.
        mean: Batched xxpp means, shape ``(batch, 2M, 1)``.
        wires: Validated, unique measured mode indices, in herald order.
        herald: Nonnegative integer photon numbers, one for each measured mode.
        cutoff: Positive output dimension of each remaining mode.
        den_mat: Return a density matrix when True, otherwise a ket.
            A ket requires every Gaussian input in the batch to be pure.

    Returns:
        Conditional Fock tensor (batch first, ket axes before bra axes), event
        probabilities of shape ``(batch,)``.
        Probabilities are computed from measured marginal moments and do not
        depend on the output cutoff. No normalization is applied.
    """
    a, b, c = gaussian_to_bargmann(cov, mean)
    nmode = cov.shape[-1] // 2
    if not den_mat:
        tol = 100 * torch.finfo(cov.dtype).eps
        pure = bool((a[:, :nmode, nmode:].detach().abs() <= tol).all())
        if not pure:
            raise ValueError('A ket requires a pure Gaussian input; set den_mat=True for mixed inputs')
    selected = dict(zip(wires, herald, strict=True))
    bounds = tuple(selected[i] + 1 if i in selected else cutoff for i in range(nmode))
    indices = tuple(selected.get(i, slice(None)) for i in range(nmode))
    if not den_mat:
        values = _fock_coefficients(a[:, :nmode, :nmode], b[:, :nmode], c.sqrt(), bounds)
        state = values[(slice(None), *indices)]
    elif nmode - len(wires) == 1:
        state = _single_mode_density(a, b, c, wires, herald, cutoff)
    else:
        values = _fock_coefficients(a, b, c, bounds * 2)
        state = values[(slice(None), *indices, *indices)]
    if not wires:
        probability = torch.ones_like(c)
    else:
        marginal = list(wires) + [i + nmode for i in wires]
        ma, mb, mc = gaussian_to_bargmann(cov[:, marginal][:, :, marginal], mean[:, marginal])
        coefficients = _fock_coefficients(ma, mb, mc, tuple(h + 1 for h in herald) * 2)
        probability = coefficients[(slice(None), *herald, *herald)].real
    return state, probability
