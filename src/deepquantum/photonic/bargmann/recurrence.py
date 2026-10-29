"""Renormalized Hermite (Fock coefficient) recurrences for Gaussian Bargmann kernels.

See https://arxiv.org/pdf/2504.10455v3 Section A.3.1, Eqs. (54)-(57).
"""

import math
from functools import lru_cache

import torch

# Bounds include recurrence indices and intermediates, not only the output.
_MAX_WORK_BYTES = 256 * 1024**2


def _check_work(bounds: tuple[int, ...], batch: int, element_size: int, polynomial_size: int = 1) -> None:
    size = math.prod(bounds)
    # Conservative working-set estimate; autograd/framework overhead may add more.
    estimate = size * (80 * max(len(bounds), 1) + 4 * batch * element_size * polynomial_size)
    if estimate > _MAX_WORK_BYTES:
        raise MemoryError('Bargmann recurrence exceeds the working-set limit; reduce cutoff or herald photon numbers')


@lru_cache(maxsize=4)
def _recurrence_plan(bounds: tuple[int, ...]) -> tuple:
    """Cache only discrete CPU indices and constant weights, never an autograd graph.

    See https://arxiv.org/pdf/2504.10455v3 Section A.3.1, Eq. (57).
    """
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
        # Eq. (57), with one pivot i per target; Eq. (58) averages over all pivots.
        # Target t = selected, i = mode, k = t - e_i; dependencies have lower total degree.
        previous = selected.clone()
        previous[row, mode] -= 1
        previous_ids = (previous * strides).sum(-1)
        valid = previous > 0
        lower = (previous_ids[:, None] - strides).clamp_min(0)
        number = selected[row, mode].double()
        # weights_j = sqrt((t_j - delta_ij) / t_i); negative-index terms have zero weight.
        weights = (previous / number[:, None]).sqrt() * valid
        layers.append((ids, mode, previous_ids, lower, weights, number.rsqrt()))
    return tuple(layers)


def _device_layers(bounds: tuple[int, ...], a: torch.Tensor):
    for layer in _recurrence_plan(bounds):
        yield tuple(t.to(device=a.device, dtype=a.real.dtype if t.is_floating_point() else t.dtype) for t in layer)


def _fock_coefficients(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, bounds: tuple[int, ...]) -> torch.Tensor:
    r"""Evaluate normalized Taylor coefficients on a rectangular index set.

    Uses Eqs. (54)-(55) for coefficient normalization and Eq. (57) for recursion.

    For multi-indices :math:`k`, with :math:`k!=\prod_j k_j!`, use

    .. math::

        G(\xi)=c e^{\xi^T A\xi/2+b^T\xi}
        =\sum_k g_k\frac{\xi^k}{\sqrt{k!}},\qquad
        \partial_iG=(b_i+\sum_j A_{ij}\xi_j)G.

    Matching powers gives Eq. (57) of the paper:

    .. math::

        g_{k+e_i}=\frac{b_i g_k+\sum_j A_{ij}\sqrt{k_j}\,g_{k-e_j}}
        {\sqrt{k_i+1}},\qquad g_0=c.

    Negative-index coefficients are zero. In code, the target is :math:`t=k+e_i`:
    ``inverse_sqrt`` is :math:`1/\sqrt{t_i}` and ``weights`` contains
    :math:`\sqrt{(t_j-\delta_{ij})/t_i}`. Thus ``terms`` implements the two summands.
    For a pure-state kernel :math:`g_n=\psi_n`; for a density kernel :math:`g_{(n,m)}=\rho_{n,m}`.
    These are Fock coefficients, not ordinary Taylor coefficients :math:`g_k/\sqrt{k!}`;
    no further factorial division or state normalization is needed here.

    See https://arxiv.org/pdf/2504.10455v3 Section A.3.1, Eqs. (54)-(55) and (57).
    """
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
