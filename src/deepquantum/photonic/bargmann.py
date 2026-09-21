"""Differentiable Gaussian Bargmann kernels and PNR conditional Fock coefficients.

Density kernels use all ket variables followed by all bra variables. Quadratures
use DeepQuantum's xxpp ordering and current hbar/kappa units. No full multimode
Fock state or environment purification is constructed.

See https://arxiv.org/pdf/2504.10455v3, "The stellar decomposition of Gaussian
quantum states", Sections 3.1, A.3.1 and A.4.1. Equation numbers refer to v3.
The polynomial specialization follows from Eq. (57); it does not construct
the stellar decomposition in Eqs. (21)-(27).
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
    r"""Convert Gaussian moments to the density kernel ``c exp(z.T A z / 2 + b.T z)``.

    Implements the Gaussian moment conversion with the unit and variable conventions below.

    With :math:`s=2\kappa^2/\hbar`, :math:`V=\mathrm{cov}` and :math:`d=\mathrm{mean}`, define
    :math:`W=2^{-1/2}\left(\begin{smallmatrix}I&iI\\I&-iI\end{smallmatrix}\right)`
    and :math:`J=\left(\begin{smallmatrix}0&I\\I&0\end{smallmatrix}\right)`:

    .. math::

        \Gamma=sV+I/2,\qquad Q=W\Gamma W^\dagger,\qquad \beta=\sqrt{s}Wd,

    .. math::

        A=(I-Q^{-1})J,\qquad b=Q^{-1}\beta,\qquad
        c=\frac{\exp(-\beta^\dagger Q^{-1}\beta/2)}{\sqrt{\det\Gamma}}.

    Here :math:`Q` is the Husimi covariance matrix, not the scalar Husimi Q function.
    For formal variables :math:`\xi=(z,w)`, the kernel is

    .. math::

        K_\rho(z,w)=c\exp(\xi^T A\xi/2+b^T\xi)
        =\sum_{n,m}\rho_{n,m}\frac{z^n w^m}{\sqrt{n!m!}}.

    On the coherent-state diagonal, :math:`z=\alpha^*` and :math:`w=\alpha`.
    Eq. (45) of the paper orders variables as :math:`(w,z)`; here they are :math:`(z,w)`.
    Thus :math:`A=J A_{\mathrm{paper}}J` and :math:`b=J b_{\mathrm{paper}}`.
    Use :math:`\hbar_{\mathrm{paper}}=\hbar/(2\kappa^2)` with the same moments.
    This explains the right-hand :math:`J` in :math:`A` and its absence in :math:`b`.
    The kernel form is Eq. (51) of the paper.

    See https://arxiv.org/pdf/2504.10455v3 Sections A.1, A.2 and A.4.1,
    Eqs. (45), (51) and (68)-(71).

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
    # A is complex symmetric (transpose, not conjugate transpose).
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


def _single_mode_density(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, wires: tuple[int, ...], herald: tuple[int, ...], cutoff: int
) -> torch.Tensor:
    r"""Extract one remaining mode via a finite polynomial times a two-variable Gaussian.

    Let :math:`x=(z,w)` be the retained ket/bra variables and :math:`y` the measured ones.
    With :math:`K=A_{HH}`, :math:`L=A_{HR}` and :math:`G(x)=c e^{x^T A_{RR}x/2+b_R^Tx}`,

    .. math::

        K_\rho(x,y)=G(x)e^{y^TKy/2+(b_H+Lx)^Ty}
        =G(x)\sum_q P_q(x)\frac{y^q}{\sqrt{q!}}.

    The recurrence above, now with polynomial-valued linear terms, yields

    .. math::

        P_{q+e_i}(x)=\frac{[b_{H,i}+(Lx)_i]P_q(x)
        +\sum_j K_{ij}\sqrt{q_j}P_{q-e_j}(x)}{\sqrt{q_i+1}},\qquad P_0=1.

    PNR projection selects :math:`q=(h,h)`, so the conditional kernel is :math:`G(x)P_{(h,h)}(x)`.
    The polynomial has total degree at most :math:`D=2\sum_j h_j`.
    Derived by replacing the linear term in Eq. (57) of the paper, with a polynomial.
    Projection follows Eq. (25) on each density-kernel side. The polynomial
    recurrence and factorial convolution are local derivations, not numbered
    equations quoted from the paper.

    See https://arxiv.org/pdf/2504.10455v3 Sections 3.1 and A.3.1, Eqs. (25) and (57).
    """
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
    # km, cross, bm are K, L, b_H; photons concatenates (h, h), rather than doubling h.
    km = a[:, measured][:, :, measured]
    cross = a[:, measured][:, :, output]
    bm = b[:, measured]
    # poly[..., u, v] stores ordinary monomial coefficients of z^u w^v, with P_0 = 1.
    poly = a.new_ones(batch, 1, 1, 1)
    poly = functional.pad(poly, (0, degree, 0, degree))
    prev_ids = torch.zeros(1, dtype=torch.long, device=a.device)
    old_ids, old_poly = prev_ids, poly
    layers = _device_layers(bounds, a) if bounds else ()
    for ids, mode, previous, lower, weights, inverse_sqrt in layers:
        before = poly[:, torch.searchsorted(prev_ids, previous)]
        # Shifts multiply P by z or w in (b_H + L x) P.
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
    # If P = sum p_uv z^u w^v and base_nm are normalized Fock coefficients of G,
    # result_nm = sum_{u<=n,v<=m} p_uv base_{n-u,m-v} sqrt(n! m! / ((n-u)! (m-v)!)).
    # lgamma(n + 1) = log(n!) supplies this change-of-basis factor stably.
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
    r"""Return an unnormalized PNR conditional tensor and its marginal event probability.

    PNR projection follows Eqs. (22), (25) and (28); for density matrices
    apply it on both sides, as in Eq. (8).

    With measured modes :math:`H`, retained modes :math:`R` and photon pattern :math:`h`, projection gives

    .. math::

        \widetilde\psi_n=\psi_{(h,n)},\qquad
        \widetilde\rho_{n,m}=\rho_{(h,n),(h,m)},\qquad
        p_h=\langle h|\operatorname{Tr}_R(\rho)|h\rangle.

    The tuples above illustrate an H,R reordering; returned axes retain their
    original mode order. Gaussian marginal moments are obtained by selecting
    the measured xxpp rows/columns, then converting that marginal afresh.
    Selecting a subblock of the full Bargmann A would instead project the other
    modes onto vacuum, and would not compute a partial trace.
    
    See https://arxiv.org/pdf/2504.10455v3 Sections 2.1 and 3.1, Eqs. (8), (22), (25) and (28).
    See https://the-walrus.readthedocs.io/en/latest/gbs.html for Gaussian marginals.

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
        # For a physical pure Gaussian state A = A_psi direct-sum A_psi.conj(); A_ket,bra = 0.
        pure = bool((a[:, :nmode, nmode:].detach().abs() <= tol).all())
        if not pure:
            raise ValueError('A ket requires a pure Gaussian input; set den_mat=True for mixed inputs')
    selected = dict(zip(wires, herald, strict=True))
    bounds = tuple(selected[i] + 1 if i in selected else cutoff for i in range(nmode))
    indices = tuple(selected.get(i, slice(None)) for i in range(nmode))
    if not den_mat:
        # c_rho = |c_psi|^2; sqrt(c) fixes the unobservable global phase by c_psi > 0.
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
        # p_h = (rho_H)_{h,h} sums over all retained occupations, independent of output cutoff.
        marginal = list(wires) + [i + nmode for i in wires]
        ma, mb, mc = gaussian_to_bargmann(cov[:, marginal][:, :, marginal], mean[:, marginal])
        coefficients = _fock_coefficients(ma, mb, mc, tuple(h + 1 for h in herald) * 2)
        probability = coefficients[(slice(None), *herald, *herald)].real
    return state, probability
