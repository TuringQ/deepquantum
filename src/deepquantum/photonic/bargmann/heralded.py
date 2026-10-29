"""Differentiable Gaussian Bargmann kernels and PNR conditional Fock coefficients.

Density kernels use all ket variables followed by all bra variables. Quadratures
use DeepQuantum's xxpp ordering and current hbar/kappa units. No full multimode
Fock state or environment purification is constructed.

See https://arxiv.org/pdf/2504.10455v3, "The stellar decomposition of Gaussian
quantum states", Sections 3.1, A.3.1 and A.4.1. Equation numbers refer to v3.
The polynomial specialization follows from Eq. (57); it does not construct
the stellar decomposition in Eqs. (21)-(27).
"""

import torch
from torch.nn import functional

from .moments import gaussian_to_bargmann
from .recurrence import _check_work, _device_layers, _fock_coefficients


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
