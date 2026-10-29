"""Conversion from Gaussian moments to Bargmann density kernels.

Quadratures use DeepQuantum's xxpp ordering and current hbar/kappa units.
"""

import torch

import deepquantum.photonic as dqp

from ..qmath import quadrature_to_ladder


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
