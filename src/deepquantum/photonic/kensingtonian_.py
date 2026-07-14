"""Functions for Kensingtonian."""

import torch
from torch import vmap


def _check_matrix(matrix: torch.Tensor) -> int:
    """Check the matrix shape and return the number of modes."""
    assert matrix.ndim == 2, 'Input matrix should be 2D'
    assert matrix.shape[-2] == matrix.shape[-1], 'Input matrix should be square'
    assert matrix.shape[-1] % 2 == 0, 'Input matrix dimension should be even'
    assert not matrix.is_complex(), 'Input matrix should be real'
    return matrix.shape[-1] // 2


def _check_clicks(clicks: torch.Tensor, nmode: int, num_detectors: int) -> torch.Tensor:
    """Check the click-counting pattern."""
    clicks = clicks.reshape(-1).long()
    assert clicks.shape == (nmode,), f'Click pattern should have shape ({nmode},)'
    assert (clicks >= 0).all(), 'Click numbers should be non-negative'
    assert (clicks <= num_detectors).all(), 'Click numbers should not exceed num_detectors'
    return clicks


def _enumerate_d(clicks: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Enumerate the mixed-radix vectors d with 0 <= d_i <= k_i."""
    bases = clicks + 1
    strides = torch.cat([torch.ones(1, dtype=torch.long, device=clicks.device), bases[:-1]]).cumprod(dim=0)
    idx = torch.arange(start, end, dtype=torch.long, device=clicks.device).unsqueeze(1)
    return (idx // strides) % bases


def _multinomial_log_terms(clicks: torch.Tensor, d_vectors: torch.Tensor, num_detectors: int) -> torch.Tensor:
    """Return the per-mode log multinomial coefficients in Eq. (27)."""
    device = d_vectors.device
    dtype = d_vectors.dtype
    clicks = clicks.cpu().double()
    d_vectors = d_vectors.cpu().double()
    n = torch.tensor(float(num_detectors), dtype=torch.double)
    return (
        torch.lgamma(n + 1)
        - torch.lgamma(n - clicks + 1)
        - torch.lgamma(clicks - d_vectors + 1)
        - torch.lgamma(d_vectors + 1)
    ).to(device=device, dtype=dtype)


def _coefficient_table(clicks: torch.Tensor, num_detectors: int, dtype: torch.dtype) -> torch.Tensor:
    """Precompute the per-mode coefficient factors for all possible d_i."""
    device = clicks.device
    d_values = torch.arange(int(clicks.max().item()) + 1, dtype=dtype, device=device)
    log_mult = _multinomial_log_terms(clicks.unsqueeze(-1), d_values.unsqueeze(0), num_detectors)
    d_safe = d_values.clamp_min(1)
    n_over_d = torch.where(d_values > 0, num_detectors / d_safe, 1)
    return torch.exp(log_mult) * n_over_d


def _kensingtonian_terms(
    sigma_inv: torch.Tensor,
    gamma: torch.Tensor | None,
    clicks: torch.Tensor,
    d_vectors: torch.Tensor,
    num_detectors: int,
    coefficient_table: torch.Tensor,
) -> torch.Tensor:
    """Calculate the sum of Kensingtonian terms for a chunk of d vectors."""
    active = d_vectors > 0
    parity = (clicks.sum() - d_vectors.sum(dim=-1)) % 2
    abs_coeffs = coefficient_table.gather(1, d_vectors.mT).mT.contiguous().prod(dim=-1)
    d_vectors = d_vectors.to(sigma_inv.dtype)
    d_safe = d_vectors.clamp_min(1)
    # Eq. (27): multinomial, product N / d_j, and alternating sign.
    signs = torch.where(parity == 0, 1, -1)
    coeffs = signs.to(sigma_inv.dtype) * abs_coeffs

    # active is the complement of Z. Identity blocks replace deleted Z modes.
    active_float = active.to(sigma_inv.dtype)
    mask2 = torch.cat([active_float, active_float], dim=-1)
    outer_mask = mask2.unsqueeze(-1) * mask2.unsqueeze(-2)
    # Eq. (28): D_Z has (N - d_i) / d_i on both coordinates of each active mode.
    t = active_float * (num_detectors - d_vectors) / d_safe
    diag = torch.cat([t, t], dim=-1) + (1 - mask2)
    b_mats = sigma_inv.unsqueeze(0) * outer_mask
    b_mats.diagonal(dim1=-2, dim2=-1).add_(diag)
    chol = torch.linalg.cholesky(b_mats)
    det_factor = chol.diagonal(dim1=-2, dim2=-1).prod(dim=-1)
    if gamma is None:
        loop_factor = 1
    else:
        gamma_masked = gamma.unsqueeze(0) * mask2
        transformed = torch.linalg.solve_triangular(chol, gamma_masked.unsqueeze(-1), upper=False).squeeze(-1)
        quad = transformed.square().sum(dim=-1) / 2
        loop_factor = torch.exp(quad)
    return (coeffs * loop_factor / det_factor).sum()


def kensingtonian(
    matrix: torch.Tensor,
    clicks: torch.Tensor | list[int],
    num_detectors: int,
    gamma: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    r"""Calculate the Kensingtonian or loop Kensingtonian.

    See https://arxiv.org/abs/2305.00853 Eq. (27)-(29) and Appendix C.

    Args:
        matrix: The real input matrix :math:`O=I-\Sigma^{-1}` in DeepQuantum's ``xxpp`` ordering.
        clicks: The click-counting pattern :math:`k`.
        num_detectors: The number of threshold detectors in each click-counting detector.
        gamma: The loop vector in the same convention as :func:`torontonian`, namely
            :math:`\gamma=\sqrt{2}\alpha^T\Sigma^{-1}`, where :math:`\alpha` is the real displacement vector
            in the dimensionless quadrature representation.
            Default: ``None``
        chunk_size: The number of ``d`` vectors evaluated per chunk. Default: ``None``
    """
    nmode = _check_matrix(matrix)
    assert num_detectors >= 1, 'num_detectors should be positive'
    assert chunk_size is None or chunk_size > 0, 'chunk_size should be positive'
    clicks = torch.as_tensor(clicks, dtype=torch.long, device=matrix.device)
    clicks = _check_clicks(clicks, nmode, num_detectors)
    identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    sigma_inv = identity - matrix
    if gamma is not None:
        gamma = torch.as_tensor(gamma, device=matrix.device)
        assert not gamma.is_complex(), 'Input gamma should be real'
        gamma = gamma.to(matrix.dtype).reshape(-1)
        assert gamma.shape == (2 * nmode,), f'gamma should have shape ({2 * nmode},)'
    clicked_modes = (clicks > 0).nonzero(as_tuple=False).reshape(-1)
    if clicked_modes.numel() == 0:
        return matrix.new_ones(())
    idx = torch.cat([clicked_modes, clicked_modes + nmode])
    sigma_inv = sigma_inv[idx[:, None], idx]
    gamma = gamma[idx] if gamma is not None else None
    clicks = clicks[clicked_modes]
    coefficient_table = _coefficient_table(clicks, num_detectors, matrix.dtype)
    total_terms = int((clicks + 1).prod().item())
    if chunk_size is None:
        chunk_size = total_terms
    ken = matrix.new_zeros(())
    for start in range(0, total_terms, chunk_size):
        end = min(start + chunk_size, total_terms)
        d_vectors = _enumerate_d(clicks, start, end)
        ken = ken + _kensingtonian_terms(sigma_inv, gamma, clicks, d_vectors, num_detectors, coefficient_table)
    return ken


def kensingtonian_batch(
    matrix: torch.Tensor,
    clicks: torch.Tensor | list[int],
    num_detectors: int,
    gamma: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Calculate the batch Kensingtonian for one click pattern."""
    assert matrix.dim() == 3, 'Input tensor should be in batched size'
    assert matrix.shape[-2] == matrix.shape[-1]
    assert matrix.shape[-1] % 2 == 0, 'Input matrix dimension should be even'
    if gamma is None:
        return vmap(kensingtonian, in_dims=(0, None, None, None, None))(
            matrix, clicks, num_detectors, gamma, chunk_size
        )
    return vmap(kensingtonian, in_dims=(0, None, None, 0, None))(matrix, clicks, num_detectors, gamma, chunk_size)
