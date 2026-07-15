"""Functions for Kensingtonian."""

from functools import lru_cache
from math import prod
from typing import NamedTuple

import torch

_MAX_CACHED_TERM_DATA_ELEMENTS = 2**17


class _ModeData(NamedTuple):
    """Click-dependent mode indices used to reduce the input matrix."""

    clicked_modes: torch.Tensor
    quadrature_indices: torch.Tensor
    reduced_clicks: tuple[int, ...]

    def to(self, device: torch.device) -> '_ModeData':
        """Move tensor fields to a device."""
        if self.clicked_modes.device == device:
            return self
        return _ModeData(
            clicked_modes=self.clicked_modes.to(device),
            quadrature_indices=self.quadrature_indices.to(device),
            reduced_clicks=self.reduced_clicks,
        )


class _TermData(NamedTuple):
    """Matrix-independent data for a batch of Kensingtonian terms."""

    coefficients: torch.Tensor
    mask: torch.Tensor
    outer_mask: torch.Tensor
    diagonal: torch.Tensor

    def to(self, device: torch.device) -> '_TermData':
        """Move tensor fields to a device."""
        if self.coefficients.device == device:
            return self
        return _TermData(*(value.to(device) for value in self))


def _check_matrix(matrix: torch.Tensor) -> int:
    """Check the matrix shape and return the number of modes."""
    assert matrix.ndim in (2, 3), 'Input matrix should be 2D or 3D'
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


def _build_term_data(
    clicks: torch.Tensor,
    d_vectors: torch.Tensor,
    num_detectors: int,
    coefficient_table: torch.Tensor,
) -> _TermData:
    """Build the matrix-independent data for a chunk of Kensingtonian terms."""
    active = d_vectors > 0
    parity = (clicks.sum() - d_vectors.sum(dim=-1)) % 2
    abs_coeffs = coefficient_table.gather(1, d_vectors.mT).mT.contiguous().prod(dim=-1)
    dtype = coefficient_table.dtype
    d_vectors = d_vectors.to(dtype)
    d_safe = d_vectors.clamp_min(1)
    # Eq. (27): multinomial, product N / d_j, and alternating sign.
    signs = torch.where(parity == 0, 1, -1)
    coefficients = signs.to(dtype) * abs_coeffs
    # The active mask is the complement of Z; identity blocks replace deleted modes.
    active_float = active.to(dtype)
    mask = torch.cat([active_float, active_float], dim=-1)
    outer_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    # Eq. (28): D_Z has (N - d_i) / d_i on both coordinates of each active mode.
    diagonal_values = active_float * (num_detectors - d_vectors) / d_safe
    diagonal = torch.cat([diagonal_values, diagonal_values], dim=-1) + (1 - mask)
    return _TermData(coefficients=coefficients, mask=mask, outer_mask=outer_mask, diagonal=diagonal)


@lru_cache(maxsize=64)
def _cached_term_data(
    clicks: tuple[int, ...],
    num_detectors: int,
    dtype: torch.dtype,
) -> _TermData | None:
    """Build and cache bounded CPU term data for one reduced click pattern."""
    total_terms = prod(click + 1 for click in clicks)
    nquadrature = 2 * len(clicks)
    num_elements = total_terms * (nquadrature**2 + 2 * nquadrature + 1)
    if num_elements > _MAX_CACHED_TERM_DATA_ELEMENTS:
        return None
    with torch.inference_mode(False), torch.no_grad():
        clicks_tensor = torch.tensor(clicks, dtype=torch.long)
        d_vectors = _enumerate_d(clicks_tensor, 0, total_terms)
        coefficient_table = _coefficient_table(clicks_tensor, num_detectors, dtype)
        return _build_term_data(clicks_tensor, d_vectors, num_detectors, coefficient_table)


@lru_cache(maxsize=64)
def _cached_mode_data(clicks: tuple[int, ...]) -> _ModeData:
    """Cache clicked modes, quadrature indices and reduced clicks on CPU."""
    with torch.inference_mode(False), torch.no_grad():
        clicks_tensor = torch.tensor(clicks, dtype=torch.long)
        clicked_modes = (clicks_tensor > 0).nonzero(as_tuple=False).reshape(-1)
        quadrature_indices = torch.cat([clicked_modes, clicked_modes + len(clicks)])
        reduced_clicks = tuple(click for click in clicks if click > 0)
        return _ModeData(
            clicked_modes=clicked_modes,
            quadrature_indices=quadrature_indices,
            reduced_clicks=reduced_clicks,
        )


def _kensingtonian_terms(
    sigma_inv: torch.Tensor,
    gamma: torch.Tensor | None,
    term_data: _TermData,
) -> torch.Tensor:
    """Calculate the sum of a chunk of Kensingtonian terms for a matrix batch."""
    b_mats = sigma_inv.unsqueeze(1) * term_data.outer_mask.unsqueeze(0)
    b_mats.diagonal(dim1=-2, dim2=-1).add_(term_data.diagonal)
    chol = torch.linalg.cholesky(b_mats)
    det_factor = chol.diagonal(dim1=-2, dim2=-1).prod(dim=-1)
    if gamma is None:
        loop_factor = 1
    else:
        gamma_masked = gamma.unsqueeze(1) * term_data.mask.unsqueeze(0)
        transformed = torch.linalg.solve_triangular(chol, gamma_masked.unsqueeze(-1), upper=False).squeeze(-1)
        quad = transformed.square().sum(dim=-1)
        loop_factor = torch.exp(quad)
    terms = term_data.coefficients * loop_factor / det_factor
    return terms.sum(dim=-1)


def _calculate_kensingtonian(
    matrix: torch.Tensor,
    clicks: torch.Tensor,
    num_detectors: int,
    gamma: torch.Tensor | None,
    chunk_size: int | None,
) -> torch.Tensor:
    """Calculate a matrix batch that shares one click pattern."""
    mode_data = _cached_mode_data(tuple(clicks.tolist())).to(clicks.device)
    if mode_data.clicked_modes.numel() == 0:
        return matrix.new_ones(matrix.shape[0])
    identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    sigma_inv = identity - matrix
    quadrature_indices = mode_data.quadrature_indices
    sigma_inv = sigma_inv[:, quadrature_indices[:, None], quadrature_indices]
    gamma = gamma[:, quadrature_indices] if gamma is not None else None
    clicks = clicks[mode_data.clicked_modes]
    reduced_clicks = mode_data.reduced_clicks

    term_data = _cached_term_data(reduced_clicks, num_detectors, matrix.dtype) if chunk_size is None else None
    if term_data is not None:
        return _kensingtonian_terms(sigma_inv, gamma, term_data.to(matrix.device))

    total_terms = prod(click + 1 for click in reduced_clicks)
    coefficient_table = _coefficient_table(clicks, num_detectors, matrix.dtype)
    if chunk_size is None:
        chunk_size = total_terms
    ken = matrix.new_zeros(matrix.shape[0])
    for start in range(0, total_terms, chunk_size):
        end = min(start + chunk_size, total_terms)
        d_vectors = _enumerate_d(clicks, start, end)
        term_data = _build_term_data(clicks, d_vectors, num_detectors, coefficient_table)
        ken = ken + _kensingtonian_terms(sigma_inv, gamma, term_data)
    return ken


def kensingtonian(
    matrix: torch.Tensor,
    clicks: torch.Tensor | list[int],
    num_detectors: int,
    gamma: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    r"""Calculate a single or batch Kensingtonian or loop Kensingtonian.

    See https://arxiv.org/abs/2305.00853 Eq. (27)-(29) and Appendix C.

    Args:
        matrix: A real matrix or matrix batch :math:`O=I-\Sigma^{-1}` in ``xxpp`` ordering.
        clicks: The click-counting pattern :math:`k`.
        num_detectors: The number of threshold detectors in each click-counting detector.
        gamma: The precontracted loop vector :math:`\gamma=\alpha^T\Sigma^{-1}`, where :math:`\alpha` is
            the real displacement vector in the dimensionless quadrature representation. Default: ``None``
        chunk_size: The number of ``d`` vectors evaluated per chunk. Default: ``None``

    Note:
        Use ``float64`` inputs for high click counts or other cancellation-sensitive cases.
    """
    nmode = _check_matrix(matrix)
    is_single = matrix.ndim == 2
    if is_single:
        matrix = matrix.unsqueeze(0)
    batch_size = matrix.shape[0]
    assert num_detectors >= 1, 'num_detectors should be positive'
    assert chunk_size is None or chunk_size > 0, 'chunk_size should be positive'
    clicks = torch.as_tensor(clicks, dtype=torch.long, device=matrix.device)
    clicks = _check_clicks(clicks, nmode, num_detectors)
    if gamma is not None:
        gamma = torch.as_tensor(gamma, device=matrix.device)
        assert not gamma.is_complex(), 'Input gamma should be real'
        expected_shape = (batch_size, 2 * nmode)
        if gamma.ndim == 1:
            gamma = gamma.unsqueeze(0)
        assert tuple(gamma.shape) in ((1, 2 * nmode), expected_shape), (
            f'gamma should have shape ({2 * nmode},), (1, {2 * nmode}), or {expected_shape}'
        )
        gamma = gamma.to(matrix.dtype).expand(expected_shape)
    result = _calculate_kensingtonian(matrix, clicks, num_detectors, gamma, chunk_size)
    return result[0] if is_single else result
