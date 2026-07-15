"""Functions for real- and complex-representation Kensingtonians."""

from collections import defaultdict
from functools import cache
from itertools import product
from math import comb

import torch


@cache
def _d_term_data(
    clicks: tuple[int, ...], num_detectors: int
) -> tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...], float], ...]:
    """Return matrix-independent data for all mixed-radix ``d`` terms."""
    size = 2 * len(clicks)
    ranges = [range(click + 1) for click in clicks]
    data = []
    for d_tuple in product(*ranges):
        sign = -1 if sum(click - d for click, d in zip(clicks, d_tuple, strict=True)) % 2 else 1
        coeff = 1
        for click, d in zip(clicks, d_tuple, strict=True):
            coeff *= comb(num_detectors, click) * comb(click, d)
        keep_modes = tuple(i for i, d in enumerate(d_tuple) if d > 0)
        prefactor = 1.0
        for i in keep_modes:
            prefactor *= num_detectors / d_tuple[i]
        idx_tuple = tuple(sorted(keep_modes + tuple(i + size // 2 for i in keep_modes)))
        dz_values = tuple((num_detectors - d_tuple[i]) / d_tuple[i] for i in keep_modes)
        data.append((d_tuple, idx_tuple, dz_values + dz_values, sign * coeff * prefactor))
    return tuple(data)


@cache
def _grouped_term_data(
    clicks: tuple[int, ...], num_detectors: int
) -> tuple[
    tuple[
        int,
        tuple[tuple[tuple[int, ...], tuple[float, ...], float], ...],
    ],
    ...,
]:
    """Group terms by submatrix size for batched linear algebra."""
    groups = defaultdict(list)
    for _, idx, dz_values, scalar in _d_term_data(clicks, num_detectors):
        groups[len(idx)].append((idx, dz_values, scalar))
    return tuple((size, tuple(groups[size])) for size in sorted(groups))


def _kensingtonian_same_clicks(
    matrix: torch.Tensor,
    clicks: tuple[int, ...],
    num_detectors: int,
    gamma: torch.Tensor | None,
    chunk_size: int | None,
) -> torch.Tensor:
    """Evaluate a matrix batch whose samples share one click pattern."""
    batch_size, size, _ = matrix.shape
    identity = torch.eye(size, dtype=matrix.dtype, device=matrix.device)
    sigma_inv = identity - matrix

    result = matrix.new_zeros(batch_size)
    for submatrix_size, terms in _grouped_term_data(clicks, num_detectors):
        if submatrix_size == 0:
            result = result + sum(term[2] for term in terms)
            continue

        group_chunk_size = len(terms) if chunk_size is None else chunk_size
        for start in range(0, len(terms), group_chunk_size):
            term_chunk = terms[start : start + group_chunk_size]
            idx = torch.tensor(
                [term[0] for term in term_chunk],
                dtype=torch.long,
                device=matrix.device,
            )
            dz_values = torch.tensor(
                [term[1] for term in term_chunk],
                dtype=matrix.dtype,
                device=matrix.device,
            )
            scalars = torch.tensor(
                [term[2] for term in term_chunk],
                dtype=matrix.dtype,
                device=matrix.device,
            )

            det_mat = sigma_inv[:, idx[:, :, None], idx[:, None, :]]
            det_mat.diagonal(dim1=-2, dim2=-1).add_(dz_values.unsqueeze(0))
            det_term = torch.linalg.det(det_mat)

            if gamma is None:
                numerator = scalars.unsqueeze(0)
            else:
                rhs = gamma[:, idx]
                solution = torch.linalg.solve(det_mat, rhs.conj().unsqueeze(-1)).squeeze(-1)
                quad = (rhs * solution).sum(dim=-1) / 2
                numerator = torch.exp(quad) * scalars.unsqueeze(0)

            result = result + (numerator / torch.sqrt(det_term)).sum(dim=-1)
    return result


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
        matrix: The input matrix :math:`O=I-\Sigma^{-1}` in either
            DeepQuantum's real ``xxpp`` ordering or complex ladder ordering.
        clicks: The click-counting pattern :math:`k`.
        num_detectors: The number of threshold detectors in each
            click-counting detector.
        gamma: The loop vector in the same convention as
            :func:`torontonian`, namely
            :math:`\gamma=\sqrt{2}\alpha^T\Sigma^{-1}`.
            Default: ``None``
        chunk_size: The maximum number of ``d`` terms evaluated together
            within one same-size group. Default: ``None``
    """
    assert matrix.dim() == 2, 'Input matrix should be 2D'
    assert matrix.shape[-2] == matrix.shape[-1], 'Input matrix should be square'
    assert matrix.shape[-1] % 2 == 0, 'Input matrix dimension should be even'
    assert num_detectors >= 1, 'num_detectors should be positive'
    assert chunk_size is None or chunk_size > 0, 'chunk_size should be positive'

    nmode = matrix.shape[-1] // 2
    clicks = torch.as_tensor(clicks, dtype=torch.long, device=matrix.device)
    clicks = clicks.reshape(-1)
    assert clicks.shape == (nmode,), f'Click pattern should have shape ({nmode},)'
    assert (clicks >= 0).all(), 'Click numbers should be non-negative'
    assert (clicks <= num_detectors).all(), 'Click numbers should not exceed num_detectors'

    if gamma is not None:
        gamma = torch.as_tensor(gamma, device=matrix.device)
        assert matrix.is_complex() or not gamma.is_complex(), 'Complex gamma requires a complex matrix'
        gamma = gamma.to(matrix.dtype).reshape(-1)
        assert gamma.shape == (2 * nmode,), f'gamma should have shape ({2 * nmode},)'

    clicks_tuple = tuple(clicks.tolist())
    return _kensingtonian_same_clicks(
        matrix.unsqueeze(0),
        clicks_tuple,
        num_detectors,
        None if gamma is None else gamma.unsqueeze(0),
        chunk_size,
    )[0]


def kensingtonian_batch(
    matrix: torch.Tensor,
    clicks: torch.Tensor | list[int],
    num_detectors: int,
    gamma: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Calculate batched Kensingtonians for one or more click patterns."""
    assert matrix.dim() == 3, 'Input tensor should be in batched size'
    assert matrix.shape[-2] == matrix.shape[-1], 'Input matrices should be square'
    assert matrix.shape[-1] % 2 == 0, 'Input matrix dimension should be even'
    assert num_detectors >= 1, 'num_detectors should be positive'
    assert chunk_size is None or chunk_size > 0, 'chunk_size should be positive'

    batch_size = matrix.shape[0]
    nmode = matrix.shape[-1] // 2
    clicks = torch.as_tensor(clicks, dtype=torch.long, device=matrix.device)
    if clicks.dim() == 1:
        clicks = clicks.expand(batch_size, -1)
    assert clicks.shape == (batch_size, nmode), f'Click patterns should have shape ({batch_size}, {nmode})'
    assert (clicks >= 0).all(), 'Click numbers should be non-negative'
    assert (clicks <= num_detectors).all(), 'Click numbers should not exceed num_detectors'

    if gamma is not None:
        gamma = torch.as_tensor(gamma, device=matrix.device)
        assert matrix.is_complex() or not gamma.is_complex(), 'Complex gamma requires complex matrices'
        gamma = gamma.to(matrix.dtype)
        if gamma.dim() == 1:
            gamma = gamma.expand(batch_size, -1)
        assert gamma.shape == (batch_size, 2 * nmode), f'gamma should have shape ({batch_size}, {2 * nmode})'

    pattern_groups = defaultdict(list)
    for sample, pattern in enumerate(clicks.tolist()):
        pattern_groups[tuple(pattern)].append(sample)

    indices = []
    values = []
    for pattern, sample_indices in pattern_groups.items():
        index = torch.tensor(sample_indices, dtype=torch.long, device=matrix.device)
        gamma_group = None if gamma is None else gamma.index_select(0, index)
        values.append(
            _kensingtonian_same_clicks(
                matrix.index_select(0, index),
                pattern,
                num_detectors,
                gamma_group,
                chunk_size,
            )
        )
        indices.append(index)

    order = torch.cat(indices).argsort()
    return torch.cat(values)[order]
