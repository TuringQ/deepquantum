"""Functions for Kensingtonian.

The Kensingtonian appears in click-counting Gaussian boson sampling.
See Phys. Rev. A 109, 023708 (2024), Eq. (27) and Appendix C.
"""

from collections import defaultdict
from functools import cache
from itertools import product
from math import comb

import torch


@cache
def _d_term_data(
    clicks: tuple[int, ...], num_detectors: int
) -> tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...], float], ...]:
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
        dz_values_tuple = dz_values + dz_values
        data.append((d_tuple, idx_tuple, dz_values_tuple, sign * coeff * prefactor))
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


def _kensingtonian_term(
    matrix: torch.Tensor,
    base_full: torch.Tensor,
    idx: torch.Tensor,
    dz_values: torch.Tensor,
    scalar: float,
    beta: torch.Tensor | None,
) -> torch.Tensor:
    if idx.numel() == 0:
        det_term = matrix.new_tensor(1)
        exp_term = matrix.new_tensor(1)
    else:
        base = base_full[idx][:, idx]
        det_mat = base.clone()
        det_mat.diagonal().add_(dz_values)
        det_term = torch.linalg.det(det_mat)

        if beta is None:
            exp_term = matrix.new_tensor(1)
        else:
            rhs = beta[idx]
            quad = rhs @ torch.linalg.solve(det_mat, rhs)
            exp_term = torch.exp(quad)

    return matrix.new_tensor(scalar) * exp_term / torch.sqrt(det_term)


def _kensingtonian_same_clicks(
    matrix: torch.Tensor,
    clicks: tuple[int, ...],
    num_detectors: int,
    alpha: torch.Tensor | None,
) -> torch.Tensor:
    """Evaluate a matrix batch whose samples share one click pattern."""
    batch_size, size, _ = matrix.shape
    identity = torch.eye(size, dtype=matrix.dtype, device=matrix.device)
    base_full = identity - matrix
    beta = None
    if alpha is not None:
        beta = (base_full @ alpha.unsqueeze(-1)).squeeze(-1)

    result = matrix.new_zeros(batch_size)
    for submatrix_size, terms in _grouped_term_data(clicks, num_detectors):
        if submatrix_size == 0:
            result = result + sum(term[2] for term in terms)
            continue

        # Creating these tensors once per group replaces per-term tensor
        # construction and lets det/solve process all terms and samples at once.
        idx = torch.tensor([term[0] for term in terms], dtype=torch.long, device=matrix.device)
        dz_values = torch.tensor([term[1] for term in terms], dtype=matrix.dtype, device=matrix.device)
        scalars = torch.tensor([term[2] for term in terms], dtype=matrix.dtype, device=matrix.device)

        det_mat = base_full[:, idx[:, :, None], idx[:, None, :]]
        det_mat.diagonal(dim1=-2, dim2=-1).add_(dz_values.unsqueeze(0))
        det_term = torch.linalg.det(det_mat)

        if beta is None:
            numerator = scalars.unsqueeze(0)
        else:
            rhs = beta[:, idx]
            solution = torch.linalg.solve(det_mat, rhs.unsqueeze(-1)).squeeze(-1)
            quad = (rhs * solution).sum(dim=-1)
            numerator = torch.exp(quad) * scalars.unsqueeze(0)

        result = result + (numerator / torch.sqrt(det_term)).sum(dim=-1)
    return result


def kensingtonian(
    matrix: torch.Tensor,
    clicks: torch.Tensor,
    num_detectors: int,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate the Kensingtonian or loop Kensingtonian.

    Args:
        matrix: A 2M x 2M matrix A.
        clicks: Length-M click pattern k, with 0 <= k_i <= num_detectors.
        num_detectors: Number N of threshold detectors in each balanced click-counting detector.
        alpha: Optional length-2M displacement vector for the loop Kensingtonian.

    Returns:
        Ken[A] for ``alpha is None`` and lken[A, alpha] otherwise.
    """
    assert matrix.dim() == 2
    assert matrix.shape[-2] == matrix.shape[-1]
    assert matrix.shape[-1] % 2 == 0
    assert num_detectors >= 1

    m = matrix.shape[-1] // 2
    clicks = torch.as_tensor(clicks, dtype=torch.long, device=matrix.device)
    assert clicks.shape == (m,)
    assert torch.all(clicks >= 0)
    assert torch.all(clicks <= num_detectors)

    if alpha is not None:
        assert alpha.shape == (2 * m,)
        alpha = alpha.to(dtype=matrix.dtype, device=matrix.device)

    clicks_tuple = tuple(clicks.tolist())
    return _kensingtonian_same_clicks(
        matrix.unsqueeze(0),
        clicks_tuple,
        num_detectors,
        None if alpha is None else alpha.unsqueeze(0),
    )[0]


def kensingtonian_batch(
    matrix: torch.Tensor,
    clicks: torch.Tensor,
    num_detectors: int,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate batched Kensingtonians."""
    assert matrix.dim() == 3
    assert matrix.shape[-2] == matrix.shape[-1]
    assert matrix.shape[-1] % 2 == 0
    assert num_detectors >= 1

    batch_size = matrix.shape[0]
    m = matrix.shape[-1] // 2
    clicks = torch.as_tensor(clicks, dtype=torch.long, device=matrix.device)
    if clicks.dim() == 1:
        clicks = clicks.expand(batch_size, -1)
    assert clicks.shape == (batch_size, m)
    assert torch.all(clicks >= 0)
    assert torch.all(clicks <= num_detectors)

    if alpha is not None:
        alpha = torch.as_tensor(alpha, dtype=matrix.dtype, device=matrix.device)
        if alpha.dim() == 1:
            alpha = alpha.expand(batch_size, -1)
        assert alpha.shape == (batch_size, 2 * m)

    # Group samples by click pattern. This keeps support for heterogeneous
    # patterns while vectorizing the common homogeneous-batch case.
    pattern_groups = defaultdict(list)
    for sample, pattern in enumerate(clicks.tolist()):
        pattern_groups[tuple(pattern)].append(sample)

    indices = []
    values = []
    for pattern, sample_indices in pattern_groups.items():
        index = torch.tensor(sample_indices, dtype=torch.long, device=matrix.device)
        alpha_group = None if alpha is None else alpha.index_select(0, index)
        values.append(_kensingtonian_same_clicks(matrix.index_select(0, index), pattern, num_detectors, alpha_group))
        indices.append(index)

    order = torch.cat(indices).argsort()
    return torch.cat(values)[order]
