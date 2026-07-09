"""Functions for Kensingtonian"""

from itertools import product

import torch


def _multinomial_click_coeff(num_detectors: int, click: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Return N! / ((N-k)! (k-d)! d!) for each mode."""
    n = torch.as_tensor(num_detectors, dtype=click.dtype, device=click.device)
    return torch.exp(
        torch.lgamma(n + 1) - torch.lgamma(n - click + 1) - torch.lgamma(click - d + 1) - torch.lgamma(d + 1)
    )


def _mode_submat(mat: torch.Tensor, keep_modes: torch.Tensor) -> torch.Tensor:
    """Keep both quadratures of each selected mode."""
    if keep_modes.numel() == 0:
        return mat.new_empty((0, 0))
    idx1 = keep_modes
    idx2 = idx1 + mat.shape[-1] // 2
    idx = torch.sort(torch.cat([idx1, idx2]))[0]
    return mat[idx][:, idx]


def _mode_subvec(vec: torch.Tensor, keep_modes: torch.Tensor) -> torch.Tensor:
    """Keep both quadratures of each selected mode from a vector."""
    if keep_modes.numel() == 0:
        return vec.new_empty((0,))
    idx1 = keep_modes
    idx2 = idx1 + vec.shape[-1] // 2
    idx = torch.sort(torch.cat([idx1, idx2]))[0]
    return vec[idx]


def _dz_diag(d: torch.Tensor, keep_modes: torch.Tensor, num_detectors: int) -> torch.Tensor:
    """Construct D_Z from Eq. (28) on the retained modes."""
    values = (num_detectors - d[keep_modes]) / d[keep_modes]
    return torch.cat([values, values]).diag()


def _kensingtonian_term(
    matrix: torch.Tensor,
    clicks: torch.Tensor,
    d: torch.Tensor,
    num_detectors: int,
    alpha: torch.Tensor | None,
) -> torch.Tensor:
    sign_power = torch.sum(clicks - d).to(torch.long)
    sign = -1 if int(sign_power.item()) % 2 else 1
    coeff = _multinomial_click_coeff(num_detectors, clicks, d).prod()
    keep_modes = torch.nonzero(d > 0, as_tuple=False).flatten()
    if keep_modes.numel() == 0:
        det_term = matrix.new_tensor(1)
        exp_term = matrix.new_tensor(1)
    else:
        size = matrix.shape[-1]
        identity = torch.eye(size, dtype=matrix.dtype, device=matrix.device)
        base = _mode_submat(identity - matrix, keep_modes)
        dz = _dz_diag(d, keep_modes, num_detectors).to(matrix.dtype)
        det_mat = base + dz
        det_term = torch.linalg.det(det_mat)
        if alpha is None:
            exp_term = matrix.new_tensor(1)
        else:
            rhs = _mode_subvec((identity - matrix) @ alpha, keep_modes)
            quad = rhs @ torch.linalg.solve(det_mat, rhs)
            exp_term = torch.exp(quad)
    prefactor = (num_detectors / d[keep_modes]).prod() if keep_modes.numel() > 0 else matrix.new_tensor(1)
    return sign * coeff.to(matrix.dtype) * prefactor.to(matrix.dtype) * exp_term / torch.sqrt(det_term)


def kensingtonian(
    matrix: torch.Tensor,
    clicks: torch.Tensor,
    num_detectors: int,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate the Kensingtonian or loop Kensingtonian"""
    assert matrix.dim() == 2
    assert matrix.shape[-2] == matrix.shape[-1]
    assert matrix.shape[-1] % 2 == 0
    assert num_detectors >= 1
    m = matrix.shape[-1] // 2
    clicks = torch.as_tensor(clicks, dtype=torch.long, device=matrix.device)
    assert clicks.shape == (m,)
    assert torch.all(clicks >= 0)
    assert torch.all(clicks <= num_detectors)
    clicks_float = clicks.to(matrix.real.dtype)
    if alpha is not None:
        assert alpha.shape == (2 * m,)
        alpha = alpha.to(dtype=matrix.dtype, device=matrix.device)
    ken = matrix.new_tensor(0)
    ranges = [range(int(click.item()) + 1) for click in clicks]
    for d_tuple in product(*ranges):
        d = torch.tensor(d_tuple, dtype=clicks_float.dtype, device=matrix.device)
        ken = ken + _kensingtonian_term(matrix, clicks_float, d, num_detectors, alpha)
    return ken


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
    clicks = torch.as_tensor(clicks, dtype=torch.long, device=matrix.device)
    if clicks.dim() == 1:
        clicks = clicks.expand(matrix.shape[0], -1)
    if alpha is not None:
        alpha = torch.as_tensor(alpha, dtype=matrix.dtype, device=matrix.device)
        if alpha.dim() == 1:
            alpha = alpha.expand(matrix.shape[0], -1)
    values = []
    for i in range(matrix.shape[0]):
        alpha_i = None if alpha is None else alpha[i]
        values.append(kensingtonian(matrix[i], clicks[i], num_detectors, alpha_i))
    return torch.stack(values)
