"""
functions for torontonian
"""

from typing import Optional

import torch

from .qmath import get_powerset


def get_submat_tor(a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Get the sub-matrix for torontonian calculation."""
    idx1 = z
    idx2 = idx1 + a.shape[-1] // 2
    idx = torch.cat([idx1, idx2])
    idx = torch.sort(idx)[0]
    if a.dim() == 1:
        return a[idx]
    if a.dim() == 2:
        return a[idx][:, idx]


def _tor_helper(submat: torch.Tensor, sub_gamma: torch.Tensor) -> torch.Tensor:
    size = submat.shape[-1]
    cov_q_inv = torch.eye(size, dtype=submat.dtype, device=submat.device) - submat
    exp_term = sub_gamma @ torch.linalg.solve(cov_q_inv, sub_gamma.conj()) / 2
    return torch.exp(exp_term) / torch.sqrt(torch.linalg.det(cov_q_inv))


def torontonian(o_mat: torch.Tensor, gamma: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Calculate the torontonian function for the given matrix.

    See https://research-information.bris.ac.uk/ws/portalfiles/portal/329011096/thesis.pdf Eq.(3.54)
    """
    size = o_mat.shape[-1]
    if gamma is None:
        gamma = torch.zeros(size, dtype=o_mat.dtype, device=o_mat.device)
    m = size // 2
    powerset = get_powerset(m)
    tor = (-1) ** m
    for i in range(1, len(powerset)):
        y_sets = torch.tensor(powerset[i], device=o_mat.device)
        num_y = len(y_sets[0])
        sub_mats = torch.vmap(get_submat_tor, in_dims=(None, 0))(o_mat, y_sets)
        sub_gammas = torch.vmap(get_submat_tor, in_dims=(None, 0))(gamma, y_sets)
        coeff = torch.vmap(_tor_helper)(sub_mats, sub_gammas)
        coeff_sum = (-1) ** (m - num_y) * coeff.sum()
        tor += coeff_sum
    return tor


def torontonian_batch(o_mat: torch.Tensor, gamma: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Calculate the batch torontonian."""
    assert o_mat.dim() == 3, 'Input tensor should be in batched size'
    assert o_mat.shape[-2] == o_mat.shape[-1]
    assert o_mat.shape[-1] % 2 == 0, 'Input matrix dimension should be even'
    if gamma is None: # torontonian case
        tors = torch.vmap(torontonian, in_dims=(0, None))(o_mat, gamma)
    else: # loop torontonian case
        tors = torch.vmap(torontonian, in_dims=(0, 0))(o_mat, gamma)
    return tors
