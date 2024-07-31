"""
functions for torontonian

"""
import itertools
import torch

from .qmath import get_powersets

def get_submat_tor(a, z):
    """Get submat for torontonian calculation"""
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z)
    len_ = a.size()[-1]
    idx1 = z
    idx2 = idx1 + int((len_)/2)
    idx = torch.cat([idx1, idx2])
    idx = torch.sort(idx)[0]
    if a.dim() == 1:
        return a[idx]
    if a.dim() == 2:
        return a[idx][:, idx]


def _tor_helper(submat, sub_gamma):
    size = submat.size()[-1]
    temp = torch.eye(size, device = submat.device)-submat
    # inv_temp = torch.linalg.inv(temp)
    sub_gamma = sub_gamma.to(temp.device, temp.dtype)
    exp_term  = sub_gamma @ torch.linalg.solve(temp, sub_gamma.conj())/2
    return torch.exp(exp_term)/torch.sqrt(torch.linalg.det(temp + 0j))

def torontonian(o_mat: torch.Tensor, gamma=None) -> torch.Tensor:
    """
    Calculate the torontonian function for given matrix.

    See https://research-information.bris.ac.uk/ws/portalfiles/portal/329011096/thesis.pdf Eq.(3.54)
    """
    if gamma is None:
        gamma = torch.zeros(len(o_mat))
    assert len(o_mat) % 2 == 0, 'input matrix dimension should be even '
    m = len(o_mat) // 2
    z_sets = get_powersets(m)
    tor = (-1) ** m
    for i in range(1, len(z_sets)):
        subset = z_sets[i]
        num_ = len(subset[0])
        sub_mats = torch.vmap(get_submat_tor, in_dims=(None, 0))(o_mat, torch.tensor(subset))
        sub_gammas = torch.vmap(get_submat_tor, in_dims=(None, 0))(gamma, torch.tensor(subset))
        coeff = torch.vmap(_tor_helper)(sub_mats, sub_gammas)
        coeff_sum = (-1) ** (m - num_) * coeff.sum()
        tor = tor + coeff_sum
    return tor

def torontonian_batch(A: torch.Tensor, gamma=None) -> torch.Tensor:
    """
    Calculate the batch torontonian
    """
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A)
    assert A.dim()==3, 'Input tensor should be in batched size'
    if gamma is None: # torontonian case
        tors = torch.vmap(torontonian, in_dims=(0, None))(A, gamma)
    else: # loop torontonian case
        if not isinstance(gamma, torch.Tensor):
            gamma = torch.tensor(gamma)
        tors = torch.vmap(torontonian, in_dims=(0, 0))(A, gamma)
    return tors
