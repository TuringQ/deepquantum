import itertools
import torch

def get_subsets(n):
    """Get powerset of [0, 1, ... , n-1]"""
    subsets = [ ]
    for k in range(n+1):
        subset = [ ]
        for i in itertools.combinations(range(n), k):
            subset.append(list(i))
        subsets.append(subset)
    return subsets

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

def torontonian(o_mat, gamma=None):
    """
    Calculate the torontonian function for given matrix.

    See https://research-information.bris.ac.uk/ws/portalfiles/portal/329011096/thesis.pdf Eq.(3.54)
    """
    if not isinstance(o_mat, torch.Tensor):
        o_mat = torch.tensor(o_mat)
    if gamma is None:
        gamma = torch.zeros(len(o_mat))
    if not isinstance(gamma, torch.Tensor):
        gamma = torch.tensor(gamma)
    assert len(o_mat) % 2 == 0, 'input matrix dimension should be even '
    m = len(o_mat) // 2
    z_sets = get_subsets(m)
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
