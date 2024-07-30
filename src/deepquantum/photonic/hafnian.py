"""
functions for hafnian
"""
import numpy as np
import torch
from scipy import special
from collections import Counter

from .qmath import get_subsets

def integer_partition(m, n):
    """Partition the integer m into lists of integers <= n"""
    results = []
    def back_track(m, n, result=[]):
        if m == 0:
            results.append(result)
            return
        if m < 0 or n == 0:
            return
        back_track(m, n - 1, result)
        back_track(m - n, n, result + [n])
    back_track(m, n)
    return results

def count_unique_permutations(nums):
    """" Count the number of unique permutations of a list of numbers."""
    def backtrack(counter):
        if sum(counter.values()) == 0:
            return 1
        total_count = 0
        for key in counter:
            if counter[key] > 0:
                counter[key] -= 1
                total_count += backtrack(counter)
                counter[key] += 1
        return total_count
    return backtrack(Counter(nums))

def get_submat_haf(a, z):
    """
    Get submat for hafnian calculation

    See https://arxiv.org/abs/1805.12498  paragraph after Eq.(3.20)
    """

    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z)
    idx1 = 2 * z
    idx2 = idx1+1
    idx = torch.cat([idx1, idx2])
    idx = torch.sort(idx)[0]
    submat = a[idx][:, idx]
    return submat

def p_labda(submat, power, loop=False):
    """Return the coefficient of polynomial."""
    sigma_x = torch.tensor([[0, 1], [1, 0]], device = submat.device)
    len_ = int(submat.size()[-1] / 2)
    x_mat = torch.block_diag(*[sigma_x]*len_)
    x_mat = x_mat.to(submat.dtype)
    xaz = x_mat @ submat
    eigen = torch.linalg.eig(xaz)[0] #eigen decomposition
    trace_list = torch.stack([(eigen ** i).sum() for i in range(0, power+1)])
    int_partition = integer_partition(power, power)
    if loop: #loop hafnian case
        v = torch.diag(submat)
        diag_contribution = torch.stack([v @ torch.linalg.matrix_power(xaz, i-1) @ x_mat @ v.reshape(-1,1)/2 for i in range(1, power+1)])
        diag_contribution = diag_contribution.squeeze()
        coeff =  0
        for k in range(len(int_partition)):
            temp = int_partition[k]
            prefactor = count_unique_permutations(temp)
            poly_list = trace_list[temp]/(2 * torch.tensor(temp)) + diag_contribution[np.array(temp)-1]
            poly_prod = poly_list.prod()
            coeff = coeff + prefactor/special.factorial(len(temp)) * poly_prod
            # coeff = coeff + prefactor/torch.exp(torch.lgamma(torch.tensor(len(temp)+1))) * poly_prod
    else:
        coeff =  0
        for k in range(len(int_partition)):
            temp = int_partition[k]
            prefactor = count_unique_permutations(temp)
            trace_prod = trace_list[temp].prod()
            coeff = coeff + prefactor/special.factorial(len(temp)) * trace_prod / (2**len(temp)*torch.tensor(temp).prod())
            # coeff = coeff + prefactor/torch.exp(torch.lgamma(torch.tensor(len(temp)+1))) * trace_prod / (2**len(temp)*torch.tensor(temp).prod())

    return coeff

def hafnian(A: torch.Tensor, loop=False) -> torch.Tensor:
    """
    Calculate the hafnian for symmetrix matrix, using eigenvalue-trace method

    See https://arxiv.org/abs/2108.01622 Eq.(B3)
    """
#     # vmap over torch.allclose isn't supported yet
#     assert torch.allclose(A, A.mT, rtol=rtol, atol=atol), 'the input matrix should be symmetric'
    size = A.size()[-1]
    if size % 2 == 1:  # consider odd case
        A = torch.block_diag(torch.tensor(1, device = A.device), A)
        size = A.size()[-1]
    if size == 0:
        return 1
    if size == 2:
        if loop:
            return A[0, 1] + A[0, 0] * A[1, 1]
        else:
            return A[0, 1]
    power = len(A)//2
    haf = 0
    z_sets = get_subsets(power)
    for i in range(1, len(z_sets)):
        subset = z_sets[i]
        num_ = len(subset[0])
        sub_mats = torch.vmap(get_submat_haf, in_dims=(None, 0))(A, torch.tensor(subset))
        coeff = torch.vmap(p_labda, in_dims=(0, None, None))(sub_mats, power, loop)
        coeff_sum = (-1) ** (power-num_) * coeff.sum()
        haf = haf + coeff_sum
    return haf

def hafnian_batch(A: torch.Tensor, loop=False) -> torch.Tensor:
    """
    Calculate the batch hafnian
    """
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A)
    assert A.dim()==3, 'Input tensor should be in batched size'
    hafs = torch.vmap(hafnian, in_dims=(0, None))(A, loop)
    return hafs


