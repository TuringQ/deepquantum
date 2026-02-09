"""
functions for hafnian
"""

from collections import Counter
from functools import cache

import numpy as np
import torch
from scipy.special import factorial

from .qmath import get_powerset


@cache
def integer_partition(remaining: int, max_num: int) -> list:
    """Generate all unique integer partitions of m using integers up to n."""
    if remaining == 0:
        return [[]]
    if remaining < 0 or max_num == 0:
        return []
    result = []
    if remaining >= max_num:
        for part in integer_partition(remaining - max_num, max_num):
            result.append([max_num] + part)
    result.extend(integer_partition(remaining, max_num - 1))
    return result


def count_unique_permutations(nums: list | np.ndarray) -> np.float64:
    """Count the number of unique permutations of a list of numbers."""
    total_permutations = factorial(len(nums))
    num_counts = Counter(nums)
    repetitions = 1
    for count in num_counts.values():
        repetitions *= factorial(count)
    unique_permutations = total_permutations // repetitions
    return unique_permutations


def get_submat_haf(a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Get the sub-matrix for hafnian calculation.

    See https://arxiv.org/abs/1805.12498 paragraph after Eq.(3.20)
    """
    idx1 = 2 * z
    idx2 = idx1 + 1
    idx = torch.cat([idx1, idx2])
    idx = torch.sort(idx)[0]
    submat = a[idx][:, idx]
    return submat


def poly_lambda(submat: torch.Tensor, int_partition: list, power: int, loop: bool = False) -> torch.Tensor:
    """Get the coefficient of the polynomial.

    See https://arxiv.org/abs/1805.12498 Eq.(3.26) (noting that Eq.(3.26) contains a typo) and
    https://research-information.bris.ac.uk/ws/portalfiles/portal/329011096/thesis.pdf Eq.(3.80)
    """
    size = submat.shape[-1]
    identity = torch.eye(size, dtype=submat.dtype, device=submat.device)
    x_mat = identity.reshape(size // 2, 2, size).flip(1).reshape(size, size)
    xaz = x_mat @ submat
    # trace power calculation
    traces = []
    x = xaz.new_ones(xaz.shape[-1]).diag()
    traces.append(torch.trace(x))
    for _ in range(power):
        x = x @ xaz
        traces.append(torch.trace(x))
    trace_list = torch.stack(traces)
    coeff = 0
    if loop:  # loop hafnian case
        v = torch.diag(submat)
        xv = x_mat @ v / 2
        # matrix power calculation
        diag_term = []
        x = xaz.new_ones(xaz.shape[-1]).diag()
        diag_term.append(v @ x @ xv)
        for _ in range(power - 1):
            x = x @ xaz
            diag_term.append(v @ x @ xv)
        diag_term = torch.stack(diag_term)
    for orders in int_partition:
        ncount = count_unique_permutations(orders)
        orders = torch.tensor(orders, device=submat.device)
        poly_list = trace_list[orders] / (2 * orders)
        if loop:
            poly_list += diag_term[orders - 1]
        poly_prod = poly_list.prod()
        coeff += ncount / factorial(len(orders)) * poly_prod
    return coeff


def hafnian(matrix: torch.Tensor, loop: bool = False) -> torch.Tensor:
    """Calculate the hafnian for symmetric matrix, using the eigenvalue-trace method.

    See https://arxiv.org/abs/2108.01622 Eq.(B3)
    """
    size = matrix.shape[-1]
    if size % 2 == 1:
        if loop:
            matrix = torch.block_diag(torch.tensor(1, dtype=matrix.dtype, device=matrix.device), matrix)
            size = matrix.shape[-1]
        else:
            return torch.tensor(0, dtype=matrix.dtype, device=matrix.device)
    if size == 0:
        return torch.tensor(1, dtype=matrix.dtype, device=matrix.device)
    if size == 2:
        if loop:
            return matrix[0, 1] + matrix[0, 0] * matrix[1, 1]
        else:
            return matrix[0, 1]
    power = size // 2
    haf = 0
    powerset = get_powerset(power)
    int_partition = integer_partition(power, power)
    for i in range(1, len(powerset)):
        z_sets = torch.tensor(powerset[i], device=matrix.device)
        num_z = len(z_sets[0])
        submats = torch.vmap(get_submat_haf, in_dims=(None, 0))(matrix, z_sets)
        coeff = torch.vmap(poly_lambda, in_dims=(0, None, None, None))(submats, int_partition, power, loop)
        coeff_sum = (-1) ** (power - num_z) * coeff.sum()
        haf += coeff_sum
    return haf


def hafnian_batch(matrix: torch.Tensor, loop: bool = False) -> torch.Tensor:
    """Calculate the batch hafnian."""
    assert matrix.dim() == 3, 'Input tensor should be in batched size'
    batch = matrix.shape[0]
    for i in range(batch):
        assert torch.allclose(matrix[i], matrix[i].mT)
    hafs = torch.vmap(hafnian, in_dims=(0, None))(matrix, loop)
    return hafs
