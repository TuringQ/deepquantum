"""
Common functions
"""

import itertools
from typing import Callable, Dict, List, Tuple

import copy
import numpy as np
import torch
from collections import defaultdict, Counter
from scipy import special
from torch import vmap
from tqdm import tqdm

import deepquantum.photonic as dqp
from ..qmath import is_unitary


def dirac_ket(matrix: torch.Tensor) -> Dict:
    """Convert the batched Fock state tensor to the dictionary of Dirac ket."""
    ket_dict = {}
    for i in range(matrix.shape[0]): # consider batch i
        state_i = matrix[i]
        abs_state = abs(state_i)
        # get the largest k values with abs(amplitudes)
        top_k = torch.topk(abs_state.flatten(), k=min(len(abs_state), 5), largest=True).values
        idx_all = []
        ket_lst = []
        for amp in top_k:
            idx = torch.nonzero(abs_state == amp)[0].tolist()
            idx_all.append(idx)
            # after finding the indx, set the value to 0, avoid the same abs values
            abs_state[tuple(idx)] = 0
            state_b = ''.join(map(str, idx))
            if amp > 0:
                state_str = f' + ({state_i[tuple(idx)]:6.3f})|{state_b}>'
                ket_lst.append(state_str)
            ket = ''.join(ket_lst)
        batch_i = f'state_{i}'
        ket_dict[batch_i] = ket[3:]
    return ket_dict


def sort_dict_fock_basis(state_dict: Dict, idx: int = 0) -> Dict:
    """Sort the dictionary of Fock basis states in the descending order of probs."""
    sort_list = sorted(state_dict.items(), key=lambda t: abs(t[1][idx]), reverse=True)
    sorted_dict = {}
    for key, value in sort_list:
        sorted_dict[key] = value
    return sorted_dict


def sub_matrix(u: torch.Tensor, input_state: torch.Tensor, output_state: torch.Tensor) -> torch.Tensor:
    """Get the submatrix for calculating the transfer amplitude and transfer probs from the given matrix,
    the input state and the output state. The rows are chosen according to the output state and the columns
    are chosen according to the input state.

    Args:
        u (torch.Tensor): The unitary matrix.
        input_state (torch.Tensor): The input state.
        output_state (torch.Tensor): The output state.
    """
    def set_copy_indx(state: torch.Tensor) -> List:
        """Pick up indices from the nonezero elements of state.
        The repeat times depend on the nonezero value.
        """
        inds_nonzero = torch.nonzero(state)
        temp_ind = []
        for i in range(len(inds_nonzero)):
            temp1 = inds_nonzero[i]
            temp = state[inds_nonzero][i]
            temp_ind = temp_ind + [int(temp1)] * (int(temp))
        return temp_ind

    indx1 = set_copy_indx(input_state)
    indx2 = set_copy_indx(output_state)
    u1 = u[[indx2]]         # choose rows from the output
    u2 = u1[:, [indx1]]     # choose columns from the input
    return u2.squeeze(1)


def permanent(mat: torch.Tensor) -> torch.Tensor:
    """Calculate the permanent."""
    shape = mat.shape
    if len(mat.size()) == 0:
        return mat
    if shape[0] == 1:
        return mat[0, 0]
    if shape[0] == 2:
        return mat[0, 0] * mat[1, 1] + mat[0, 1] * mat[1, 0]
    if shape[0] == 3:
        return (mat[0, 2] * mat[1, 1] * mat[2, 0]
                + mat[0, 1] * mat[1, 2] * mat[2, 0]
                + mat[0, 2] * mat[1, 0] * mat[2, 1]
                + mat[0, 0] * mat[1, 2] * mat[2, 1]
                + mat[0, 1] * mat[1, 0] * mat[2, 2]
                + mat[0, 0] * mat[1, 1] * mat[2, 2])
    return permanent_ryser(mat)


def create_subset(num_coincidence: int) -> List:
    r"""Create all subsets from :math:`\{1,2,...,n\}`."""
    subsets = []
    for k in range(1, num_coincidence + 1):
        comb_lst = []
        for comb in itertools.combinations(range(num_coincidence), k):
            comb_lst.append(list(comb))
        temp = torch.tensor(comb_lst).reshape(len(comb_lst), k)
        subsets.append(temp)
    return subsets


def permanent_ryser(mat: torch.Tensor) -> torch.Tensor:
    """Calculate the permanent by Ryser's formula."""
    def helper(subset: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
        num_elements = subset.numel()
        s = torch.sum(mat[:, subset], dim=1)
        value_times = torch.prod(s) * (-1) ** num_elements
        return value_times

    num_coincidence = mat.size()[0]
    sets = create_subset(num_coincidence)
    value_perm = 0
    for subset in sets:
        temp_value = vmap(helper, in_dims=(0, None))(subset, mat)
        value_perm += temp_value.sum()
    value_perm *= (-1) ** num_coincidence
    return value_perm


def product_factorial(state: torch.Tensor) -> torch.Tensor:
    """Get the product of the factorial from the Fock state, i.e., :math:`|s_1,s_2,...s_n> --> s_1!*s_2!*...s_n!`."""
    fac = special.factorial(state.cpu())
    if not isinstance(fac, torch.Tensor):
        fac = torch.tensor(fac)
    return fac.prod(axis=-1, keepdims=True)


def fock_combinations(nmode: int, nphoton: int) -> List:
    """Generate all possible combinations of Fock states for a given number of modes and photons.

    Args:
        nmode (int): The number of modes in the system.
        nphoton (int): The total number of photons in the system.

    Returns:
        List[List[int]]: A list of all possible Fock states, each represented by a list of
        occupation numbers for each mode.

    Examples:
        >>> fock_combinations(2, 3)
        [[0, 3], [1, 2], [2, 1], [3, 0]]
        >>> fock_combinations(3, 2)
        [[0, 0, 2], [0, 1, 1], [0, 2, 0], [1, 0, 1], [1, 1, 0], [2, 0, 0]]
    """
    result = []
    def backtrack(state: List[int], length: int, num_sum: int) -> None:
        """A helper function that uses backtracking to generate all possible Fock states.

        Args:
            state (List[int]): The current Fock state being constructed.
            length (int): The remaining number of modes to be filled.
            num_sum (int): The remaining number of photons to be distributed.
        """
        if length == 0:
            if num_sum == 0:
                result.append(state)
            return
        for i in range(num_sum + 1):
            backtrack(state + [i], length - 1, num_sum - i)

    backtrack([], nmode, nphoton)
    return result


def xxpp_to_xpxp(matrix: torch.Tensor) -> torch.Tensor:
    """Transform the representation in ``xxpp`` ordering to the representation in ``xpxp`` ordering."""
    nmode = matrix.shape[-2] // 2
    # transformation matrix
    t = torch.zeros([2 * nmode] * 2, dtype=matrix.dtype, device=matrix.device)
    for i in range(2 * nmode):
        if i % 2 == 0:
            t[i][i // 2] = 1
        else:
            t[i][i // 2 + nmode] = 1
    # new matrix in xpxp ordering
    if matrix.shape[-1] == 2 * nmode:
        return t @ matrix @ t.mT
    elif matrix.shape[-1] == 1:
        return t @ matrix


def xpxp_to_xxpp(matrix: torch.Tensor) -> torch.Tensor:
    """Transform the representation in ``xpxp`` ordering to the representation in ``xxpp`` ordering."""
    nmode = matrix.shape[-2] // 2
    # transformation matrix
    t = torch.zeros([2 * nmode] * 2, dtype=matrix.dtype, device=matrix.device)
    for i in range(2 * nmode):
        if i < nmode:
            t[i][2 * i] = 1
        else:
            t[i][2 * (i - nmode) + 1] = 1
    # new matrix in xxpp ordering
    if matrix.shape[-1] == 2 * nmode:
        return t @ matrix @ t.mT
    elif matrix.shape[-1] == 1:
        return t @ matrix


def quadrature_to_ladder(matrix: torch.Tensor) -> torch.Tensor:
    """Transform the representation in ``xxpp`` ordering to the representation in ``aa^+`` ordering."""
    nmode = matrix.shape[-2] // 2
    matrix = matrix + 0j
    identity = torch.eye(nmode, dtype=matrix.dtype, device=matrix.device)
    omega = torch.cat([torch.cat([identity, identity * 1j], dim=-1),
                       torch.cat([identity, identity * -1j], dim=-1)]) * dqp.kappa / dqp.hbar ** 0.5
    if matrix.shape[-1] == 2 * nmode:
        return omega @ matrix @ omega.mH
    elif matrix.shape[-1] == 1:
        return omega @ matrix


def ladder_to_quadrature(matrix: torch.Tensor) -> torch.Tensor:
    """Transform the representation in ``aa^+`` ordering to the representation in ``xxpp`` ordering."""
    nmode = matrix.shape[-2] // 2
    matrix = matrix + 0j
    identity = torch.eye(nmode, dtype=matrix.dtype, device=matrix.device)
    omega = torch.cat([torch.cat([identity, identity], dim=-1),
                       torch.cat([identity * -1j, identity * 1j], dim=-1)]) * dqp.hbar ** 0.5 / (2 * dqp.kappa)
    if matrix.shape[-1] == 2 * nmode:
        return (omega @ matrix @ omega.mH).real
    elif matrix.shape[-1] == 1:
        return (omega @ matrix).real


def photon_number_mean_var(cov: torch.Tensor, mean: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the expectation value and variance of the photon number for single-mode Gaussian states."""
    coef = dqp.kappa ** 2 / dqp.hbar
    cov = cov.reshape(-1, 2, 2)
    mean = mean.reshape(-1, 2, 1)
    exp = coef * (vmap(torch.trace)(cov) + (mean.mT @ mean).squeeze()) - 1 / 2
    var = coef ** 2 * (vmap(torch.trace)(cov @ cov) + 2 * (mean.mT @ cov @ mean).squeeze()) * 2 - 1 / 4
    return exp, var


def takagi(a: torch.Tensor):
    """Tagaki decomposition for symmetric complex matrix.

    See https://math.stackexchange.com/questions/2026110/
    """
    size = a.size()[0]
    a_2 = torch.block_diag(-a.real, a.real)
    if torch.is_complex(a):
        a_2[:size, size:] = a.imag
        a_2[size:, :size] = a.imag
    s, u = torch.linalg.eigh(a_2)
    diag = s[size:] # s already sorted
    v = u[:, size:][size:] + 1j * u[:, size:][:size]
    if is_unitary(v):
        return v, diag
    else: # consider degeneracy case
        idx_zero = torch.where(abs(s) < 1e-5)[0]
        idx_max  = max(idx_zero) + 1
        temp = abs(u[:size, idx_max:]) ** 2 + abs(u[size:, idx_max:]) ** 2
        sum_rhalf = temp.sum(1)
        idx_lt_1 = torch.where(abs(sum_rhalf - 1) > 1e-6)[0]
        r = size - (2 * size - idx_max)
        # find the correct combination
        for i in itertools.combinations(idx_zero, r):
            u_temp = u[:, list(i)]
            temp2 = abs(u_temp[idx_lt_1]) ** 2 + abs(u_temp[idx_lt_1 + size]) ** 2
            sum_lhalf = temp2.sum(1)
            sum_total = sum_lhalf + sum_rhalf[idx_lt_1]
            if torch.allclose(sum_total, torch.ones(len(idx_lt_1), dtype=sum_total.dtype, device=sum_total.device)):
                u_half = torch.cat([u[:, list(i)], u[:, idx_max:]], dim=1)
                v = u_half[size:] + 1j * u_half[:size]
                if is_unitary(v):
                    return v, diag


def sample_sc_mcmc(prob_func: Callable,
                   proposal_sampler: Callable,
                   shots: int = 1024,
                   num_chain: int = 5) -> defaultdict:
    """Get the samples of the probability distribution function via SC-MCMC method."""
    samples_chain = []
    merged_samples = defaultdict(int)
    cache_prob = {}
    shots_lst = [shots // num_chain] * num_chain
    shots_lst[-1] += shots % num_chain
    for trial in range(num_chain):
        cache = []
        len_cache = min(shots_lst)
        if shots_lst[trial] > 1e5:
            len_cache = 4000
        samples = []
        # random start
        sample_0 = proposal_sampler()
        cache.append(sample_0)
        sample_max = sample_0
        if tuple(sample_max.tolist()) in cache_prob:
            prob_max = cache_prob[tuple(sample_max.tolist())]
        else:
            prob_max = prob_func(sample_0)
            cache_prob[tuple(sample_max.tolist())] = prob_max
        dict_sample = defaultdict(int)
        for i in tqdm(range(1, shots_lst[trial]), desc=f'chain {trial+1}', ncols=80, colour='green'):
            sample_i = proposal_sampler()
            if tuple(sample_i.tolist()) in cache_prob:
                prob_i = cache_prob[tuple(sample_i.tolist())]
            else:
                prob_i = prob_func(sample_i)
                cache_prob[tuple(sample_i.tolist())] = prob_i
            rand_num = torch.rand(1)
            samples.append(sample_i)
            # MCMC transfer to new state
            if prob_i / prob_max > rand_num:
                sample_max = sample_i
                prob_max = prob_i
            if i < len_cache: # cache not full
                cache.append(sample_max)
            else: # full
                idx = np.random.randint(0, len_cache)
                out_sample = copy.deepcopy(cache[idx])
                cache[idx] = sample_max
                out_sample_key = tuple(out_sample.tolist())
                if out_sample_key in dict_sample:
                    dict_sample[out_sample_key] = dict_sample[out_sample_key] + 1
                else:
                    dict_sample[out_sample_key] = 1
        # clear the cache
        for i in range(len_cache):
            out_sample = cache[i]
            out_sample_key = tuple(out_sample.tolist())
            if out_sample_key in dict_sample:
                dict_sample[out_sample_key] = dict_sample[out_sample_key] + 1
            else:
                dict_sample[out_sample_key] = 1
        samples_chain.append(dict_sample)
        for key, value in dict_sample.items():
            merged_samples[key] += value
    return merged_samples

def get_subsets(n):
    '''Get powerset of [0, 1, ... , n-1]'''
    subsets = [ ]
    for k in range(n+1):
        subset = [ ]
        for i in itertools.combinations(range(n), k):
            subset.append(list(i))
        subsets.append(subset)
    return subsets

def integer_partition(m, n):
    """ integer partition"""
    results = [ ]
    def back_trace(m, n, result=[]):
        if m == 0:
            results.append(result)
            return
        if m < 0 or n == 0:
            return
        back_trace(m, n - 1, result)
        back_trace(m - n, n, result + [n])
    back_trace(m, n)
    return results

def count_unique_permutations(nums):
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
    '''Get submat for hafnian calculation'''
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z)
    idx1 = 2 * z
    idx2 = idx1+1
    idx = torch.cat([idx1, idx2])
    idx = torch.sort(idx)[0]
    submat = a[idx][:, idx]
    return submat

def get_submat_tor(a, z):
    '''Get submat for torontonian calculation'''
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

def p_labda(submat, power, if_loop=False):
    """
    Return the coefficient of polynomial.
    """
    sigma_x = torch.tensor([[0, 1], [1, 0]])
    len_ = int(submat.size()[-1]/2)
    x_mat = torch.block_diag(*[sigma_x]*len_)
    x_mat = x_mat.to(submat.dtype)
    xaz = x_mat @ submat
    eigen = torch.linalg.eig(xaz)[0] #eigen decomposition
    trace_list = torch.stack([(eigen ** i).sum() for i in range(0, power+1)])
    int_partition = integer_partition(power, power)
    num_permu = []
    for i in int_partition:
        num_permu.append(count_unique_permutations(i))
    if if_loop: #loop hafnian case
        v = torch.diag(submat)
        diag_contribution = torch.stack([v @ torch.linalg.matrix_power(xaz, i-1) @ x_mat @ v.reshape(-1,1)/2 for i in range(1, power+1)])
        diag_contribution = diag_contribution.squeeze()
        coeff =  0
        for k in range(len(int_partition)):
            prefactor = num_permu[k]
            temp = int_partition[k]
            poly_list = trace_list[temp]/(2 * torch.tensor(temp)) + diag_contribution[np.array(temp)-1]
            poly_prod = poly_list.prod()
            coeff = coeff + prefactor/special.factorial(len(temp)) * poly_prod
    else:
        coeff =  0
        for k in range(len(int_partition)):
            prefactor = num_permu[k]
            temp = int_partition[k]
            trace_prod = trace_list[temp].prod()
            coeff = coeff + prefactor/special.factorial(len(temp)) * trace_prod / (2**len(temp)*torch.tensor(temp).prod())
    return coeff

def hafnian(A, if_loop = False, rtol=1e-05, atol=1e-06):
    """
    Calculate the hafnian for symmetrix matrix, using eigenvalue-trace method

    See https://arxiv.org/abs/2108.01622 Eq.(B3)
    """
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A)
#     # vmap over torch.allclose isn't supported yet
#     assert torch.allclose(A, A.mT, rtol=rtol, atol=atol), 'the input matrix should be symmetric'
    size = A.size()[-1]
    if size == 0:
        return 1
    if size == 2:
        if if_loop:
            return A[0, 1] + A[0, 0] * A[1, 1]
        else:
            return A[0, 1]
    if A.size()[-1] % 2 == 1:  # consider odd case
        A = torch.block_diag(torch.tensor(1), A)
    power = len(A)//2
    haf = 0
    z_sets = get_subsets(power)
    for i in range(1, len(z_sets)):
        subset = z_sets[i]
        num_ = len(subset[0])
        sub_mats = torch.vmap(get_submat_haf, in_dims=(None, 0))(A, torch.tensor(subset))
        coeff = torch.vmap(p_labda, in_dims=(0, None, None))(sub_mats, power, if_loop)
        coeff_sum = (-1) ** (power-num_) * coeff.sum()
        haf = haf + coeff_sum
    return haf

def _tor_helper(submat, sub_gamma):
    size = submat.size()[-1]
    temp = torch.eye(size)-submat
    inv_temp = torch.linalg.inv(temp)
    sub_gamma = sub_gamma.to(inv_temp.dtype)
    exp_term  = sub_gamma @ inv_temp @ sub_gamma.conj()/2
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
    tor = (-1) ** m * torch.exp(gamma @ gamma.conj() / 2)
    for i in range(1, len(z_sets)):
        subset = z_sets[i]
        num_ = len(subset[0])
        sub_mats = torch.vmap(get_submat_tor, in_dims=(None, 0))(o_mat, torch.tensor(subset))
        sub_gammas = torch.vmap(get_submat_tor, in_dims=(None, 0))(gamma, torch.tensor(subset))
        coeff = torch.vmap(_tor_helper)(sub_mats, sub_gammas)
        coeff_sum = (-1) ** (m - num_) * coeff.sum()
        tor = tor + coeff_sum
    return tor


