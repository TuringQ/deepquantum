"""
Common functions
"""

import itertools
import warnings
from typing import Dict, Generator, List, Optional, Tuple

import torch
from torch import vmap
from torch.distributions.multivariate_normal import MultivariateNormal

import deepquantum.photonic as dqp
from ..qmath import is_unitary
from .utils import mem_to_chunksize


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
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore') # local warning
        u1 = torch.repeat_interleave(u, output_state, dim=0)
        u2 = torch.repeat_interleave(u1, input_state, dim=-1)
    return u2


def permanent(mat: torch.Tensor) -> torch.Tensor:
    """Calculate the permanent."""
    shape = mat.shape
    if mat.numel() == 0:
        if shape[0] == shape[1] == 0:
            return torch.tensor(1, dtype=mat.dtype, device=mat.device)
        else:
            return torch.tensor(0, dtype=mat.dtype, device=mat.device)
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


def create_subset(num_coincidence: int) -> Generator[torch.Tensor, None, None]:
    r"""Create all subsets from :math:`\{1,2,...,n\}`."""
    for k in range(1, num_coincidence + 1):
        comb_lst = []
        for comb in itertools.combinations(range(num_coincidence), k):
            comb_lst.append(list(comb))
        yield torch.tensor(comb_lst).reshape(len(comb_lst), k)

def get_powerset(n: int) -> List:
    r"""Get the powerset of :math:`\{0,1,...,n-1\}`."""
    powerset = []
    for k in range(n + 1):
        subset = []
        for i in itertools.combinations(range(n), k):
            subset.append(list(i))
        powerset.append(subset)
    return powerset


def permanent_ryser(mat: torch.Tensor) -> torch.Tensor:
    """Calculate the permanent by Ryser's formula."""
    def helper(subset: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
        num_elements = subset.numel()
        s = torch.sum(mat[:, subset], dim=-1)
        value_times = torch.prod(s) * (-1) ** num_elements
        return value_times

    num_coincidence = mat.size()[0]
    value_perm = 0
    chunk_size = mem_to_chunksize(mat.device, mat.dtype)
    for subset in create_subset(num_coincidence):
        temp_value = vmap(helper, in_dims=(0, None), chunk_size=chunk_size)(subset, mat)
        value_perm += temp_value.sum()
    value_perm *= (-1) ** num_coincidence
    return value_perm


def product_factorial(state: torch.Tensor) -> torch.Tensor:
    """Get the product of the factorial from the Fock state, i.e., :math:`|s_1,s_2,...s_n> -> s_1!s_2!...s_n!`."""
    return torch.exp(torch.lgamma(state.double() + 1).sum(-1, keepdim=True)) # nature log gamma function


def fock_combinations(nmode: int, nphoton: int, cutoff: Optional[int] = None, nancilla: int = 0) -> List:
    """Generate all possible combinations of Fock states for a given number of modes, photons, and cutoff.

    Args:
        nmode (int): The number of modes in the system.
        nphoton (int): The total number of photons in the system.
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        nancilla (int, optional): The number of ancilla modes (NOT limited by ``cutoff``). Default: ``0``

    Returns:
        List[List[int]]: A list of all possible Fock states, each represented by a list of
        occupation numbers for each mode.

    Examples:
        >>> fock_combinations(2, 3)
        [[0, 3], [1, 2], [2, 1], [3, 0]]
        >>> fock_combinations(3, 2)
        [[0, 0, 2], [0, 1, 1], [0, 2, 0], [1, 0, 1], [1, 1, 0], [2, 0, 0]]
        >>> fock_combinations(4, 4, 2)
        [[1, 1, 1, 1]]
    """
    if cutoff is None:
        cutoff = nphoton + 1
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
        # Determine the effective length for cutoff
        effective_length = length - nancilla
        # skip iterations if remaining photons exceed the remaining cutoff
        if nancilla == 0 and num_sum > (cutoff - 1) * effective_length:
            return
        for i in range(min((num_sum + 1), cutoff) if effective_length > 0 else (num_sum + 1)):
            backtrack(state + [i], length - 1, num_sum - i)

    backtrack([], nmode, nphoton)
    return result


def shift_func(l: List, nstep: int) -> List:
    """Shift a list by a number of steps.

    If ``nstep`` is positive, it shifts to the left.
    """
    if len(l) <= 1:
        return l
    nstep = nstep % len(l)
    return l[nstep:] + l[:nstep]


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


def _photon_number_mean_var_gaussian(cov: torch.Tensor, mean: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the expectation value and variance of the photon number for single-mode Gaussian states."""
    coef = dqp.kappa ** 2 / dqp.hbar
    cov = cov.reshape(-1, 2, 2)
    mean = mean.reshape(-1, 2, 1)
    exp = coef * (vmap(torch.trace)(cov) + (mean.mT @ mean).squeeze()) - 1 / 2
    var = coef**2 * (vmap(torch.trace)(cov @ cov) + 2 * (mean.mT @ cov.to(mean.dtype) @ mean).squeeze()) * 2 - 1 / 4
    return exp, var


def _photon_number_mean_var_bosonic(
    cov: torch.Tensor,
    mean: torch.Tensor,
    weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the expectation value and variance of the photon number for single-mode Bosonic states."""
    shape_cov = cov.shape
    shape_mean = mean.shape
    cov = cov.reshape(*shape_cov[:2], 2, 2).reshape(-1, 2, 2)
    mean = mean.reshape(*shape_mean[:2], 2, 1).reshape(-1, 2, 1)
    exp_gaussian, var_gaussian = _photon_number_mean_var_gaussian(cov, mean)
    exp_gaussian = exp_gaussian.reshape(*shape_cov[:2])
    var_gaussian = var_gaussian.reshape(*shape_cov[:2])
    exp = (weight * exp_gaussian).sum(-1)
    var = (weight * var_gaussian).sum(-1) + (weight * exp_gaussian**2).sum(-1) - exp ** 2
    assert torch.allclose(exp.imag, torch.zeros(1))
    assert torch.allclose(var.imag, torch.zeros(1))
    return exp.real, var.real


def photon_number_mean_var(
    cov: torch.Tensor,
    mean: torch.Tensor,
    weight: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the expectation value and variance of the photon number for single-mode Gaussian (Bosonic) states."""
    if weight is None:
        return  _photon_number_mean_var_gaussian(cov, mean)
    else:
        return _photon_number_mean_var_bosonic(cov, mean, weight)


def takagi(a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


def sample_reject_bosonic(
    cov: torch.Tensor,
    mean: torch.Tensor,
    weight: torch.Tensor,
    cov_m: torch.Tensor,
    shots: int
) -> torch.Tensor:
    """Get the samples of the Bosonic states via rejection sampling.

    See https://arxiv.org/abs/2103.05530 Algorithm 1 in Section VI B
    """
    if cov.ndim == 3:
        cov = cov.unsqueeze(0)
    if mean.ndim == 3:
        mean = mean.unsqueeze(0)
    if weight.ndim == 1:
        weight = weight.unsqueeze(0)
    assert cov.ndim == mean.ndim == 4
    assert weight.ndim == 2
    batch = cov.shape[0]
    rst = [cov.new_empty(0)] * batch
    batches = list(range(batch))
    count_shots = [0] * batch
    shots_tmp = shots
    mask = (weight.real > 0) | (abs(weight.imag) > 1e-8) | (abs(mean.imag) > 1e-8).any(-2).squeeze(-1)
    exp_real = torch.exp(mean.imag.mT @ torch.linalg.solve(cov_m + cov, mean.imag) / 2).squeeze()
    c_tilde = mask * abs(weight) * exp_real
    while len(batches) > 0:
        cov_rest = cov[batches]
        mean_rest = mean[batches]
        cov_t = cov_m + cov_rest
        m0 = torch.multinomial(c_tilde[batches], 1).reshape(-1) # (batch)
        cov_m0 = cov[batches, m0]
        mean_m0 = mean[batches, m0].squeeze(-1).real
        dist_g = MultivariateNormal(mean_rest.squeeze(-1).real, cov_t) # (batch, ncomb, 2 * nmode)
        r0 = MultivariateNormal(mean_m0, cov_m + cov_m0).sample([shots_tmp]) # (shots, batch, 2 * nmode)
        prob_g = dist_g.log_prob(r0.unsqueeze(-2)).exp() # (shots, batch, ncomb, 2 * nmode) -> (shots, batch, ncomb)
        g_r0 = (c_tilde[batches] * prob_g).sum(-1) # (shots, batch)
        y0 = torch.rand_like(g_r0) * g_r0
        rm = r0.unsqueeze(-1).unsqueeze(-3) # (shots, batch, 2 * nmode) -> (shots, batch, 1, 2 * nmode, 1)
        # (shots, batch, ncomb)
        exp_imag = torch.exp((rm - mean_rest.real).mT @ torch.linalg.solve(cov_t, mean_rest.imag) * 1j).squeeze()
        # Eq.(70-71)
        p_r0 = (weight[batches] * exp_real[batches] * prob_g * exp_imag).sum(-1) # (shots, batch)
        assert torch.allclose(p_r0.imag, p_r0.imag.new_zeros(1))
        idx_shots, idx_batch = torch.where((y0 <= p_r0.real))
        batches_done = []
        for i in range(len(batches)):
            idx = batches[i]
            rst[idx] = torch.cat([rst[idx], r0[idx_shots[idx_batch==i], i]]) # (shots, 2 * nmode)
            count_shots[idx] = len(rst[idx])
            if count_shots[idx] >= shots:
                batches_done.append(idx)
                rst[idx] = rst[idx][:shots]
        for i in batches_done:
            batches.remove(i)
        shots_tmp = shots - min(count_shots)
    return torch.stack(rst) # (batch, shots, 2 * nmode)


def align_shape(cov: torch.Tensor, mean: torch.Tensor, weight: torch.Tensor) -> List[torch.Tensor]:
    """Align the shape for Bosonic state."""
    assert cov.ndim == mean.ndim == 4
    assert weight.ndim == 2
    ncomb = weight.shape[-1]
    if cov.shape[1] == 1:
        cov = cov.expand(-1, ncomb, -1, -1)
    if mean.shape[1] == 1:
        mean = mean.expand(-1, ncomb, -1, -1)
    if weight.shape[0] == 1:
        weight = weight.expand(cov.shape[0], -1)
    return [cov, mean, weight]
