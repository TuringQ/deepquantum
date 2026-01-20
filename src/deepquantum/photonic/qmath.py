"""
Common functions
"""

import itertools
import warnings
from collections import Counter
from typing import Dict, Generator, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch
from matplotlib import cm
from torch import vmap
from torch.distributions.multivariate_normal import MultivariateNormal

import deepquantum.photonic as dqp
from ..qmath import list_to_decimal, decimal_to_list, is_unitary, partial_trace, block_sample
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


def ladder_ops(cutoff: int, dtype = torch.cfloat, device = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the matrix representation of the annihilation and creation operators."""
    sqrt = torch.arange(1, cutoff, dtype=dtype, device=device) ** 0.5
    a = torch.diag(sqrt, diagonal=1)
    ad = a.mH # share the memory
    return a, ad


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
    idx = torch.arange(2 * nmode, device=matrix.device).reshape(2, nmode).T.flatten()
    if matrix.shape[-1] == 2 * nmode:
        return matrix[..., idx[:, None], idx]
    elif matrix.shape[-1] == 1:
        return matrix[..., idx, :]


def xpxp_to_xxpp(matrix: torch.Tensor) -> torch.Tensor:
    """Transform the representation in ``xpxp`` ordering to the representation in ``xxpp`` ordering."""
    nmode = matrix.shape[-2] // 2
    idx = torch.arange(2 * nmode, device=matrix.device).reshape(nmode, 2).T.flatten()
    if matrix.shape[-1] == 2 * nmode:
        return matrix[..., idx[:, None], idx]
    elif matrix.shape[-1] == 1:
        return matrix[..., idx, :]


def quadrature_to_ladder(tensor: torch.Tensor, symplectic: bool = False) -> torch.Tensor:
    """Transform the representation in ``xxpp`` ordering to the representation in ``aaa^+a^+`` ordering.

    Args:
        tensor (torch.Tensor): The input tensor in ``xxpp`` ordering.
        symplectic (bool, optional): Whether the transformation is applied for symplectic matrix or Gaussian state.
            Default: ``False`` (which means covariance matrix or displacement vector)
    """
    nmode = tensor.shape[-2] // 2
    tensor = tensor + 0j
    identity = torch.eye(nmode, dtype=tensor.dtype, device=tensor.device)
    omega = torch.cat([torch.cat([identity, identity * 1j], dim=-1),
                       torch.cat([identity, identity * -1j], dim=-1)])
    if tensor.shape[-1] == 2 * nmode:
        if symplectic:
            return omega @ tensor @ omega.mH / 2 # inversed omega
        else:
            return omega @ tensor @ omega.mH * dqp.kappa**2 / dqp.hbar
    elif tensor.shape[-1] == 1:
        return omega @ tensor * dqp.kappa / dqp.hbar**0.5


def ladder_to_quadrature(tensor: torch.Tensor, symplectic: bool = False) -> torch.Tensor:
    """Transform the representation in ``aaa^+a^+`` ordering to the representation in ``xxpp`` ordering.

    Args:
        tensor (torch.Tensor): The input tensor in ``aaa^+a^+`` ordering.
        symplectic (bool, optional): Whether the transformation is applied for symplectic matrix or Gaussian state.
            Default: ``False`` (which means covariance matrix or displacement vector)
    """
    nmode = tensor.shape[-2] // 2
    tensor = tensor + 0j
    identity = torch.eye(nmode, dtype=tensor.dtype, device=tensor.device)
    omega = torch.cat([torch.cat([identity, identity], dim=-1),
                       torch.cat([identity * -1j, identity * 1j], dim=-1)])
    if tensor.shape[-1] == 2 * nmode:
        if symplectic:
            return (omega @ tensor @ omega.mH).real / 2 # inversed omega
        else:
            return (omega @ tensor @ omega.mH).real * dqp.hbar / (4 * dqp.kappa**2)
    elif tensor.shape[-1] == 1:
        return (omega @ tensor).real * dqp.hbar**0.5 / (2 * dqp.kappa)


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
    exp_gaussian = exp_gaussian.reshape(shape_cov[:2])
    var_gaussian = var_gaussian.reshape(shape_cov[:2])
    exp = (weight * exp_gaussian).sum(-1)
    var = (weight * var_gaussian).sum(-1) + (weight * exp_gaussian**2).sum(-1) - exp ** 2
    zeros = cov.new_zeros(1)
    assert torch.allclose(exp.imag, zeros, atol=1e-6)
    assert torch.allclose(var.imag, zeros, atol=1e-6)
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
    """Tagaki decomposition for a symmetric complex matrix.

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


def sqrtm_herm(mat: torch.Tensor) -> torch.Tensor:
    """Compute the positive matrix square root of a Hermitian matrix using eigenvalue decomposition."""
    lambd, mat_q = torch.linalg.eigh(mat)
    return mat_q @ lambd.sqrt().diag_embed().to(mat_q.dtype) @ mat_q.mH


def schur_anti_symm_even(mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Schur decomposition for a real antisymmetric and even-dimensional matrix.

    This function decomposes a real antisymmetric matrix :math:`A` into the form :math:`A = O T O^T`,
    where :math:`O` is an orthogonal matrix and :math:`T` is a block-diagonal matrix
    with :math:`2 \times 2` antisymmetric blocks.
    """
    assert torch.allclose(mat, -mat.mT, rtol=1e-5, atol=1e-5)
    n = len(mat)
    hermitian = mat * -1j
    lambd, u = torch.linalg.eigh(hermitian)
    mat_t = torch.zeros_like(mat)
    idx1 = torch.arange(0, n, 2, device=mat.device)
    idx2 = torch.arange(1, n, 2, device=mat.device)
    # positive value is above the diagonal and in ascending order
    mat_t[idx1, idx2] = lambd[n//2:]
    mat_t[idx2, idx1] = -lambd[n//2:]
    mat_o = torch.zeros_like(mat)
    mat_o[:, ::2] = u[:, n//2:].real
    mat_o[:, 1::2] = u[:, n//2:].imag
    norm = torch.linalg.vector_norm(mat_o, dim=0, keepdim=True)
    mat_o = mat_o / norm
    return mat_t, mat_o


def williamson(cov: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Williamson decomposition.

    This function decomposes a real symmetric and even-dimensional positive definite matrix :math:`V`
    into the form :math:`V = S D S^T`, where :math:`S` is a symplectic matrix and
    :math:`D` is a diagonal matrix with the symplectic eigenvalues.

    See https://arxiv.org/pdf/2403.04596 Section VII.
    """
    assert torch.allclose(cov, cov.mT, rtol=1e-5, atol=1e-5)
    nmode = cov.shape[-1] // 2
    omega = cov.new_ones(nmode)
    omega = torch.cat([-omega, omega]).diag_embed()
    omega = omega.reshape(2, nmode, 2 * nmode).flip(0).reshape(2 * nmode, 2 * nmode) # symplectic form
    vals = torch.linalg.eigvalsh(cov)
    assert torch.all(vals > 0), 'Matrix must be positive definite.'
    cov_sqrt = sqrtm_herm(cov)
    cov_sqrt_inv = cov_sqrt.inverse()
    psi = cov_sqrt_inv @ omega @ cov_sqrt_inv # antisymmetric
    mat_t, o_tilde = schur_anti_symm_even(psi)
    idx_perm = torch.arange(2 * nmode, device=cov.device).reshape(nmode, 2).T.flatten()
    mat_t_xxpp = mat_t[:, idx_perm][idx_perm]
    mat_o = o_tilde[:, idx_perm]
    idx = torch.arange(nmode, device=cov.device)
    phi = mat_t_xxpp[idx, idx + nmode]
    phi2 = torch.cat([phi, phi])
    diag = (1 / phi2).diag_embed()
    mat_s = cov_sqrt @ mat_o @ phi2.sqrt().diag_embed()
    return diag, mat_s


def measure_fock_tensor(
    state: torch.Tensor,
    shots: int = 1024,
    with_prob: bool = False,
    wires: Union[int, List[int], None] = None,
    block_size: int = 2 ** 24
) -> Union[Dict, List[Dict]]:
    r"""Measure the batched Fock state tensors.

    Args:
        state (torch.Tensor): The quantum state to measure. It should be a tensor of shape
            :math:`(\text{batch}, \text{cutoff}, ..., \text{cutoff})`.
        shots (int, optional): The number of times to sample from the quantum state. Default: 1024
        with_prob (bool, optional): A flag that indicates whether to return the probabilities along with
            the number of occurrences. Default: ``False``
        wires (int, List[int] or None, optional): The wires to measure. It can be an integer or a list of
            integers specifying the indices of the wires. Default: ``None`` (which means all wires are
            measured)
        block_size (int, optional): The block size for sampling. Default: 2 ** 24
    """
    # pylint: disable=import-outside-toplevel
    from .state import FockState
    shape = state.shape
    batch = shape[0]
    cutoff = shape[-1]
    nmode = len(shape) - 1
    if wires is not None:
        if isinstance(wires, int):
            wires = [wires]
        assert isinstance(wires, list)
        wires = sorted(wires)
        pm_shape = list(range(nmode))
        for w in wires:
            pm_shape.remove(w)
        pm_shape = wires + pm_shape
    nwires = len(wires) if wires else nmode
    results_tot = []
    for i in range(batch):
        probs = torch.abs(state[i]) ** 2
        if wires is not None:
            probs = probs.permute(pm_shape).reshape([cutoff] * nwires + [-1]).sum(-1)
        probs = probs.reshape(-1)
        # Perform block sampling to reduce memory consumption
        samples = Counter(block_sample(probs, shots, block_size))
        results = {FockState(decimal_to_list(key, cutoff, nwires)): value for key, value in samples.items()}
        if with_prob:
            for k in results:
                index = list_to_decimal(k.state, cutoff)
                results[k] = results[k], probs[index]
        results_tot.append(results)
    if batch == 1:
        return results_tot[0]
    else:
        return results_tot


def sample_homodyne_fock(
    state: torch.Tensor,
    wire: int,
    nmode: int,
    cutoff: int,
    shots: int = 1,
    den_mat: bool = False,
    x_range: float = 15,
    nbin: int = 100000
) -> torch.Tensor:
    """Get the samples of homodyne measurement for batched Fock state tensors on one mode."""
    coef = 2 * dqp.kappa**2 / dqp.hbar
    if den_mat:
        state = state.reshape(-1, cutoff ** nmode, cutoff ** nmode)
    else:
        state = state.reshape(-1, cutoff ** nmode, 1)
        state = state @ state.mH
    trace_lst = [i for i in range(nmode) if i != wire]
    reduced_dm = partial_trace(state, nmode, trace_lst, cutoff) # (batch, cutoff, cutoff)
    orders = torch.arange(cutoff, dtype=state.real.dtype, device=state.device).reshape(-1, 1) # (cutoff, 1)
    # with dimension \sqrt{m\omega\hbar}
    xs = torch.linspace(-x_range, x_range, nbin, dtype=state.real.dtype, device=state.device) # (nbin)
    h_vals = torch.special.hermite_polynomial_h(coef**0.5 * xs, orders) #（cutoff, nbin)
    # H_n / \sqrt{2^n * n!}
    h_vals = h_vals / torch.sqrt(2**orders * torch.exp(torch.lgamma(orders.double() + 1))).to(orders.dtype)
    h_mat = h_vals.reshape(1, cutoff, nbin) * h_vals.reshape(cutoff, 1, nbin) # (cutoff, cutoff, nbin)
    h_terms = reduced_dm.unsqueeze(-1) * h_mat # (batch, cutoff, cutoff, nbin)
    probs = (h_terms.sum(dim=[-3, -2]) * torch.exp(-coef * xs**2)).real # (batch, nbin)
    probs = abs(probs)
    probs[probs < 1e-10] = 0
    indices = torch.multinomial(probs.reshape(-1, nbin), num_samples=shots, replacement=True) # (batch, shots)
    samples = xs[indices]
    return samples.unsqueeze(-1) # (batch, shots, 1)


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
    exp_real = torch.exp(mean.imag.mT @ torch.linalg.solve(cov_m + cov, mean.imag) / 2).squeeze(-2, -1)
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
    ncomb = weight.shape[-1]
    if cov.ndim == mean.ndim == 4 and weight.ndim == 2:
        if cov.shape[1] == 1:
            cov = cov.expand(-1, ncomb, -1, -1)
        if mean.shape[1] == 1:
            mean = mean.expand(-1, ncomb, -1, -1)
        if weight.shape[0] == 1:
            weight = weight.expand(cov.shape[0], -1)
    elif cov.ndim == mean.ndim == 3 and weight.ndim == 1:
        if cov.shape[0] == 1:
            cov = cov.expand(ncomb, -1, -1)
        if mean.shape[0] == 1:
            mean = mean.expand(ncomb, -1, -1)
    return [cov, mean, weight]

def fock_to_wigner(
    state: torch.Tensor,
    wire: int,
    nmode: int,
    cutoff: int,
    den_mat: bool = False,
    xrange: Union[int, List] = 10,
    prange: Union[int, List] = 10,
    npoints: Union[int, List] = 200,
    plot: bool = True,
    k: int = 0
) -> torch.Tensor:
    """Compute the Wigner function W(q, p) from a Fock tensor state or density matrix
    using the iterative method.

    See https://qutip.org/docs/4.7/modules/qutip/wigner.html

    Args:
        state(List): The input Fock state tensor or density matrix.
        wire (int): The wigner function for given wire.
        nmode(int): The mode number of the Fock state.
        xrange (int or List, optional): The range of quadrature q. Default: 10
        prange (int or List, optional): The range of quadrature p. Default: 10
        npoints(int or List, optional): The number of discretization points for quadratures. Default: 200
        plot (bool, optional): Whether to plot the wigner function. Default: ``True``
        k (int, optional): The wigner function of kth batch to plot. Default: 0
    """
    if den_mat:
        rho = state.reshape(-1, cutoff**nmode, cutoff**nmode)
    else:
        state = state.reshape(-1, cutoff**nmode, 1)
        rho = state @ state.mH
    trace_lst = [i for i in range(nmode) if i != wire]
    reduced_dm = partial_trace(rho, nmode, trace_lst, cutoff) # (batch, cutoff, cutoff)
    if reduced_dm.ndim == 2:
        reduced_dm = reduced_dm.unsqueeze(0)
    if isinstance(xrange, int):
        xlist = [-xrange, xrange]
    else:
        xlist = xrange
    if isinstance(prange, int):
        plist = [-prange, prange]
    else:
        plist = prange
    if isinstance(npoints, int):
        xlist.append(npoints)
        plist.append(npoints)
    else:
        xlist.append(npoints[0])
        plist.append(npoints[1])
    assert len(xlist) == len(plist) == 3
    xvec = torch.linspace(*xlist, dtype=torch.double)
    pvec = torch.linspace(*plist, dtype=torch.double)
    coef = 2 * dqp.kappa ** 2 / dqp.hbar
    xvec = coef ** 0.5 * xvec
    pvec = coef ** 0.5 * pvec
    xlist, plist = torch.meshgrid(xvec, pvec, indexing='ij')
    # alpha = (q + i p) / sqrt(2)
    alpha = (xlist + 1.0j * plist) / rho.new_tensor(2.0).sqrt()
    w_list = xlist.new_zeros(cutoff, xlist.shape[-2], xlist.shape[-1]) * 1j
    w_00 = coef * torch.exp(-2 * abs(alpha)**2) / torch.pi
    w_list[0] = w_00
    w  = reduced_dm[:,0,0].reshape(-1,1,1) * w_list[0]
    # First row: W_{0i}
    for i in range(1, cutoff):
        # For numerical stability, it is recommended to use cutoff < 80
        w_list[i] = 2 * alpha * w_list[i-1] / rho.new_tensor(i).sqrt()
        w += 2 * (reduced_dm[:,0,i].reshape(-1,1,1) * w_list[i]).real
    # Remaining rows: W_{ij}, i ≥ 1
    for i in range(1, cutoff):
        # Diagonal element W_{ii}
        sqrt_i = rho.new_tensor(i).sqrt()
        temp = w_list[i].clone()
        w_list[i] = (2 * alpha.conj() * temp - sqrt_i * w_list[i-1]) / sqrt_i
        w += reduced_dm[:,i,i].reshape(-1,1,1) * w_list[i]
        # Off-diagonal elements W_{ij}, j > i
        for j in range(i+1, cutoff):
            sqrt_j = rho.new_tensor(j).sqrt()
            temp2 = (2 * alpha * w_list[j-1] - sqrt_i * temp) / sqrt_j
            temp = w_list[j].clone()
            w_list[j] = temp2
            w += 2 * (reduced_dm[:,i,j].reshape(-1,1,1) * w_list[j]).real
    if plot:
        plot_wigner(w, xvec, pvec, k=k)
    return w

def cv_to_wigner(
    state: List,
    wire: int,
    xrange: Union[int, List] = 10,
    prange: Union[int, List] = 10,
    npoints: Union[int, List] = 200,
    plot: bool = True,
    k: int = 0
):
    """Get the discretized Wigner function of the specified mode.

    Args:
        state(List): The input Gaussianstate or BosonicState.
        wire (int): The wigner function for given wire.
        xrange (int or List, optional): The range of quadrature q. Default: 10
        prange (int or List, optional): The range of quadrature p. Default: 10
        npoints(int or List, optional): The number of discretization points for quadratures. Default: 200
        plot (bool, optional): Whether to plot the wigner function. Default: ``True``
        k (int, optional): The wigner function of kth batch to plot. Default: 0
    """
    if isinstance(xrange, int):
        xlist = [-xrange, xrange]
    else:
        xlist = xrange
    if isinstance(prange, int):
        plist = [-prange, prange]
    else:
        plist = prange
    if isinstance(npoints, int):
        xlist.append(npoints)
        plist.append(npoints)
    else:
        xlist.append(npoints[0])
        plist.append(npoints[1])
    assert len(xlist) == len(plist) == 3
    xvec = torch.linspace(*xlist, dtype=torch.double)
    pvec = torch.linspace(*plist, dtype=torch.double)
    grid_x, grid_y = torch.meshgrid(xvec, pvec, indexing='ij')
    coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)]).mT
    coords2 = coords.unsqueeze(1).unsqueeze(2) # (npoints, 1, 1, 2)
    coords3 = coords.unsqueeze(-1).unsqueeze(-3)
    if not isinstance(wire, torch.Tensor):
        wire = torch.tensor(wire).reshape(1)
    cov, mean = state[:2]
    mean = mean.to(torch.complex128)
    if cov.ndim == 2:
        cov = cov.unsqueeze(0)
    if mean.ndim == 2:
        mean = mean.unsqueeze(0)
    if cov.ndim == 3:
        cov = cov.unsqueeze(1)
    if mean.ndim == 3:
        mean = mean.unsqueeze(1)
    if len(state)==2:
        weight = cov.new_tensor(1.).reshape(1)
    if len(state)==3:
        weight = state[-1]
    cov, mean, weight = align_shape(cov, mean, weight)
    nmode = cov.shape[-1]//2
    idx = torch.cat([wire, wire + nmode]) # xxpp order
    cov  = cov[..., idx[:, None], idx]
    mean = mean[..., idx, :]
    gauss_b = MultivariateNormal(mean.squeeze(-1).real, cov) # mean shape: (batch, ncomb, 2)
    prob_g = gauss_b.log_prob(coords2).exp() # (npoints, batch, ncomb)
    exp_real = torch.exp(mean.imag.mT @ torch.linalg.solve(cov, mean.imag) / 2).squeeze(-2, -1) # (batch, ncomb)
    # (batch, npoints, ncomb)
    exp_imag = torch.exp((coords3 - mean.real.unsqueeze(1)).mT @
                            torch.linalg.solve(cov, mean.imag).unsqueeze(1) * 1j).squeeze(-2, -1)
    wigner_vals = exp_real.unsqueeze(-2) * prob_g.permute(1, 0, 2) * exp_imag * weight.unsqueeze(-2)
    wigner_vals = wigner_vals.sum(dim=2).reshape(-1, len(xvec), len(pvec)).real
    # normalize the wigner function
    dx_ = xvec[1] - xvec[0]
    dp_ = pvec[1] - pvec[0]
    total_integral = torch.sum(wigner_vals, dim=[1,2]) * dx_ * dp_
    wigner_vals = wigner_vals / total_integral.reshape(-1, 1, 1)
    if plot:
        plot_wigner(wigner_vals, xvec, pvec, k=k)
    return wigner_vals

def plot_wigner(
    wigner: torch.Tensor,
    xvec: torch.Tensor,
    pvec: torch.Tensor,
    k: int=0
):
    """Plot a 2D contour and a 3D surface of a discretized Wigner function W(x, p).

    Args:
        wigner (torch.Tensor): Discretized Wigner values with shape (batch, len(xvec), len(pvec)).
        xvec (torch.Tensor): 1D grid for quadrature q.
        pvec (torch.Tensor): 1D grid for quadrature p.
        k (int, optional): Batch index to plot. Default: 0.
    """
    grid_x, grid_y = torch.meshgrid(xvec, pvec, indexing='ij')
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    ax1 = plt.subplot(121)
    plt.xlabel('Quadrature q')
    plt.ylabel('Quadrature p')
    plt.contourf(grid_x.cpu(), grid_y.cpu(), wigner[k].cpu(), 60, cmap=cm.RdBu)
    plt.colorbar(shrink=0.5)
    ax2 = plt.subplot(122, projection='3d')
    surf = ax2.plot_surface(grid_x.cpu(), grid_y.cpu(), wigner[k].cpu(), cmap=cm.RdBu, alpha=0.8)
    ax2.set_xlabel('Quadrature q')
    ax2.set_ylabel('Quadrature p')
    ax2.set_zlabel('W(q,p)')
    plt.tight_layout()
    plt.show()