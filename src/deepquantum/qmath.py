"""
Common functions
"""

import copy
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn, vmap
from tqdm import tqdm

if TYPE_CHECKING:
    from .layer import Observable


def is_power_of_two(n: int) -> bool:
    """Check if an integer is a power of two."""
    def f(x):
        if x < 2:
            return False
        elif x & (x-1) == 0:
            return True
        return False

    return np.vectorize(f)(n)


def is_power(n: int, base: int) -> bool:
    """Check if an integer is a power of the given base."""
    if n <= 0 or base <= 0 or base == 1:
        return False
    if n == 1:
        return True
    while n % base == 0:
        n //= base
    return n == 1


def int_to_bitstring(x: int, n: int, debug: bool = False) -> str:
    """Convert from integer to bit string."""
    assert isinstance(x, int)
    assert isinstance(n, int)
    if x < 2 ** n:
        # remove '0b'
        s = bin(x)[2:]
        if len(s) <= n:
            s = '0' * (n - len(s)) + s
    else:
        if debug:
            print(f'Quantum register ({n}) overflowed for {x}.')
        s = bin(x)[-n:]
    return s


def list_to_decimal(digits: List[int], base: int) -> int:
    """Convert from list of digits to decimal integer."""
    result = 0
    for digit in digits:
        assert 0 <= digit < base, 'Invalid digit for the given base'
        result = result * base + digit
    return result


def decimal_to_list(n: int, base: int, ndigit: Optional[int] = None) -> List[int]:
    """Convert from decimal integer to list of digits."""
    assert base >= 2, 'Base must be at least 2'
    if n == 0:
        if isinstance(ndigit, int):
            return [0] * ndigit
        else:
            return [0]
    digits = []
    num = abs(n)
    while num > 0:
        num, remainder = divmod(num, base)
        digits.insert(0, remainder)
    if ndigit is not None:
        digits = [0] * (ndigit - len(digits)) + digits
    return digits


def inverse_permutation(permute_shape: List[int]) -> List[int]:
    """Calculate the inversed permutation.

    Args:
        permute_shape (List[int]): Shape of permutation.

    Returns:
        List[int]: A list of integers that is the inverse of ``permute_shape``.
    """
    # find the index of each element in the range of the list length
    return [permute_shape.index(i) for i in range(len(permute_shape))]


def is_unitary(matrix: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-4) -> bool:
    """Check if a tensor is a unitary matrix.

    Args:
        matrix (torch.Tensor): Square matrix.

    Returns:
        bool: ``True`` if ``matrix`` is unitary, ``False`` otherwise.
    """
    if matrix.shape[-1] != matrix.shape[-2]:
        return False
    conj_trans = matrix.t().conj()
    product = torch.matmul(matrix, conj_trans)
    return torch.allclose(product, torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device),
                          rtol=rtol, atol=atol)


def is_density_matrix(rho: torch.Tensor) -> bool:
    """Check if a tensor is a valid density matrix.

    A density matrix is a positive semi-definite Hermitian matrix with trace one.

    Args:
        rho (torch.Tensor): The tensor to check. It can be either 2D or 3D. If 3D, the first dimension
            is assumed to be the batch dimension.

    Returns:
        bool: ``True`` if the tensor is a density matrix, ``False`` otherwise.
    """
    if not isinstance(rho, torch.Tensor):
        return False
    if rho.ndim not in (2, 3):
        return False
    if not is_power_of_two(rho.shape[-2]):
        return False
    if not is_power_of_two(rho.shape[-1]):
        return False
    if rho.ndim == 2:
        rho = rho.unsqueeze(0)
    # Check if the tensor is Hermitian
    hermitian = torch.allclose(rho, rho.mH)
    if not hermitian:
        return False
    # Check if the trace of each matrix is one
    trace_one = torch.allclose(vmap(torch.trace)(rho), torch.tensor(1.0, dtype=rho.dtype, device=rho.device))
    if not trace_one:
        return False
    # Check if the eigenvalues of each matrix are non-negative
    positive_semi_definite = torch.all(torch.linalg.eig(rho)[0].real >= 0).item()
    if not positive_semi_definite:
        return False
    return True


def is_positive_definite(mat: torch.Tensor) -> bool:
    """Check if the matrix is positive definite"""
    is_herm = torch.equal(mat, mat.mH)
    diag = torch.linalg.eigvalsh(mat)
    return is_herm and torch.all(diag > 0).item()


def safe_inverse(x: Any, epsilon: float = 1e-12) -> Any:
    """Safe inversion."""
    return x / (x ** 2 + epsilon)


class SVD(torch.autograd.Function):
    """Customized backward of SVD for better numerical stability.

    Modified from https://github.com/wangleiphy/tensorgrad/blob/master/tensornets/adlib/svd.py
    See https://readpaper.com/paper/2971614414
    """
    generate_vmap_rule = True

    # pylint: disable=arguments-renamed
    @staticmethod
    def forward(a):
        u, s, vh = torch.linalg.svd(a, full_matrices=False)
        s = s.to(u.dtype)
        # ctx.save_for_backward(u, s, vh)
        return u, s, vh

    # setup_context is responsible for calling methods and/or assigning to
    # the ctx object. Please do not do additional compute (e.g. add
    # Tensors together) in setup_context.
    # https://pytorch.org/docs/master/notes/extending.func.html
    @staticmethod
    def setup_context(ctx, inputs, output):
        # a = inputs
        u, s, vh = output
        ctx.save_for_backward(u, s, vh)

    @staticmethod
    def backward(ctx, du, ds, dvh):
        u, s, vh = ctx.saved_tensors
        uh = u.mH
        v = vh.mH
        dv = dvh.mH
        m = u.shape[-2]
        n = v.shape[-2]
        ns = s.shape[-1]

        f = s.unsqueeze(-2) ** 2 - s.unsqueeze(-1) ** 2
        f = safe_inverse(f)
        f.diagonal(dim1=-2, dim2=-1).fill_(0)

        j = f * (uh @ du)
        k = f * (vh @ dv)
        l = (vh @ dv).diagonal(dim1=-2, dim2=-1).diag_embed()
        s_inv = safe_inverse(s).diag_embed()
        # pylint: disable=line-too-long
        da = u @ (ds.diag_embed() + (j + j.mH) @ s.diag_embed() + s.diag_embed() @ (k + k.mH) + s_inv @ (l.mH - l) / 2) @ vh
        if m > ns:
            da += (torch.eye(m, dtype=du.dtype, device=du.device) - u @ uh) @ du @ s_inv @ vh
        if n > ns:
            da += u @ s_inv @ dvh @ (torch.eye(n, dtype=du.dtype, device=du.device) - v @ vh)
        return da


# from tensorcircuit
def torchqr_grad(a, q, r, dq, dr):
    """Get the gradient for QR."""
    qr_epsilon = 1e-8

    if r.shape[-2] > r.shape[-1] and q.shape[-2] == q.shape[-1]:
        raise NotImplementedError(
            'QrGrad not implemented when nrows > ncols '
            'and full_matrices is true. Received r.shape='
            f'{r.shape} with nrows={r.shape[-2]}'
            f'and ncols={r.shape[-1]}.'
        )

    def _triangular_solve(x, r):
        """Equivalent to matmul(x, adjoint(matrix_inverse(r))) if r is upper-tri."""
        return torch.linalg.solve_triangular(
            r, x.adjoint(), upper=True, unitriangular=False
        ).adjoint()

    def _qr_grad_square_and_deep_matrices(q, r, dq, dr):
        """Get the gradient for matrix orders num_rows >= num_cols and full_matrices is false."""

        # Modification begins
        rdiag = torch.linalg.diagonal(r)
        # if abs(rdiag[i]) < qr_epsilon then rdiag[i] = qr_epsilon otherwise keep the old value
        qr_epsilon_diag = torch.ones_like(rdiag) * qr_epsilon
        rdiag = torch.where(rdiag.abs() < qr_epsilon, qr_epsilon_diag, rdiag)
        r = torch.diagonal_scatter(r, rdiag, dim1=-2, dim2=-1)
        # delta_dq = math_ops.matmul(q, math_ops.matmul(dr, tf.linalg.adjoint(delta_r)))
        # dq = dq + delta_dq
        # Modification ends

        qdq = torch.matmul(q.adjoint(), dq)
        qdq_ = qdq - qdq.adjoint()
        rdr = torch.matmul(r, dr.adjoint())
        rdr_ = rdr - rdr.adjoint()
        tril = torch.tril(qdq_ + rdr_)

        grad_a = torch.matmul(q, dr + _triangular_solve(tril, r))
        grad_b = _triangular_solve(dq - torch.matmul(q, qdq), r)
        ret = grad_a + grad_b

        if q.is_complex():
            m = rdr - qdq.adjoint()
            eyem = torch.diagonal_scatter(
                torch.zeros_like(m), torch.linalg.diagonal(m), dim1=-2, dim2=-1
            )
            correction = eyem - torch.real(eyem).to(dtype=q.dtype)
            ret = ret + _triangular_solve(torch.matmul(q, correction.adjoint()), r)

        return ret

    num_rows, num_cols = q.shape[-2], r.shape[-1]

    if num_rows >= num_cols:
        return _qr_grad_square_and_deep_matrices(q, r, dq, dr)

    y = a[..., :, num_rows:]
    u = r[..., :, :num_rows]
    dv = dr[..., :, num_rows:]
    du = dr[..., :, :num_rows]
    dy = torch.matmul(q, dv)
    dx = _qr_grad_square_and_deep_matrices(q, u, dq + torch.matmul(y, dv.adjoint()), du)
    return torch.cat([dx, dy], dim=-1)


# from tensorcircuit
class QR(torch.autograd.Function):
    """Customized backward of QR for better numerical stability."""
    generate_vmap_rule = True

    # pylint: disable=arguments-renamed
    @staticmethod
    def forward(a):
        q, r = torch.linalg.qr(a, mode='reduced')
        # ctx.save_for_backward(a, q, r)
        return q, r

    # setup_context is responsible for calling methods and/or assigning to
    # the ctx object. Please do not do additional compute (e.g. add
    # Tensors together) in setup_context.
    # https://pytorch.org/docs/master/notes/extending.func.html
    @staticmethod
    def setup_context(ctx, inputs, output):
        (a,) = inputs
        q, r = output
        # Tensors must be saved via ctx.save_for_backward. Please do not
        # assign them directly onto the ctx object.
        ctx.save_for_backward(a, q, r)
        # Non-tensors may be saved by assigning them as attributes on the ctx object.
        # ctx.dim = dim

    @staticmethod
    def backward(ctx, dq, dr):
        a, q, r = ctx.saved_tensors
        return torchqr_grad(a, q, r, dq, dr)


svd = SVD.apply
qr = QR.apply

def split_tensor(tensor: torch.Tensor, center_left: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a tensor by QR."""
    if center_left:
        q, r = qr(tensor.mH)
        return r.mH, q.mH
    else:
        return qr(tensor)


def state_to_tensors(state: torch.Tensor, nsite: int, qudit: int = 2) -> List[torch.Tensor]:
    """Convert a quantum state to a list of tensors."""
    state = state.reshape([qudit] * nsite)
    tensors = []
    nleft = 1
    for _ in range(nsite - 1):
        u, state = split_tensor(state.reshape(nleft * qudit, -1), center_left=False)
        tensors.append(u.reshape(nleft, qudit, -1))
        nleft = state.shape[0]
    u, state = split_tensor(state.reshape(nleft * qudit, -1), center_left=False)
    assert state.shape == (1, 1)
    tensors.append(u.reshape(nleft, qudit, -1) * state[0, 0])
    return tensors


def slice_state_vector(
    state: torch.Tensor,
    nqubit: int,
    wires: List[int],
    bits: str,
    normalize: bool = True
) -> torch.Tensor:
    """Get the sliced state vectors according to ``wires`` and ``bits``."""
    if len(bits) == 1:
        bits = bits * len(wires)
    assert len(wires) == len(bits)
    wires = [i + 1 for i in wires]
    state = state.reshape([-1] + [2] * nqubit)
    batch = state.shape[0]
    permute_shape = list(range(nqubit + 1))
    for i in wires:
        permute_shape.remove(i)
    permute_shape = wires + permute_shape
    state = state.permute(permute_shape)
    for b in bits:
        b = int(b)
        assert b in (0, 1)
        state = state[b]
    state = state.reshape(batch, -1)
    if normalize:
        state = nn.functional.normalize(state, p=2, dim=-1)
    return state


def multi_kron(lst: List[torch.Tensor]) -> torch.Tensor:
    """Calculate the Kronecker/tensor/outer product for a list of tensors.

    Args:
        lst (List[torch.Tensor]): A list of tensors.

    Returns:
        torch.Tensor: The Kronecker/tensor/outer product of the input.
    """
    n = len(lst)
    if n == 1:
        return lst[0].contiguous()
    else:
        mid = n // 2
        rst = torch.kron(multi_kron(lst[0:mid]), multi_kron(lst[mid:]))
        return rst.contiguous()


def partial_trace(rho: torch.Tensor, nqudit: int, trace_lst: List[int], qudit: int = 2) -> torch.Tensor:
    r"""Calculate the partial trace for a batch of density matrices.

    Args:
        rho (torch.Tensor): Density matrices with the shape of
            :math:`(\text{batch}, \text{qudit}^{\text{nqudit}}, \text{qudit}^{\text{nqudit}})`.
        nqudit (int): Total number of qudits.
        trace_lst (List[int]): A list of qudits to be traced.
        qudit (int, optional): The dimension of the qudits. Default: 2

    Returns:
        torch.Tensor: Reduced density matrices.
    """
    if rho.ndim == 2:
        rho = rho.unsqueeze(0)
    assert rho.ndim == 3
    assert rho.shape[1] == rho.shape[2] == qudit ** nqudit
    b = rho.shape[0]
    n = len(trace_lst)
    trace_lst = [i + 1 for i in trace_lst]
    trace_lst2 = [i + nqudit for i in trace_lst]
    trace_lst += trace_lst2
    permute_shape = list(range(2 * nqudit + 1))
    for i in trace_lst:
        permute_shape.remove(i)
    permute_shape += trace_lst
    rho = rho.reshape([b] + [qudit] * 2 * nqudit).permute(permute_shape).reshape(-1, qudit ** n, qudit ** n)
    rho = rho.diagonal(dim1=-2, dim2=-1).sum(-1)
    return rho.reshape(b, qudit ** (nqudit - n), qudit ** (nqudit - n)).squeeze(0)


def amplitude_encoding(data: Any, nqubit: int) -> torch.Tensor:
    r"""Encode data into quantum states using amplitude encoding.

    This function takes a batch of data and encodes each sample into a quantum state using amplitude encoding.
    The quantum state is represented by a complex-valued tensor of shape :math:`(\text{batch}, 2^{\text{nqubit}})`.
    The data is normalized to have unit norm along the last dimension before encoding. If the data size is smaller
    than :math:`2^{\text{nqubit}}`, the remaining amplitudes are set to zero. If the data size is larger than
    :math:`2^{\text{nqubit}}`, only the first :math:`2^{\text{nqubit}}` elements are used.

    Args:
        data (torch.Tensor or array-like): The input data to be encoded. It should have shape
            :math:`(\text{batch}, ...)` where :math:`...` can be any dimensions. If it is not a torch.Tensor object,
            it will be converted to one.
        nqubit (int): The number of qubits to use for encoding.

    Returns:
        torch.Tensor: The encoded quantum states as complex-valued tensors of shape
        :math:`(\text{batch}, 2^{\text{nqubit}}, 1)`.

    Examples:
        >>> data = [[0.5, 0.5], [0.7, 0.3]]
        >>> amplitude_encoding(data, nqubit=2)
        tensor([[[0.7071+0.j],
                [0.7071+0.j],
                [0.0000+0.j],
                [0.0000+0.j]],
                [[0.9487+0.j],
                [0.3162+0.j],
                [0.0000+0.j],
                [0.0000+0.j]]])
    """
    if not isinstance(data, (torch.Tensor, nn.Parameter)):
        data = torch.tensor(data)
    if data.ndim == 1 or (data.ndim == 2 and data.shape[-1] == 1):
        batch = 1
    else:
        batch = data.shape[0]
    data = data.reshape(batch, -1)
    size = data.shape[1]
    n = 2 ** nqubit
    state = torch.zeros(batch, n, dtype=data.dtype, device=data.device)
    data = nn.functional.normalize(data[:, :n], p=2, dim=-1)
    if n > size:
        state[:, :size] = data[:, :]
    else:
        state[:, :] = data[:, :]
    return state.unsqueeze(-1)


def evolve_state(
    state: torch.Tensor,
    matrix: torch.Tensor,
    nqudit: int,
    wires: List[int],
    qudit: int = 2
) -> torch.Tensor:
    """Perform the evolution of quantum states.

    Args:
        state (torch.Tensor): The batched state tensor.
        matrix (torch.Tensor): The evolution matrix.
        nqudit (int): The number of the qudits.
        wires (List[int]): The indices of the qudits that the quantum operation acts on.
        qudit (int, optional): The dimension of the qudits. Default: 2
    """
    nt = len(wires)
    wires = [i + 1 for i in wires]
    pm_shape = list(range(nqudit + 1))
    for i in wires:
        pm_shape.remove(i)
    pm_shape = wires + pm_shape
    state = state.permute(pm_shape).reshape(qudit ** nt, -1)
    state = (matrix @ state).reshape([qudit] * nt + [-1] + [qudit] * (nqudit - nt))
    state = state.permute(inverse_permutation(pm_shape))
    return state


def evolve_den_mat(
    state: torch.Tensor,
    matrix: torch.Tensor,
    nqudit: int,
    wires: List[int],
    qudit: int = 2
) -> torch.Tensor:
    """Perform the evolution of density matrices.

    Args:
        state (torch.Tensor): The batched state tensor.
        matrix (torch.Tensor): The evolution matrix.
        nqudit (int): The number of the qudits.
        wires (List[int]): The indices of the qudits that the quantum operation acts on.
        qudit (int, optional): The dimension of the qudits. Default: 2
    """
    nt = len(wires)
    # left multiply
    wires1 = [i + 1 for i in wires]
    pm_shape = list(range(2 * nqudit + 1))
    for i in wires1:
        pm_shape.remove(i)
    pm_shape = wires1 + pm_shape
    state = state.permute(pm_shape).reshape(qudit ** nt, -1)
    state = (matrix @ state).reshape([qudit] * nt + [-1] + [qudit] * (2 * nqudit - nt))
    state = state.permute(inverse_permutation(pm_shape))
    # right multiply
    wires2 = [i + 1 + nqudit for i in wires]
    pm_shape = list(range(2 * nqudit + 1))
    for i in wires2:
        pm_shape.remove(i)
    pm_shape = wires2 + pm_shape
    state = state.permute(pm_shape).reshape(qudit ** nt, -1)
    state = (matrix.conj() @ state).reshape([qudit] * nt + [-1] + [qudit] * (2 * nqudit - nt))
    state = state.permute(inverse_permutation(pm_shape))
    return state


def block_sample(probs: torch.Tensor, shots: int = 1024, block_size: int = 2 ** 24) -> List:
    """Sample from a probability distribution using block sampling.

    Args:
        probs (torch.Tensor): The probability distribution to sample from.
        shots (int, optional): The number of samples to draw. Default: 1024
        block_size (int, optional): The block size for sampling. Default: 2 ** 24
    """
    samples = []
    num_blocks = int(np.ceil(len(probs) / block_size))
    probs_block = torch.zeros(num_blocks, device=probs.device)
    start = (num_blocks - 1) * block_size
    end = min(num_blocks * block_size, len(probs))
    probs_block[:-1] = probs[:start].reshape(num_blocks - 1, block_size).sum(1)
    probs_block[-1] = probs[start:end].sum()
    blocks = torch.multinomial(probs_block, shots, replacement=True).cpu().numpy()
    block_dict = Counter(blocks)
    for idx_block, shots_block in block_dict.items():
        start = idx_block * block_size
        end = min((idx_block + 1) * block_size, len(probs))
        samples_block = torch.multinomial(probs[start:end], shots_block, replacement=True)
        samples.extend((samples_block + start).cpu().numpy())
    return samples


def measure(
    state: torch.Tensor,
    shots: int = 1024,
    with_prob: bool = False,
    wires: Union[int, List[int], None] = None,
    den_mat: bool = False,
    block_size: int = 2 ** 24
) -> Union[Dict, List[Dict]]:
    r"""A function that performs a measurement on a quantum state and returns the results.

    The measurement is done by sampling from the probability distribution of the quantum state. The results
    are given as a dictionary or a list of dictionaries, where each key is a bit string representing the
    measurement outcome, and each value is either the number of occurrences or a tuple of the number of
    occurrences and the probability.

    Args:
        state (torch.Tensor): The quantum state to measure. It can be a tensor of shape :math:`(2^n,)` or
            :math:`(2^n, 1)` representing a state vector, or a tensor of shape :math:`(\text{batch}, 2^n)`
            or :math:`(\text{batch}, 2^n, 1)` representing a batch of state vectors. It can also be a tensor
            of shape :math:`(2^n, 2^n)` representing a density matrix or :math:`(\text{batch}, 2^n, 2^n)`
            representing a batch of density matrices.
        shots (int, optional): The number of times to sample from the quantum state. Default: 1024
        with_prob (bool, optional): A flag that indicates whether to return the probabilities along with
            the number of occurrences. Default: ``False``
        wires (int, List[int] or None, optional): The wires to measure. It can be an integer or a list of
            integers specifying the indices of the wires. Default: ``None`` (which means all wires are
            measured)
        den_mat (bool, optional): Whether the state is a density matrix or not. Default: ``False``
        block_size (int, optional): The block size for sampling. Default: 2 ** 24

    Returns:
        Union[Dict, List[Dict]]: The measurement results. If the state is a single state vector, it returns
        a dictionary where each key is a bit string representing the measurement outcome, and each value
        is either the number of occurrences or a tuple of the number of occurrences and the probability.
        If the state is a batch of state vectors, it returns a list of dictionaries with the same format
        for each state vector in the batch.
    """
    if den_mat:
        assert is_density_matrix(state), 'Please input density matrices'
        state = state.diagonal(dim1=-2, dim2=-1)
    if state.ndim == 1 or (state.ndim == 2 and state.shape[-1] == 1):
        batch = 1
    else:
        batch = state.shape[0]
    state = state.reshape(batch, -1)
    assert is_power_of_two(state.shape[-1]), 'The length of the quantum state is not in the form of 2^n'
    n = int(np.log2(state.shape[-1]))
    if wires is not None:
        if isinstance(wires, int):
            wires = [wires]
        assert isinstance(wires, list)
        wires = sorted(wires)
        pm_shape = list(range(n))
        for w in wires:
            pm_shape.remove(w)
        pm_shape = wires + pm_shape
    num_bits = len(wires) if wires else n
    results_tot = []
    for i in range(batch):
        if den_mat:
            probs = torch.abs(state[i])
        else:
            probs = torch.abs(state[i]) ** 2
        if wires is not None:
            probs = probs.reshape([2] * n).permute(pm_shape).reshape([2] * len(wires) + [-1]).sum(-1).reshape(-1)
        # Perform block sampling to reduce memory consumption
        samples = Counter(block_sample(probs, shots, block_size))
        results = {bin(key)[2:].zfill(num_bits): value for key, value in samples.items()}
        if with_prob:
            for k in results:
                index = int(k, 2)
                results[k] = results[k], probs[index]
        results_tot.append(results)
    if batch == 1:
        return results_tot[0]
    else:
        return results_tot


def sample_sc_mcmc(
    prob_func: Callable,
    proposal_sampler: Callable,
    shots: int = 1024,
    num_chain: int = 5
) -> defaultdict:
    """Get the samples of the probability distribution function via SC-MCMC method."""
    samples_chain = []
    merged_samples = defaultdict(int)
    cache_prob = {}
    if shots <= 0:
        return merged_samples
    elif shots < num_chain:
        num_chain = shots
    shots_lst = [shots // num_chain] * num_chain
    shots_lst[-1] += shots % num_chain
    for trial in range(num_chain):
        cache = []
        len_cache = min(shots_lst)
        if shots_lst[trial] > 1e5:
            len_cache = 4000
        # random start
        sample_0 = proposal_sampler()
        if not isinstance(sample_0, str):
            if prob_func(sample_0) < 1e-12: # avoid the samples with almost-zero probability
                sample_0 = tuple([0] * len(sample_0))
            while prob_func(sample_0) < 1e-9:
                sample_0 = proposal_sampler()
        cache.append(sample_0)
        sample_max = sample_0
        if sample_max in cache_prob:
            prob_max = cache_prob[sample_max]
        else:
            prob_max = prob_func(sample_0)
            cache_prob[sample_max] = prob_max
        dict_sample = defaultdict(int)
        for i in tqdm(range(1, shots_lst[trial]), desc=f'chain {trial+1}', ncols=80, colour='green'):
            sample_i = proposal_sampler()
            if sample_i in cache_prob:
                prob_i = cache_prob[sample_i]
            else:
                prob_i = prob_func(sample_i)
                cache_prob[sample_i] = prob_i
            rand_num = torch.rand(1, device=prob_i.device)
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
                out_sample_key = out_sample
                if out_sample_key in dict_sample:
                    dict_sample[out_sample_key] = dict_sample[out_sample_key] + 1
                else:
                    dict_sample[out_sample_key] = 1
        # clear the cache
        for i in range(len_cache):
            out_sample = cache[i]
            out_sample_key = out_sample
            if out_sample_key in dict_sample:
                dict_sample[out_sample_key] = dict_sample[out_sample_key] + 1
            else:
                dict_sample[out_sample_key] = 1
        samples_chain.append(dict_sample)
        for key, value in dict_sample.items():
            merged_samples[key] += value
    return merged_samples


def get_prob_mps(mps_lst: List[torch.Tensor], wire: int) -> torch.Tensor:
    """Calculate the probability distribution (|0⟩ and |1⟩ probabilities) for a specific wire in an MPS.

    This function computes the probability of measuring |0⟩ and |1⟩ for the k-th qubit in a quantum state
    represented as a Matrix Product State (MPS). It does this by:
    1. Contracting the tensors to the left of the target tensor
    2. Contracting the tensors to the right of the target tensor
    3. Computing the final contraction with the target tensor

    Args:
        wire (int): Index of the target qubit to compute probabilities for
        mps_lst (List[torch.Tensor]): List of MPS tensors representing the quantum state
            Each 3-dimensional tensor should have shape (bond_dim_left, physical_dim, bond_dim_right)

    Returns:
        torch.Tensor: A tensor containing [P(|0⟩), P(|1⟩)] probabilities for the target qubit
    """
    def contract_conjugate_pair(tensors: List[torch.Tensor]) -> torch.Tensor:
        """Contract a list of MPS tensors with their conjugates.

        This helper function performs the contraction between a list of MPS tensors
        and their complex conjugates, which is needed for probability calculation.

        Args:
            tensors (List[torch.Tensor]): List of MPS tensors to contract

        Returns:
            torch.Tensor: Contracted tensor
        """
        if not tensors:  # Handle empty tensor list case
            return torch.tensor(1).reshape(1, 1, 1, 1).to(mps_lst[0].dtype).to(mps_lst[0].device)

        # Contract first tensor with its conjugate
        contracted = torch.tensordot(tensors[0].conj(), tensors[0], dims=([1], [1]))
        contracted = contracted.permute(0, 2, 1, 3) # (left_c, left, right_c, right)

        # Iteratively contract remaining tensors
        for tensor in tensors[1:]:
            pair_contracted = torch.tensordot(tensor.conj(), tensor, dims=([1], [1]))
            pair_contracted = pair_contracted.permute(0, 2, 1, 3)
            contracted = torch.tensordot(contracted, pair_contracted, dims=([2, 3], [0, 1]))

        return contracted

    # Split MPS into left and right parts relative to target qubit
    left_tensors = mps_lst[:wire] if wire > 0 else []
    right_tensors = mps_lst[wire + 1:] if wire < len(mps_lst) - 1 else []
    target_tensor = mps_lst[wire]

    # Contract left and right parts separately
    left_contracted = contract_conjugate_pair(left_tensors)
    right_contracted = contract_conjugate_pair(right_tensors)

    # Perform final contractions with target qubit tensor
    temp1 = torch.tensordot(left_contracted, target_tensor.conj(), dims=([2], [0]))
    temp2 = torch.tensordot(temp1, target_tensor, dims=([2], [0]))
    final_tensor = torch.tensordot(right_contracted, temp2, dims=([0, 1], [3, 5])).squeeze()

    # Extract probabilities from diagonal elements
    probabilities = final_tensor.diagonal().real
    return torch.clamp(probabilities, min=0)  # Returns [P(|0⟩), P(|1⟩)]


def inner_product_mps(
    tensors0: List[torch.Tensor],
    tensors1: List[torch.Tensor],
    form: str = 'norm'
) -> Union[torch.Tensor, List[torch.Tensor]]:
    r"""Computes the inner product of two matrix product states.

    Args:
        tensors0 (List[torch.Tensor]): The tensors of the first MPS, each with shape :math:`(..., d_0, d_1, d_2)`,
            where :math:`d_0` is the bond dimension of the left site, :math:`d_1` is the physical dimension,
            and :math:`d_2` is the bond dimension of the right site.
        tensors1 (List[torch.Tensor]): The tensors of the second MPS, each with shape :math:`(..., d_0, d_1, d_2)`,
            where :math:`d_0` is the bond dimension of the left site, :math:`d_1` is the physical dimension,
            and :math:`d_2` is the bond dimension of the right site.
        form (str, optional): The form of the output. If ``'log'``, returns the logarithm of the absolute value
            of the inner product. If ``'list'``, returns a list of norms at each step. Otherwise, returns the
            inner product as a scalar. Default: ``'norm'``

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: The inner product of the two MPS, or a list of norms at each step.

    Raises:
        AssertionError: If the tensors have incompatible shapes or lengths.
    """
    assert tensors0[0].shape[-3] == tensors0[-1].shape[-1]
    assert tensors1[0].shape[-3] == tensors1[-1].shape[-1]
    assert len(tensors0) == len(tensors1)

    v0 = torch.eye(tensors0[0].shape[-3], dtype=tensors0[0].dtype, device=tensors0[0].device)
    v1 = torch.eye(tensors1[0].shape[-3], dtype=tensors0[0].dtype, device=tensors0[0].device)
    v = torch.kron(v0, v1).reshape([tensors0[0].shape[-3], tensors1[0].shape[-3],
                                    tensors0[0].shape[-3], tensors1[0].shape[-3]])
    norm_list = []
    for n in range(len(tensors0)):
        v = torch.einsum('...uvap,...adb,...pdq->...uvbq', v, tensors0[n].conj(), tensors1[n])
        norm_v = v.norm(p=2, dim=[-4,-3,-2,-1], keepdim=True)
        v = v / norm_v
        norm_list.append(norm_v.squeeze())
    if v.numel() > 1:
        norm1 = torch.einsum('...acac->...', v)
        norm_list.append(norm1)
    else:
        norm_list.append(v[0, 0, 0, 0])
    if form == 'log':
        norm = 0.0
        for x in norm_list:
            norm = norm + torch.log(x.abs())
    elif form == 'list':
        return norm_list
    else:
        norm = 1.0
        for x in norm_list:
            norm = norm * x
    return norm


def expectation(
    state: Union[torch.Tensor, List[torch.Tensor]],
    observable: 'Observable',
    den_mat: bool = False,
    chi: Optional[int] = None
) -> torch.Tensor:
    """A function that calculates the expectation value of an observable on a quantum state.

    The expectation value is the average measurement outcome of the observable on the quantum state.
    It is a real number that represents the mean of the probability distribution of the measurement outcomes.

    Args:
        state (torch.Tensor or List[torch.Tensor]): The quantum state to measure. It can be a list of tensors
            representing a matrix product state, or a tensor representing a density matrix or a state vector.
        observable (Observable): The observable to measure. It is an instance of ``Observable`` class that
            implements the measurement basis and the corresponding gates.
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        chi (int or None, optional): The bond dimension of the matrix product state. It is only used
            when the state is a list of tensors. Default: ``None`` (which means no truncation)

    Returns:
        torch.Tensor: The expectation value of the observable on the quantum state. It is a scalar tensor
        with real values.
    """
    # pylint: disable=import-outside-toplevel
    if isinstance(state, list):
        from .state import MatrixProductState
        mps = MatrixProductState(nsite=len(state), state=state, chi=chi)
        return inner_product_mps(state, observable(mps).tensors).real
    if den_mat:
        expval = (observable.get_unitary() @ state).diagonal(dim1=-2, dim2=-1).sum(-1).real
    else:
        expval = state.mH @ observable(state)
        expval = expval.squeeze(-1).squeeze(-1).real
    return expval


def sample2expval(sample: Dict) -> torch.Tensor:
    """Get the expectation value according to the measurement results."""
    total = 0
    exp = 0
    for bitstring, ncount in sample.items():
        coeff = (-1) ** (bitstring.count('1') % 2)
        exp += ncount * coeff
        total += ncount
    return torch.tensor([exp / total])


def meyer_wallach_measure(state_tsr: torch.Tensor) -> torch.Tensor:
    r"""Calculate Meyer-Wallach entanglement measure.

    See https://readpaper.com/paper/2945680873 Eq.(19)

    Args:
        state_tsr (torch.Tensor): Input with the shape of :math:`(\text{batch}, 2, ..., 2)`.

    Returns:
        torch.Tensor: The value of Meyer-Wallach measure.
    """
    nqubit = len(state_tsr.shape) - 1
    batch = state_tsr.shape[0]
    rst = 0
    for i in range(nqubit):
        s1 = linear_map_mw(state_tsr, i, 0).reshape(batch, -1, 1)
        s2 = linear_map_mw(state_tsr, i, 1).reshape(batch, -1, 1)
        rst += generalized_distance(s1, s2).reshape(-1)
    return rst * 4 / nqubit


def linear_map_mw(state_tsr: torch.Tensor, j: int, b: int) -> torch.Tensor:
    r"""Calculate the linear mapping for Meyer-Wallach measure.

    See https://readpaper.com/paper/2945680873 Eq.(18)

    Note:
        Project on state with local projectors on the ``j`` th qubit.
        See https://arxiv.org/pdf/quant-ph/0305094.pdf Eq.(2)

    Args:
        state_tsr (torch.Tensor): Input with the shape of :math:`(\text{batch}, 2, ..., 2)`.
        j (int): The ``j`` th qubit to project on, from :math:`0` to :math:`\text{nqubit}-1`.
        b (int): The basis of projection, :math:`\ket{0}` or :math:`\ket{1}`.

    Returns:
        torch.Tensor: Non-normalized state tensor after the linear mapping.
    """
    assert b in (0, 1), 'b must be 0 or 1'
    n = len(state_tsr.shape)
    assert j < n - 1, 'j can not exceed nqubit'
    permute_shape = list(range(n))
    permute_shape.remove(j + 1)
    permute_shape = [0] + [j + 1] + permute_shape[1:]
    return state_tsr.permute(permute_shape)[:, b]


def generalized_distance(state1: torch.Tensor, state2: torch.Tensor) -> torch.Tensor:
    r"""Calculate the generalized distance.

    See https://readpaper.com/paper/2945680873 Eq.(20)

    Note:
        Implemented according to https://arxiv.org/pdf/quant-ph/0310137.pdf Eq.(4)

    Args:
        state1 (torch.Tensor): Input with the shape of :math:`(\text{batch}, 2^n, 1)`.
        state2 (torch.Tensor): Input with the shape of :math:`(\text{batch}, 2^n, 1)`.

    Returns:
        torch.Tensor: The generalized distance.
    """
    return ((state1.mH @ state1) * (state2.mH @ state2) - (state1.mH @ state2) * (state2.mH @ state1)).real


def meyer_wallach_measure_brennen(state_tsr: torch.Tensor) -> torch.Tensor:
    r"""Calculate Meyer-Wallach entanglement measure, proposed by Brennen.

    See https://arxiv.org/pdf/quant-ph/0305094.pdf Eq.(6)

    Note:
        This implementation is slower than ``meyer_wallach_measure`` when :math:`\text{nqubit} \ge 8`.

    Args:
        state_tsr (torch.Tensor): Input with the shape of :math:`(\text{batch}, 2, ..., 2)`.

    Returns:
        torch.Tensor: The value of Meyer-Wallach measure.
    """
    nqubit = len(state_tsr.shape) - 1
    batch = state_tsr.shape[0]
    rho = state_tsr.reshape(batch, -1, 1) @ state_tsr.conj().reshape(batch, 1, -1)
    rst = 0
    for i in range(nqubit):
        trace_list = list(range(nqubit))
        trace_list.remove(i)
        rho_i = partial_trace(rho, nqubit, trace_list)
        rho_i = rho_i @ rho_i
        trace_rho_i = rho_i.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1).real
        rst += trace_rho_i
    return 2 * (1 - rst / nqubit)
