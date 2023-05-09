import torch
import torch.nn as nn
import numpy as np
import random
from collections import Counter
from typing import List
from torch import vmap


def is_power_of_two(n):
    def f(x):
        if x < 2:
            return False
        elif x & (x-1) == 0:
            return True
        return False
    
    return np.vectorize(f)(n)


def inverse_permutation(permute_shape):
    # permute_shape is a list of integers
    # return a list of integers that is the inverse of permute_shape
    # find the index of each element in the range of the list length
    return [permute_shape.index(i) for i in range(len(permute_shape))]


def is_unitary(matrix):
    # matrix is a torch tensor of complex numbers
    # return True if matrix is unitary, False otherwise
    # calculate the conjugate transpose of matrix
    assert matrix.shape[-1] == matrix.shape[-2]
    conj_trans = matrix.t().conj()
    # calculate the product of matrix and conj_trans
    product = torch.matmul(matrix, conj_trans)
    # compare product and identity matrix using torch.allclose function
    return torch.allclose(product, torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device))


def is_density_matrix(rho: torch.Tensor) -> bool:
    """Check if a tensor is a valid density matrix.

    A density matrix is a positive semi-definite Hermitian matrix with trace one.

    Args:
        rho (torch.Tensor): The tensor to check. It can be either 2D or 3D. If 3D, the first dimension is assumed to be the batch dimension.

    Returns:
        bool: True if the tensor is a density matrix, False otherwise.

    Raises:
        AssertionError: If the tensor is not of type torch.Tensor or has an invalid number of dimensions.
    """
    assert type(rho) == torch.Tensor
    assert rho.ndim in (2, 3)
    assert is_power_of_two(rho.shape[-2]) and is_power_of_two(rho.shape[-1])
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


def safe_inverse(x, epsilon=1e-12):
    return x / (x ** 2 + epsilon)


class SVD(torch.autograd.Function):
    # modified from https://github.com/wangleiphy/tensorgrad/blob/master/tensornets/adlib/svd.py
    # See https://readpaper.com/paper/2971614414
    @staticmethod
    def forward(ctx, A):
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        S = S.to(U.dtype)
        ctx.save_for_backward(U, S, Vh)
        return U, S, Vh

    @staticmethod
    def backward(ctx, dU, dS, dVh):
        U, S, Vh = ctx.saved_tensors
        Uh = U.mH
        V = Vh.mH
        dV = dVh.mH
        m = U.shape[-2]
        n = V.shape[-2]
        ns = S.shape[-1]

        F = (S.unsqueeze(-2) ** 2 - S.unsqueeze(-1) ** 2)
        F = safe_inverse(F)
        F.diagonal(dim1=-2, dim2=-1).fill_(0)

        J = F * (Uh @ dU)
        K = F * (Vh @ dV)
        L = (Vh @ dV).diagonal(dim1=-2, dim2=-1).diag_embed()
        S_inv = safe_inverse(S).diag_embed()
        dA = U @ (dS.diag_embed() + (J + J.mH) @ S.diag_embed() + S.diag_embed() @ (K + K.mH) + S_inv @ (L.mH - L) / 2) @ Vh
        if (m > ns):
            dA += (torch.eye(m, dtype=dU.dtype, device=dU.device) - U @ Uh) @ dU @ S_inv @ Vh 
        if (n > ns):
            dA += U @ S_inv @ dVh @ (torch.eye(n, dtype=dU.dtype, device=dU.device) - V @ Vh)
        return dA


def multi_kron(lst: List[torch.Tensor]) -> torch.Tensor:
    """Calculate the Kronecker/tensor/outer product for a list of tensors
    
    Args:
        lst: a list of tensors
    
    Returns:
        torch.Tensor: the Kronecker/tensor/outer product of the input
    """
    n = len(lst)
    if n == 1:
        return lst[0]
    else:
        mid = n // 2
        rst = torch.kron(multi_kron(lst[0:mid]), multi_kron(lst[mid:]))
        return rst


def partial_trace(rho: torch.Tensor, N: int, trace_lst: List) -> torch.Tensor:
    """Calculate the partial trace for a batch of density matrices
    
    Args:
        rho: density matrices with the shape of (batch, 2^N, 2^N)
        N: total number of qubits
        trace_lst: a list of qubits to be traced
    
    Returns:
        torch.Tensor: reduced density matrices with the shape of (batch, 2^n, 2^n)
    """
    if rho.ndim == 2:
        rho = rho.unsqueeze(0)
    assert rho.ndim == 3
    assert rho.shape[1] == 2 ** N and rho.shape[2] == 2 ** N
    b = rho.shape[0]
    n = len(trace_lst)
    trace_lst = [i + 1 for i in trace_lst]
    trace_lst2 = [i + N for i in trace_lst]
    trace_lst = trace_lst + trace_lst2
    permute_shape = list(range(2 * N + 1))
    for i in trace_lst:
        permute_shape.remove(i)
    permute_shape = permute_shape + trace_lst
    rho = rho.reshape([b] + [2] * 2 * N).permute(permute_shape).reshape(-1, 2 ** n, 2 ** n)
    rho = rho.diagonal(dim1=-2, dim2=-1).sum(-1)
    return rho.reshape(b, 2 ** (N - n), 2 ** (N - n)).squeeze(0)


def amplitude_encoding(data, nqubit: int) -> torch.Tensor:
    """Encode data into quantum states using amplitude encoding.

    This function takes a batch of data and encodes each sample into a quantum state
    using amplitude encoding. The quantum state is represented by a complex-valued tensor
    of shape (batch_size, 2**nqubit). The data is normalized to have unit norm along the last dimension
    before encoding. If the data size is smaller than 2**nqubit, the remaining amplitudes are set to zero.
    If the data size is larger than 2**nqubit, only the first 2**nqubit elements are used.

    Parameters
    ----------
    data : torch.Tensor or array-like
        The input data to be encoded. It should have shape (batch_size, ...) where ... can be any dimensions.
        If it is not a torch.Tensor object, it will be converted to one.
    nqubit : int
        The number of qubits to use for encoding.

    Returns
    -------
    torch.Tensor
        The encoded quantum states as complex-valued tensors of shape (batch_size, 2**nqubit, 1).

    Examples
    --------
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
    if type(data) != torch.Tensor and type(data) != torch.nn.parameter.Parameter:
        data = torch.tensor(data)
    if data.ndim == 1 or (data.ndim == 2 and data.shape[-1] == 1):
        batch = 1
    else:
        batch = data.shape[0]
    data = data.reshape(batch, -1)
    size = data.shape[1]
    n = 2 ** nqubit
    state = torch.zeros(batch, n, dtype=torch.cfloat, device=data.device)
    data = nn.functional.normalize(data[:, :n], p=2, dim=-1)
    if n > size:
        state[:, :size] = data[:, :]
    else:
        state[:, :] = data[:, :size]
    return state.unsqueeze(-1)


def measure(state, shots=1024, with_prob=False, wires=None):
    if state.ndim == 1 or (state.ndim == 2 and state.shape[-1] == 1):
        batch = 1
    else:
        batch = state.shape[0]
    state = state.reshape(batch, -1)
    assert is_power_of_two(state.shape[-1]), 'The length of the quantum state is not in the form of 2^n'
    n = int(np.log2(state.shape[-1]))
    if wires == None:
        bit_strings = [format(i, f'0{n}b') for i in range(2 ** n)]
    else:
        assert type(wires) in (int, list)
        if type(wires) == int:
            wires = [wires]
        bit_strings = [format(i, f'0{len(wires)}b') for i in range(2 ** len(wires))]
    results_tot = []
    for i in range(batch):
        probs = torch.abs(state[i]) ** 2
        if wires != None:
            wires.sort()
            pm_shape = list(range(n))
            for w in wires:
                pm_shape.remove(w)
            pm_shape = wires + pm_shape
            probs = probs.reshape([2] * n).permute(pm_shape).reshape([2] * len(wires) + [-1]).sum(-1).reshape(-1)
        samples = random.choices(bit_strings, weights=probs, k=shots)
        results = dict(Counter(samples))
        if with_prob:
            for k in results:
                index = int(k, 2)
                results[k] = results[k], probs[index]
        results_tot.append(results)
    if batch == 1:
        return results_tot[0]
    else:
        return results_tot


def expectation(state, observable, den_mat=False):
    if den_mat:
        expval = (observable.get_unitary() @ state).diagonal(dim1=-2, dim2=-1).sum(-1).real
    else:
        expval = state.mH @ observable(state)
        expval = expval.squeeze(-1).squeeze(-1).real
    return expval


def inner_product_mps(tensors0, tensors1, form='norm'):
    # form: 'log' or 'list'
    assert tensors0[0].shape[0] == tensors0[-1].shape[-1]
    assert tensors1[0].shape[0] == tensors1[-1].shape[-1]
    assert len(tensors0) == len(tensors1)

    v0 = torch.eye(tensors0[0].shape[0], dtype=tensors0[0].dtype, device=tensors0[0].device)
    v1 = torch.eye(tensors1[0].shape[0], dtype=tensors0[0].dtype, device=tensors0[0].device)
    v = torch.kron(v0, v1).reshape([tensors0[0].shape[0], tensors1[0].shape[0],
                                    tensors0[0].shape[0], tensors1[0].shape[0]])
    norm_list = []
    for n in range(len(tensors0)):
        v = torch.einsum('...uvap,...adb,...pdq->...uvbq', v, tensors0[n].conj(), tensors1[n])
        norm_list.append(v.norm())
        v = v / norm_list[-1]
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


def Meyer_Wallach_measure(state_tsr: torch.Tensor) -> torch.Tensor:
    """Calculate Meyer-Wallach entanglement measure
    
    See https://readpaper.com/paper/2945680873 Eq.(19)
    
    Args:
        state_tsr: input with the shape of (batch, 2, ..., 2)
    
    Returns:
        torch.Tensor: the value of Meyer-Wallach measure
    """
    nqubit = len(state_tsr.shape) - 1
    batch = state_tsr.shape[0]
    rst = 0
    for i in range(nqubit):
        s1 = linear_map_MW(state_tsr, i, 0).reshape(batch, -1, 1)
        s2 = linear_map_MW(state_tsr, i, 1).reshape(batch, -1, 1)
        rst += generalized_distance(s1, s2).reshape(-1)
    return rst * 4 / nqubit


def linear_map_MW(state_tsr: torch.Tensor, j: int, b: int) -> torch.Tensor:
    """Calculate the linear mapping for Meyer-Wallach measure

    See https://readpaper.com/paper/2945680873 Eq.(18)

    Note:
        Project on state with local projectors on the `j`th qubit
        See https://arxiv.org/pdf/quant-ph/0305094.pdf Eq.(2)

    Args:
        state_tsr: input with the shape of (batch, 2, ..., 2)
        j: the `j`th qubit to project on, from 0 to nqubit - 1
        b: the project basis, |0> or |1>
    
    Returns:
        torch.Tensor: non-normalized state tensor after the linear mapping
    """
    assert b == 0 or b == 1, 'b must be 0 or 1'
    n = len(state_tsr.shape)
    assert j < n - 1, 'j can not exceed nqubit'
    permute_shape = list(range(n))
    permute_shape.remove(j + 1)
    permute_shape = [0] + [j + 1] + permute_shape[1:]
    return state_tsr.permute(permute_shape)[:, b]


def generalized_distance(state1: torch.Tensor, state2: torch.Tensor) -> torch.Tensor:
    """Calculate the generalized distance

    See https://readpaper.com/paper/2945680873 Eq.(20)
    Implemented according to https://arxiv.org/pdf/quant-ph/0310137.pdf Eq.(4)
    
    Args:
        state1: input with the shape of (batch, 2^n, 1)
        state2: input with the shape of (batch, 2^n, 1)

    Returns:
        torch.Tensor: the generalized distance
    """
    return ((state1.mH @ state1) * (state2.mH @ state2) - (state1.mH @ state2) * (state2.mH @ state1)).real


def Meyer_Wallach_measure_Brennen(state_tsr: torch.Tensor) -> torch.Tensor:
    """Calculate Meyer-Wallach entanglement measure, proposed by Brennen
    
    See https://arxiv.org/pdf/quant-ph/0305094.pdf Eq.(6)

    Note:
        This implementation is slower than `Meyer_Wallach_measure` when nqubit >= 8
    
    Args:
        state_tsr: input with the shape of (batch, 2, ..., 2)
    
    Returns:
        torch.Tensor: the value of Meyer-Wallach measure
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