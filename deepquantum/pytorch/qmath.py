import torch
from typing import List


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
    rho = rho.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
    return rho.reshape(b, 2 ** (N - n), 2 ** (N - n))


def Meyer_Wallach_measure(MPS: torch.Tensor) -> torch.Tensor:
    """Calculate Meyer-Wallach entanglement measure
    
    See https://readpaper.com/paper/2945680873 Eq.(19)
    
    Args:
        MPS: input with the shape of (batch, 2, ..., 2, 1)
    
    Returns:
        torch.Tensor: the value of Meyer-Wallach measure
    """
    nqubit = len(MPS.shape) - 2
    batch = MPS.shape[0]
    rst = 0
    for i in range(nqubit):
        s1 = linear_map_MW(MPS, i, 0).reshape(batch, -1, 1)
        s2 = linear_map_MW(MPS, i, 1).reshape(batch, -1, 1)
        rst += generalized_distance(s1, s2).reshape(-1)
    return rst * 4 / nqubit


def linear_map_MW(MPS: torch.Tensor, j: int, b: int) -> torch.Tensor:
    """Calculate the linear mapping for Meyer-Wallach measure

    See https://readpaper.com/paper/2945680873 Eq.(18)

    Note:
        Project on state `MPS` with local projectors on the `j`th qubit
        See https://arxiv.org/pdf/quant-ph/0305094.pdf Eq.(2)

    Args:
        MPS: input with the shape of (batch, 2, ..., 2, 1)
        j: the `j`th qubit to project on, from 0 to nqubit - 1
        b: the project basis, |0> or |1>
    
    Returns:
        torch.Tensor: non-normalized MPS after the linear mapping
    """
    assert b == 0 or b == 1, "b must be 0 or 1"
    n = len(MPS.shape)
    assert j < n - 2, "j can not exceed nqubit"
    permute_shape = list(range(n))
    permute_shape.remove(j + 1)
    permute_shape = [0] + [j + 1] + permute_shape[1:]
    return MPS.permute(permute_shape)[:, b]


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
    state1_dag = torch.conj(state1).transpose(-1,-2)
    state2_dag = torch.conj(state2).transpose(-1,-2)
    rst = torch.bmm(state1_dag, state1) * torch.bmm(state2_dag, state2) \
        - torch.bmm(state1_dag, state2) * torch.bmm(state2_dag, state1)
    return rst


def Meyer_Wallach_measure_Brennen(MPS: torch.Tensor) -> torch.Tensor:
    """Calculate Meyer-Wallach entanglement measure, proposed by Brennen
    
    See https://arxiv.org/pdf/quant-ph/0305094.pdf Eq.(6)

    Note:
        This implementation is slower than `Meyer_Wallach_measure` when nqubit >= 8
    
    Args:
        MPS: input with the shape of (batch, 2, ..., 2, 1)
    
    Returns:
        torch.Tensor: the value of Meyer-Wallach measure
    """
    nqubit = len(MPS.shape) - 2
    batch = MPS.shape[0]
    rho = MPS.reshape(batch, -1, 1) @ torch.conj(MPS.reshape(batch, 1, -1))
    rst = 0
    for i in range(nqubit):
        trace_list = list(range(nqubit))
        trace_list.remove(i)
        rho_i = partial_trace(rho, nqubit, trace_list)
        rho_i = rho_i @ rho_i
        trace_rho_i = rho_i.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
        rst += trace_rho_i
    return 2 * (1 - rst / nqubit)