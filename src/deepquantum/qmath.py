"""
Common functions
"""

import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn, vmap


def is_power_of_two(n: int) -> bool:
    """Check if an integer is power of two."""
    def f(x):
        if x < 2:
            return False
        elif x & (x-1) == 0:
            return True
        return False

    return np.vectorize(f)(n)


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


def partial_trace(rho: torch.Tensor, nqubit: int, trace_lst: List[int]) -> torch.Tensor:
    r"""Calculate the partial trace for a batch of density matrices.

    Args:
        rho (torch.Tensor): Density matrices with the shape of
            :math:`(\text{batch}, 2^{\text{nqubit}}, 2^{\text{nqubit}})`.
        nqubit (int): Total number of qubits.
        trace_lst (List[int]): A list of qubits to be traced.

    Returns:
        torch.Tensor: Reduced density matrices.
    """
    if rho.ndim == 2:
        rho = rho.unsqueeze(0)
    assert rho.ndim == 3
    assert rho.shape[1] == rho.shape[2] == 2 ** nqubit
    b = rho.shape[0]
    n = len(trace_lst)
    trace_lst = [i + 1 for i in trace_lst]
    trace_lst2 = [i + nqubit for i in trace_lst]
    trace_lst += trace_lst2
    permute_shape = list(range(2 * nqubit + 1))
    for i in trace_lst:
        permute_shape.remove(i)
    permute_shape += trace_lst
    rho = rho.reshape([b] + [2] * 2 * nqubit).permute(permute_shape).reshape(-1, 2 ** n, 2 ** n)
    rho = rho.diagonal(dim1=-2, dim2=-1).sum(-1)
    return rho.reshape(b, 2 ** (nqubit - n), 2 ** (nqubit - n)).squeeze(0)


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
    state = torch.zeros(batch, n, dtype=torch.cfloat, device=data.device)
    data = nn.functional.normalize(data[:, :n], p=2, dim=-1)
    if n > size:
        state[:, :size] = data[:, :]
    else:
        state[:, :] = data[:, :]
    return state.unsqueeze(-1)


def measure(
    state: torch.Tensor,
    shots: int = 1024,
    with_prob: bool = False,
    wires: Union[int, List[int], None] = None,
    den_mat: bool = False
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
    if wires is None:
        bit_strings = [format(i, f'0{n}b') for i in range(2 ** n)]
    else:
        assert isinstance(wires, (int, list))
        if isinstance(wires, int):
            wires = [wires]
        bit_strings = [format(i, f'0{len(wires)}b') for i in range(2 ** len(wires))]
    results_tot = []
    for i in range(batch):
        if den_mat:
            probs = torch.abs(state[i])
        else:
            probs = torch.abs(state[i]) ** 2
        if wires is not None:
            wires = sorted(wires)
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


# pylint: disable=wrong-import-position
from .layer import Observable

def expectation(
    state: Union[torch.Tensor, List[torch.Tensor]],
    observable: Observable,
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
