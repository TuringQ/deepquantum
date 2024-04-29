"""
Common functions
"""

import itertools
from typing import Dict, List

import torch
from scipy import special
from torch import vmap


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
    return torch.squeeze(u2)


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
    """Create all subsets from {1,2,...,n}."""
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


def xxpp_to_xpxp(matrix):
    """Transform the representation in xxpp ordering to the representation in xpxp ordering."""
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix).squeeze()
    nmode = int(matrix.size()[0] / 2)
    # transformation matrix
    t = torch.zeros([2 * nmode] * 2, dtype=matrix.dtype)
    for i in range(2 * nmode):
        if i % 2 == 0:
            t[i][i // 2] = 1
        else:
            t[i][i // 2 + nmode] = 1
    # new matrix in xpxp ordering
    if matrix.numel() == (2 * nmode) ** 2:
        new_mat = t @ matrix @ t.mT
    elif matrix.numel() == 2 * nmode:
        new_mat = t @ matrix.reshape(-1, 1)
        new_mat = new_mat.reshape(-1, 1).squeeze()
    return new_mat


def xpxp_to_xxpp(matrix):
    """Transform the representation in xpxp ordering to the representation in xxpp ordering."""
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix).squeeze()
    nmode = int(matrix.size()[0] / 2)
    # transformation matrix
    t = torch.zeros([2 * nmode] * 2, dtype=matrix.dtype)
    for i in range(2 * nmode):
        if i < nmode:
            t[i][2 * i] = 1
        else:
            t[i][2 * (i - nmode) + 1] = 1
    # new matrix in xxpp ordering
    if matrix.numel() == (2 * nmode) ** 2:
        new_mat = t @ matrix @ t.mT
    elif matrix.numel() == 2 * nmode:
        new_mat = t @ matrix.reshape(-1, 1)
        new_mat = new_mat.reshape(-1, 1).squeeze()
    return new_mat
