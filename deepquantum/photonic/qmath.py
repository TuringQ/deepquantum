import itertools
from typing import Any, List, Optional, Union

import torch
from scipy import special
from torch import vmap


def dirac_ket(matrix: torch.Tensor) -> str:
    """
    the dirac state output with batch
    """
    ket_dict = {}
    for i in range(matrix.shape[0]): # consider batch i
        state_i = matrix[i]
        abs_state = abs(state_i)
        # get largest k values with abs(amplitudes)
        top_k = torch.topk(abs_state.flatten(), k=min(len(abs_state), 5), largest=True).values
        idx_all = []
        ket_repr_i = ''
        for amp in top_k:
            idx = torch.nonzero(abs_state == amp)[0].tolist()
            idx_all.append(idx)
            # after finding the indx, set the value to 0, avoid the same abs values
            abs_state[tuple(idx)] = 0
            lst1 = list(map(lambda x:str(x), idx))
            if amp > 0:
                state_str = f'({state_i[tuple(idx)]:6.3f})' + '|' + ''.join(lst1) + '>'
                ket_repr_i = ket_repr_i + ' + ' + state_str
        batch_i = 'state_' + f'{i}'
        ket_dict[batch_i] = ket_repr_i[3:]
    return ket_dict


def sort_dict_fock_basis(state_dict, idx = 0):
    """Sort the dictionary of Fock basis states in descending order of probs
    """
    sort_list = sorted(state_dict.items(),  key=lambda t: abs(t[1][idx]), reverse=True)
    sorted_dict = {}
    for key, value in sort_list:
        sorted_dict[key] = value
    return sorted_dict


def sub_matrix(u, input_state, output_state):
    """Get the submatrix for calculating the transfer amplitude and transfer probs from the given matrix,
    the input state and the output state. The rows are chosen according to the output state and the columns
    are chosen according to the input state.

    u: torch.tensor, the unitary matrix for the circuit or component
    """
    def set_copy_indx(state):
        """
        picking up indices from the nonezero elements of state,
        repeat times depend on the nonezero value
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


def permanent(mat):
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


def create_subset(num_coincidence):
    """
    all subset from {1,2,...n}
    """
    subsets = []
    for k in range(1, num_coincidence + 1):
        comb_lst = []
        for comb in itertools.combinations(range(num_coincidence), k):
            comb_lst.append(list(comb))
        temp = torch.tensor(comb_lst).reshape(len(comb_lst), k)
        subsets.append(temp)
    return subsets


def permanent_ryser(mat):
    def helper(subset, mat):
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


def product_factorial(state):
    """
    return the product of the factorial of each element
    |s_1,s_2,...s_n> --> s_1!*s_2!*...s_n!
    """
    return special.factorial(state).prod(axis=-1, keepdims=True)


def fock_combinations(nmode, nphoton):
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
    def backtrack(state, length, num_sum):
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
