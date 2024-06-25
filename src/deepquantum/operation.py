"""
Base classes
"""

# pylint: disable=unused-import
import warnings
from copy import copy
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from .qmath import inverse_permutation, state_to_tensors
from .state import MatrixProductState


class Operation(nn.Module):
    r"""A base class for quantum operations.

    Args:
        name (str or None, optional): The name of the quantum operation. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`. Default: ``False``
    """
    def __init__(
        self,
        name: Optional[str] = None,
        nqubit: int = 1,
        wires: Union[int, List, None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__()
        self.name = name
        self.nqubit = nqubit
        self.wires = wires
        self.den_mat = den_mat
        self.tsr_mode = tsr_mode
        self.npara = 0

    def tensor_rep(self, x: torch.Tensor) -> torch.Tensor:
        """Get the tensor representation of the state."""
        if self.den_mat:
            assert x.shape[-1] == x.shape[-2] == 2 ** self.nqubit
            return x.reshape([-1] + [2] * 2 * self.nqubit)
        else:
            if x.ndim == 1:
                assert x.shape[-1] == 2 ** self.nqubit
            else:
                assert x.shape[-1] == 2 ** self.nqubit or x.shape[-2] == 2 ** self.nqubit
            return x.reshape([-1] + [2] * self.nqubit)

    def vector_rep(self, x: torch.Tensor) -> torch.Tensor:
        """Get the vector representation of the state."""
        return x.reshape(-1, 2 ** self.nqubit, 1)

    def matrix_rep(self, x: torch.Tensor) -> torch.Tensor:
        """Get the density matrix representation of the state."""
        return x.reshape(-1, 2 ** self.nqubit, 2 ** self.nqubit)

    def get_unitary(self) -> torch.Tensor:
        """Get the global unitary matrix."""
        raise NotImplementedError

    def init_para(self) -> None:
        """Initialize the parameters."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass."""
        if self.tsr_mode:
            return self.tensor_rep(x)
        else:
            if self.den_mat:
                return self.matrix_rep(x)
            else:
                return self.vector_rep(x)

    def _convert_indices(self, indices: Union[int, List[int]]) -> List[int]:
        """Convert and check the indices of the qubits."""
        if isinstance(indices, int):
            indices = [indices]
        assert isinstance(indices, list), 'Invalid input type'
        assert all(isinstance(i, int) for i in indices), 'Invalid input type'
        if len(indices) > 0:
            assert min(indices) > -1 and max(indices) < self.nqubit, 'Invalid input'
        assert len(set(indices)) == len(indices), 'Invalid input'
        return indices

    def _check_minmax(self, minmax: List[int]) -> None:
        """Check the minimum and maximum indices of the qubits."""
        assert isinstance(minmax, list)
        assert len(minmax) == 2
        assert all(isinstance(i, int) for i in minmax)
        assert -1 < minmax[0] <= minmax[1] < self.nqubit


class Gate(Operation):
    r"""A base class for quantum gates.

    Args:
        name (str or None, optional): The name of the gate. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`. Default: ``False``
    """
    # include default names in QASM
    _qasm_new_gate = ['c3x', 'c4x']

    def __init__(
        self,
        name: Optional[str] = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        self.nqubit = nqubit
        if wires is None:
            wires = [0]
        if controls is None:
            controls = []
        wires = self._convert_indices(wires)
        controls = self._convert_indices(controls)
        for wire in wires:
            assert wire not in controls, 'Use repeated wires'
        if condition:
            assert len(controls) > 0
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.controls = controls
        self.condition = condition

    def update_matrix(self) -> torch.Tensor:
        """Update the local unitary matrix."""
        return self.matrix

    def op_state(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass for state vectors."""
        matrix = self.update_matrix()
        if self.controls == []:
            x = self.op_state_base(x=x, matrix=matrix)
        else:
            x = self.op_state_control(x=x, matrix=matrix)
        if not self.tsr_mode:
            x = self.vector_rep(x).squeeze(0)
        return x

    def op_state_base(self, x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of a gate for state vectors."""
        nt = len(self.wires)
        wires = [i + 1 for i in self.wires]
        pm_shape = list(range(self.nqubit + 1))
        for i in wires:
            pm_shape.remove(i)
        pm_shape = wires + pm_shape
        x = x.permute(pm_shape).reshape(2 ** nt, -1)
        x = (matrix @ x).reshape([2] * nt + [-1] + [2] * (self.nqubit - nt))
        x = x.permute(inverse_permutation(pm_shape))
        return x

    def op_state_control(self, x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of a controlled gate for state vectors."""
        nt = len(self.wires)
        nc = len(self.controls)
        wires = [i + 1 for i in self.wires]
        controls = [i + 1 for i in self.controls]
        pm_shape = list(range(self.nqubit + 1))
        for i in wires:
            pm_shape.remove(i)
        for i in controls:
            pm_shape.remove(i)
        pm_shape = wires + pm_shape + controls
        state1 = x.permute(pm_shape).reshape(2 ** nt, -1, 2 ** nc)
        state2 = (matrix @ state1[:, :, -1]).unsqueeze(-1)
        state1 = torch.cat([state1[:, :, :-1], state2], dim=-1)
        state1 = state1.reshape([2] * nt + [-1] + [2] * (self.nqubit - nt - nc) + [2] * nc)
        x = state1.permute(inverse_permutation(pm_shape))
        return x

    def op_den_mat(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass for density matrices."""
        matrix = self.update_matrix()
        if self.controls == []:
            x = self.op_den_mat_base(x=x, matrix=matrix)
        else:
            x = self.op_den_mat_control(x=x, matrix=matrix)
        if not self.tsr_mode:
            x = self.matrix_rep(x).squeeze(0)
        return x

    def op_den_mat_base(self, x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of a gate for density matrices."""
        nt = len(self.wires)
        # left multiply
        wires = [i + 1 for i in self.wires]
        pm_shape = list(range(2 * self.nqubit + 1))
        for i in wires:
            pm_shape.remove(i)
        pm_shape = wires + pm_shape
        x = x.permute(pm_shape).reshape(2 ** nt, -1)
        x = (matrix @ x).reshape([2] * nt + [-1] + [2] * (2 * self.nqubit - nt))
        x = x.permute(inverse_permutation(pm_shape))
        # right multiply
        wires = [i + 1 + self.nqubit for i in self.wires]
        pm_shape = list(range(2 * self.nqubit + 1))
        for i in wires:
            pm_shape.remove(i)
        pm_shape = wires + pm_shape
        x = x.permute(pm_shape).reshape(2 ** nt, -1)
        x = (matrix.conj() @ x).reshape([2] * nt + [-1] + [2] * (2 * self.nqubit - nt))
        x = x.permute(inverse_permutation(pm_shape))
        return x

    def op_den_mat_control(self, x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of a controlled gate for density matrices."""
        nt = len(self.wires)
        nc = len(self.controls)
        # left multiply
        wires = [i + 1 for i in self.wires]
        controls = [i + 1 for i in self.controls]
        pm_shape = list(range(2 * self.nqubit + 1))
        for i in wires:
            pm_shape.remove(i)
        for i in controls:
            pm_shape.remove(i)
        pm_shape = wires + pm_shape + controls
        state1 = x.permute(pm_shape).reshape(2 ** nt, -1, 2 ** nc)
        state2 = (matrix @ state1[:, :, -1]).unsqueeze(-1)
        state1 = torch.cat([state1[:, :, :-1], state2], dim=-1)
        state1 = state1.reshape([2] * nt + [-1] + [2] * (2 * self.nqubit - nt - nc) + [2] * nc)
        x = state1.permute(inverse_permutation(pm_shape))
        # right multiply
        wires = [i + 1 + self.nqubit for i in self.wires]
        controls = [i + 1 + self.nqubit for i in self.controls]
        pm_shape = list(range(2 * self.nqubit + 1))
        for i in wires:
            pm_shape.remove(i)
        for i in controls:
            pm_shape.remove(i)
        pm_shape = wires + pm_shape + controls
        state1 = x.permute(pm_shape).reshape(2 ** nt, -1, 2 ** nc)
        state2 = (matrix.conj() @ state1[:, :, -1]).unsqueeze(-1)
        state1 = torch.cat([state1[:, :, :-1], state2], dim=-1)
        state1 = state1.reshape([2] * nt + [-1] + [2] * (2 * self.nqubit - nt - nc) + [2] * nc)
        x = state1.permute(inverse_permutation(pm_shape))
        return x

    def forward(self, x: Union[torch.Tensor, MatrixProductState]) -> Union[torch.Tensor, MatrixProductState]:
        """Perform a forward pass."""
        if isinstance(x, MatrixProductState):
            return self.op_mps(x)
        if not self.tsr_mode:
            x = self.tensor_rep(x)
        if self.den_mat:
            assert x.ndim == 2 * self.nqubit + 1
            return self.op_den_mat(x)
        else:
            assert x.ndim == self.nqubit + 1
            return self.op_state(x)

    def inverse(self) -> 'Gate':
        """Get the inversed gate."""
        return self

    def extra_repr(self) -> str:
        s = f'wires={self.wires}'
        if self.controls == []:
            return s
        else:
            return s + f', controls={self.controls}'

    @staticmethod
    def _reset_qasm_new_gate() -> None:
        Gate._qasm_new_gate = ['c3x', 'c4x']

    def _qasm_cond_measure(self) -> str:
        qasm_str = ''
        for control in self.controls:
            qasm_str += f'measure q[{control}] -> c[{control}];\n'
        qasm_str += 'if(c==1) '
        return qasm_str

    def _qasm_customized(self, name: str) -> str:
        """Get QASM for multi-controlled gates."""
        name = name.lower()
        if len(self.controls) > 2:
            name = f'c{len(self.controls)}' + name
        else:
            name = 'c' * len(self.controls) + name
        # warnings.warn(f'{name} is an empty gate and should be only used to draw circuit.')
        qasm_lst1 = [f'gate {name} ']
        qasm_lst2 = [f'{name} ']
        for i, wire in enumerate(self.controls + self.wires):
            qasm_lst1.append(f'q{i},')
            qasm_lst2.append(f'q[{wire}],')
        qasm_str1 = ''.join(qasm_lst1)[:-1] + ' { }\n'
        qasm_str2 = ''.join(qasm_lst2)[:-1] + ';\n'
        if name not in Gate._qasm_new_gate:
            Gate._qasm_new_gate.append(name)
            return qasm_str1 + qasm_str2
        else:
            return qasm_str2

    def get_mpo(self) -> Tuple[List[torch.Tensor], int]:
        r"""Convert gate to MPO form with identities at empty sites.

        Note:
            If sites are not adjacent, insert identities in the middle, i.e.,

            >>>      |       |            |   |   |
            >>>    --A---x---B--   ->   --A---I---B--
            >>>      |       |            |   |   |

            where

            >>>         a
            >>>         |
            >>>    --i--I--j--
            >>>         |
            >>>         b

            means :math:`\delta_{i,j} \delta_{a,b}`
        """
        index = self.wires + self.controls
        index_left = min(index)
        nindex = len(index)
        index_sort = sorted(index)
        # convert index to a list of integers from 0 to nindex-1
        s = {x: i for i, x in enumerate(index_sort)}
        index_local = [s[x] for x in index]
        # use shallow copy to share parameters
        gate_copy = copy(self)
        gate_copy.nqubit = nindex
        gate_copy.wires = index_local[:len(gate_copy.wires)]
        gate_copy.controls = index_local[len(gate_copy.wires):]
        u = gate_copy.get_unitary()
        # transform gate from (out1, out2, ..., in1, in2 ...) to (out1, in1, out2, in2, ...)
        order = list(np.arange(2 * nindex).reshape((2, nindex)).T.flatten())
        u = u.reshape([2] * 2 * nindex).permute(order).reshape([4] * nindex)
        main_tensors = state_to_tensors(u, nsite=nindex, qudit=4)
        # each tensor is in shape of (i, a, b, j)
        tensors = []
        previous_i = None
        for i, main_tensor in zip(index_sort, main_tensors):
            # insert identities in the middle
            if previous_i is not None:
                for _ in range(previous_i + 1, i):
                    chi = tensors[-1].shape[-1]
                    identity = torch.eye(chi * 2, dtype=self.matrix.dtype, device=self.matrix.device)
                    tensors.append(identity.reshape(chi, 2, chi, 2).permute(0, 1, 3, 2))
            nleft, _, nright = main_tensor.shape
            tensors.append(main_tensor.reshape(nleft, 2, 2, nright))
            previous_i = i
        return tensors, index_left

    def op_mps(self, mps: MatrixProductState) -> MatrixProductState:
        """Perform a forward pass for the ``MatrixProductState``."""
        mpo_tensors, left = self.get_mpo()
        right = left + len(mpo_tensors) - 1
        diff_left = abs(left - mps.center)
        diff_right = abs(right - mps.center)
        center_left = diff_left < diff_right
        if center_left:
            end1 = left
            end2 = right
        else:
            end1 = right
            end2 = left
        wires = list(range(left, right + 1))
        out = MatrixProductState(nsite=mps.nsite, state=mps.tensors, chi=mps.chi, normalize=mps.normalize)
        out.center_orthogonalization(end1, dc=-1, normalize=out.normalize)
        out.apply_mpo(mpo_tensors, wires)
        out.center_orthogonalization(end2, dc=-1, normalize=out.normalize)
        out.center_orthogonalization(end1, dc=out.chi, normalize=out.normalize)
        return out


class Layer(Operation):
    r"""A base class for quantum layers.

    Args:
        name (str, optional): The name of the layer. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`. Default: ``False``
    """
    def __init__(
        self,
        name: Optional[str] = None,
        nqubit: int = 1,
        wires: Union[int, List, None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name=name, nqubit=nqubit, wires=None, den_mat=den_mat, tsr_mode=tsr_mode)
        if wires is None:
            wires = [[0]]
        self.wires = self._convert_indices(wires)
        self.gates = nn.Sequential()

    def get_unitary(self) -> torch.Tensor:
        """Get the global unitary matrix."""
        u = None
        for gate in self.gates:
            if u is None:
                u = gate.get_unitary()
            else:
                u = gate.get_unitary() @ u
        return u

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        count = 0
        for gate in self.gates:
            if inputs is None:
                gate.init_para()
            else:
                gate.init_para(inputs[count:count+gate.npara])
            count += gate.npara

    def update_npara(self) -> None:
        """Update the number of parameters."""
        self.npara = 0
        for gate in self.gates:
            self.npara += gate.npara

    def forward(self, x: Union[torch.Tensor, MatrixProductState]) -> Union[torch.Tensor, MatrixProductState]:
        """Perform a forward pass."""
        if isinstance(x, MatrixProductState):
            return self.gates(x)
        if not self.tsr_mode:
            x = self.tensor_rep(x)
        x = self.gates(x)
        if not self.tsr_mode:
            if self.den_mat:
                return self.matrix_rep(x).squeeze(0)
            else:
                return self.vector_rep(x).squeeze(0)
        return x

    def inverse(self) -> 'Layer':
        """Get the inversed gate."""
        return self

    def _convert_indices(self, indices: Union[int, List]) -> List[List[int]]:
        if isinstance(indices, int):
            indices = [[indices]]
        assert isinstance(indices, list), 'Invalid input type'
        if all(isinstance(i, int) for i in indices):
            indices = [[i] for i in indices]
        assert all(isinstance(i, list) for i in indices), 'Invalid input type'
        for idx in indices:
            assert all(isinstance(i, int) for i in idx), 'Invalid input type'
            assert min(idx) > -1 and max(idx) < self.nqubit, 'Invalid input'
            assert len(set(idx)) == len(idx), 'Invalid input'
        return indices

    def _qasm(self) -> str:
        lst = []
        for gate in self.gates:
            # pylint: disable=protected-access
            lst.append(gate._qasm())
        return ''.join(lst)
