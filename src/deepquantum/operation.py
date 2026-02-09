"""
Base classes
"""

from copy import copy
from typing import Any

import numpy as np
import torch
from torch import nn, vmap

from .distributed import dist_many_targ_gate
from .qmath import evolve_den_mat, evolve_state, inverse_permutation, state_to_tensors
from .state import DistributedQubitState, MatrixProductState


class Operation(nn.Module):
    r"""A base class for quantum operations.

    Args:
        name (str or None, optional): The name of the quantum operation. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`. Default: ``False``
    """

    def __init__(
        self,
        name: str | None = None,
        nqubit: int = 1,
        wires: int | list[int] | None = None,
        den_mat: bool = False,
        tsr_mode: bool = False,
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
            assert x.shape[-1] == x.shape[-2] == 2**self.nqubit
            return x.reshape([-1] + [2] * 2 * self.nqubit)
        else:
            if x.ndim == 1:
                assert x.shape[-1] == 2**self.nqubit
            else:
                assert x.shape[-1] == 2**self.nqubit or x.shape[-2] == 2**self.nqubit
            return x.reshape([-1] + [2] * self.nqubit)

    def vector_rep(self, x: torch.Tensor) -> torch.Tensor:
        """Get the vector representation of the state."""
        return x.reshape(-1, 2**self.nqubit, 1)

    def matrix_rep(self, x: torch.Tensor) -> torch.Tensor:
        """Get the density matrix representation of the state."""
        return x.reshape(-1, 2**self.nqubit, 2**self.nqubit)

    def get_unitary(self) -> torch.Tensor:
        """Get the global unitary matrix."""
        raise NotImplementedError

    def init_para(self) -> None:
        """Initialize the parameters."""
        pass

    def set_nqubit(self, nqubit: int) -> None:
        """Set the number of qubits of the ``Operation``."""
        self.nqubit = nqubit

    def set_wires(self, wires: int | list[int]) -> None:
        """Set the wires of the ``Operation``."""
        self.wires = self._convert_indices(wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass."""
        if self.tsr_mode:
            return self.tensor_rep(x)
        else:
            if self.den_mat:
                return self.matrix_rep(x)
            else:
                return self.vector_rep(x)

    def _convert_indices(self, indices: int | list[int]) -> list[int]:
        """Convert and check the indices of the qubits."""
        if isinstance(indices, int):
            indices = [indices]
        assert isinstance(indices, list), 'Invalid input type'
        assert all(isinstance(i, int) for i in indices), 'Invalid input type'
        if len(indices) > 0:
            assert min(indices) > -1 and max(indices) < self.nqubit, 'Invalid input'
        assert len(set(indices)) == len(indices), 'Invalid input'
        return indices

    def _check_minmax(self, minmax: list[int]) -> None:
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
        name: str | None = None,
        nqubit: int = 1,
        wires: int | list[int] | None = None,
        controls: int | list[int] | None = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False,
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
        # MBQC
        self.nodes = self.wires
        self.nancilla = 1

    def to(self, arg: Any) -> 'Gate':
        """Set dtype or device of the ``Gate``."""
        if arg == torch.float:
            if self.npara == 0:
                self.matrix = self.matrix.to(torch.cfloat)
            elif self.npara > 0:
                super().to(torch.float)
        elif arg == torch.double:
            if self.npara == 0:
                self.matrix = self.matrix.to(torch.cdouble)
            elif self.npara > 0:
                super().to(torch.double)
        else:
            super().to(arg)
        return self

    def set_controls(self, controls: int | list[int]) -> None:
        """Set the control wires of the ``Operation``."""
        self.controls = self._convert_indices(controls)

    def get_matrix(self, inputs: Any) -> torch.Tensor:
        """Get the local unitary matrix."""
        return self.matrix

    def update_matrix(self) -> torch.Tensor:
        """Update the local unitary matrix."""
        return self.matrix

    def _real_wrapper(self, x: Any) -> torch.Tensor:
        mat = self.get_matrix(x)
        return torch.view_as_real(mat)

    def get_derivative(self, inputs: Any) -> torch.Tensor:
        """Get the derivative of the local unitary matrix."""
        return torch.zeros_like(self.matrix)

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
        return evolve_state(x, matrix, self.nqubit, self.wires)

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
        x = x.permute(pm_shape).reshape(2**nt, -1, 2**nc)
        x = torch.cat([x[:, :, :-1], (matrix @ x[:, :, -1]).unsqueeze(-1)], dim=-1)
        x = x.reshape([2] * nt + [-1] + [2] * (self.nqubit - nt - nc) + [2] * nc)
        x = x.permute(inverse_permutation(pm_shape))
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
        return evolve_den_mat(x, matrix, self.nqubit, self.wires)

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
        x = x.permute(pm_shape).reshape(2**nt, -1, 2**nc)
        x = torch.cat([x[:, :, :-1], (matrix @ x[:, :, -1]).unsqueeze(-1)], dim=-1)
        x = x.reshape([2] * nt + [-1] + [2] * (2 * self.nqubit - nt - nc) + [2] * nc)
        x = x.permute(inverse_permutation(pm_shape))
        # right multiply
        wires = [i + 1 + self.nqubit for i in self.wires]
        controls = [i + 1 + self.nqubit for i in self.controls]
        pm_shape = list(range(2 * self.nqubit + 1))
        for i in wires:
            pm_shape.remove(i)
        for i in controls:
            pm_shape.remove(i)
        pm_shape = wires + pm_shape + controls
        x = x.permute(pm_shape).reshape(2**nt, -1, 2**nc)
        x = torch.cat([x[:, :, :-1], (matrix.conj() @ x[:, :, -1]).unsqueeze(-1)], dim=-1)
        x = x.reshape([2] * nt + [-1] + [2] * (2 * self.nqubit - nt - nc) + [2] * nc)
        x = x.permute(inverse_permutation(pm_shape))
        return x

    def op_dist_state(self, x: DistributedQubitState) -> DistributedQubitState:
        """Perform a forward pass of a gate for a distributed state vector."""
        wires = self.controls + self.wires
        matrix = self.update_matrix()
        identity = matrix.new_ones(2 ** len(wires) - 2 ** len(self.wires)).diag_embed()
        unitary = torch.block_diag(identity, matrix)
        targets = [self.nqubit - wire - 1 for wire in wires]
        return dist_many_targ_gate(x, targets, unitary)

    def forward(
        self, x: torch.Tensor | MatrixProductState | DistributedQubitState
    ) -> torch.Tensor | MatrixProductState | DistributedQubitState:
        """Perform a forward pass."""
        if isinstance(x, MatrixProductState):
            return self.op_mps(x)
        elif isinstance(x, DistributedQubitState):
            return self.op_dist_state(x)
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

    def qpd(self, label: int | None = None) -> 'Gate':
        """Get the quasiprobability-decomposition representation."""
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
        qasm_lst1 = [f'opaque {name} ']
        qasm_lst2 = [f'{name} ']
        for i, wire in enumerate(self.controls + self.wires):
            qasm_lst1.append(f'q{i},')
            qasm_lst2.append(f'q[{wire}],')
        qasm_str1 = ''.join(qasm_lst1)[:-1] + ';\n'
        qasm_str2 = ''.join(qasm_lst2)[:-1] + ';\n'
        if name not in Gate._qasm_new_gate:
            Gate._qasm_new_gate.append(name)
            return qasm_str1 + qasm_str2
        else:
            return qasm_str2

    def _qasm(self) -> str:
        return self._qasm_customized(self.name)

    def get_mpo(self) -> tuple[list[torch.Tensor], int]:
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
        gate_copy.wires = index_local[: len(gate_copy.wires)]
        gate_copy.controls = index_local[len(gate_copy.wires) :]
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
                    identity = torch.eye(chi * 2, dtype=u.dtype, device=u.device)
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
        out.center = mps.center
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
        wires (int, List[int], List[List[int]] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`. Default: ``False``
    """

    def __init__(
        self,
        name: str | None = None,
        nqubit: int = 1,
        wires: int | list[int] | list[list[int]] | None = None,
        den_mat: bool = False,
        tsr_mode: bool = False,
    ) -> None:
        super().__init__(name=name, nqubit=nqubit, wires=None, den_mat=den_mat, tsr_mode=tsr_mode)
        if wires is None:
            wires = [[0]]
        self.wires = self._convert_indices(wires)
        self.gates = nn.Sequential()
        # MBQC
        self.nodes = copy(self.wires)

    def to(self, arg: Any) -> 'Layer':
        """Set dtype or device of the ``Layer``."""
        for gate in self.gates:
            gate.to(arg)
        return self

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
                gate.init_para(inputs[count : count + gate.npara])
            count += gate.npara

    def update_npara(self) -> None:
        """Update the number of parameters."""
        self.npara = 0
        for gate in self.gates:
            self.npara += gate.npara

    def set_nqubit(self, nqubit: int) -> None:
        """Set the number of qubits of the ``Layer``."""
        self.nqubit = nqubit
        for gate in self.gates:
            gate.nqubit = nqubit

    def set_wires(self, wires: int | list[int] | list[list[int]]) -> None:
        """Set the wires of the ``Layer``."""
        self.wires = self._convert_indices(wires)
        for i, gate in enumerate(self.gates):
            gate.wires = self.wires[i]

    def forward(
        self, x: torch.Tensor | MatrixProductState | DistributedQubitState
    ) -> torch.Tensor | MatrixProductState | DistributedQubitState:
        """Perform a forward pass."""
        if isinstance(x, (MatrixProductState, DistributedQubitState)):
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
        """Get the inversed layer."""
        return self

    def _convert_indices(self, indices: int | list) -> list[list[int]]:
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

    def pattern(self, nodes: list[list[int]], ancilla: list[list[int]]) -> nn.Sequential:
        """Get the MBQC pattern."""
        assert len(nodes) == len(ancilla) == len(self.gates)
        cmds = nn.Sequential()
        for i, gate in enumerate(self.gates):
            cmds.extend(gate.pattern(nodes[i], ancilla[i]))
            self.nodes[i] = gate.nodes
        return cmds


class Channel(Operation):
    r"""A base class for quantum channels.

    Args:
        inputs (Any, optional): The parameter of the channel. Default: ``None``
        name (str or None, optional): The name of the channel. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`. Default: ``False``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """

    # include default names in QASM
    _qasm_new_gate = []

    def __init__(
        self,
        inputs: Any = None,
        name: str | None = None,
        nqubit: int = 1,
        wires: int | list[int] | None = None,
        tsr_mode: bool = False,
        requires_grad: bool = False,
    ) -> None:
        self.nqubit = nqubit
        if wires is None:
            wires = [0]
        wires = self._convert_indices(wires)
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=True, tsr_mode=tsr_mode)
        self.npara = 1
        self.requires_grad = requires_grad
        self.init_para(inputs)

    @property
    def prob(self):
        """The error probability."""
        return torch.sin(self.theta) ** 2

    def inputs_to_tensor(self, inputs: Any = None) -> torch.Tensor:
        """Convert inputs to torch.Tensor."""
        while isinstance(inputs, list):
            inputs = inputs[0]
        if inputs is None:
            inputs = torch.rand(1)[0] * torch.pi
        elif not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float)
        return inputs

    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Update the local Kraus matrices acting on density matrices."""
        raise self.matrix

    def update_matrix(self) -> torch.Tensor:
        """Update the local Kraus matrices acting on density matrices."""
        matrix = self.get_matrix(self.theta)
        self.matrix = matrix.detach()
        return matrix

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        theta = self.inputs_to_tensor(inputs)
        if self.requires_grad:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('theta', theta)
        self.update_matrix()

    def op_den_mat(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass for density matrices."""
        matrix = self.update_matrix()
        x = vmap(evolve_den_mat, in_dims=(None, 0, None, None))(x, matrix, self.nqubit, self.wires).sum(0)
        if not self.tsr_mode:
            x = self.matrix_rep(x).squeeze(0)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass."""
        if not self.tsr_mode:
            x = self.tensor_rep(x)
        assert x.ndim == 2 * self.nqubit + 1
        return self.op_den_mat(x)

    def extra_repr(self) -> str:
        return f'wires={self.wires}, probability={self.prob.item()}'

    @staticmethod
    def _reset_qasm_new_gate() -> None:
        Channel._qasm_new_gate = []

    def _qasm_customized(self, name: str) -> str:
        """Get QASM for channels."""
        name = name.lower()
        qasm_lst1 = [f'opaque {name} ']
        qasm_lst2 = [f'{name} ']
        for i, wire in enumerate(self.wires):
            qasm_lst1.append(f'q{i},')
            qasm_lst2.append(f'q[{wire}],')
        qasm_str1 = ''.join(qasm_lst1)[:-1] + ';\n'
        qasm_str2 = ''.join(qasm_lst2)[:-1] + ';\n'
        if name not in Channel._qasm_new_gate:
            Channel._qasm_new_gate.append(name)
            return qasm_str1 + qasm_str2
        else:
            return qasm_str2

    def _qasm(self) -> str:
        return self._qasm_customized(self.name)


class GateQPD(Gate):
    r"""A base class for quasiprobability-decomposition gates.

    Args:
        bases (List[Tuple[nn.Sequential, ...]]): A list of tuples describing the operations probabilistically used to
            simulate an ideal quantum operation.
        coeffs (List[float]): The coefficients for quasiprobability representation.
        label (int or None, optional): The label of the gate. Default: ``None``
        name (str or None, optional): The name of the quantum operation. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`. Default: ``False``
    """

    def __init__(
        self,
        bases: list[tuple[nn.Sequential, ...]],
        coeffs: list[float],
        label: int | None = None,
        name: str | None = None,
        nqubit: int = 1,
        wires: int | list[int] | None = None,
        den_mat: bool = False,
        tsr_mode: bool = False,
    ) -> None:
        self.nqubit = nqubit
        if wires is None:
            wires = [0]
        wires = self._convert_indices(wires)
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.bases = bases
        self.coeffs = coeffs
        self.label = label
        self.idx = 0

    def to(self, arg: Any) -> 'GateQPD':
        """Set dtype or device of the ``GateQPD``."""
        for basis in self.bases:
            for ops in basis:
                for op in ops:
                    op.to(arg)
        return self

    def set_nqubit(self, nqubit: int) -> None:
        """Set the number of qubits of the ``GateQPD``."""
        self.nqubit = nqubit
        for basis in self.bases:
            for ops in basis:
                for op in ops:
                    op.nqubit = nqubit

    def set_wires(self, wires: int | list[int]) -> None:
        """Set the wires of the ``GateQPD``."""
        self.wires = self._convert_indices(wires)
        for basis in self.bases:
            for i, ops in enumerate(basis):
                for op in ops:
                    op.set_wires(self.wires[i])

    def forward(self, x: torch.Tensor, idx: int | None = None) -> torch.Tensor:
        """Perform a forward pass.

        Args:
            x (torch.Tensor): The input tensor.
            idx (int, optional): The index of the operation to be applied. Default: 0
        """
        if idx is not None:
            self.idx = idx
        if not self.tsr_mode:
            x = self.tensor_rep(x)
        for ops in self.bases[self.idx]:
            x = ops(x)
        if not self.tsr_mode:
            if self.den_mat:
                x = self.matrix_rep(x).squeeze(0)
            else:
                x = self.vector_rep(x).squeeze(0)
        return x


class MeasureQPD(Operation):
    """A operation for denoting a QPD measurement location.

    Args:
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
    """

    def __init__(self, nqubit: int = 1, wires: int | list[int] | None = None) -> None:
        self.nqubit = nqubit
        if wires is None:
            wires = [0]
        wires = self._convert_indices(wires)
        super().__init__(name='MeasureQPD', nqubit=nqubit, wires=wires)

    def forward(self, x: Any) -> Any:
        """Perform a forward pass."""
        return x
