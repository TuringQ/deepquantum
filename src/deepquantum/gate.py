"""
Quantum gates
"""

from copy import copy
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.autograd.functional import jacobian

from .distributed import dist_one_targ_gate, dist_many_ctrl_one_targ_gate, dist_swap_gate
from .operation import Gate
from .qmath import multi_kron, is_unitary, inverse_permutation, svd
from .state import DistributedQubitState

class SingleGate(Gate):
    r"""A base class for single-qubit gates.

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
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
    """
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
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=controls, condition=condition,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        assert len(self.wires) == 1
        # MBQC
        self.nancilla = 2

    def get_unitary(self) -> torch.Tensor:
        """Get the global unitary matrix."""
        matrix = self.update_matrix()
        identity = torch.eye(2, dtype=matrix.dtype, device=matrix.device)
        if self.controls == []:
            lst = [identity] * self.nqubit
            lst[self.wires[0]] = matrix
            return multi_kron(lst)
        else:
            oneone = torch.tensor([[0, 0], [0, 1]], dtype=matrix.dtype, device=matrix.device)
            lst1 = [identity] * self.nqubit
            lst2 = [identity] * self.nqubit
            lst3 = [identity] * self.nqubit
            for i in self.controls:
                lst2[i] = oneone
                lst3[i] = oneone
            lst3[self.wires[0]] = matrix
            return multi_kron(lst1) - multi_kron(lst2) + multi_kron(lst3)

    def op_dist_state(self, x: DistributedQubitState) -> DistributedQubitState:
        """Perform a forward pass of a gate for a distributed state vector."""
        target = self.nqubit - self.wires[0] - 1
        matrix = self.update_matrix()
        if len(self.controls) > 0:
            controls = [self.nqubit - control - 1 for control in self.controls]
            return dist_many_ctrl_one_targ_gate(x, controls, target, matrix)
        else:
            return dist_one_targ_gate(x, target, matrix)


class DoubleGate(Gate):
    r"""A base class for two-qubit gates.

    Args:
        name (str or None, optional): The name of the gate. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 2
        wires (List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
    """
    def __init__(
        self,
        name: Optional[str] = None,
        nqubit: int = 2,
        wires: Optional[List[int]] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        if wires is None:
            wires = [0, 1]
        assert len(wires) == 2
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=controls, condition=condition,
                         den_mat=den_mat, tsr_mode=tsr_mode)

    def get_unitary(self) -> torch.Tensor:
        """Get the global unitary matrix."""
        matrix = self.update_matrix()
        identity = torch.eye(2, dtype=matrix.dtype, device=matrix.device)
        zerozero = torch.tensor([[1, 0], [0, 0]], dtype=matrix.dtype, device=matrix.device)
        zeroone  = torch.tensor([[0, 1], [0, 0]], dtype=matrix.dtype, device=matrix.device)
        onezero  = torch.tensor([[0, 0], [1, 0]], dtype=matrix.dtype, device=matrix.device)
        oneone   = torch.tensor([[0, 0], [0, 1]], dtype=matrix.dtype, device=matrix.device)
        if self.controls == []:
            lst1 = [identity] * self.nqubit
            lst2 = [identity] * self.nqubit
            lst3 = [identity] * self.nqubit
            lst4 = [identity] * self.nqubit

            lst1[self.wires[0]] = zerozero
            lst1[self.wires[1]] = matrix[0:2, 0:2]

            lst2[self.wires[0]] = zeroone
            lst2[self.wires[1]] = matrix[0:2, 2:4]

            lst3[self.wires[0]] = onezero
            lst3[self.wires[1]] = matrix[2:4, 0:2]

            lst4[self.wires[0]] = oneone
            lst4[self.wires[1]] = matrix[2:4, 2:4]
            return multi_kron(lst1) + multi_kron(lst2) + multi_kron(lst3) + multi_kron(lst4)
        else:
            lst1 = [identity] * self.nqubit
            lst2 = [identity] * self.nqubit
            lst3 = [identity] * self.nqubit
            lst4 = [identity] * self.nqubit
            lst5 = [identity] * self.nqubit
            lst6 = [identity] * self.nqubit
            for i in self.controls:
                lst2[i] = oneone

                lst3[i] = oneone
                lst4[i] = oneone
                lst5[i] = oneone
                lst6[i] = oneone

            lst3[self.wires[0]] = zerozero
            lst3[self.wires[1]] = matrix[0:2, 0:2]

            lst4[self.wires[0]] = zeroone
            lst4[self.wires[1]] = matrix[0:2, 2:4]

            lst5[self.wires[0]] = onezero
            lst5[self.wires[1]] = matrix[2:4, 0:2]

            lst6[self.wires[0]] = oneone
            lst6[self.wires[1]] = matrix[2:4, 2:4]
            return multi_kron(lst1) - multi_kron(lst2) + \
                   multi_kron(lst3) + multi_kron(lst4) + multi_kron(lst5) + multi_kron(lst6)


class DoubleControlGate(DoubleGate):
    r"""A base class for two-qubit controlled gates.

    Args:
        name (str or None, optional): The name of the gate. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 2
        wires (List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
    """
    def __init__(
        self,
        name: Optional[str] = None,
        nqubit: int = 2,
        wires: Optional[List[int]] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=None, condition=False,
                         den_mat=den_mat, tsr_mode=tsr_mode)

    def get_unitary(self) -> torch.Tensor:
        """Get the global unitary matrix."""
        matrix = self.update_matrix()
        identity = torch.eye(2, dtype=matrix.dtype, device=matrix.device)
        zerozero = torch.tensor([[1, 0], [0, 0]], dtype=matrix.dtype, device=matrix.device)
        oneone   = torch.tensor([[0, 0], [0, 1]], dtype=matrix.dtype, device=matrix.device)
        lst1 = [identity] * self.nqubit
        lst2 = [identity] * self.nqubit

        lst1[self.wires[0]] = zerozero

        lst2[self.wires[0]] = oneone
        lst2[self.wires[1]] = matrix[2:4, 2:4]
        return multi_kron(lst1) + multi_kron(lst2)


class TripleGate(Gate):
    r"""A base class for three-qubit gates.

    Args:
        name (str or None, optional): The name of the gate. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 3
        wires (List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
    """
    def __init__(
        self,
        name: Optional[str] = None,
        nqubit: int = 3,
        wires: Optional[List[int]] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        if wires is None:
            wires = [0, 1, 2]
        assert len(wires) == 3
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=controls, condition=condition,
                         den_mat=den_mat, tsr_mode=tsr_mode)


class ArbitraryGate(Gate):
    r"""A base class for customized gates.

     Args:
        name (str or None, optional): The name of the gate. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        minmax (List[int] or None, optional): The minimum and maximum indices of the qubits that the quantum
            operation acts on. Only valid when ``wires`` is ``None``. Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
    """
    def __init__(
        self,
        name: Optional[str] = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        minmax: Optional[List[int]] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        self.nqubit = nqubit
        if wires is None:
            if minmax is None:
                minmax = [0, nqubit - 1]
            self._check_minmax(minmax)
            wires = list(range(minmax[0], minmax[1] + 1))
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=controls, condition=False,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.minmax = [min(self.wires), max(self.wires)]
        # whether the wires are consecutive integers
        self.local = True
        for i in range(len(self.wires) - 1):
            if self.wires[i + 1] - self.wires[i] != 1:
                self.local = False
                break
        self.inv_mode = False

    def get_unitary(self) -> torch.Tensor:
        """Get the global unitary matrix."""
        if self.local:
            matrix = self.update_matrix()
            identity = torch.eye(2, dtype=matrix.dtype, device=matrix.device)
            lst = [identity] * (self.nqubit - len(self.wires) + 1)
            lst[self.wires[0]] = matrix
            return multi_kron(lst)
        else:
            matrix = self.update_matrix()
            identity = torch.eye(2 ** self.nqubit, dtype=matrix.dtype, device=matrix.device)
            identity = identity.reshape([2 ** self.nqubit] + [2] * self.nqubit)
            return self.op_state_base(identity, matrix).reshape(2 ** self.nqubit, 2 ** self.nqubit).T

    def inverse(self) -> 'ArbitraryGate':
        """Get the inversed gate."""
        if isinstance(self.name, str):
            name = self.name + '_dagger'
        else:
            name = self.name
        gate = copy(self)
        gate.inv_mode = not self.inv_mode
        gate.name = name
        return gate

    def _qasm(self) -> str:
        return self._qasm_customized(self.name)


class ParametricSingleGate(SingleGate):
    r"""A base class for single-qubit gates with parameters.

    Args:
        name (str or None, optional): The name of the gate. Default: ``None``
        inputs (Any, optional): The parameters of the gate. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
        requires_grad (bool, optional): Whether the parameters are ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        name: Optional[str] = None,
        inputs: Any = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=controls, condition=condition,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.npara = 1
        self.requires_grad = requires_grad
        self.inv_mode = False
        self.init_para(inputs)

    def inputs_to_tensor(self, inputs: Any = None) -> torch.Tensor:
        """Convert inputs to torch.Tensor."""
        while isinstance(inputs, list):
            inputs = inputs[0]
        if inputs is None:
            inputs = torch.rand(1)[0] * 4 * torch.pi
        elif not isinstance(inputs, (torch.Tensor, nn.Parameter)):
            inputs = torch.tensor(inputs, dtype=torch.float)
        return inputs

    def update_matrix(self) -> torch.Tensor:
        """Update the local unitary matrix."""
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
        matrix = self.get_matrix(theta)
        self.matrix = matrix.detach()
        return matrix

    def get_derivative(self, theta: Any) -> torch.Tensor:
        """Get the derivative of the local unitary matrix."""
        theta = self.inputs_to_tensor(theta).squeeze()
        du_dx = jacobian(self._real_wrapper, theta)
        return du_dx[..., 0] + du_dx[..., 1] * 1j

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        theta = self.inputs_to_tensor(inputs)
        if self.requires_grad:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('theta', theta)
        self.update_matrix()

    def inverse(self) -> 'ParametricSingleGate':
        """Get the inversed gate."""
        gate = copy(self)
        gate.inv_mode = not self.inv_mode
        return gate

    def extra_repr(self) -> str:
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
        s = f'wires={self.wires}, theta={theta.item()}'
        if self.controls == []:
            return s
        else:
            return s + f', controls={self.controls}'


class ParametricDoubleGate(DoubleGate):
    r"""A base class for two-qubit gates with parameters.

    Args:
        name (str or None, optional): The name of the gate. Default: ``None``
        inputs (Any, optional): The parameters of the gate. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 2
        wires (List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
        requires_grad (bool, optional): Whether the parameters are ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        name: Optional[str] = None,
        inputs: Any = None,
        nqubit: int = 2,
        wires: Optional[List[int]] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=controls, condition=condition,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.npara = 1
        self.requires_grad = requires_grad
        self.inv_mode = False
        self.init_para(inputs)

    def inputs_to_tensor(self, inputs: Any = None) -> torch.Tensor:
        """Convert inputs to torch.Tensor."""
        while isinstance(inputs, list):
            inputs = inputs[0]
        if inputs is None:
            inputs = torch.rand(1)[0] * 4 * torch.pi
        elif not isinstance(inputs, (torch.Tensor, nn.Parameter)):
            inputs = torch.tensor(inputs, dtype=torch.float)
        return inputs

    def update_matrix(self) -> torch.Tensor:
        """Update the local unitary matrix."""
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
        matrix = self.get_matrix(theta)
        self.matrix = matrix.detach()
        return matrix

    def get_derivative(self, theta: Any) -> torch.Tensor:
        """Get the derivative of the local unitary matrix."""
        theta = self.inputs_to_tensor(theta).squeeze()
        du_dx = jacobian(self._real_wrapper, theta)
        return du_dx[..., 0] + du_dx[..., 1] * 1j

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        theta = self.inputs_to_tensor(inputs)
        if self.requires_grad:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('theta', theta)
        self.update_matrix()

    def inverse(self) -> 'ParametricDoubleGate':
        """Get the inversed gate."""
        gate = copy(self)
        gate.inv_mode = not self.inv_mode
        return gate

    def extra_repr(self) -> str:
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
        s = f'wires={self.wires}, theta={theta.item()}'
        if self.controls == []:
            return s
        else:
            return s + f', controls={self.controls}'


class U3Gate(ParametricSingleGate):
    r"""U3 gate, a generic single-qubit rotation gate with 3 angles.

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        U_3(\theta, \phi, \lambda) =
            \begin{pmatrix}
                \cos\left(\th\right)          & -e^{i\lambda}\sin\left(\th\right) \\
                e^{i\phi}\sin\left(\th\right) & e^{i(\phi+\lambda)}\cos\left(\th\right)
            \end{pmatrix}

    Args:
        inputs (Any, optional): The parameters of the gate (:math:`\theta`, :math:`\phi` and :math:`\lambda`).
            Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
        requires_grad (bool, optional): Whether the parameters are ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name='U3Gate', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         condition=condition, den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)
        self.npara = 3

    def inputs_to_tensor(self, inputs: Any = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert inputs to torch.Tensor."""
        if inputs is None:
            theta = torch.rand(1)[0] * torch.pi
            phi   = torch.rand(1)[0] * 2 * torch.pi
            lambd = torch.rand(1)[0] * 2 * torch.pi
        else:
            theta = inputs[0]
            phi   = inputs[1]
            lambd = inputs[2]
        if not isinstance(theta, (torch.Tensor, nn.Parameter)):
            theta = torch.tensor(theta, dtype=torch.float)
        if not isinstance(phi, (torch.Tensor, nn.Parameter)):
            phi = torch.tensor(phi, dtype=torch.float)
        if not isinstance(lambd, (torch.Tensor, nn.Parameter)):
            lambd = torch.tensor(lambd, dtype=torch.float)
        return theta, phi, lambd

    def get_matrix(self, theta: Any, phi: Any, lambd: Any) -> torch.Tensor:
        """Get the local unitary matrix."""
        theta, phi, lambd = self.inputs_to_tensor([theta, phi, lambd])
        cos_t = torch.cos(theta / 2)
        sin_t = torch.sin(theta / 2)
        e_il  = torch.exp(1j * lambd)
        e_ip  = torch.exp(1j * phi)
        e_ipl = torch.exp(1j * (phi + lambd))
        return torch.stack([cos_t, -e_il * sin_t, e_ip * sin_t, e_ipl * cos_t]).reshape(2, 2)

    def update_matrix(self) -> torch.Tensor:
        """Update the local unitary matrix."""
        if self.inv_mode:
            theta = -self.theta
            phi   = -self.lambd
            lambd = -self.phi
        else:
            theta = self.theta
            phi   = self.phi
            lambd = self.lambd
        matrix = self.get_matrix(theta, phi, lambd)
        self.matrix = matrix.detach()
        return matrix

    def _real_wrapper(self, x: torch.Tensor) -> torch.Tensor:
        mat = self.get_matrix(x[0], x[1], x[2])
        return torch.view_as_real(mat)

    def get_derivative(self, inputs: Any) -> torch.Tensor:
        """Get the derivatives of the local unitary matrix."""
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float)
        inputs = inputs.reshape(self.npara)
        du_dx = jacobian(self._real_wrapper, inputs).permute(3, 0, 1, 2)
        return du_dx[..., 0] + du_dx[..., 1] * 1j

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        theta, phi, lambd = self.inputs_to_tensor(inputs)
        if self.requires_grad:
            self.theta = nn.Parameter(theta)
            self.phi   = nn.Parameter(phi)
            self.lambd = nn.Parameter(lambd)
        else:
            self.register_buffer('theta', theta)
            self.register_buffer('phi', phi)
            self.register_buffer('lambd', lambd)
        self.update_matrix()

    def extra_repr(self) -> str:
        if self.inv_mode:
            theta = -self.theta
            phi   = -self.lambd
            lambd = -self.phi
        else:
            theta = self.theta
            phi   = self.phi
            lambd = self.lambd
        s = f'wires={self.wires}, theta={theta.item()}, phi={phi.item()}, lambda={lambd.item()}'
        if self.controls == []:
            return s
        else:
            return s + f', controls={self.controls}'

    def _qasm(self) -> str:
        if self.inv_mode:
            theta = -self.theta
            phi   = -self.lambd
            lambd = -self.phi
        else:
            theta = self.theta
            phi   = self.phi
            lambd = self.lambd
        if self.condition:
            return self._qasm_cond_measure() + f'u({theta.item()},{phi.item()},{lambd.item()}) q{self.wires};\n'
        if self.controls == []:
            return f'u({theta.item()},{phi.item()},{lambd.item()}) q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'cu({theta.item()},{phi.item()},{lambd.item()},0.0) q{self.controls},q{self.wires};\n'
        else:
            return self._qasm_customized('u')


class PhaseShift(ParametricSingleGate):
    r"""Phase shift gate.

    **Matrix Representation:**

    .. math::

        P(\theta) =
            \begin{pmatrix}
                1 & 0 \\
                0 & e^{i\theta}
            \end{pmatrix}

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name='PhaseShift', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         condition=condition, den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)

    def inputs_to_tensor(self, inputs: Any = None) -> torch.Tensor:
        """Convert inputs to torch.Tensor."""
        while isinstance(inputs, list):
            inputs = inputs[0]
        if inputs is None:
            inputs = torch.rand(1)[0] * 2 * torch.pi
        elif not isinstance(inputs, (torch.Tensor, nn.Parameter)):
            inputs = torch.tensor(inputs, dtype=torch.float)
        return inputs

    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Get the local unitary matrix."""
        theta = self.inputs_to_tensor(theta)
        m1 = torch.eye(1, dtype=theta.dtype, device=theta.device)
        e_it = torch.exp(1j * theta)
        return torch.block_diag(m1, e_it).reshape(2, 2)

    def _qasm(self) -> str:
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
        if self.condition:
            return self._qasm_cond_measure() + f'p({theta.item()}) q{self.wires};\n'
        if self.controls == []:
            return f'p({theta.item()}) q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'cp({theta.item()}) q{self.controls},q{self.wires};\n'
        else:
            return self._qasm_customized('p')


class Identity(Gate):
    r"""Identity gate.

    **Matrix Representation:**

    .. math::

        I = \begin{pmatrix}
                1 & 0 \\
                0 & 1
            \end{pmatrix}

    Args:
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='Identity', nqubit=nqubit, wires=wires, controls=None, condition=False,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.eye(2 ** self.nqubit, dtype=torch.cfloat))

    def get_unitary(self) -> torch.Tensor:
        """Get the global unitary matrix."""
        return self.matrix

    def forward(self, x: Any) -> Any:
        """Perform a forward pass."""
        return x


class PauliX(SingleGate):
    r"""PauliX gate.

    **Matrix Representation:**

    .. math::

        X = \begin{pmatrix}
                0 & 1 \\
                1 & 0
            \end{pmatrix}

    Args:
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='PauliX', nqubit=nqubit, wires=wires, controls=controls, condition=condition,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat))

    def _qasm(self) -> str:
        if self.condition:
            return self._qasm_cond_measure() + f'x q{self.wires};\n'
        if self.controls == []:
            return f'x q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'cx q{self.controls},q{self.wires};\n'
        elif len(self.controls) == 2:
            return f'ccx q[{self.controls[0]}],q[{self.controls[1]}],q{self.wires};\n'
        else:
            return self._qasm_customized('x')

    def pattern(self, nodes: Union[int, List[int]], ancilla: List[int]) -> nn.Sequential:
        """Get the MBQC pattern."""
        from .mbqc import Node, Entanglement, Measurement, Correction
        if isinstance(nodes, list):
            assert len(nodes) == len(self.wires)
            nodes = nodes[0]
        assert len(ancilla) == self.nancilla
        cmds = nn.Sequential()
        cmds.append(Node(ancilla))
        cmds.append(Entanglement(nodes, ancilla[0]))
        cmds.append(Entanglement(ancilla[0], ancilla[1]))
        cmds.append(Measurement(nodes))
        cmds.append(Measurement(ancilla[0], angle=-torch.pi))
        cmds.append(Correction(ancilla[1], basis='x', domain=ancilla[0]))
        cmds.append(Correction(ancilla[1], basis='z', domain=nodes))
        self.nodes = [ancilla[1]]
        return cmds


class PauliY(SingleGate):
    r"""PauliY gate.

    **Matrix Representation:**

    .. math::

        Y = \begin{pmatrix}
                0 & -i \\
                i & 0
            \end{pmatrix}

    Args:
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='PauliY', nqubit=nqubit, wires=wires, controls=controls, condition=condition,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[0, -1j], [1j, 0]]))
        # MBQC
        self.nancilla = 4

    def _qasm(self) -> str:
        if self.condition:
            return self._qasm_cond_measure() + f'y q{self.wires};\n'
        if self.controls == []:
            return f'y q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'cy q{self.controls},q{self.wires};\n'
        else:
            return self._qasm_customized('y')

    def pattern(self, nodes: Union[int, List[int]], ancilla: List[int]) -> nn.Sequential:
        """Get the MBQC pattern."""
        from .mbqc import Node, Entanglement, Measurement, Correction
        if isinstance(nodes, list):
            assert len(nodes) == len(self.wires)
            nodes = nodes[0]
        assert len(ancilla) == self.nancilla
        cmds = nn.Sequential()
        cmds.append(Node(ancilla))
        cmds.append(Entanglement(nodes, ancilla[0]))
        cmds.append(Entanglement(ancilla[0], ancilla[1]))
        cmds.append(Entanglement(ancilla[1], ancilla[2]))
        cmds.append(Entanglement(ancilla[2], ancilla[3]))
        cmds.append(Measurement(nodes, angle=torch.pi/2))
        cmds.append(Measurement(ancilla[0], angle=torch.pi, s_domain=nodes))
        cmds.append(Measurement(ancilla[1], angle=-torch.pi/2, s_domain=nodes))
        cmds.append(Measurement(ancilla[2]))
        cmds.append(Correction(ancilla[3], basis='x', domain=[ancilla[0], ancilla[2]]))
        cmds.append(Correction(ancilla[3], basis='z', domain=[ancilla[0], ancilla[1]]))
        self.nodes = [ancilla[3]]
        return cmds


class PauliZ(SingleGate):
    r"""PauliZ gate.

    **Matrix Representation:**

    .. math::

        Z = \begin{pmatrix}
                1 & 0 \\
                0 & -1
            \end{pmatrix}

    Args:
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='PauliZ', nqubit=nqubit, wires=wires, controls=controls, condition=condition,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat))

    def _qasm(self) -> str:
        if self.condition:
            return self._qasm_cond_measure() + f'z q{self.wires};\n'
        if self.controls == []:
            return f'z q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'cz q{self.controls},q{self.wires};\n'
        else:
            return self._qasm_customized('z')

    def pattern(self, nodes: Union[int, List[int]], ancilla: List[int]) -> nn.Sequential:
        """Get the MBQC pattern."""
        from .mbqc import Node, Entanglement, Measurement, Correction
        if isinstance(nodes, list):
            assert len(nodes) == len(self.wires)
            nodes = nodes[0]
        assert len(ancilla) == self.nancilla
        cmds = nn.Sequential()
        cmds.append(Node(ancilla))
        cmds.append(Entanglement(nodes, ancilla[0]))
        cmds.append(Entanglement(ancilla[0], ancilla[1]))
        cmds.append(Measurement(nodes, angle=-torch.pi))
        cmds.append(Measurement(ancilla[0]))
        cmds.append(Correction(ancilla[1], basis='x', domain=ancilla[0]))
        cmds.append(Correction(ancilla[1], basis='z', domain=nodes))
        self.nodes = [ancilla[1]]
        return cmds


class Hadamard(SingleGate):
    r"""Hadamard gate.

    **Matrix Representation:**

    .. math::

        H = \frac{1}{\sqrt{2}}
            \begin{pmatrix}
                1 & 1 \\
                1 & -1
            \end{pmatrix}

    Args:
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='Hadamard', nqubit=nqubit, wires=wires, controls=controls, condition=condition,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat) / 2 ** 0.5)
        # MBQC
        self.nancilla = 1

    def _qasm(self) -> str:
        if self.condition:
            return self._qasm_cond_measure() + f'h q{self.wires};\n'
        if self.controls == []:
            return f'h q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'ch q{self.controls},q{self.wires};\n'
        else:
            return self._qasm_customized('h')

    def pattern(self, nodes: Union[int, List[int]], ancilla: Union[int, List[int]]) -> nn.Sequential:
        """Get the MBQC pattern."""
        from .mbqc import Node, Entanglement, Measurement, Correction
        if isinstance(nodes, list):
            assert len(nodes) == len(self.wires)
            nodes = nodes[0]
        if isinstance(ancilla, list):
            assert len(ancilla) == self.nancilla
            ancilla = ancilla[0]
        cmds = nn.Sequential()
        cmds.append(Node(ancilla))
        cmds.append(Entanglement(nodes, ancilla))
        cmds.append(Measurement(nodes))
        cmds.append(Correction(ancilla, basis='x', domain=nodes))
        self.nodes = [ancilla]
        return cmds


class SGate(SingleGate):
    r"""S gate.

    **Matrix Representation:**

    .. math::

        S = \begin{pmatrix}
                1 & 0 \\
                0 & i
            \end{pmatrix}

    Args:
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='SGate', nqubit=nqubit, wires=wires, controls=controls, condition=condition,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0], [0, 1j]]))

    def inverse(self) -> 'SDaggerGate':
        """Get the inversed gate."""
        return SDaggerGate(nqubit=self.nqubit, wires=self.wires, controls=self.controls,
                           condition=self.condition, den_mat=self.den_mat, tsr_mode=self.tsr_mode)

    def _qasm(self) -> str:
        if self.condition:
            return self._qasm_cond_measure() + f's q{self.wires};\n'
        if self.controls == []:
            return f's q{self.wires};\n'
        elif len(self.controls) == 1:
            qasm_str1 = ''
            qasm_str2 = f'cs q{self.controls},q{self.wires};\n'
            # pylint: disable=protected-access
            if 'cs' not in Gate._qasm_new_gate:
                qasm_str1 = 'gate cs q0,q1 { p(pi/4) q0; cx q0,q1; p(-pi/4) q1; cx q0,q1; p(pi/4) q1; }\n'
                Gate._qasm_new_gate.append('cs')
            return qasm_str1 + qasm_str2
        else:
            return self._qasm_customized('s')

    def pattern(self, nodes: Union[int, List[int]], ancilla: List[int]) -> nn.Sequential:
        """Get the MBQC pattern."""
        from .mbqc import Node, Entanglement, Measurement, Correction
        if isinstance(nodes, list):
            assert len(nodes) == len(self.wires)
            nodes = nodes[0]
        assert len(ancilla) == self.nancilla
        cmds = nn.Sequential()
        cmds.append(Node(ancilla))
        cmds.append(Entanglement(nodes, ancilla[0]))
        cmds.append(Entanglement(ancilla[0], ancilla[1]))
        cmds.append(Measurement(nodes, angle=-torch.pi/2))
        cmds.append(Measurement(ancilla[0]))
        cmds.append(Correction(ancilla[1], basis='x', domain=ancilla[0]))
        cmds.append(Correction(ancilla[1], basis='z', domain=nodes))
        self.nodes = [ancilla[1]]
        return cmds


class SDaggerGate(SingleGate):
    r"""S dagger gate.

    **Matrix Representation:**

    .. math::

        S^{\dagger} =
            \begin{pmatrix}
                1 & 0 \\
                0 & -i
            \end{pmatrix}

    Args:
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='SDaggerGate', nqubit=nqubit, wires=wires, controls=controls,
                         condition=condition, den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0], [0, -1j]]))

    def inverse(self) -> SGate:
        """Get the inversed gate."""
        return SGate(nqubit=self.nqubit, wires=self.wires, controls=self.controls,
                     condition=self.condition, den_mat=self.den_mat, tsr_mode=self.tsr_mode)

    def _qasm(self) -> str:
        if self.condition:
            return self._qasm_cond_measure() + f'sdg q{self.wires};\n'
        if self.controls == []:
            return f'sdg q{self.wires};\n'
        elif len(self.controls) == 1:
            qasm_str1 = ''
            qasm_str2 = f'csdg q{self.controls},q{self.wires};\n'
            # pylint: disable=protected-access
            if 'csdg' not in Gate._qasm_new_gate:
                qasm_str1 = 'gate csdg q0,q1 { p(-pi/4) q0; cx q0,q1; p(pi/4) q1; cx q0,q1; p(-pi/4) q1; }\n'
                Gate._qasm_new_gate.append('csdg')
            return qasm_str1 + qasm_str2
        else:
            return self._qasm_customized('sdg')


class TGate(SingleGate):
    r"""T gate.

    **Matrix Representation:**

    .. math::

        T = \begin{pmatrix}
                1 & 0 \\
                0 & e^{i\pi/4}
            \end{pmatrix}

    Args:
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='TGate', nqubit=nqubit, wires=wires, controls=controls, condition=condition,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0], [0, (1 + 1j) / 2 ** 0.5]]))

    def inverse(self) -> 'TDaggerGate':
        """Get the inversed gate."""
        return TDaggerGate(nqubit=self.nqubit, wires=self.wires, controls=self.controls,
                           condition=self.condition, den_mat=self.den_mat, tsr_mode=self.tsr_mode)

    def _qasm(self) -> str:
        if self.condition:
            return self._qasm_cond_measure() + f't q{self.wires};\n'
        if self.controls == []:
            return f't q{self.wires};\n'
        else:
            return self._qasm_customized('t')


class TDaggerGate(SingleGate):
    r"""T dagger gate.

    **Matrix Representation:**

    .. math::

        T^{\dagger} =
            \begin{pmatrix}
                1 & 0 \\
                0 & e^{-i\pi/4}
            \end{pmatrix}

    Args:
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='TDaggerGate', nqubit=nqubit, wires=wires, controls=controls,
                         condition=condition, den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0], [0, (1 - 1j) / 2 ** 0.5]]))

    def inverse(self) -> TGate:
        """Get the inversed gate."""
        return TGate(nqubit=self.nqubit, wires=self.wires, controls=self.controls,
                     condition=self.condition, den_mat=self.den_mat, tsr_mode=self.tsr_mode)

    def _qasm(self) -> str:
        if self.condition:
            return self._qasm_cond_measure() + f'tdg q{self.wires};\n'
        if self.controls == []:
            return f'tdg q{self.wires};\n'
        else:
            return self._qasm_customized('tdg')


class Rx(ParametricSingleGate):
    r"""Rx gate, rotation around x-axis.

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        R_x(\theta) =
            \begin{pmatrix}
                \cos\left(\th\right)   & -i\sin\left(\th\right) \\
                -i\sin\left(\th\right) & \cos\left(\th\right)
            \end{pmatrix}

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name='Rx', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         condition=condition, den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)
        # MBQC
        self.idx_enc = [4] # index of the commands to encode

    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Get the local unitary matrix."""
        theta = self.inputs_to_tensor(theta)
        cos  = torch.cos(theta / 2)
        isin = torch.sin(theta / 2) * 1j
        return torch.stack([cos, -isin, -isin, cos]).reshape(2, 2)

    def _qasm(self) -> str:
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
        if self.condition:
            return self._qasm_cond_measure() + f'rx({theta.item()}) q{self.wires};\n'
        if self.controls == []:
            return f'rx({theta.item()}) q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'crx({theta.item()}) q{self.controls},q{self.wires};\n'
        else:
            return self._qasm_customized('rx')

    def pattern(
        self,
        nodes: Union[int, List[int]],
        ancilla: List[int],
        angle: Any,
        requires_grad: bool = False
    ) -> nn.Sequential:
        """Get the MBQC pattern."""
        from .mbqc import Node, Entanglement, Measurement, Correction
        if isinstance(nodes, list):
            assert len(nodes) == len(self.wires)
            nodes = nodes[0]
        assert len(ancilla) == self.nancilla
        cmds = nn.Sequential()
        cmds.append(Node(ancilla))
        cmds.append(Entanglement(nodes, ancilla[0]))
        cmds.append(Entanglement(ancilla[0], ancilla[1]))
        cmds.append(Measurement(nodes))
        cmds.append(Measurement(ancilla[0], angle=-angle, s_domain=nodes, requires_grad=requires_grad))
        cmds.append(Correction(ancilla[1], basis='x', domain=ancilla[0]))
        cmds.append(Correction(ancilla[1], basis='z', domain=nodes))
        self.nodes = [ancilla[1]]
        return cmds


class Ry(ParametricSingleGate):
    r"""Ry gate, rotation around y-axis.

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        R_y(\theta) =
            \begin{pmatrix}
                \cos\left(\th\right) & -\sin\left(\th\right) \\
                \sin\left(\th\right) & \cos\left(\th\right)
            \end{pmatrix}

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name='Ry', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         condition=condition, den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)
        # MBQC
        self.nancilla = 4
        self.idx_enc = [6] # index of the commands to encode

    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Get the local unitary matrix."""
        theta = self.inputs_to_tensor(theta)
        cos = torch.cos(theta / 2)
        sin = torch.sin(theta / 2)
        return torch.stack([cos, -sin, sin, cos]).reshape(2, 2) + 0j

    def _qasm(self) -> str:
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
        if self.condition:
            return self._qasm_cond_measure() + f'ry({theta.item()}) q{self.wires};\n'
        if self.controls == []:
            return f'ry({theta.item()}) q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'cry({theta.item()}) q{self.controls},q{self.wires};\n'
        else:
            return self._qasm_customized('ry')

    def pattern(
        self,
        nodes: Union[int, List[int]],
        ancilla: List[int],
        angle: Any,
        requires_grad: bool = False
    ) -> nn.Sequential:
        """Get the MBQC pattern."""
        from .mbqc import Node, Entanglement, Measurement, Correction
        if isinstance(nodes, list):
            assert len(nodes) == len(self.wires)
            nodes = nodes[0]
        assert len(ancilla) == self.nancilla
        cmds = nn.Sequential()
        cmds.append(Node(ancilla))
        cmds.append(Entanglement(nodes, ancilla[0]))
        cmds.append(Entanglement(ancilla[0], ancilla[1]))
        cmds.append(Entanglement(ancilla[1], ancilla[2]))
        cmds.append(Entanglement(ancilla[2], ancilla[3]))
        cmds.append(Measurement(nodes, angle=torch.pi / 2))
        cmds.append(Measurement(ancilla[0], angle=-angle, s_domain=nodes, requires_grad=requires_grad))
        cmds.append(Measurement(ancilla[1], angle=-torch.pi / 2, s_domain=nodes))
        cmds.append(Measurement(ancilla[2]))
        cmds.append(Correction(ancilla[3], basis='x', domain=[ancilla[0], ancilla[2]]))
        cmds.append(Correction(ancilla[3], basis='z', domain=[ancilla[0], ancilla[1]]))
        self.nodes = [ancilla[3]]
        return cmds


class Rz(ParametricSingleGate):
    r"""Rz gate, rotation around z-axis.

    **Matrix Representation:**

    .. math::

        R_z(\theta) =
            \begin{pmatrix}
                e^{-i\frac{\theta}{2}} & 0 \\
                0                      & e^{i\frac{\theta}{2}}
            \end{pmatrix}

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name='Rz', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         condition=condition, den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)
        # MBQC
        self.idx_enc = [3] # index of the commands to encode

    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Get the local unitary matrix."""
        theta = self.inputs_to_tensor(theta)
        e_m_it = torch.exp(-1j * theta / 2)
        e_it = torch.exp(1j * theta / 2)
        return torch.stack([e_m_it, e_it]).reshape(-1).diag_embed().reshape(2, 2)

    def _qasm(self) -> str:
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
        if self.condition:
            return self._qasm_cond_measure() + f'rz({theta.item()}) q{self.wires};\n'
        if self.controls == []:
            return f'rz({theta.item()}) q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'crz({theta.item()}) q{self.controls},q{self.wires};\n'
        else:
            return self._qasm_customized('rz')

    def pattern(
        self,
        nodes: Union[int, List[int]],
        ancilla: List[int],
        angle: Any,
        requires_grad: bool = False
    ) -> nn.Sequential:
        """Get the MBQC pattern."""
        from .mbqc import Node, Entanglement, Measurement, Correction
        if isinstance(nodes, list):
            assert len(nodes) == len(self.wires)
            nodes = nodes[0]
        assert len(ancilla) == self.nancilla
        cmds = nn.Sequential()
        cmds.append(Node(ancilla))
        cmds.append(Entanglement(nodes, ancilla[0]))
        cmds.append(Entanglement(ancilla[0], ancilla[1]))
        cmds.append(Measurement(nodes, angle=-angle, requires_grad=requires_grad))
        cmds.append(Measurement(ancilla[0]))
        cmds.append(Correction(ancilla[1], basis='x', domain=ancilla[0]))
        cmds.append(Correction(ancilla[1], basis='z', domain=nodes))
        self.nodes = [ancilla[1]]
        return cmds


class ProjectionJ(ParametricSingleGate):
    r"""Matrix J to project the measurement basis.

    The matrix :math:`J(\theta)` transforms the measurement in the XY, YZ, or ZX plane to Z measurement.
    The transformation means :math:`M(\theta)|\psi\rangle = M_z J(\theta)|\psi\rangle`,
    where :math:`M_z` is the Z-basis measurement.

    **Matrix Representation:**

    .. math::

        J_{\text{XY}}(\theta) = \frac{1}{\sqrt{2}}
            \begin{pmatrix}
                1 &  e^{-i\theta} \\
                1 & -e^{-i\theta}
            \end{pmatrix}

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        J_{\text{YZ}}(\theta) = \frac{1}{\sqrt{2}}
            \begin{pmatrix}
                \cos(\th) + \sin(\th) & -i(\cos(\th) - \sin(\th)) \\
                \cos(\th) - \sin(\th) &  i(\cos(\th) + \sin(\th))
            \end{pmatrix}

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        J_{\text{ZX}}(\theta) =
            \begin{pmatrix}
                \cos(\th) &  \sin(\th) \\
                \sin(\th) & -\cos(\th)
            \end{pmatrix}

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        plane (str, optional): The measurement plane (``'xy'``, ``'yz'``, or ``'zx'``). Default: ``'xy'``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        plane: str = 'xy',
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        self.plane = plane.lower()
        super().__init__(name='ProjectionJ', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         condition=condition, den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)

    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Get the local unitary matrix."""
        theta = self.inputs_to_tensor(theta)
        if self.plane in ['xy', 'yx']:
            one = torch.ones_like(theta)
            e_m_theta = torch.exp(-1j * theta)
            return torch.stack([one, e_m_theta, one, -e_m_theta]).reshape(2, 2) / 2 ** 0.5
        elif self.plane in ['yz', 'zy']:
            c_p_s = torch.cos(theta / 2) + torch.sin(theta / 2)
            c_m_s = torch.cos(theta / 2) - torch.sin(theta / 2)
            return torch.stack([c_p_s, -1j * c_m_s, c_m_s, 1j * c_p_s]).reshape(2, 2) / 2 ** 0.5
        elif self.plane in ['zx', 'xz']:
            cos = torch.cos(theta / 2)
            sin = torch.sin(theta / 2)
            return torch.stack([cos, sin, sin, -cos]).reshape(2, 2) + 0j
        else:
            raise ValueError(f'Unsupported measurement plane: {self.plane}')

    def extra_repr(self) -> str:
        return super().extra_repr() + f', plane={self.plane}'

    def _qasm(self) -> str:
        if self.condition:
            name = 'j'
            qasm_lst1 = [f'opaque {name} ']
            qasm_lst2 = [f'{name} ']
            for i, wire in enumerate(self.wires):
                qasm_lst1.append(f'q{i},')
                qasm_lst2.append(f'q[{wire}],')
            qasm_str1 = ''.join(qasm_lst1)[:-1] + ';\n'
            qasm_str2 = ''.join(qasm_lst2)[:-1] + ';\n'
            # pylint: disable=protected-access
            if name not in Gate._qasm_new_gate:
                Gate._qasm_new_gate.append(name)
                return qasm_str1 + self._qasm_cond_measure() + qasm_str2
            else:
                return self._qasm_cond_measure() + qasm_str2
        return self._qasm_customized('j')


class CombinedSingleGate(SingleGate):
    r"""Combined single-qubit gate.

    Args:
        gates (List[SingleGate]): The list of single-qubit gates.
        name (str or None, optional): The name of the gate. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
    """
    def __init__(
        self,
        gates: List[SingleGate],
        name: Optional[str] = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=controls, condition=condition,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        for gate in gates:
            gate.nqubit = self.nqubit
            gate.wires = self.wires
            gate.controls = self.controls
            gate.condition = self.condition
            gate.den_mat = self.den_mat
            gate.tsr_mode = self.tsr_mode
        self.gates = nn.ModuleList(gates)
        self.update_npara()
        self.update_matrix()

    def get_matrix(self) -> torch.Tensor:
        """Get the local unitary matrix."""
        matrix = None
        for gate in self.gates:
            if matrix is None:
                matrix = gate.update_matrix()
            else:
                matrix = gate.update_matrix() @ matrix
        return matrix

    def update_matrix(self) -> torch.Tensor:
        """Update the local unitary matrix."""
        matrix = self.get_matrix()
        self.matrix = matrix.detach()
        return matrix

    def get_derivative(self, inputs: Any) -> torch.Tensor:
        """Get the derivatives of the local unitary matrix."""
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float)
        inputs = inputs.reshape(self.npara)
        derivatives = []
        count = 0
        for gate in self.gates:
            du_dx = gate.get_derivative(inputs[count:count+gate.npara])
            if du_dx.ndim == 2:
                du_dx = du_dx.unsqueeze(0)
            derivatives.append(du_dx)
            count += gate.npara
        return torch.cat(derivatives)

    def update_npara(self) -> None:
        """Update the number of parameters."""
        self.npara = 0
        for gate in self.gates:
            self.npara += gate.npara

    def add(self, gate: SingleGate) -> None:
        """Add a single-qubit gate to the list and update the local unitary matrix."""
        gate.nqubit = self.nqubit
        gate.wires = self.wires
        gate.controls = self.controls
        gate.condition = self.condition
        gate.den_mat = self.den_mat
        gate.tsr_mode = self.tsr_mode
        self.gates.append(gate)
        self.matrix = gate.matrix @ self.matrix
        self.npara += gate.npara

    def inverse(self) -> 'CombinedSingleGate':
        """Get the inversed gate."""
        gates = nn.ModuleList()
        for gate in reversed(self.gates):
            gates.append(gate.inverse())
        return CombinedSingleGate(gates=gates, name=self.name, nqubit=self.nqubit, wires=self.wires,
                                  controls=self.controls, condition=self.condition, den_mat=self.den_mat,
                                  tsr_mode=self.tsr_mode)

    def _qasm(self) -> str:
        lst = []
        for gate in self.gates:
            # pylint: disable=protected-access
            lst.append(gate._qasm())
        return ''.join(lst)


class CNOT(DoubleControlGate):
    r"""CNOT gate.

    **Matrix Representation:**

    .. math::

        \text{CNOT} =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 1 \\
                0 & 0 & 1 & 0
            \end{pmatrix}

    Args:
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 2
        wires (List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 2,
        wires: Optional[List[int]] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='CNOT', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0, 0, 0],
                                                     [0, 1, 0, 0],
                                                     [0, 0, 0, 1],
                                                     [0, 0, 1, 0]]) + 0j)
        # MBQC
        self.nancilla = 2

    def _qasm(self) -> str:
        return f'cx q[{self.wires[0]}],q[{self.wires[1]}];\n'

    def pattern(self, nodes: List[int], ancilla: List[int]) -> nn.Sequential:
        """Get the MBQC pattern."""
        from .mbqc import Node, Entanglement, Measurement, Correction
        assert len(nodes) == len(self.wires)
        assert len(ancilla) == self.nancilla
        control = nodes[0]
        target = nodes[1]
        cmds = nn.Sequential()
        cmds.append(Node(ancilla))
        cmds.append(Entanglement(target, ancilla[0]))
        cmds.append(Entanglement(control, ancilla[0]))
        cmds.append(Entanglement(ancilla[0], ancilla[1]))
        cmds.append(Measurement(target))
        cmds.append(Measurement(ancilla[0]))
        cmds.append(Correction(ancilla[1], basis='x', domain=ancilla[0]))
        cmds.append(Correction(ancilla[1], basis='z', domain=target))
        cmds.append(Correction(control, basis='z', domain=target))
        self.nodes = [control, ancilla[1]]
        return cmds


class Swap(DoubleGate):
    r"""Swap gate.

    **Matrix Representation:**

    .. math::
        \text{SWAP} =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}

    Args:
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 2
        wires (List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 2,
        wires: Optional[List[int]] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='Swap', nqubit=nqubit, wires=wires, controls=controls, condition=condition,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0, 0, 0],
                                                     [0, 0, 1, 0],
                                                     [0, 1, 0, 0],
                                                     [0, 0, 0, 1]]) + 0j)

    def op_dist_state(self, x: DistributedQubitState) -> DistributedQubitState:
        """Perform a forward pass of a gate for a distributed state vector."""
        if len(self.controls) > 0:
            return super().op_dist_state(x)
        else:
            return dist_swap_gate(x, self.nqubit - self.wires[0] - 1, self.nqubit - self.wires[1] - 1)

    def _qasm(self) -> str:
        if self.condition:
            return self._qasm_cond_measure() + f'swap q[{self.wires[0]}],q[{self.wires[1]}];\n'
        if self.controls == []:
            return f'swap q[{self.wires[0]}],q[{self.wires[1]}];\n'
        elif len(self.controls) == 1:
            return f'cswap q{self.controls},q[{self.wires[0]}],q[{self.wires[1]}];\n'
        else:
            return self._qasm_customized('swap')


class Rxx(ParametricDoubleGate):
    r"""Rxx gate.

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        R_{xx}(\theta) = \exp\left(-i \th X{\otimes}X\right) =
            \begin{pmatrix}
                \cos\left(\th\right)   & 0                      & 0                      & -i\sin\left(\th\right) \\
                0                      & \cos\left(\th\right)   & -i\sin\left(\th\right) & 0                      \\
                0                      & -i\sin\left(\th\right) & \cos\left(\th\right)   & 0                      \\
                -i\sin\left(\th\right) & 0                      & 0                      & \cos\left(\th\right)
            \end{pmatrix}

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 2
        wires (List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 2,
        wires: Optional[List[int]] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name='Rxx', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         condition=condition, den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)

    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Get the local unitary matrix."""
        theta = self.inputs_to_tensor(theta)
        cos  = torch.cos(theta / 2)
        isin = torch.sin(theta / 2) * 1j
        m1 = torch.stack([cos, cos, cos, cos]).reshape(-1).diag_embed().reshape(4, 4)
        m2 = torch.stack([-isin, -isin, -isin, -isin]).reshape(-1).diag_embed().fliplr().reshape(4, 4)
        return m1 + m2

    def _qasm(self) -> str:
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
        if self.condition:
            return self._qasm_cond_measure() + f'rxx({theta.item()}) q[{self.wires[0]}],q[{self.wires[1]}];\n'
        if self.controls == []:
            return f'rxx({theta.item()}) q[{self.wires[0]}],q[{self.wires[1]}];\n'
        else:
            return self._qasm_customized('rxx')


class Ryy(ParametricDoubleGate):
    r"""Ryy gate.

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        R_{yy}(\theta) = \exp\left(-i \th Y{\otimes}Y\right) =
            \begin{pmatrix}
                \cos\left(\th\right)  & 0                      & 0                      & i\sin\left(\th\right) \\
                0                     & \cos\left(\th\right)   & -i\sin\left(\th\right) & 0                     \\
                0                     & -i\sin\left(\th\right) & \cos\left(\th\right)   & 0                     \\
                i\sin\left(\th\right) & 0                      & 0                      & \cos\left(\th\right)
            \end{pmatrix}

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 2
        wires (List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 2,
        wires: Optional[List[int]] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name='Ryy', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         condition=condition, den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)

    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Get the local unitary matrix."""
        theta = self.inputs_to_tensor(theta)
        cos  = torch.cos(theta / 2)
        isin = torch.sin(theta / 2) * 1j
        m1 = torch.stack([cos, cos, cos, cos]).reshape(-1).diag_embed().reshape(4, 4)
        m2 = torch.stack([isin, -isin, -isin, isin]).reshape(-1).diag_embed().fliplr().reshape(4, 4)
        return m1 + m2

    def _qasm(self) -> str:
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
        qasm_str1 = ''
        qasm_str2 = f'ryy({theta.item()}) q[{self.wires[0]}],q[{self.wires[1]}];\n'
        # pylint: disable=protected-access
        if 'ryy' not in Gate._qasm_new_gate:
            # pylint: disable=line-too-long
            qasm_str1 = 'gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(param0) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }\n'
            Gate._qasm_new_gate.append('ryy')
        if self.condition:
            return qasm_str1 + self._qasm_cond_measure() + qasm_str2
        if self.controls == []:
            return qasm_str1 + qasm_str2
        else:
            return self._qasm_customized('ryy')


class Rzz(ParametricDoubleGate):
    r"""Rzz gate.

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        R_{zz}(\theta) = \exp\left(-i \th Z{\otimes}Z\right) =
            \begin{pmatrix}
                e^{-i \th} & 0         & 0         & 0 \\
                0          & e^{i \th} & 0         & 0 \\
                0          & 0         & e^{i \th} & 0 \\
                0          & 0         & 0         & e^{-i \th}
            \end{pmatrix}

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 2
        wires (List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 2,
        wires: Optional[List[int]] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name='Rzz', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         condition=condition, den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)

    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Get the local unitary matrix."""
        theta = self.inputs_to_tensor(theta)
        e_m_it = torch.exp(-1j * theta / 2)
        e_it = torch.exp(1j * theta / 2)
        return torch.stack([e_m_it, e_it, e_it, e_m_it]).reshape(-1).diag_embed().reshape(4, 4)

    def _qasm(self) -> str:
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
        if self.condition:
            return self._qasm_cond_measure() + f'rzz({theta.item()}) q[{self.wires[0]}],q[{self.wires[1]}];\n'
        if self.controls == []:
            return f'rzz({theta.item()}) q[{self.wires[0]}],q[{self.wires[1]}];\n'
        else:
            return self._qasm_customized('rzz')


class Rxy(ParametricDoubleGate):
    r"""Rxy gate.

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        R_{xy}(\theta) = \exp\left(-i \th X{\otimes}Y\right) =
            \begin{pmatrix}
                1 & 0                      & 0                      & 0 \\
                0 & \cos\left(\th\right)   & -i\sin\left(\th\right) & 0 \\
                0 & -i\sin\left(\th\right) & \cos\left(\th\right)   & 0 \\
                0 & 0                      & 0                      & 1
            \end{pmatrix}

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 2
        wires (List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 2,
        wires: Optional[List[int]] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name='Rxy', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         condition=condition, den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)

    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Get the local unitary matrix."""
        theta = self.inputs_to_tensor(theta)
        cos  = torch.cos(theta / 2)
        isin = torch.sin(theta / 2) * 1j
        m1 = torch.eye(1, dtype=theta.dtype, device=theta.device)
        m2 = torch.stack([cos, -isin, -isin, cos]).reshape(2, 2)
        return torch.block_diag(m1, m2, m1)

    def _qasm(self) -> str:
        if self.condition:
            name = 'rxy'
            qasm_lst1 = [f'opaque {name} ']
            qasm_lst2 = [f'{name} ']
            for i, wire in enumerate(self.wires):
                qasm_lst1.append(f'q{i},')
                qasm_lst2.append(f'q[{wire}],')
            qasm_str1 = ''.join(qasm_lst1)[:-1] + ';\n'
            qasm_str2 = ''.join(qasm_lst2)[:-1] + ';\n'
            # pylint: disable=protected-access
            if name not in Gate._qasm_new_gate:
                Gate._qasm_new_gate.append(name)
                return qasm_str1 + self._qasm_cond_measure() + qasm_str2
            else:
                return self._qasm_cond_measure() + qasm_str2
        return self._qasm_customized('rxy')


class ReconfigurableBeamSplitter(ParametricDoubleGate):
    r"""Reconfigurable Beam Splitter gate.

    **Matrix Representation:**

    .. math::

        \text{RBS}(\theta) =
            \begin{pmatrix}
                1 & 0                        & 0                       & 0 \\
                0 & \cos\left(\theta\right)  & \sin\left(\theta\right) & 0 \\
                0 & -\sin\left(\theta\right) & \cos\left(\theta\right) & 0 \\
                0 & 0                        & 0                       & 1
            \end{pmatrix}

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 2
        wires (List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        condition (bool, optional): Whether to use ``controls`` as conditional measurement. Default: ``False``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 2,
        wires: Optional[List[int]] = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name='ReconfigurableBeamSplitter', inputs=inputs, nqubit=nqubit, wires=wires,
                         controls=controls, condition=condition, den_mat=den_mat, tsr_mode=tsr_mode,
                         requires_grad=requires_grad)

    def inputs_to_tensor(self, inputs: Any = None) -> torch.Tensor:
        """Convert inputs to torch.Tensor."""
        while isinstance(inputs, list):
            inputs = inputs[0]
        if inputs is None:
            inputs = torch.rand(1)[0] * 2 * torch.pi
        elif not isinstance(inputs, (torch.Tensor, nn.Parameter)):
            inputs = torch.tensor(inputs, dtype=torch.float)
        return inputs

    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Get the local unitary matrix."""
        theta = self.inputs_to_tensor(theta)
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        m1 = torch.eye(1, dtype=theta.dtype, device=theta.device)
        m2 = torch.stack([cos, sin, -sin, cos]).reshape(2, 2) + 0j
        return torch.block_diag(m1, m2, m1)

    def _qasm(self) -> str:
        if self.condition:
            name = 'rbs'
            qasm_lst1 = [f'opaque {name} ']
            qasm_lst2 = [f'{name} ']
            for i, wire in enumerate(self.wires):
                qasm_lst1.append(f'q{i},')
                qasm_lst2.append(f'q[{wire}],')
            qasm_str1 = ''.join(qasm_lst1)[:-1] + ';\n'
            qasm_str2 = ''.join(qasm_lst2)[:-1] + ';\n'
            # pylint: disable=protected-access
            if name not in Gate._qasm_new_gate:
                Gate._qasm_new_gate.append(name)
                return qasm_str1 + self._qasm_cond_measure() + qasm_str2
            else:
                return self._qasm_cond_measure() + qasm_str2
        return self._qasm_customized('rbs')


class Toffoli(TripleGate):
    r"""Toffoli gate.

    **Matrix Representation:**

    .. math::
        \text{Toffoli} =
           \begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
            \end{pmatrix}

    Args:
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 3
        wires (List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 3,
        wires: Optional[List[int]] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='Toffoli', nqubit=nqubit, wires=wires, controls=None, condition=False,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0],
                                                     [0, 1, 0, 0, 0, 0, 0, 0],
                                                     [0, 0, 1, 0, 0, 0, 0, 0],
                                                     [0, 0, 0, 1, 0, 0, 0, 0],
                                                     [0, 0, 0, 0, 1, 0, 0, 0],
                                                     [0, 0, 0, 0, 0, 1, 0, 0],
                                                     [0, 0, 0, 0, 0, 0, 0, 1],
                                                     [0, 0, 0, 0, 0, 0, 1, 0]]) + 0j)
        # MBQC
        self.nancilla = 18

    def get_unitary(self) -> torch.Tensor:
        """Get the global unitary matrix."""
        matrix = self.update_matrix()
        identity = torch.eye(2, dtype=matrix.dtype, device=matrix.device)
        oneone = torch.tensor([[0, 0], [0, 1]], dtype=matrix.dtype, device=matrix.device)
        lst1 = [identity] * self.nqubit
        lst2 = [identity] * self.nqubit
        lst3 = [identity] * self.nqubit

        lst2[self.wires[0]] = oneone
        lst2[self.wires[1]] = oneone

        lst3[self.wires[0]] = oneone
        lst3[self.wires[1]] = oneone
        lst3[self.wires[2]] = matrix[-2:, -2:]
        return multi_kron(lst1) - multi_kron(lst2) + multi_kron(lst3)

    def _qasm(self) -> str:
        return f'ccx q[{self.wires[0]}],q[{self.wires[1]}],q[{self.wires[2]}];\n'

    def pattern(self, nodes: List[int], ancilla: List[int]) -> nn.Sequential:
        """Get the MBQC pattern."""
        from .mbqc import Node, Entanglement, Measurement, Correction
        assert len(nodes) == len(self.wires)
        assert len(ancilla) == self.nancilla
        control1 = nodes[0]
        control2 = nodes[1]
        target = nodes[2]
        cmds = nn.Sequential()
        cmds.append(Node(ancilla))
        cmds.append(Entanglement(target, ancilla[0]))
        cmds.append(Entanglement(ancilla[0], ancilla[1]))
        cmds.append(Entanglement(ancilla[1], ancilla[2]))
        cmds.append(Entanglement(ancilla[1], control2))
        cmds.append(Entanglement(control1, ancilla[14]))
        cmds.append(Entanglement(ancilla[2], ancilla[3]))
        cmds.append(Entanglement(ancilla[14], ancilla[4]))
        cmds.append(Entanglement(ancilla[3], ancilla[5]))
        cmds.append(Entanglement(ancilla[3], ancilla[4]))
        cmds.append(Entanglement(ancilla[5], ancilla[6]))
        cmds.append(Entanglement(control2, ancilla[6]))
        cmds.append(Entanglement(control2, ancilla[9]))
        cmds.append(Entanglement(ancilla[6], ancilla[7]))
        cmds.append(Entanglement(ancilla[9], ancilla[4]))
        cmds.append(Entanglement(ancilla[9], ancilla[10]))
        cmds.append(Entanglement(ancilla[7], ancilla[8]))
        cmds.append(Entanglement(ancilla[10], ancilla[11]))
        cmds.append(Entanglement(ancilla[4], ancilla[8]))
        cmds.append(Entanglement(ancilla[4], ancilla[11]))
        cmds.append(Entanglement(ancilla[4], ancilla[16]))
        cmds.append(Entanglement(ancilla[8], ancilla[12]))
        cmds.append(Entanglement(ancilla[11], ancilla[15]))
        cmds.append(Entanglement(ancilla[12], ancilla[13]))
        cmds.append(Entanglement(ancilla[16], ancilla[17]))
        cmds.append(Measurement(target))
        cmds.append(Measurement(ancilla[0], s_domain=target))
        cmds.append(Measurement(ancilla[1], s_domain=ancilla[0]))
        cmds.append(Measurement(control1))
        cmds.append(Measurement(ancilla[2], angle=-torch.pi * 7 / 4, s_domain=[ancilla[1], target]))
        cmds.append(Measurement(ancilla[14], s_domain=control1))
        cmds.append(Measurement(ancilla[3], s_domain=[ancilla[2], ancilla[0]]))
        cmds.append(Measurement(ancilla[5], angle=-torch.pi / 4,
                                s_domain=[ancilla[3], ancilla[1], ancilla[14], target]))
        cmds.append(Measurement(control2, angle=-torch.pi / 4))
        cmds.append(Measurement(ancilla[6], s_domain=[ancilla[5], ancilla[2], ancilla[0]]))
        cmds.append(Measurement(ancilla[9], s_domain=[control2, ancilla[5], ancilla[2]]))
        cmds.append(Measurement(ancilla[7], angle=-torch.pi * 7 / 4,
                                s_domain=[ancilla[6], ancilla[3], ancilla[1], ancilla[14], target]))
        cmds.append(Measurement(ancilla[10], angle=-torch.pi * 7 / 4, s_domain=[ancilla[9], ancilla[14]]))
        cmds.append(Measurement(ancilla[4], angle=-torch.pi / 4, s_domain=ancilla[14]))
        cmds.append(Measurement(ancilla[8], s_domain=[ancilla[7], ancilla[5], ancilla[2], ancilla[0]]))
        cmds.append(Measurement(ancilla[11], s_domain=[ancilla[10], control2, ancilla[5], ancilla[2]]))
        cmds.append(Measurement(ancilla[12], angle=-torch.pi / 4,
                                s_domain=[ancilla[8], ancilla[6], ancilla[3], ancilla[1], target]))
        cmds.append(Measurement(ancilla[16],
                                s_domain=[ancilla[4], control1, ancilla[2], control2, ancilla[7],
                                          ancilla[10], ancilla[2], control2, ancilla[5]]))
        cmds.append(Correction(ancilla[17], basis='x', domain=[ancilla[14], ancilla[16]]))
        cmds.append(Correction(ancilla[15], basis='x', domain=[ancilla[9], ancilla[11]]))
        cmds.append(Correction(ancilla[13], basis='x',
                               domain=[ancilla[0], ancilla[2], ancilla[5], ancilla[7], ancilla[12]]))
        cmds.append(Correction(ancilla[17], basis='z',
                               domain=[ancilla[4], ancilla[5], ancilla[7], ancilla[10], control1]))
        cmds.append(Correction(ancilla[15], basis='z', domain=[control2, ancilla[2], ancilla[5], ancilla[10]]))
        cmds.append(Correction(ancilla[13], basis='z',
                               domain=[ancilla[1], ancilla[3], ancilla[6], ancilla[8], target]))
        self.nodes = [ancilla[17], ancilla[15], ancilla[13]]
        return cmds


class Fredkin(TripleGate):
    r"""Fredkin gate.

    **Matrix Representation:**

    .. math::
        \text{Fredkin} =
           \begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
            \end{pmatrix}

    Args:
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 3
        wires (List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 3,
        wires: Optional[List[int]] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='Fredkin', nqubit=nqubit, wires=wires, controls=None, condition=False,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0],
                                                     [0, 1, 0, 0, 0, 0, 0, 0],
                                                     [0, 0, 1, 0, 0, 0, 0, 0],
                                                     [0, 0, 0, 1, 0, 0, 0, 0],
                                                     [0, 0, 0, 0, 1, 0, 0, 0],
                                                     [0, 0, 0, 0, 0, 0, 1, 0],
                                                     [0, 0, 0, 0, 0, 1, 0, 0],
                                                     [0, 0, 0, 0, 0, 0, 0, 1]]) + 0j)

    def get_unitary(self) -> torch.Tensor:
        """Get the global unitary matrix."""
        matrix = self.update_matrix()
        identity = torch.eye(2, dtype=matrix.dtype, device=matrix.device)
        zerozero = torch.tensor([[1, 0], [0, 0]], dtype=matrix.dtype, device=matrix.device)
        zeroone  = torch.tensor([[0, 1], [0, 0]], dtype=matrix.dtype, device=matrix.device)
        onezero  = torch.tensor([[0, 0], [1, 0]], dtype=matrix.dtype, device=matrix.device)
        oneone   = torch.tensor([[0, 0], [0, 1]], dtype=matrix.dtype, device=matrix.device)
        lst1 = [identity] * self.nqubit
        lst2 = [identity] * self.nqubit
        lst3 = [identity] * self.nqubit
        lst4 = [identity] * self.nqubit
        lst5 = [identity] * self.nqubit

        lst1[self.wires[0]] = zerozero

        lst2[self.wires[0]] = oneone
        lst2[self.wires[1]] = zerozero
        lst2[self.wires[2]] = matrix[-4:-2, -4:-2]

        lst3[self.wires[0]] = oneone
        lst3[self.wires[1]] = zeroone
        lst3[self.wires[2]] = matrix[-4:-2, -2:]

        lst4[self.wires[0]] = oneone
        lst4[self.wires[1]] = onezero
        lst4[self.wires[2]] = matrix[-2:, -4:-2]

        lst5[self.wires[0]] = oneone
        lst5[self.wires[1]] = oneone
        lst5[self.wires[2]] = matrix[-2:, -2:]
        return multi_kron(lst1) + multi_kron(lst2) + multi_kron(lst3) + multi_kron(lst4) + multi_kron(lst5)

    def _qasm(self) -> str:
        return f'cswap q[{self.wires[0]}],q[{self.wires[1]}],q[{self.wires[2]}];\n'


class UAnyGate(ArbitraryGate):
    r"""Arbitrary unitary gate.

    Args:
        unitary (Any): Any given unitary matrix.
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        minmax (List[int] or None, optional): The minimum and maximum indices of the qubits that the quantum
            operation acts on. Only valid when ``wires`` is ``None``. Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        name (str, optional): The name of the gate. Default: ``'UAnyGate'``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
    """
    def __init__(
        self,
        unitary: Any,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        minmax: Optional[List[int]] = None,
        controls: Union[int, List[int], None] = None,
        name: str = 'UAnyGate',
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name=name, nqubit=nqubit, wires=wires, minmax=minmax, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        if not isinstance(unitary, torch.Tensor):
            unitary = torch.tensor(unitary, dtype=torch.cfloat).reshape(-1, 2 ** len(self.wires))
        assert unitary.dtype in (torch.cfloat, torch.cdouble)
        assert unitary.shape[-1] == unitary.shape[-2] == 2 ** len(self.wires)
        assert is_unitary(unitary)
        self.register_buffer('matrix', unitary)

    def update_matrix(self) -> torch.Tensor:
        """Update the local unitary matrix."""
        if self.inv_mode:
            return self.matrix.mH
        else:
            return self.matrix


class LatentGate(ArbitraryGate):
    r"""Latent gate.

    Args:
        inputs (Any, optional): Any given real matrix.
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        minmax (List[int] or None, optional): The minimum and maximum indices of the qubits that the quantum
            operation acts on. Only valid when ``wires`` is ``None``. Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        name (str, optional): The name of the gate. Default: ``'LatentGate'``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
        requires_grad (bool, optional): Whether the parameters are ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        minmax: Optional[List[int]] = None,
        controls: Union[int, List[int], None] = None,
        name: str = 'LatentGate',
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name=name, nqubit=nqubit, wires=wires, minmax=minmax, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.requires_grad = requires_grad
        self.init_para(inputs)

    def inputs_to_tensor(self, inputs: Any = None) -> torch.Tensor:
        """Convert inputs to torch.Tensor."""
        if inputs is None:
            inputs = torch.randn(2 ** len(self.wires), 2 ** len(self.wires))
        elif not isinstance(inputs, (torch.Tensor, nn.Parameter)):
            inputs = torch.tensor(inputs, dtype=torch.float)
        return inputs.reshape(2 ** len(self.wires), 2 ** len(self.wires))

    def get_matrix(self, inputs: Any) -> torch.Tensor:
        """Get the local unitary matrix."""
        latent = self.inputs_to_tensor(inputs) + 0j
        u, _, vh = svd(latent)
        return u @ vh

    def update_matrix(self) -> torch.Tensor:
        """Update the local unitary matrix."""
        if self.inv_mode:
            latent = self.latent.mH
        else:
            latent = self.latent
        matrix = self.get_matrix(latent)
        assert matrix.shape[-1] == matrix.shape[-2] == 2 ** len(self.wires)
        self.matrix = matrix.detach()
        return matrix

    def get_derivative(self, latent: Any) -> torch.Tensor:
        """Get the derivatives of the local unitary matrix."""
        latent = self.inputs_to_tensor(latent)
        du_dx = jacobian(self._real_wrapper, latent).permute(3, 4, 0, 1, 2)
        return du_dx[..., 0] + du_dx[..., 1] * 1j

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        latent = self.inputs_to_tensor(inputs)
        if self.requires_grad:
            self.latent = nn.Parameter(latent)
        else:
            self.register_buffer('latent', latent)
        self.update_matrix()
        self.npara = self.latent.numel()


class HamiltonianGate(ArbitraryGate):
    r"""Hamiltonian gate.

    Args:
        hamiltonian (Any): The Hamiltonian. It can be a list, e.g., ``[[0.5, 'x0y1'], [-1, 'z3y1']]`` for
            :math:`0.5 * \sigma^x_0 \otimes \sigma^y_1 - \sigma^y_1 \otimes \sigma^z_3`. It can also be
            a torch.Tensor when ``wires`` or ``minmax`` is specified.
        t (Any, optional): The evolution time. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Only valid when ``hamiltonian`` is not a list. Default: ``None``
        minmax (List[int] or None, optional): The minimum and maximum indices of the qubits that the quantum
            operation acts on. Only valid when ``hamiltonian`` is not a list and ``wires`` is ``None``.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        name (str, optional): The name of the gate. Default: ``'HamiltonianGate'``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        hamiltonian: Any,
        t: Any = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        minmax: Optional[List[int]] = None,
        controls: Union[int, List[int], None] = None,
        name: str = 'HamiltonianGate',
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        self.nqubit = nqubit
        self.ham_lst = None
        if isinstance(hamiltonian, list):
            self.ham_lst = hamiltonian
            wires = None
            minmax = self.get_minmax(hamiltonian)
        super().__init__(name=name, nqubit=nqubit, wires=wires, minmax=minmax, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.npara = 1
        self.requires_grad = requires_grad
        self.register_buffer('x', PauliX().matrix)
        self.register_buffer('y', PauliY().matrix)
        self.register_buffer('z', PauliZ().matrix)
        self.init_para([hamiltonian, t])

    def to(self, arg: Any) -> 'HamiltonianGate':
        """Set dtype or device of the ``HamiltonianGate``."""
        if arg == torch.float:
            self.x = self.x.to(torch.cfloat)
            self.y = self.y.to(torch.cfloat)
            self.z = self.z.to(torch.cfloat)
            self.ham_tsr = self.ham_tsr.to(torch.cfloat)
            self.t = self.t.to(torch.float)
        elif arg == torch.double:
            self.x = self.x.to(torch.cdouble)
            self.y = self.y.to(torch.cdouble)
            self.z = self.z.to(torch.cdouble)
            self.ham_tsr = self.ham_tsr.to(torch.cdouble)
            self.t = self.t.to(torch.double)
        else:
            super().to(arg)
        return self

    def _convert_hamiltonian(self, hamiltonian: List) -> List[List]:
        """Convert and check the list representation of the Hamiltonian."""
        if len(hamiltonian) == 2 and isinstance(hamiltonian[1], str):
            hamiltonian = [hamiltonian]
        assert all(isinstance(i, list) for i in hamiltonian), 'Invalid input type'
        for pair in hamiltonian:
            assert isinstance(pair[1], str), 'Invalid input type'
        return hamiltonian

    def get_minmax(self, hamiltonian: List) -> List[int]:
        """Get ``minmax`` according to the Hamiltonian."""
        hamiltonian = self._convert_hamiltonian(hamiltonian)
        minmax = [self.nqubit - 1, 0]
        for pair in hamiltonian:
            wires = pair[1][1::2]
            for i in wires:
                i = int(i)
                if i < minmax[0]:
                    minmax[0] = i
                if i > minmax[1]:
                    minmax[1] = i
        return minmax

    def inputs_to_tensor(self, inputs: Optional[List] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert inputs to torch.Tensor."""
        if inputs is None:
            t = torch.rand(1)[0]
            return self.ham_tsr, t
        ham, t = inputs
        if ham is None:
            ham_tsr = self.ham_tsr
        elif isinstance(ham, list):
            ham = self._convert_hamiltonian(ham)
            pauli_dict = {'x': self.x, 'y': self.y, 'z': self.z}
            identity = torch.eye(2, dtype=self.x.dtype, device=self.x.device)
            minmax = self.get_minmax(ham)
            ham_tsr = None
            for pair in ham:
                lst = [identity] * self.nqubit
                coeff = pair[0]
                basis = pair[1][::2]
                wires = pair[1][1::2]
                for wire, key in zip(wires, basis):
                    wire = int(wire)
                    key = key.lower()
                    lst[wire] = pauli_dict[key]
                if ham_tsr is None:
                    ham_tsr = multi_kron(lst[minmax[0]:minmax[1]+1]) * coeff
                else:
                    ham_tsr += multi_kron(lst[minmax[0]:minmax[1]+1]) * coeff
        elif not isinstance(ham, torch.Tensor):
            ham_tsr = torch.tensor(ham, dtype=self.x.dtype, device=self.x.device)
        else:
            ham_tsr = ham
        assert torch.allclose(ham_tsr, ham_tsr.mH)
        if t is None:
            t = torch.rand(1)[0]
        elif not isinstance(t, (torch.Tensor, nn.Parameter)):
            t = torch.tensor(t, dtype=torch.float)
        return ham_tsr, t

    def get_matrix(self, hamiltonian: Any, t: Any) -> torch.Tensor:
        """Get the local unitary matrix."""
        ham, t = self.inputs_to_tensor([hamiltonian, t])
        matrix = torch.linalg.matrix_exp(-1j * ham * t)
        return matrix

    def update_matrix(self) -> torch.Tensor:
        """Update the local unitary matrix."""
        if self.inv_mode:
            t = -self.t
        else:
            t = self.t
        matrix = self.get_matrix(self.ham_tsr, t)
        assert matrix.shape[-1] == matrix.shape[-2] == 2 ** len(self.wires)
        self.matrix = matrix.detach()
        return matrix

    def _real_wrapper(self, x: Any) -> torch.Tensor:
        mat = self.get_matrix(self.ham_tsr, x)
        return torch.view_as_real(mat)

    def get_derivative(self, t: Any) -> torch.Tensor:
        """Get the derivative of the local unitary matrix."""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float)
        t = t.squeeze()
        du_dx = jacobian(self._real_wrapper, t)
        return du_dx[..., 0] + du_dx[..., 1] * 1j

    def init_para(self, inputs: Optional[List] = None) -> None:
        """Initialize the parameters."""
        ham, t = self.inputs_to_tensor(inputs)
        self.register_buffer('ham_tsr', ham)
        if self.requires_grad:
            self.t = nn.Parameter(t)
        else:
            self.register_buffer('t', t)
        self.update_matrix()


class Reset(Gate):
    r"""Reset.

    Args:
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`.
            Default: ``False``
    """
    def __init__(self, nqubit: int = 1, wires: Union[int, List[int], None] = None, tsr_mode: bool = False) -> None:
        if wires is None:
            wires = list(range(nqubit))
        super().__init__(name='Reset', nqubit=nqubit, wires=wires, tsr_mode=tsr_mode)

    def to(self, arg: Any) -> 'Reset':
        """Set dtype or device of the ``Reset``."""
        return self

    def op_state(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass for state vectors."""
        if len(self.wires) == self.nqubit:
            x = torch.zeros_like(self.vector_rep(x))
            x[:, 0] = 1
            x = self.tensor_rep(x)
        else:
            for wire in self.wires:
                pm_shape = list(range(1, self.nqubit + 1))
                pm_shape.remove(wire + 1)
                pm_shape = [wire + 1] + pm_shape + [0]
                x = x.permute(pm_shape) # (2, ..., 2, batch)
                probs = (x.abs() ** 2).sum(list(range(1, self.nqubit))) # (2, batch)
                mask = 1 - torch.sign(probs[0]) # (batch)
                norm = torch.sqrt(probs[0] + mask)
                state0 = ((1 - mask) * x[0] + mask * x[1]) / norm
                state1 = torch.zeros_like(state0)
                x = torch.stack([state0, state1])
                x = x.permute(inverse_permutation(pm_shape))
        if not self.tsr_mode:
            x = self.vector_rep(x).squeeze(0)
        return x

    def _qasm(self) -> str:
        qasm_lst = []
        for wire in self.wires:
            qasm_lst.append(f'reset q[{wire}];\n')
        return ''.join(qasm_lst)


class Barrier(Gate):
    """Barrier.

    Args:
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        name (str, optional): The name of the gate. Default: ``'Barrier'``
    """
    def __init__(self, nqubit: int = 1, wires: Union[int, List[int], None] = None, name: str = 'Barrier') -> None:
        if wires is None:
            wires = list(range(nqubit))
        super().__init__(name=name, nqubit=nqubit, wires=wires)
        self.nancilla = 0

    def to(self, arg: Any) -> 'Barrier':
        """Set dtype or device of the ``Barrier``."""
        return self

    def forward(self, x: Any) -> Any:
        """Perform a forward pass."""
        return x

    def _qasm(self) -> str:
        qasm_lst = ['barrier ']
        for wire in self.wires:
            qasm_lst.append(f'q[{wire}],')
        return ''.join(qasm_lst)[:-1] + ';\n'

    def pattern(self, nodes: List[int], ancilla: List[int]) -> nn.Sequential:
        """Get the MBQC pattern."""
        assert len(ancilla) == self.nancilla
        self.nodes = nodes
        return nn.Sequential()


class WireCut(Barrier):
    """Wire Cut.

    Args:
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int or List[int], optional): The indices of the qubits that the quantum operation acts on. Default: 0
    """
    def __init__(self, nqubit: int = 1, wires: Union[int, List[int]] = 0) -> None:
        super().__init__(name='WireCut', nqubit=nqubit, wires=wires)
