"""
Quantum gates
"""

from copy import copy
from typing import Any, List, Optional, Union

import torch
from torch import nn

from .operation import Gate
from .qmath import multi_kron, is_unitary, svd


class SingleGate(Gate):
    """A base class for single-qubit gates.
    
    Args:
        name (str, optional): The given name of `SingleGate`. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
    
    """
    def __init__(
        self,
        name: Optional[str] = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        assert len(self.wires) == 1

    def get_unitary(self):
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


class DoubleGate(Gate):
    """A base class for two-qubit gates.

    Args:
        name (str, optional): The given name of `DoubleGate`. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 2
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``

    """

    def __init__(
        self,
        name: Optional[str] = None,
        nqubit: int = 2,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) ->None:
        if wires is None:
            wires = [0, 1]
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        assert len(self.wires) == 2

    def get_unitary(self):
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
    """A base class for two-qubit controlled gates.

    Args:
        name (str, optional): The given name of `DoubleControlGate`. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 2
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
    """
    def __init__(
        self,
        name: Optional[str]=None,
        nqubit: int = 2,
        wires: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) ->None:
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=None,
                         den_mat=den_mat, tsr_mode=tsr_mode)

    def get_unitary(self):
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
    """A base class for three-qubit gates.

    Args:
        name (str, optional): The given name of `Triple`. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 3
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
    """
    def __init__(
        self,
        name: Optional[str] = None,
        nqubit: int = 3,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        if wires is None:
            wires = [0, 1, 2]
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        assert len(self.wires) == 3


class ArbitraryGate(Gate):
    """A base class for customized gates.

     Args:
        name (str, optional): The given name of `Arbitrary`. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        minmax (List or None, optional): The minmum and maximum indices of qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``

    """
    def __init__(
        self,
        name: Optional[str] = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        minmax: Optional[List[int]] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        if isinstance(wires, int):
            wires = [wires]
        if wires is None:
            if minmax is None:
                minmax = [0, nqubit - 1]
            assert isinstance(minmax, list)
            assert len(minmax) == 2
            assert all(isinstance(i, int) for i in minmax)
            assert minmax[0] > -1 and minmax[0] <= minmax[1] and minmax[1] < nqubit
            wires = list(range(minmax[0], minmax[1] + 1))
        self.minmax = minmax
        self.inv_mode = False
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=None,
                         den_mat=den_mat, tsr_mode=tsr_mode)

    def get_unitary(self):
        if self.minmax is not None:
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

    def inverse(self):
        if isinstance(self.name, str):
            name = self.name + '_dagger'
        else:
            name = self.name
        gate = copy(self)
        gate.inv_mode = not self.inv_mode
        gate.name = name
        return gate

    def _qasm(self):
        return self._qasm_customized(self.name)


class ParametricSingleGate(SingleGate):
    """A base class for single-qubit gates with parameters.

    Args:
        name (str, optional): The given name of `ParametricSingleGate`. Default: ``None``
        inputs (Any, optional): The parameter for `ParametricSingleGate` . Default: ``None``
        nqubit (int, optional): The number of qubits that the `ParametricSingleGate` acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
        requires_grad (bool, optional): Whether the parameter of `ParametricSingleGate` is `nn.Parameter` or `buffer`.
            Default: ``False`` (which means the parameter is `buffer`).
    """
    def __init__(
        self,
        name: Optional[str] = None,
        inputs: Any = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.npara = 1
        self.requires_grad = requires_grad
        self.inv_mode = False
        self.init_para(inputs=inputs)

    def inputs_to_tensor(self, inputs=None):
        while isinstance(inputs, list):
            inputs = inputs[0]
        if inputs is None:
            inputs = torch.rand(1)[0] * 4 * torch.pi
        elif not isinstance(inputs, (torch.Tensor, nn.Parameter)):
            inputs = torch.tensor(inputs, dtype=torch.float)
        return inputs

    def update_matrix(self):
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
        matrix = self.get_matrix(theta)
        self.matrix = matrix.detach()
        return matrix

    def init_para(self, inputs=None):
        theta = self.inputs_to_tensor(inputs=inputs)
        if self.requires_grad:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('theta', theta)
        self.update_matrix()

    def inverse(self):
        gate = copy(self)
        gate.inv_mode = not self.inv_mode
        return gate

    def extra_repr(self):
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
    """A base class for two-qubit gates with parameters.

    Args:
        name (str, optional): The given name of `ParametricDoubleGate`. Default: ``None``
        inputs (Any, optional): The parameter for `ParametricDoubleGate`. Default: ``None``
        nqubit (int, optional): The number of qubits that the `ParametricDoubleGate` acts on. Default: 2
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
        requires_grad (bool, optional): Whether the parameter of `ParametricDoubleGate` is `nn.Parameter` or `buffer`.
            Default: ``False`` (which means the parameter is `buffer`).

    """
    def __init__(
        self,
        name: Optional[str] = None,
        inputs: Any = None,
        nqubit: int = 2,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.npara = 1
        self.requires_grad = requires_grad
        self.inv_mode = False
        self.init_para(inputs=inputs)

    def inputs_to_tensor(self, inputs=None):
        while isinstance(inputs, list):
            inputs = inputs[0]
        if inputs is None:
            inputs = torch.rand(1)[0] * 4 * torch.pi
        elif not isinstance(inputs, (torch.Tensor, nn.Parameter)):
            inputs = torch.tensor(inputs, dtype=torch.float)
        return inputs

    def update_matrix(self):
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
        matrix = self.get_matrix(theta)
        self.matrix = matrix.detach()
        return matrix

    def init_para(self, inputs=None):
        theta = self.inputs_to_tensor(inputs=inputs)
        if self.requires_grad:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('theta', theta)
        self.update_matrix()

    def inverse(self):
        gate = copy(self)
        gate.inv_mode = not self.inv_mode
        return gate

    def extra_repr(self):
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
    r"""U3 gate, a generic single-qubit rotation gate with 3 Euler angles.

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        U3(\theta, \phi, \lambda) =
            \begin{pmatrix}
                \cos\left(\th\right)          & -e^{i\lambda}\sin\left(\th\right) \\
                e^{i\phi}\sin\left(\th\right) & e^{i(\phi+\lambda)}\cos\left(\th\right)
            \end{pmatrix}
    
            
    Args:
        inputs (Any, optional): 3 rotation Euler angles [\theta, \phi, \lambda]. Default: ``None``
        nqubit (int, optional): The number of qubits that the `ParametricDoubleGate` acts on. Default: 2
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
        requires_grad (bool, optional): Whether the parameter of `U3Gate` is `nn.Parameter` or `buffer`.
            Default: ``False`` (which means the parameter is `buffer`).
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name='U3Gate', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)
        self.npara = 3

    def inputs_to_tensor(self, inputs=None):
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

    def get_matrix(self, theta, phi, lambd):
        theta, phi, lambd = self.inputs_to_tensor([theta, phi, lambd])
        cos_t = torch.cos(theta / 2)
        sin_t = torch.sin(theta / 2)
        e_il  = torch.exp(1j * lambd)
        e_ip  = torch.exp(1j * phi)
        e_ipl = torch.exp(1j * (phi + lambd))
        return torch.stack([cos_t, -e_il * sin_t, e_ip * sin_t, e_ipl * cos_t]).reshape(2, 2)

    def update_matrix(self):
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

    def init_para(self, inputs=None):
        theta, phi, lambd = self.inputs_to_tensor(inputs=inputs)
        if self.requires_grad:
            self.theta = nn.Parameter(theta)
            self.phi   = nn.Parameter(phi)
            self.lambd = nn.Parameter(lambd)
        else:
            self.register_buffer('theta', theta)
            self.register_buffer('phi', phi)
            self.register_buffer('lambd', lambd)
        self.update_matrix()

    def extra_repr(self):
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

    def _qasm(self):
        if self.inv_mode:
            theta = -self.theta
            phi   = -self.lambd
            lambd = -self.phi
        else:
            theta = self.theta
            phi   = self.phi
            lambd = self.lambd
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
        inputs (Any, optional): the phase angle parameter for `PhaseShift`. Default: ``None``
        nqubit (int, optional): The number of qubits that the `PhaseShift` gate acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
        requires_grad (bool, optional): Whether the parameter of `PhaseShift` is `nn.Parameter` or `buffer`.
            Default: ``False`` (which means the parameter is `buffer`).

    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name='PhaseShift', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)

    def inputs_to_tensor(self, inputs=None):
        while isinstance(inputs, list):
            inputs = inputs[0]
        if inputs is None:
            inputs = torch.rand(1)[0] * 2 * torch.pi
        elif not isinstance(inputs, (torch.Tensor, nn.Parameter)):
            inputs = torch.tensor(inputs, dtype=torch.float)
        return inputs

    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        m1 = torch.eye(1, dtype=theta.dtype, device=theta.device)
        e_it = torch.exp(1j * theta)
        return torch.block_diag(m1, e_it)

    def _qasm(self):
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
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

            I =
            \begin{pmatrix}
                1 & 0 \\
                0 & 1
            \end{pmatrix}

    Args:
        nqubit (int, optional): The number of qubits that the `Identity` gate acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:


        super().__init__(name='Identity', nqubit=nqubit, wires=wires, controls=None,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.eye(2 ** self.nqubit, dtype=torch.cfloat))

    def get_unitary(self):
        return self.matrix

    def forward(self, x):
        return x


class PauliX(SingleGate):
    r"""PauliX gate.

    **Matrix Representation:**

        .. math::

            X =
            \begin{pmatrix}
                0 & 1 \\
                1 & 0
            \end{pmatrix}

    Args:
        nqubit (int, optional): The number of qubits that the `PauliX` gate acts on. Default: 1
        wires (int, List or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='PauliX', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat))

    def _qasm(self):
        if self.controls == []:
            return f'x q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'cx q{self.controls},q{self.wires};\n'
        elif len(self.controls) == 2:
            return f'ccx q[{self.controls[0]}],q[{self.controls[1]}],q{self.wires};\n'
        else:
            return self._qasm_customized('x')


class PauliY(SingleGate):
    r"""PauliY gate.

    **Matrix Representation:**

        .. math::

            Y =
            \begin{pmatrix}
                0 & -i \\
                i & 0
            \end{pmatrix}


    Args:
        nqubit (int, optional): The number of qubits that the `PauliY` gate acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='PauliY', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[0, -1j], [1j, 0]]))

    def _qasm(self):
        if self.controls == []:
            return f'y q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'cy q{self.controls},q{self.wires};\n'
        else:
            return self._qasm_customized('y')

class PauliZ(SingleGate):
    r"""PauliZ gate.

    **Matrix Representation:**

        .. math::

            Z =
            \begin{pmatrix}
                1 & 0\\
                0 & -1
            \end{pmatrix}

    Args:
        nqubit (int, optional): The number of qubits that the `PauliZ` gate acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='PauliZ', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat))

    def _qasm(self):
        if self.controls == []:
            return f'z q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'cz q{self.controls},q{self.wires};\n'
        else:
            return self._qasm_customized('z')


class Hadamard(SingleGate):
    r"""Hadamard gate.

     **Matrix Representation:**

        .. math::

            H =
            \begin{pmatrix}
                \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2}\\
                \frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2}
            \end{pmatrix}

    Args:
        nqubit (int, optional): The number of qubits that the `Hadamard` gate acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='Hadamard', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat) / 2 ** 0.5)

    def _qasm(self):
        if self.controls == []:
            return f'h q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'ch q{self.controls},q{self.wires};\n'
        else:
            return self._qasm_customized('h')


class SGate(SingleGate):
    r"""S gate.

    **Matrix Representation:**

        .. math::

            S =
            \begin{pmatrix}
                1 & 0\\
                0 & i
            \end{pmatrix}

    Args:
        nqubit (int, optional): The number of qubits that the `SGate` gate acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='SGate', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0], [0, 1j]]))

    def inverse(self):
        return SDaggerGate(nqubit=self.nqubit, wires=self.wires, controls=self.controls,
                           den_mat=self.den_mat, tsr_mode=self.tsr_mode)

    def _qasm(self):
        if self.controls == []:
            return f's q{self.wires};\n'
        elif len(self.controls) == 1:
            qasm_str1 = ''
            qasm_str2 = f'cs q{self.controls},q{self.wires};\n'
            if 'cs' not in Gate.qasm_new_gate:
                qasm_str1 += 'gate cs q0,q1 { p(pi/4) q0; cx q0,q1; p(-pi/4) q1; cx q0,q1; p(pi/4) q1; }\n'
                Gate.qasm_new_gate.append('cs')
            return qasm_str1 + qasm_str2
        else:
            return self._qasm_customized('s')


class SDaggerGate(SingleGate):
    r"""Sdagger gate.

    **Matrix Representation:**

        .. math::

            S^{\dag}=
            \begin{pmatrix}
                1 & 0\\
                0 & -i
            \end{pmatrix}

    Args:
        nqubit (int, optional): The number of qubits that the `SdaggerGate` gate acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='SDaggerGate', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0], [0, -1j]]))

    def inverse(self):
        return SGate(nqubit=self.nqubit, wires=self.wires, controls=self.controls,
                     den_mat=self.den_mat, tsr_mode=self.tsr_mode)

    def _qasm(self):
        if self.controls == []:
            return f'sdg q{self.wires};\n'
        elif len(self.controls) == 1:
            qasm_str1 = ''
            qasm_str2 = f'csdg q{self.controls},q{self.wires};\n'
            if 'csdg' not in Gate.qasm_new_gate:
                qasm_str1 += 'gate csdg q0,q1 { p(-pi/4) q0; cx q0,q1; p(pi/4) q1; cx q0,q1; p(-pi/4) q1; }\n'
                Gate.qasm_new_gate.append('csdg')
            return qasm_str1 + qasm_str2
        else:
            return self._qasm_customized('sdg')


class TGate(SingleGate):
    r"""TGate.

     **Matrix Representation:**

        .. math::

            T =
            \begin{pmatrix}
                1 & 0 \\
                0 & e^{i\pi/4}
            \end{pmatrix}

    Args:
        nqubit (int, optional): The number of qubits that the `TGate` gate acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls:Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='TGate', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0], [0, (1 + 1j) / 2 ** 0.5]]))

    def inverse(self):
        return TDaggerGate(nqubit=self.nqubit, wires=self.wires, controls=self.controls,
                           den_mat=self.den_mat, tsr_mode=self.tsr_mode)

    def _qasm(self):
        if self.controls == []:
            return f't q{self.wires};\n'
        else:
            return self._qasm_customized('t')


class TDaggerGate(SingleGate):
    r"""TDaggerGate.

    **Matrix Representation:**

        .. math::

            T^{\dag} =
                \begin{pmatrix}
                    1 & 0 \\
                    0 & e^{-i\pi/4}
                \end{pmatrix}
    
    Args:
        nqubit (int, optional): The number of qubits that the `TDaggerGate` gate acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``

    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ):
        super().__init__(name='TDaggerGate', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0], [0, (1 - 1j) / 2 ** 0.5]]))

    def inverse(self):
        return TGate(nqubit=self.nqubit, wires=self.wires, controls=self.controls,
                     den_mat=self.den_mat, tsr_mode=self.tsr_mode)

    def _qasm(self):
        if self.controls == []:
            return f'tdg q{self.wires};\n'
        else:
            return self._qasm_customized('tdg')


class Rx(ParametricSingleGate):
    r"""Rx gate, rotation around x-axis.

     **Matrix Representation:**

        .. math::

            \newcommand{\th}{\frac{\theta}{2}}

            Rx(\theta) = 
                \begin{pmatrix}
                    \cos\left(\th\right)   & -i\sin\left(\th\right) \\
                    -i\sin\left(\th\right) & \cos\left(\th\right)
                \end{pmatrix}

    Args:
        inputs (Any, optional): The rotation angle parameter for `Rx` . Default: ``None``
        nqubit (int, optional): The number of qubits that the `Rx` gate acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
        requires_grad (bool, optional): Whether the parameter of `Rx` is `nn.Parameter` or `buffer`.
            Default: ``False`` (which means the parameter is `buffer`).
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name='Rx', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)

    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        cos  = torch.cos(theta / 2)
        isin = torch.sin(theta / 2) * 1j
        return torch.stack([cos, -isin, -isin, cos]).reshape(2, 2)

    def _qasm(self):
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
        if self.controls == []:
            return f'rx({theta.item()}) q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'crx({theta.item()}) q{self.controls},q{self.wires};\n'
        else:
            return self._qasm_customized('rx')


class Ry(ParametricSingleGate):
    r"""Ry gate, rotation around y-axis.

    **Matrix Representation:**

        .. math::

            \newcommand{\th}{\frac{\theta}{2}}

            Ry(\theta) = 
                \begin{pmatrix}
                    \cos\left(\th\right) & -\sin\left(\th\right) \\
                    \sin\left(\th\right) & \cos\left(\th\right)
                \end{pmatrix}

    Args:
        inputs (Any, optional): The rotation angle parameter for `Ry` . Default: ``None``
        nqubit (int, optional): The number of qubits that the `Ry` gate acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
        requires_grad (bool, optional): Whether the parameter of `Ry` is `nn.Parameter` or `buffer`.
            Default: ``False`` (which means the parameter is `buffer`).
    """

    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name='Ry', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)

    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        cos = torch.cos(theta / 2)
        sin = torch.sin(theta / 2)
        return torch.stack([cos, -sin, sin, cos]).reshape(2, 2) + 0j

    def _qasm(self):
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
        if self.controls == []:
            return f'ry({theta.item()}) q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'cry({theta.item()}) q{self.controls},q{self.wires};\n'
        else:
            return self._qasm_customized('ry')


class Rz(ParametricSingleGate):
    r"""Rz gate, rotation around z-axis.
    
     **Matrix Representation:**

        .. math::

            RZ(\theta) =
                \begin{pmatrix}
                    e^{-i\frac{\theta}{2}} & 0 \\
                    0 & e^{i\frac{\theta}{2}}
                \end{pmatrix}

    Args:
        inputs (Any, optional): The rotation angle parameter for `Rz` . Default: ``None``
        nqubit (int, optional): The number of qubits that the `Rz` gate acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
        requires_grad (bool, optional): Whether the parameter of `Rz` is `nn.Parameter` or `buffer`.
            Default: ``False`` (which means the parameter is `buffer`).  
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name='Rz', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)

    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        e_m_it = torch.exp(-1j * theta / 2)
        e_it = torch.exp(1j * theta / 2)
        return torch.stack([e_m_it, e_it]).reshape(-1).diag_embed()

    def _qasm(self):
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
        if self.controls == []:
            return f'rz({theta.item()}) q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'crz({theta.item()}) q{self.controls},q{self.wires};\n'
        else:
            return self._qasm_customized('rz')


class CombinedSingleGate(SingleGate):
    r"""Combined single-qubit gate.

     *Matrix Representation:**

        .. math::

            CombinedSingleGate([\sigma_x,\sigma_y]) =
                \begin{pmatrix}
                    -j & 0 \\
                    0 & j
                \end{pmatrix}

    Args:
        gatelist (List[torch.Tensor]): The list of single gates.
        name (str, optional): The given name of `CombinedSingleGate`. Default: ``None``
        nqubit (int, optional): The number of qubits that the `CombinedSingleGate` gate acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
    """
    def __init__(
        self,
        gatelist: List[torch.Tensor],
        name: Optional[str]=None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.gatelist = nn.ModuleList(gatelist)
        self.update_npara()
        self.update_matrix()

    def get_matrix(self):
        matrix = None
        for gate in self.gatelist:
            if matrix is None:
                matrix = gate.update_matrix()
            else:
                matrix = gate.update_matrix() @ matrix
        return matrix

    def update_matrix(self):
        matrix = self.get_matrix()
        self.matrix = matrix.detach()
        return matrix

    def update_npara(self):
        self.npara = 0
        for gate in self.gatelist:
            self.npara += gate.npara

    def add(self, gate: SingleGate):
        self.gatelist.append(gate)
        self.matrix = gate.matrix @ self.matrix
        self.npara += gate.npara

    def inverse(self):
        gatelist = nn.ModuleList()
        for gate in reversed(self.gatelist):
            gatelist.append(gate.inverse())
        return CombinedSingleGate(gatelist=gatelist, name=self.name, nqubit=self.nqubit, wires=self.wires,
                                  controls=self.controls, den_mat=self.den_mat, tsr_mode=self.tsr_mode)

    def _qasm(self):
        lst = []
        for gate in self.gatelist:
            # pylint: disable=protected-access
            lst.append(gate._qasm())
        return ''.join(lst)


class CNOT(DoubleControlGate):
    r"""CNOT gate.

    **Matrix Representation:**

        .. math::
            CNOT = \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 1 \\
                    0 & 0 & 1 & 0
                    \end{pmatrix}
    
    Args:
        nqubit (int, optional): The number of qubits that the `CNOT` gate acts on. Default: 2
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
            
    """
    def __init__(
        self,
        nqubit: int = 2,
        wires: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='CNOT', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0, 0, 0],
                                                     [0, 1, 0, 0],
                                                     [0, 0, 0, 1],
                                                     [0, 0, 1, 0]]) + 0j)

    def _qasm(self):
        return f'cx q[{self.wires[0]}],q[{self.wires[1]}];\n'


class Swap(DoubleGate):
    r"""Swap gate.
    
    **Matrix Representation:**

        .. math::
           Swap = \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 1
                 \end{pmatrix}

    Args:
        nqubit (int, optional): The number of qubits that the `Swap` gate acts on. Default: 2
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
    
    
    """
    def __init__(
        self,
        nqubit: int = 2,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='Swap', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0, 0, 0],
                                                     [0, 0, 1, 0],
                                                     [0, 1, 0, 0],
                                                     [0, 0, 0, 1]]) + 0j)

    def _qasm(self):
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

        R_{XX}(\theta) = \exp\left(-i \th X{\otimes}X\right) =
            \begin{pmatrix}
                \cos\left(\th\right)   & 0           & 0           & -i\sin\left(\th\right) \\
                0           & \cos\left(\th\right)   & -i\sin\left(\th\right) & 0 \\
                0           & -i\sin\left(\th\right) & \cos\left(\th\right)   & 0 \\
                -i\sin\left(\th\right) & 0           & 0           & \cos\left(\th\right)
            \end{pmatrix}

    Args:
        inputs (Any, optional): The rotation angle parameter for `Rxx` . Default: ``None``
        nqubit (int, optional): The number of qubits that the `Rxx` gate acts on. Default: 2
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
        requires_grad (bool, optional): Whether the parameter of `Rxx` is `nn.Parameter` or `buffer`.
            Default: ``False`` (which means the parameter is `buffer`).

    """

    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 2,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name='Rxx', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)

    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        cos  = torch.cos(theta / 2)
        isin = torch.sin(theta / 2) * 1j
        m1 = torch.stack([cos, cos, cos, cos]).reshape(-1).diag_embed()
        m2 = torch.stack([-isin, -isin, -isin, -isin]).reshape(-1).diag_embed().fliplr()
        return m1 + m2

    def _qasm(self):
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
        if self.controls == []:
            return f'rxx({theta.item()}) q[{self.wires[0]}],q[{self.wires[1]}];\n'
        else:
            return self._qasm_customized('rxx')


class Ryy(ParametricDoubleGate):
    r"""Ryy gate.

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        R_{YY}(\theta) = \exp\left(-i \th Y{\otimes}Y\right) =
            \begin{pmatrix}
                \cos\left(\th\right)   & 0           & 0           & i\sin\left(\th\right) \\
                0           & \cos\left(\th\right)   & -i\sin\left(\th\right) & 0 \\
                0           & -i\sin\left(\th\right) & \cos\left(\th\right)   & 0 \\
                i\sin\left(\th\right) & 0           & 0           & \cos\left(\th\right)
            \end{pmatrix}

    Args:
        inputs (Any, optional): The rotation angle parameter for `Ryy` . Default: ``None``
        nqubit (int, optional): The number of qubits that the `Ryy` gate acts on. Default: 2
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
        requires_grad (bool, optional): Whether the parameter of `Ryy` is `nn.Parameter` or `buffer`.
            Default: ``False`` (which means the parameter is `buffer`).
    
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 2,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name='Ryy', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)

    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        cos  = torch.cos(theta / 2)
        isin = torch.sin(theta / 2) * 1j
        m1 = torch.stack([cos, cos, cos, cos]).reshape(-1).diag_embed()
        m2 = torch.stack([isin, -isin, -isin, isin]).reshape(-1).diag_embed().fliplr()
        return m1 + m2

    def _qasm(self):
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
        if self.controls == []:
            qasm_str1 = ''
            qasm_str2 = f'ryy({theta.item()}) q[{self.wires[0]}],q[{self.wires[1]}];\n'
            if 'ryy' not in Gate.qasm_new_gate:
                # pylint: disable=line-too-long
                qasm_str1 += 'gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(param0) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }\n'
                Gate.qasm_new_gate.append('ryy')
            return qasm_str1 + qasm_str2
        else:
            return self._qasm_customized('ryy')


class Rzz(ParametricDoubleGate):
    r"""Rzz gate.

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        R_{ZZ}(\theta) = \exp\left(-i \th Z{\otimes}Z\right) =
            \begin{pmatrix}
                e^{-i \th}  & 0           & 0           & 0 \\
                0           & e^{i \th}  & 0           & 0 \\
                0           & 0           & e^{i \th}  & 0 \\
                0           & 0           & 0           & e^{-i \th}
            \end{pmatrix}

    Args:
        inputs (Any, optional): The rotation angle parameter for `Rzz`. Default: ``None``
        nqubit (int, optional): The number of qubits that the `Rzz` gate acts on. Default: 2
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
        requires_grad (bool, optional): Whether the parameter of `Rzz` is `nn.Parameter` or `buffer`.
            Default: ``False`` (which means the parameter is `buffer`).
    """

    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 2,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name='Rzz', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)

    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        e_m_it = torch.exp(-1j * theta / 2)
        e_it = torch.exp(1j * theta / 2)
        return torch.stack([e_m_it, e_it, e_it, e_m_it]).reshape(-1).diag_embed()

    def _qasm(self):
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
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
                1  &        0             & 0           & 0 \\
                0           & \cos\left(\th\right)      & -i\sin\left(\th\right)           & 0 \\
                0           & -i\sin\left(\th\right)    & \cos\left(\th\right)  & 0 \\
                0           & 0           & 0           & 1
            \end{pmatrix}

    Args:
        inputs (Any, optional): The rotation angle parameter for `Rxy` . Default: ``None``
        nqubit (int, optional): The number of qubits that the `Rxy` gate acts on. Default: 2
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
        requires_grad (bool, optional): Whether the parameter of `Rxy` is `nn.Parameter` or `buffer`.
            Default: ``False`` (which means the parameter is `buffer`).
    
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 2,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name='Rxy', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)

    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        cos  = torch.cos(theta / 2)
        isin = torch.sin(theta / 2) * 1j
        m1 = torch.eye(1, dtype=theta.dtype, device=theta.device)
        m2 = torch.stack([cos, -isin, -isin, cos]).reshape(2, 2)
        return torch.block_diag(m1, m2, m1)

    def _qasm(self):
        return self._qasm_customized('rxy')


class Toffoli(TripleGate):
    r"""Toffoli gate.

    **Matrix Representation:**

    .. math::
        Toffoli =
           \begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
            \end{pmatrix}


    Args:

        nqubit (int, optional): The number of qubits that the `Toffoli` gate acts on. Default: 3
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``

    """
    def __init__(
        self,
        nqubit: int = 3,
        wires: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='Toffoli', nqubit=nqubit, wires=wires, controls=None,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0],
                                                     [0, 1, 0, 0, 0, 0, 0, 0],
                                                     [0, 0, 1, 0, 0, 0, 0, 0],
                                                     [0, 0, 0, 1, 0, 0, 0, 0],
                                                     [0, 0, 0, 0, 1, 0, 0, 0],
                                                     [0, 0, 0, 0, 0, 1, 0, 0],
                                                     [0, 0, 0, 0, 0, 0, 0, 1],
                                                     [0, 0, 0, 0, 0, 0, 1, 0]]) + 0j)

    def get_unitary(self):
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

    def _qasm(self):
        return f'ccx q[{self.wires[0]}],q[{self.wires[1]}],q[{self.wires[2]}];\n'


class Fredkin(TripleGate):
    r"""Fredkin gate.

    **Matrix Representation:**

    .. math::
        Fredkin =
           \begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
            \end{pmatrix}


    Args:

        nqubit (int, optional): The number of qubits that the `Fredkin` gate acts on. Default: 3
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
    
    """
    def __init__(
        self,
        nqubit: int = 3,
        wires: Union[int, List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='Fredkin', nqubit=nqubit, wires=wires, controls=None,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0],
                                                     [0, 1, 0, 0, 0, 0, 0, 0],
                                                     [0, 0, 1, 0, 0, 0, 0, 0],
                                                     [0, 0, 0, 1, 0, 0, 0, 0],
                                                     [0, 0, 0, 0, 1, 0, 0, 0],
                                                     [0, 0, 0, 0, 0, 0, 1, 0],
                                                     [0, 0, 0, 0, 0, 1, 0, 0],
                                                     [0, 0, 0, 0, 0, 0, 0, 1]]) + 0j)

    def get_unitary(self):
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

    def _qasm(self):
        return f'cswap q[{self.wires[0]}],q[{self.wires[1]}],q[{self.wires[2]}];\n'


class UAnyGate(ArbitraryGate):
    """Arbitrary unitary gate.

    Args:
        unitaty (Any, optional): Any given unitary matrix.
        nqubit (int, optional): The number of qubits that the unitary gate acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        minmax (List or None, optional): The minmum and maximum indices of  qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``

    """
    def __init__(
        self,
        unitary: Any,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        minmax: Optional[List[int]] = None,
        name: str = 'UAnyGate',
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name=name, nqubit=nqubit, wires=wires, minmax=minmax, den_mat=den_mat,
                         tsr_mode=tsr_mode)
        if not isinstance(unitary, torch.Tensor):
            unitary = torch.tensor(unitary, dtype=torch.cfloat).reshape(-1, 2 ** len(self.wires))
        assert unitary.dtype in (torch.cfloat, torch.cdouble)
        assert unitary.shape[-1] == unitary.shape[-2] == 2 ** len(self.wires)
        assert is_unitary(unitary)
        self.register_buffer('matrix', unitary)

    def update_matrix(self):
        if self.inv_mode:
            return self.matrix.mH
        else:
            return self.matrix


class LatentGate(ArbitraryGate):
    """Latent gate.

     Args:
        inputs (Any, optional): Any given real  matrix.
        nqubit (int, optional): The number of qubits that the input gate acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        minmax (ist or None, optional): The minmum and maximum indices of  qubits. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
    
    """
    def __init__(self, inputs=None, nqubit=1, wires=None, minmax=None, name='LatentGate',
                 den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, minmax=minmax, den_mat=den_mat,
                         tsr_mode=tsr_mode)
        self.requires_grad = requires_grad
        self.init_para(inputs=inputs)

    def inputs_to_tensor(self, inputs=None):
        if inputs is None:
            inputs = torch.randn(2 ** len(self.wires), 2 ** len(self.wires))
        elif not isinstance(inputs, (torch.Tensor, nn.Parameter)):
            inputs = torch.tensor(inputs, dtype=torch.float)
        assert inputs.shape[-1] == inputs.shape[-2] == 2 ** len(self.wires)
        return inputs

    def get_matrix(self, inputs):
        latent = self.inputs_to_tensor(inputs) + 0j
        u, _, vh = svd(latent)
        return u @ vh

    def update_matrix(self):
        if self.inv_mode:
            latent = self.latent.mH
        else:
            latent = self.latent
        matrix = self.get_matrix(latent)
        self.matrix = matrix.detach()
        return matrix

    def init_para(self, inputs=None):
        latent = self.inputs_to_tensor(inputs=inputs)
        if self.requires_grad:
            self.latent = nn.Parameter(latent)
        else:
            self.register_buffer('latent', latent)
        self.update_matrix()
        self.npara = self.latent.numel()


class Barrier(Gate):
    """Barrier.

     Args:
        nqubit (int, optional): The number of qubits that the gate acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
    """
    def __init__(self, nqubit=1, wires=None):
        if wires is None:
            wires = list(range(nqubit))
        super().__init__(name='Barrier', nqubit=nqubit, wires=wires)

    def forward(self, x):
        return x

    def _qasm(self):
        qasm_lst = ['barrier ']
        for wire in self.wires:
            qasm_lst.append(f'q[{wire}],')
        return ''.join(qasm_lst)[:-1] + ';\n'
