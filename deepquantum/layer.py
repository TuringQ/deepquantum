"""
Quantum layers
"""

from copy import deepcopy
from typing import List, Union, Optional, Any

import torch
from torch import nn

from .operation import Layer
from .gate import PauliX, PauliY, PauliZ, U3Gate, Hadamard, Rx, Ry, Rz, CNOT
from .qmath import multi_kron


class SingleLayer(Layer):
    """A base class for layers of single-qubit gates.
    
    Args:
        name (str, optional): The given name of `SingleLayer`. Default: ``None``
        nqubit (int, optional): The number of qubits in the quantum circuit. Default: 1
        wires (int, List[List], List[int] or None, optional): The indices of the qubits that the 
            quantum operation acts on. Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
    """
    def __init__(
        self,
        name: Optional[str] = None,
        nqubit: int = 1,
        wires: Union[int, List[List], List[int], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        if wires is None:
            wires = [[i] for i in range(nqubit)]
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        for wire in self.wires:
            assert len(wire) == 1

    def get_unitary(self):
        assert len(self.gates) > 0, 'There is no quantum gate'
        identity = torch.eye(2, dtype=torch.cfloat, device=self.gates[0].matrix.device)
        lst = [identity] * self.nqubit
        for gate in self.gates:
            lst[gate.wires[0]] = gate.update_matrix()
        return multi_kron(lst)


class DoubleLayer(Layer):
    """A base class for layers of two-qubit gates.
    
    Args:
        name (str, optional): The given name of `DoubleLayer`. Default: ``None``
        nqubit (int, optional): The number of qubits in the quantum circuit. Default: 2
        wires (List[List] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape (batch, 2, ..., 2). Default: ``False``
    """
    def __init__(
        self,
        name: Optional[str] = None,
        nqubit: int = 2,
        wires: Union[List[List], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        if wires is None:
            wires = [[i, i + 1] for i in range(0, nqubit - 1, 2)]
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        for wire in self.wires:
            assert len(wire) == 2


class Observable(SingleLayer):
    """A `Layer` that represents an observable which can be expressed by Pauli string.

    Args:
        nqubit (int, optional): The number of qubits in the quantum circuit. Default: 1
        wires (int, List[int], List[List] or None, optional): The wires to measure. Default: ``None`` (which means all 
            wires are measured)
        basis (str, optional): The measurement basis for each wire. It can be 'x', 'y', or 'z'. If only one
            character is given, it is repeated for all wires. Default: ``'z'``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        tsr_mode (bool, optional): Whether the input and output are represented by a tensor of
            shape (batch, 2, ..., 2). Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], List[List], None] = None,
        basis: str = 'z',
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='Observable', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        basis = basis.lower()
        if len(basis) == 1:
            self.basis = basis * len(self.wires)
        else:
            self.basis = basis
        assert len(self.wires) == len(self.basis), 'The number of wires is not equal to the number of bases'
        for i, wire in enumerate(self.wires):
            if self.basis[i] == 'x':
                gate = PauliX(nqubit=nqubit, wires=wire, den_mat=den_mat, tsr_mode=True)
            elif self.basis[i] == 'y':
                gate = PauliY(nqubit=nqubit, wires=wire, den_mat=den_mat, tsr_mode=True)
            elif self.basis[i] == 'z':
                gate = PauliZ(nqubit=nqubit, wires=wire, den_mat=den_mat, tsr_mode=True)
            else:
                raise ValueError('Use illegal measurement basis')
            self.gates.append(gate)


class U3Layer(SingleLayer):
    """A layer of U3 gates.

    Args:
        nqubit (int, optional): The number of qubits in the quantum circuit. Default: 1
        wires (int, List[int], List[List] or None, optional): The indices of the qubits that 
            the quantum operation acts on. Default: ``None``
        inputs (Any, optional):  The rotation Euler angles. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        tsr_mode (bool, optional): Whether the input and output are represented by a tensor of
            shape (batch, 2, ..., 2). Default: ``False``
        requires_grad (bool, optional): Whether the parameter of `U3Layer` is `nn.Parameter` or `buffer`.
            Default: ``True`` (which means the parameter is `nn.Parameter`).
        
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], List[List], None] = None,
        inputs: Any = None,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = True
    ) -> None:
        super().__init__(name='U3Layer', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        for i, wire in enumerate(self.wires):
            if inputs is None:
                thetas = None
            else:
                thetas = inputs[3*i:3*i+3]
            u3 = U3Gate(inputs=thetas, nqubit=nqubit, wires=wire, den_mat=den_mat,
                        tsr_mode=True, requires_grad=requires_grad)
            self.gates.append(u3)
            self.npara += u3.npara

    def inverse(self):
        layer = deepcopy(self)
        gates = nn.Sequential()
        for gate in self.gates[::-1]:
            gates.append(gate.inverse())
        layer.gates = gates
        layer.wires = self.wires[::-1]
        return layer


class XLayer(SingleLayer):
    """A layer of Pauli-X gates.

    Args:
        nqubit (int, optional): The number of qubits in the quantum circuit. Default: 1
        wires (int, List[int], List[List] or None, optional): The indices of the qubits that 
            the quantum operation acts on. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        tsr_mode (bool, optional): Whether the input and output are represented by a tensor of
            shape (batch, 2, ..., 2). Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], List[List], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='XLayer', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        for wire in self.wires:
            x = PauliX(nqubit=nqubit, wires=wire, den_mat=den_mat, tsr_mode=True)
            self.gates.append(x)


class YLayer(SingleLayer):
    """A layer of Pauli-Y gates.

    Args:
        nqubit (int, optional): The number of qubits in the quantum circuit. Default: 1
        wires (int, List[int], List[List] or None, optional): The indices of the qubits that 
            the quantum operation acts on. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        tsr_mode (bool, optional): Whether the input and output are represented by a tensor of
            shape (batch, 2, ..., 2). Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], List[List], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='YLayer', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        for wire in self.wires:
            y = PauliY(nqubit=nqubit, wires=wire, den_mat=den_mat, tsr_mode=True)
            self.gates.append(y)


class ZLayer(SingleLayer):
    """A layer of Pauli-Z gates.
    
    Args:
        nqubit (int, optional): The number of qubits in the quantum circuit. Default: 1
        wires (int, List[int], List[List] or None, optional): The indices of the qubits that 
            the quantum operation acts on. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        tsr_mode (bool, optional): Whether the input and output are represented by a tensor of
            shape (batch, 2, ..., 2). Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], List[List], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='ZLayer', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        for wire in self.wires:
            z = PauliZ(nqubit=nqubit, wires=wire, den_mat=den_mat, tsr_mode=True)
            self.gates.append(z)


class HLayer(SingleLayer):
    """A layer of Hadamard gates.

    Args:
        nqubit (int, optional): The number of qubits in the quantum circuit. Default: 1
        wires (int, List[int], List[List] or None, optional): The indices of the qubits that 
            the quantum operation acts on. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        tsr_mode (bool, optional): Whether the input and output are represented by a tensor of
            shape (batch, 2, ..., 2). Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], List[List], None] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name='HLayer', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        for wire in self.wires:
            h = Hadamard(nqubit=nqubit, wires=wire, den_mat=den_mat, tsr_mode=True)
            self.gates.append(h)


class RxLayer(SingleLayer):
    """A layer of Rx gates.

    Args:
        nqubit (int, optional): The number of qubits in the quantum circuit. Default: 1
        wires (int, List[int], List[List] or None, optional): The indices of the qubits that 
            the quantum operation acts on. Default: ``None``
        inputs (Any, optional):  The rotation angle parameters for `RxLayer`. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        tsr_mode (bool, optional): Whether the input and output are represented by a tensor of
            shape (batch, 2, ..., 2). Default: ``False``
        requires_grad (bool, optional): Whether the parameter of `RxLayer` is `nn.Parameter` or `buffer`.
            Default: ``True`` (which means the parameter is `nn.Parameter`).
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], List[List], None] = None,
        inputs: Any = None,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = True
    ) -> None:
        super().__init__(name='RxLayer', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        for i, wire in enumerate(self.wires):
            if inputs is None:
                theta = None
            else:
                theta = inputs[i]
            rx = Rx(inputs=theta, nqubit=nqubit, wires=wire, den_mat=den_mat,
                    tsr_mode=True, requires_grad=requires_grad)
            self.gates.append(rx)
            self.npara += rx.npara

    def inverse(self):
        layer = deepcopy(self)
        gates = nn.Sequential()
        for gate in self.gates[::-1]:
            gates.append(gate.inverse())
        layer.gates = gates
        layer.wires = self.wires[::-1]
        return layer


class RyLayer(SingleLayer):
    """A layer of Ry gates.
    
    Args:
        nqubit (int, optional): The number of qubits in the quantum circuit. Default: 1
        wires (int, List[int], List[List] or None, optional): The indices of the qubits that 
            the quantum operation acts on. Default: ``None``
        inputs (Any, optional):  The rotation angle parameters for `RyLayer`. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        tsr_mode (bool, optional): Whether the input and output are represented by a tensor of
            shape (batch, 2, ..., 2). Default: ``False``
        requires_grad (bool, optional): Whether the parameter of `RyLayer` is `nn.Parameter` or `buffer`.
            Default: ``True`` (which means the parameter is `nn.Parameter`).
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], List[List], None] = None,
        inputs: Any = None,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = True
    ) -> None:
        super().__init__(name='RyLayer', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        for i, wire in enumerate(self.wires):
            if inputs is None:
                theta = None
            else:
                theta = inputs[i]
            ry = Ry(inputs=theta, nqubit=nqubit, wires=wire, den_mat=den_mat,
                    tsr_mode=True, requires_grad=requires_grad)
            self.gates.append(ry)
            self.npara += ry.npara

    def inverse(self):
        layer = deepcopy(self)
        gates = nn.Sequential()
        for gate in self.gates[::-1]:
            gates.append(gate.inverse())
        layer.gates = gates
        layer.wires = self.wires[::-1]
        return layer


class RzLayer(SingleLayer):
    """A layer of Rz gates.
    
    Args:
        nqubit (int, optional): The number of qubits in the quantum circuit. Default: 1
        wires (int, List[int], List[List] or None, optional): The indices of the qubits that 
            the quantum operation acts on. Default: ``None``
        inputs (Any, optional):  The rotation angle parameters for `RzLayer`. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        tsr_mode (bool, optional): Whether the input and output are represented by a tensor of
            shape (batch, 2, ..., 2). Default: ``False``
        requires_grad (bool, optional): Whether the parameter of `RzLayer` is `nn.Parameter` or `buffer`.
            Default: ``True`` (which means the parameter is `nn.Parameter`).
    """
    def __init__(
        self,
        nqubit: int = 1,
        wires: Union[int, List[int], List[List], None] = None,
        inputs: Any = None,
        den_mat: bool = False,
        tsr_mode: bool = False,
        requires_grad: bool = True
    ) -> None:
        super().__init__(name='RzLayer', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        for i, wire in enumerate(self.wires):
            if inputs is None:
                theta = None
            else:
                theta = inputs[i]
            rz = Rz(inputs=theta, nqubit=nqubit, wires=wire, den_mat=den_mat,
                    tsr_mode=True, requires_grad=requires_grad)
            self.gates.append(rz)
            self.npara += rz.npara

    def inverse(self):
        layer = deepcopy(self)
        gates = nn.Sequential()
        for gate in self.gates[::-1]:
            gates.append(gate.inverse())
        layer.gates = gates
        layer.wires = self.wires[::-1]
        return layer


class CnotLayer(DoubleLayer):
    """A layer of CNOT gates.

     Args:
        nqubit (int, optional): The number of qubits in the quantum circuit. Default: 1
        wires (List[List] or None, optional): The indices of the qubits that 
            the quantum operation acts on. Default: ``None``
        name (str): The given name of `CnotLayer`. Default: 'CnotLayer'
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        tsr_mode (bool, optional): Whether the input and output are represented by a tensor of
            shape (batch, 2, ..., 2). Default: ``False``    
    """
    def __init__(
        self,
        nqubit: int = 2,
        wires: Union[List[List], None] = None,
        name: str='CnotLayer',
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        for wire in self.wires:
            cnot = CNOT(nqubit=nqubit, wires=wire, den_mat=den_mat, tsr_mode=True)
            self.gates.append(cnot)

    def inverse(self):
        wires = []
        for wire in reversed(self.wires):
            wires.append(wire)
        return CnotLayer(nqubit=self.nqubit, wires=wires, name=self.name,
                         den_mat=self.den_mat, tsr_mode=self.tsr_mode)


class CnotRing(CnotLayer):
    """A layer of CNOT gates in a cyclic way.

    Args:
        nqubit (int, optional): The number of qubits in the quantum circuit. Default: 1
        wires (List[List] or None, optional): TThe indices of the qubits that 
            the quantum operation acts on. Default: ``None``
        minmax (List[int] or None, optional): The minmum and maximum indices of qubits. Default: ``None``
        step (int): The indices distance of  the target qubit and control qubit.  Default: 1
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        tsr_mode (bool, optional): Whether the input and output are represented by a tensor of
            shape (batch, 2, ..., 2). Default: ``False``    
    """
    def __init__(
        self,
        nqubit: int = 2,
        minmax: Optional[List[int]] = None,
        step: int = 1,
        reverse: bool = False,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        if minmax is None:
            minmax = [0, nqubit-1]
        assert isinstance(minmax, list)
        assert len(minmax) == 2
        assert all(isinstance(i, int) for i in minmax)
        assert minmax[0] > -1 and minmax[0] < minmax[1] and minmax[1] < nqubit
        self.minmax = minmax
        self.step = step
        self.reverse = reverse
        nwires = minmax[1] - minmax[0] + 1
        if reverse: # from minmax[1] to minmax[0]
            wires = [[minmax[0] + i, minmax[0] + (i-step) % nwires] for i in range(minmax[1] - minmax[0], -1, -1)]
        else:
            wires = [[minmax[0] + i, minmax[0] + (i+step) % nwires] for i in range(minmax[1] - minmax[0] + 1)]
        super().__init__(nqubit=nqubit, wires=wires, name='CnotRing', den_mat=den_mat, tsr_mode=tsr_mode)
