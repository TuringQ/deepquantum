"""
Ansatze: various quantum circuits
"""

import random
from typing import Any, List, Optional, Union

import numpy as np
import torch

from .qmath import int_to_bitstring, is_unitary # avoid circular import
from .circuit import QubitCircuit
from .gate import U3Gate, Rxx, Ryy, Rzz


class Ansatz(QubitCircuit):
    """A base class for Ansatz.

    Args:
        nqubit (int): The number of qubits in the circuit.
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        minmax (List[int] or None, optional): The minimum and maximum indices of the qubits that the quantum
            operation acts on. Only valid when ``wires`` is ``None``. Default: ``None``
        ancilla (int, List[int] or None, optional): The indices of the ancilla qubits. Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        init_state (Any, optional): The initial state of the circuit. Default: ``'zeros'``
        name (str or None, optional): The name of the circuit. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        reupload (bool, optional): Whether to use data re-uploading. Default: ``False``
        mps (bool, optional): Whether to use matrix product state representation. Default: ``False``
        chi (int or None, optional): The bond dimension for matrix product state representation.
            Default: ``None``
    """
    def __init__(
        self,
        nqubit: int,
        wires: Union[int, List[int], None] = None,
        minmax: Optional[List[int]] = None,
        ancilla: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        init_state: Any = 'zeros',
        name: Optional[str] = None,
        den_mat: bool = False,
        reupload: bool = False,
        mps: bool = False,
        chi: Optional[int] = None
    ) -> None:
        super().__init__(nqubit=nqubit, init_state=init_state, name=name, den_mat=den_mat,
                         reupload=reupload, mps=mps, chi=chi)
        if wires is None:
            if minmax is None:
                minmax = [0, nqubit - 1]
            self._check_minmax(minmax)
            wires = list(range(minmax[0], minmax[1] + 1))
        if ancilla is None:
            ancilla = []
        if controls is None:
            controls = []
        wires = self._convert_indices(wires)
        ancilla = self._convert_indices(ancilla)
        controls = self._convert_indices(controls)
        for wire in wires:
            assert wire not in ancilla and wire not in controls, 'Use repeated wires'
        self.wires = sorted(wires)
        self.minmax = [min(wires), max(wires)]
        self.ancilla = ancilla
        self.controls = controls


class ControlledMultiplier(Ansatz):
    r"""Controlled multiplier.

    See https://arxiv.org/pdf/quant-ph/0205095.pdf Fig.6

    Args:
        nqubit (int): The number of qubits in the circuit.
        a (int): Number ``a`` in :math:`b+a*x \mod N`.
        mod (int): The modulus in :math:`b+a*x \mod N`.
        minmax (List[int] or None, optional): The minimum and maximum indices of the qubits that the quantum
            operation acts on. Default: ``None``
        ancilla (int, List[int] or None, optional): The indices of the ancilla qubits. Default: ``None``
        nqubitx (int or None, optional): The number of qubits in the register x.
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        mps (bool, optional): Whether to use matrix product state representation. Default: ``False``
        chi (int or None, optional): The bond dimension for matrix product state representation.
            Default: ``None``
        debug (bool, optional): Whether to print the debug information. Default: ``False``
    """
    def __init__(
        self,
        nqubit: int,
        a: int,
        mod: int,
        minmax: Optional[List[int]] = None,
        nqubitx: Optional[int] = None,
        ancilla: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        mps: bool = False,
        chi: Optional[int] = None,
        debug: bool = False
    ) -> None:
        assert isinstance(a, int)
        assert isinstance(mod, int)
        if minmax is None:
            minmax = [0, nqubit - 2]
        if nqubitx is None:
            nqubitx = len(bin(mod)) - 2
        if ancilla is None:
            ancilla = [minmax[1] + 1]
        super().__init__(nqubit=nqubit, wires=None, minmax=minmax, ancilla=ancilla, controls=controls,
                         init_state='zeros', name='ControlledMultiplier', den_mat=den_mat, mps=mps, chi=chi)
        # one extra qubit to prevent overflow
        assert len(self.wires) >= nqubitx + len(bin(mod)) - 1, 'Quantum register is not enough.'
        minmax1 = [self.minmax[0], self.minmax[0] + nqubitx - 1]
        minmax2 = [minmax1[1] + 1, minmax[1]]
        qft = QuantumFourierTransform(nqubit=nqubit, minmax=minmax2, reverse=True,
                                      den_mat=self.den_mat, mps=self.mps, chi=self.chi)
        iqft = qft.inverse()
        self.add(qft)
        k = 0
        for i in range(minmax1[1], minmax1[0] - 1, -1): # the significant bit in |x> is reversed in Fig.6
            if debug and 2**k * a >= 2 * mod:
                print(f'The number 2^{k}*{a} in {self.name} may be too large, unless the control qubit {i} is 0.')
            pma = PhiModularAdder(nqubit=nqubit, number=2**k * a, mod=mod, minmax=minmax2,
                                  ancilla=self.ancilla, controls=self.controls + [i],
                                  den_mat=self.den_mat, mps=self.mps, chi=self.chi, debug=debug)
            self.add(pma)
            k += 1
        self.add(iqft)


class ControlledUa(Ansatz):
    r"""Controlled Ua gate.

    ``a`` has a modular inverse only if ``a`` is coprime to ``mod``.
    See https://arxiv.org/pdf/quant-ph/0205095.pdf Fig.7

    Args:
        nqubit (int): The number of qubits in the circuit.
        a (int): Number ``a`` in :math:`a*x \mod N`.
        mod (int): The modulus in :math:`a*x \mod N`.
        minmax (List[int] or None, optional): The minimum and maximum indices of the qubits that the quantum
            operation acts on. Default: ``None``
        ancilla (int, List[int] or None, optional): The indices of the ancilla qubits. Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        mps (bool, optional): Whether to use matrix product state representation. Default: ``False``
        chi (int or None, optional): The bond dimension for matrix product state representation.
            Default: ``None``
        debug (bool, optional): Whether to print the debug information. Default: ``False``
    """
    def __init__(
        self,
        nqubit: int,
        a: int,
        mod: int,
        minmax: Optional[List[int]] = None,
        ancilla: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        mps: bool = False,
        chi: Optional[int] = None,
        debug: bool = False
    ) -> None:
        # |x> with n bits, |0> with n+1 bits and one extra ancilla bit
        nregister = len(bin(mod)) - 2
        nancilla = len(bin(mod))
        if minmax is None:
            minmax = [0, nregister - 1]
        if ancilla is None:
            ancilla = list(range(minmax[1] + 1, minmax[1] + 1 + nancilla))
        super().__init__(nqubit=nqubit, wires=None, minmax=minmax, ancilla=ancilla, controls=controls,
                         init_state='zeros', name='ControlledUa', den_mat=den_mat, mps=mps, chi=chi)
        assert len(self.wires) == nregister
        assert len(self.ancilla) == nancilla
        cmult = ControlledMultiplier(nqubit=nqubit, a=a, mod=mod, minmax=[self.minmax[0], self.ancilla[-2]],
                                     nqubitx=nregister, ancilla=self.ancilla[-1], controls=self.controls,
                                     den_mat=self.den_mat, mps=self.mps, chi=self.chi, debug=debug)
        self.add(cmult)
        for i in range(len(self.wires)):
            self.swap([self.wires[i], self.ancilla[i + 1]], controls=self.controls)
        a_inv = pow(a, -1, mod)
        cmult_inv = ControlledMultiplier(nqubit=nqubit, a=a_inv, mod=mod, minmax=[self.minmax[0], self.ancilla[-2]],
                                         nqubitx=nregister, ancilla=self.ancilla[-1], controls=self.controls,
                                         den_mat=self.den_mat, mps=self.mps, chi=self.chi, debug=debug).inverse()
        self.add(cmult_inv)


class HHL(Ansatz):
    r"""A quantum circuit for the HHL algorithm.

    Args:
        ncount (int): The number of counting qubits.
        mat (Any): The Hermitian matrix `A`.
        t0 (float, optional): The time parameter for the matrix exponential in units of :math:`2\pi`.
            Default: 1
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        mps (bool, optional): Whether to use matrix product state representation. Default: ``False``
        chi (int or None, optional): The bond dimension for matrix product state representation.
            Default: ``None``
        show_barrier (bool, optional): Whether to show the barriers in the circuit. Default: ``False``
    """
    def __init__(
        self,
        ncount: int,
        mat: Any,
        t0: float = 1,
        den_mat: bool = False,
        mps: bool = False,
        chi: Optional[int] = None,
        show_barrier: bool = False
    ) -> None:
        if not isinstance(mat, torch.Tensor):
            mat = torch.tensor(mat)
        t0 *= 2 * torch.pi
        unitary = torch.linalg.matrix_exp(1j * mat * t0 / 2 ** ncount)
        assert is_unitary(unitary)
        nreg_i = int(np.log2(len(unitary)))
        nqubit = 1 + ncount + nreg_i
        self.unitary = unitary
        super().__init__(nqubit=nqubit, wires=None, minmax=None, ancilla=None, controls=None,
                         init_state='zeros', name='HHL', den_mat=den_mat, mps=mps, chi=chi)
        qpe = QuantumPhaseEstimation(nqubit=nqubit, ncount=ncount, unitary=unitary, minmax=[1, nqubit-1],
                                     den_mat=self.den_mat, mps=self.mps, chi=self.chi, show_barrier=show_barrier)
        self.add(qpe)
        if show_barrier:
            self.barrier()

        for i in range(2 ** ncount):
            for j in range(ncount):
                if format(i, '0' + str(ncount) + 'b')[ncount-j-1] == '0':
                    self.x(1 + j)
            theta = 2 * torch.pi * i / 2 ** ncount
            self.ry(0, inputs=theta, controls=list(range(1, ncount+1)))
            for j in range(ncount):
                if format(i, '0' + str(ncount) + 'b')[ncount-j-1] == '0':
                    self.x(1 + j)
            if show_barrier:
                self.barrier()

        iqpe = qpe.inverse()
        self.add(iqpe)
        if show_barrier:
            self.barrier()


class NumberEncoder(Ansatz):
    """Convert number to corresponding encoding circuit.

    Args:
        nqubit (int): The number of qubits in the circuit.
        number (int): The integer for converting to bit string.
        minmax (List[int] or None, optional): The minimum and maximum indices of the qubits that the quantum
            operation acts on. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        mps (bool, optional): Whether to use matrix product state representation. Default: ``False``
        chi (int or None, optional): The bond dimension for matrix product state representation.
            Default: ``None``
    """
    def __init__(
        self,
        nqubit: int,
        number: int,
        minmax: Optional[List[int]] = None,
        den_mat: bool = False,
        mps: bool = False,
        chi: Optional[int] = None
    ) -> None:
        super().__init__(nqubit=nqubit, wires=None, minmax=minmax, ancilla=None, controls=None,
                         init_state='zeros', name='NumberEncoder', den_mat=den_mat, mps=mps, chi=chi)
        bits = int_to_bitstring(number, len(self.wires))
        for i, wire in enumerate(self.wires):
            if bits[i] == '1':
                self.x(wire)


class PhiAdder(Ansatz):
    r"""Phi adder.

    See https://arxiv.org/pdf/quant-ph/0205095.pdf Fig.2 and Fig.3

    Args:
        nqubit (int): The number of qubits in the circuit.
        number (int): Number ``a`` in :math:`\Phi(a+b)`.
        minmax (List[int] or None, optional): The minimum and maximum indices of the qubits that the quantum
            operation acts on. Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        mps (bool, optional): Whether to use matrix product state representation. Default: ``False``
        chi (int or None, optional): The bond dimension for matrix product state representation.
            Default: ``None``
        debug (bool, optional): Whether to print the debug information. Default: ``False``
    """
    def __init__(
        self,
        nqubit: int,
        number: int,
        minmax: Optional[List[int]] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        mps: bool = False,
        chi: Optional[int] = None,
        debug: bool = False
    ) -> None:
        super().__init__(nqubit=nqubit, wires=None, minmax=minmax, ancilla=None, controls=controls,
                         init_state='zeros', name='PhiAdder', den_mat=den_mat, mps=mps, chi=chi)
        bits = int_to_bitstring(number, len(self.wires), debug=debug)
        for i, wire in enumerate(self.wires):
            phi = 0
            k = 0
            for j in range(i, len(bits)):
                if bits[j] == '1':
                    phi += torch.pi / 2 ** k
                k += 1
            if phi != 0:
                self.p(wires=wire, inputs=phi, controls=self.controls)


class PhiModularAdder(Ansatz):
    r"""Phi modular adder.

    See https://arxiv.org/pdf/quant-ph/0205095.pdf Fig.5

    Args:
        nqubit (int): The number of qubits in the circuit.
        number (int): Number ``a`` in :math:`\Phi(a+b \mod N)`.
        mod (int): The modulus in :math:`\Phi(a+b \mod N)`.
        minmax (List[int] or None, optional): The minimum and maximum indices of the qubits that the quantum
            operation acts on. Default: ``None``
        ancilla (int, List[int] or None, optional): The indices of the ancilla qubits. Default: ``None``
        controls (int, List[int] or None, optional): The indices of the control qubits. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        mps (bool, optional): Whether to use matrix product state representation. Default: ``False``
        chi (int or None, optional): The bond dimension for matrix product state representation.
            Default: ``None``
        debug (bool, optional): Whether to print the debug information. Default: ``False``
    """
    def __init__(
        self,
        nqubit: int,
        number: int,
        mod: int,
        minmax: Optional[List[int]] = None,
        ancilla: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None,
        den_mat: bool = False,
        mps: bool = False,
        chi: Optional[int] = None,
        debug: bool = False
    ) -> None:
        if minmax is None:
            minmax = [0, nqubit - 2]
        if ancilla is None:
            ancilla = [minmax[1] + 1]
        super().__init__(nqubit=nqubit, wires=None, minmax=minmax, ancilla=ancilla, controls=controls,
                         init_state='zeros', name='PhiModularAdder', den_mat=den_mat, mps=mps, chi=chi)
        if debug and number >= 2 * mod:
            print(f'The number {number} in {self.name} is too large.')
        phi_add_number = PhiAdder(nqubit=nqubit, number=number, minmax=self.minmax, controls=self.controls,
                                  den_mat=self.den_mat, mps=self.mps, chi=self.chi, debug=debug)
        phi_sub_number = phi_add_number.inverse()
        phi_add_mod = PhiAdder(nqubit=nqubit, number=mod, minmax=self.minmax, controls=self.ancilla,
                               den_mat=self.den_mat, mps=self.mps, chi=self.chi, debug=debug)
        phi_sub_mod = PhiAdder(nqubit=nqubit, number=mod, minmax=self.minmax,
                               den_mat=self.den_mat, mps=self.mps, chi=self.chi, debug=debug).inverse()
        qft = QuantumFourierTransform(nqubit=nqubit, minmax=self.minmax, reverse=True,
                                      den_mat=self.den_mat, mps=self.mps, chi=self.chi)
        iqft = qft.inverse()
        self.add(phi_add_number)
        self.add(phi_sub_mod)
        self.add(iqft)
        self.cnot(self.minmax[0], self.ancilla[0])
        self.add(qft)
        self.add(phi_add_mod)
        self.add(phi_sub_number)
        self.add(iqft)
        self.x(self.minmax[0])
        self.cnot(self.minmax[0], self.ancilla[0])
        self.x(self.minmax[0])
        self.add(qft)
        self.add(phi_add_number)


class QuantumConvolutionalNeuralNetwork(Ansatz):
    """Quantum convolutional neural network.

    See https://readpaper.com/paper/4554418257818296321 Fig.1
    or https://pennylane.ai/qml/demos/tutorial_learning_few_data

    Args:
        nqubit (int): The number of qubits in the circuit.
        nlayer (int): The number of layers.
        minmax (List[int] or None, optional): The minimum and maximum indices of the qubits that the quantum
            operation acts on. Default: ``None``
        init_state (Any, optional): The initial state of the circuit. Default: ``'zeros'``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        requires_grad (bool, optional): Whether the parameters are ``nn.Parameter`` or ``buffer``.
            Default: ``True`` (which means ``nn.Parameter``)
        mps (bool, optional): Whether to use matrix product state representation. Default: ``False``
        chi (int or None, optional): The bond dimension for matrix product state representation.
            Default: ``None``
    """
    def __init__(
        self,
        nqubit: int,
        nlayer: int,
        minmax: Optional[List[int]] = None,
        init_state: Any = 'zeros',
        den_mat: bool = False,
        requires_grad: bool = True,
        mps: bool = False,
        chi: Optional[int] = None
    ) -> None:
        super().__init__(nqubit=nqubit, wires=None, minmax=minmax, ancilla=None, controls=None,
                         init_state=init_state, name='QuantumConvolutionalNeuralNetwork', den_mat=den_mat,
                         mps=mps, chi=chi)
        wires = self.wires
        self.requires_grad = requires_grad
        u1 = U3Gate(nqubit=nqubit, den_mat=den_mat, requires_grad=requires_grad)
        u2 = U3Gate(nqubit=nqubit, den_mat=den_mat, requires_grad=requires_grad)
        for i, wire in enumerate(wires[1::2]):
            self.add(u1, wires=wires[2 * i])
            self.add(u2, wires=wire)
        for _ in range(nlayer):
            self.conv(wires)
            self.pool(wires)
            wires = wires[::2]
        self.latent(wires=wires)

    def conv(self, wires: List[int]) -> None:
        rxx = Rxx(nqubit=self.nqubit, den_mat=self.den_mat, requires_grad=self.requires_grad)
        ryy = Ryy(nqubit=self.nqubit, den_mat=self.den_mat, requires_grad=self.requires_grad)
        rzz = Rzz(nqubit=self.nqubit, den_mat=self.den_mat, requires_grad=self.requires_grad)
        u1 = U3Gate(nqubit=self.nqubit, den_mat=self.den_mat, requires_grad=self.requires_grad)
        u2 = U3Gate(nqubit=self.nqubit, den_mat=self.den_mat, requires_grad=self.requires_grad)
        for start in [1, 2]:
            for i, wire in enumerate(wires[start::2]):
                self.add(rxx, wires=[wires[2 * i + start - 1], wire])
                self.add(ryy, wires=[wires[2 * i + start - 1], wire])
                self.add(rzz, wires=[wires[2 * i + start - 1], wire])
                self.add(u1, wires=wires[2 * i + start - 1])
                self.add(u2, wires=wire)

    def pool(self, wires: List[int]) -> None:
        cu = U3Gate(nqubit=self.nqubit, den_mat=self.den_mat, requires_grad=self.requires_grad)
        for i, wire in enumerate(wires[1::2]):
            self.add(cu, wires=wires[2 * i], controls=wire)


class QuantumFourierTransform(Ansatz):
    """Quantum Fourier transform.

    Args:
        nqubit (int): The number of qubits in the circuit.
        minmax (List[int] or None, optional): The minimum and maximum indices of the qubits that the quantum
            operation acts on. Default: ``None``
        reverse (bool, optional): Whether to reverse the output order. Default: ``False`` (which means
            the default output order of phase is :math:`x/2, ..., x/2^n`. If ``reverse=True``, the output order
            of phase is :math:`x/2^n, ..., x/2`)
        init_state (Any, optional): The initial state of the circuit. Default: ``'zeros'``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        mps (bool, optional): Whether to use matrix product state representation. Default: ``False``
        chi (int or None, optional): The bond dimension for matrix product state representation.
            Default: ``None``
        show_barrier (bool, optional): Whether to show the barriers in the circuit. Default: ``False``
    """
    def __init__(
        self,
        nqubit: int,
        minmax: Optional[List[int]] = None,
        reverse: bool = False,
        init_state: Any = 'zeros',
        den_mat: bool = False,
        mps: bool = False,
        chi: Optional[int] = None,
        show_barrier: bool = False
    ) -> None:
        super().__init__(nqubit=nqubit, wires=None, minmax=minmax, ancilla=None, controls=None,
                         init_state=init_state, name='QuantumFourierTransform', den_mat=den_mat,
                         mps=mps, chi=chi)
        self.reverse = reverse
        for i in self.wires:
            self.qft_block(i)
            if show_barrier:
                self.barrier(self.wires)
        if not reverse:
            for i in range(len(self.wires) // 2):
                self.swap([self.wires[i], self.wires[-1 - i]])

    def qft_block(self, n: int) -> None:
        self.h(n)
        k = 2
        for i in range(n, self.minmax[1]):
            self.cp(i + 1, n, torch.pi / 2 ** (k - 1))
            k += 1


class QuantumPhaseEstimation(Ansatz):
    """Quantum phase estimation for arbitrary unitary operator.

    Args:
        nqubit (int): The number of qubits in the circuit.
        ncount (int): The number of counting qubits.
        unitary (Any): The unitary operator.
        minmax (List[int] or None, optional): The minimum and maximum indices of the qubits that the quantum
            operation acts on. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        mps (bool, optional): Whether to use matrix product state representation. Default: ``False``
        chi (int or None, optional): The bond dimension for matrix product state representation.
            Default: ``None``
        show_barrier (bool, optional): Whether to show the barriers in the circuit. Default: ``False``
    """
    def __init__(
        self,
        nqubit: int,
        ncount: int,
        unitary: Any,
        minmax: Optional[List[int]] = None,
        den_mat: bool = False,
        mps: bool = False,
        chi: Optional[int] = None,
        show_barrier: bool = False
    ) -> None:
        if not isinstance(unitary, torch.Tensor):
            unitary = torch.tensor(unitary, dtype=torch.cfloat)
        assert is_unitary(unitary)
        nreg_i = int(np.log2(len(unitary)))
        if minmax is None:
            minmax = [0, ncount + nreg_i - 1]
        assert minmax[1] - minmax[0] == ncount + nreg_i - 1
        self.unitary = unitary
        super().__init__(nqubit=nqubit, wires=None, minmax=minmax, ancilla=None, controls=None,
                         init_state='zeros', name='QuantumPhaseEstimation', den_mat=den_mat,
                         mps=mps, chi=chi)
        wires_c = list(range(minmax[0], minmax[0] + ncount))
        wires_i = list(range(minmax[0] + ncount, minmax[1] + 1))
        self.hlayer(wires_c)
        if show_barrier:
            self.barrier()
        for i, wire in enumerate(wires_c):
            u = torch.linalg.matrix_power(self.unitary, 2 ** (ncount - 1 - i))
            self.any(unitary=u, wires=wires_i, controls=wire)
        if show_barrier:
            self.barrier()
        iqft = QuantumFourierTransform(nqubit=nqubit, minmax=[wires_c[0], wires_c[-1]], den_mat=self.den_mat,
                                       mps=self.mps, chi=self.chi, show_barrier=show_barrier).inverse()
        self.add(iqft)


class QuantumPhaseEstimationSingleQubit(Ansatz):
    """Quantum phase estimation for single-qubit gate.

    Args:
        t (int): The number of counting qubits.
        phase (Any): The phase to be estimated.
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        mps (bool, optional): Whether to use matrix product state representation. Default: ``False``
        chi (int or None, optional): The bond dimension for matrix product state representation.
            Default: ``None``
    """
    def __init__(
        self,
        t: int,
        phase: Any,
        den_mat: bool = False,
        mps: bool = False,
        chi: Optional[int] = None
    ) -> None:
        nqubit = t + 1
        self.phase = phase
        super().__init__(nqubit=nqubit, wires=None, minmax=None, ancilla=None, controls=None,
                         init_state='zeros', name='QuantumPhaseEstimationSingleQubit', den_mat=den_mat,
                         mps=mps, chi=chi)
        self.hlayer(list(range(t)))
        self.x(t)
        for i in range(t):
            self.cp(i, t, torch.pi * phase * (2 ** (t - i)))
        iqft = QuantumFourierTransform(nqubit=nqubit, minmax=[0, t - 1],
                                       den_mat=self.den_mat, mps=self.mps, chi=self.chi).inverse()
        self.add(iqft)


class RandomCircuitG3(Ansatz):
    """Random circuit of G3 family.

    Args:
        nqubit (int): The number of qubits in the circuit.
        ngate (int): The number of random gates in the circuit.
        wires (List[int] or None, optional): The indices of the qubits that the random gates act on.
            Default: ``None``
        minmax (List[int] or None, optional): The minimum and maximum indices of the qubits that the quantum
            operation acts on. Only valid when ``wires`` is ``None``. Default: ``None``
        init_state (Any, optional): The initial state of the circuit. Default: ``'zeros'``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        mps (bool, optional): Whether to use matrix product state representation. Default: ``False``
        chi (int or None, optional): The bond dimension for matrix product state representation.
            Default: ``None``
    """
    def __init__(
        self,
        nqubit: int,
        ngate: int,
        wires: Optional[List[int]] = None,
        minmax: Optional[List[int]] = None,
        init_state: Any = 'zeros',
        den_mat: bool = False,
        mps: bool = False,
        chi: Optional[int] = None
    ) -> None:
        super().__init__(nqubit=nqubit, wires=wires, minmax=minmax, ancilla=None, controls=None,
                         init_state=init_state, name='RandomCircuitG3', den_mat=den_mat, mps=mps, chi=chi)
        self.ngate = ngate
        self.gate_set = ['CNOT', 'H', 'T']
        for _ in range(ngate):
            gate = random.sample(self.gate_set, 1)[0]
            if gate == 'CNOT':
                wire = random.sample(self.wires, 2)
            else:
                wire = random.sample(self.wires, 1)
            if gate == 'CNOT':
                self.cnot(wire[0], wire[1])
            elif gate == 'H':
                self.h(wire)
            elif gate == 'T':
                self.t(wire)


class ShorCircuit(Ansatz):
    r"""Circuit for Shor's algorithm.

    Args:
        mod (int): The odd integer to be factored.
        ncount (int): The number of counting qubits.
        a (int): Any integer that satisfies :math:`1 < a < N` and :math:`\gcd(a, N) = 1`.
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        mps (bool, optional): Whether to use matrix product state representation. Default: ``False``
        chi (int or None, optional): The bond dimension for matrix product state representation.
            Default: ``None``
        debug (bool, optional): Whether to print the debug information. Default: ``False``
    """
    def __init__(
        self,
        mod: int,
        ncount: int,
        a: int,
        den_mat: bool = False,
        mps: bool = False,
        chi: Optional[int] = None,
        debug: bool = False
    ) -> None:
        nreg = len(bin(mod)) - 2
        nqubit = ncount + 2 * nreg + 2
        super().__init__(nqubit=nqubit, wires=None, minmax=None, ancilla=None, controls=None,
                         init_state='zeros', name='ShorCircuit', den_mat=den_mat, mps=mps, chi=chi)
        minmax1 = [0, ncount - 1]
        minmax2 = [ncount, ncount + nreg - 1]
        ancilla = list(range(ncount + nreg, nqubit))
        self.hlayer(list(range(ncount)))
        self.x(ncount + nreg - 1)
        n = 0
        for i in range(ncount - 1, -1, -1):
            # Compute a^{2^n} (mod N) by repeated squaring
            an = a
            for _ in range(n):
                an = an ** 2 % mod
            cua = ControlledUa(nqubit=nqubit, a=an, mod=mod, minmax=minmax2, ancilla=ancilla, controls=[i],
                               den_mat=self.den_mat, mps=self.mps, chi=self.chi, debug=debug)
            self.add(cua)
            n += 1
        iqft = QuantumFourierTransform(nqubit=nqubit, minmax=minmax1,
                                       den_mat=self.den_mat, mps=self.mps, chi=self.chi).inverse()
        self.add(iqft)


class ShorCircuitFor15(Ansatz):
    r"""Circuit for Shor's algorithm to factor number 15.

    See https://learn.qiskit.org/course/ch-algorithms/shors-algorithm

    Args:
        ncount (int): The number of counting qubits.
        a (int): Any integer that satisfies :math:`1 < a < N` and :math:`\gcd(a, N) = 1`.
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        mps (bool, optional): Whether to use matrix product state representation. Default: ``False``
        chi (int or None, optional): The bond dimension for matrix product state representation.
            Default: ``None``
    """
    def __init__(
        self,
        ncount: int,
        a: int,
        den_mat: bool = False,
        mps: bool = False,
        chi: Optional[int] = None
    ) -> None:
        mod = 15
        nreg = len(bin(mod)) - 2
        nqubit = ncount + nreg
        self.ncount = ncount
        super().__init__(nqubit=nqubit, wires=None, minmax=None, ancilla=None, controls=None,
                         init_state='zeros', name='ShorCircuitFor15', den_mat=den_mat, mps=mps, chi=chi)
        minmax = [0, ncount - 1]
        self.hlayer(list(range(ncount)))
        self.x(ncount + nreg - 1)
        n = 0
        for i in range(ncount - 1, -1, -1):
            self.cua(a, 2 ** n, i)
            n += 1
        iqft = QuantumFourierTransform(nqubit=nqubit, minmax=minmax,
                                       den_mat=self.den_mat, mps=self.mps, chi=self.chi).inverse()
        self.add(iqft)

    def cua(self, a: int, power: int, controls: Union[int, List[int], None]) -> None:
        assert a in [2, 4, 7, 8, 11, 13]
        for _ in range(power):
            if a in [2, 13]:
                self.swap([self.ncount + 2, self.ncount + 3], controls)
                self.swap([self.ncount + 1, self.ncount + 2], controls)
                self.swap([self.ncount + 0, self.ncount + 1], controls)
            if a in [7, 8]:
                self.swap([self.ncount + 0, self.ncount + 1], controls)
                self.swap([self.ncount + 1, self.ncount + 2], controls)
                self.swap([self.ncount + 2, self.ncount + 3], controls)
            if a in [4, 11]:
                self.swap([self.ncount + 1, self.ncount + 3], controls)
                self.swap([self.ncount + 0, self.ncount + 2], controls)
            if a in [7, 11, 13]:
                for q in range(4):
                    self.x(self.ncount + q, controls)
