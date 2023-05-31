import torch
from deepquantum.qmath import int_to_bitstring
from deepquantum.circuit import QubitCircuit
import random


class Ansatz(QubitCircuit):
    def __init__(self, nqubit, wires=None, minmax=None, ancilla=None, controls=None, init_state='zeros',
                 name=None, den_mat=False, mps=False, chi=None):
        super().__init__(nqubit=nqubit, init_state=init_state, name=name, den_mat=den_mat, mps=mps, chi=chi)
        if type(wires) == int:
            wires = [wires]
        if wires == None:
            if minmax == None:
                minmax = [0, nqubit - 1]
            assert type(minmax) == list
            assert len(minmax) == 2
            assert all(isinstance(i, int) for i in minmax)
            assert minmax[0] > -1 and minmax[0] <= minmax[1] and minmax[1] < nqubit
            wires = list(range(minmax[0], minmax[1] + 1))
        if type(ancilla) == int:
            ancilla = [ancilla]
        if ancilla == None:
            ancilla = []
        if type(controls) == int:
            controls = [controls]
        if controls == None:
            controls = []
        assert type(wires) == list and type(ancilla) == list and type(controls) == list, 'Invalid input type'
        assert all(isinstance(i, int) for i in wires), 'Invalid input type'
        assert all(isinstance(i, int) for i in ancilla), 'Invalid input type'
        assert all(isinstance(i, int) for i in controls), 'Invalid input type'
        assert min(wires) > -1 and max(wires) < nqubit, 'Invalid input'
        if len(ancilla) > 0:
            assert min(ancilla) > -1 and max(ancilla) < nqubit, 'Invalid input'
        if len(controls) > 0:
            assert min(controls) > -1 and max(controls) < nqubit, 'Invalid input'
        assert len(set(wires)) == len(wires), 'Invalid input'
        assert len(set(ancilla)) == len(ancilla) and len(set(controls)) == len(controls), 'Invalid input'
        for wire in wires:
            assert wire not in ancilla and wire not in controls, 'Use repeated wires'
        self.wires = sorted(wires)
        self.minmax = [min(wires), max(wires)]
        self.ancilla = ancilla
        self.controls = controls


class NumberEncoder(Ansatz):
    def __init__(self, nqubit, number, minmax=None, den_mat=False, mps=False, chi=None):
        super().__init__(nqubit=nqubit, wires=None, minmax=minmax, ancilla=None, controls=None,
                         init_state='zeros', name='NumberEncoder', den_mat=den_mat, mps=mps, chi=chi)
        bits = int_to_bitstring(number, len(self.wires))
        for i, wire in enumerate(self.wires):
            if bits[i] == '1':
                self.x(wire)


class PhiAdder(Ansatz):
    # See https://arxiv.org/pdf/quant-ph/0205095.pdf Fig.2 and Fig.3
    def __init__(self, nqubit, number, minmax=None, controls=None, den_mat=False, mps=False, chi=None):
        super().__init__(nqubit=nqubit, wires=None, minmax=minmax, ancilla=None, controls=controls,
                         init_state='zeros', name='PhiAdder', den_mat=den_mat, mps=mps, chi=chi)
        bits = int_to_bitstring(number, len(self.wires))
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
    # See https://arxiv.org/pdf/quant-ph/0205095.pdf Fig.5
    def __init__(self, nqubit, number, mod, minmax=None, ancilla=None, controls=None, den_mat=False,
                 mps=False, chi=None):
        assert number < mod
        if minmax == None:
            minmax = [0, nqubit - 2]
        if ancilla == None:
            ancilla = [minmax[1] + 1]
        super().__init__(nqubit=nqubit, wires=None, minmax=minmax, ancilla=ancilla, controls=controls,
                         init_state='zeros', name='PhiModularAdder', den_mat=den_mat, mps=mps, chi=chi)
        phi_add_number = PhiAdder(nqubit=nqubit, number=number, minmax=self.minmax, controls=self.controls)
        phi_sub_number = phi_add_number.inverse()
        phi_add_mod = PhiAdder(nqubit=nqubit, number=mod, minmax=self.minmax, controls=self.ancilla)
        phi_sub_mod = PhiAdder(nqubit=nqubit, number=mod, minmax=self.minmax).inverse()
        qft = QuantumFourierTransform(nqubit=nqubit, minmax=self.minmax, reverse=True)
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


class QuantumFourierTransform(Ansatz):
    def __init__(self, nqubit, minmax=None, reverse=False, init_state='zeros', den_mat=False,
                 mps=False, chi=None):
        super().__init__(nqubit=nqubit, wires=None, minmax=minmax, ancilla=None, controls=None,
                         init_state=init_state, name='QuantumFourierTransform', den_mat=den_mat,
                         mps=mps, chi=chi)
        # the default output order of phase is x/2, ..., x/2**n
        # if reverse=True, the output order of phase is x/2**n, ..., x/2
        self.reverse = reverse
        for i in range(self.minmax[0], self.minmax[1] + 1):
            self.qft_block(i)
        if not reverse:
            for i in range(self.minmax[0], (self.minmax[0] + self.minmax[1] + 1) // 2):
                self.swap([i, self.minmax[0] + self.minmax[1] - i])
        
    def qft_block(self, n):
        self.h(n)
        k = 2
        for i in range(n, self.minmax[1]):
            self.cp(i + 1, n, torch.pi / 2 ** (k - 1))
            k += 1
        self.barrier(self.wires)


class QuantumPhaseEstimationSingleQubit(Ansatz):
    def __init__(self, t, phase, den_mat=False, mps=False, chi=None):
        nqubit = t + 1
        self.phase = phase
        super().__init__(nqubit=nqubit, wires=None, minmax=None, ancilla=None, controls=None,
                         init_state='zeros', name='QuantumPhaseEstimationSingleQubit', den_mat=den_mat,
                         mps=mps, chi=chi)
        self.hlayer(list(range(t)))
        self.x(t)
        for i in range(t):
            self.cp(i, t, torch.pi * phase * (2 ** (t - i)))
        self.add(QuantumFourierTransform(nqubit=nqubit, minmax=[0, t - 1]).inverse())


class RandomCircuitG3(Ansatz):
    def __init__(self, nqubit, ngate, wires=None, minmax=None, init_state='zeros', den_mat=False,
                 mps=False, chi=None):
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