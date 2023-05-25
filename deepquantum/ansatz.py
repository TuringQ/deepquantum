import torch
from deepquantum.circuit import QubitCircuit
import random


class Ansatz(QubitCircuit):
    def __init__(self, nqubit, wires=None, minmax=None, init_state='zeros', name=None, den_mat=False,
                 mps=False, chi=None):
        super().__init__(nqubit=nqubit, init_state=init_state, name=name, den_mat=den_mat, mps=mps, chi=chi)
        if wires == None:
            if minmax == None:
                minmax = [0, nqubit - 1]
            assert type(minmax) == list
            assert len(minmax) == 2
            assert all(isinstance(i, int) for i in minmax)
            assert minmax[0] > -1 and minmax[0] <= minmax[1] and minmax[1] < nqubit
            wires = list(range(minmax[0], minmax[1] + 1))
        assert type(wires) == list, 'Invalid input type'
        assert all(isinstance(i, int) for i in wires), 'Invalid input type'
        assert min(wires) > -1 and max(wires) < nqubit, 'Invalid input'
        assert len(set(wires)) == len(wires), 'Invalid input'
        self.wires = wires
        self.minmax = [min(wires), max(wires)]


class RandomCircuitG3(Ansatz):
    def __init__(self, nqubit, ngate, wires=None, minmax=None, init_state='zeros', den_mat=False,
                 mps=False, chi=None):
        super().__init__(nqubit=nqubit, wires=wires, minmax=minmax, init_state=init_state,
                         name='RandomCircuitG3', den_mat=den_mat, mps=mps, chi=chi)
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


class QuantumFourierTransform(Ansatz):
    def __init__(self, nqubit, minmax=None, init_state='zeros', den_mat=False, mps=False, chi=None):
        super().__init__(nqubit=nqubit, wires=None, minmax=minmax, init_state=init_state,
                         name='QuantumFourierTransform', den_mat=den_mat, mps=mps, chi=chi)
        for i in range(self.minmax[0], self.minmax[1] + 1):
            self.qft_block(i)
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
    def __init__(self, t, phase, init_state='zeros', den_mat=False, mps=False, chi=None):
        nqubit = t + 1
        self.phase = phase
        super().__init__(nqubit=nqubit, wires=None, minmax=None, init_state=init_state,
                         name='QuantumPhaseEstimationSingleQubit', den_mat=den_mat, mps=mps, chi=chi)
        self.hlayer(list(range(t)))
        self.x(t)
        for i in range(t):
            self.qpe_block(i)
        self.add(QuantumFourierTransform(nqubit=nqubit, minmax=[0, t - 1]).inverse())
        
    def qpe_block(self, n):
        t = self.nqubit - 1
        for _ in range(2 ** (t - n - 1)):
            self.cp(n, t, 2 * torch.pi * self.phase)