from deepquantum.circuit import QubitCircuit
import random


class RandomCircuitG3(QubitCircuit):
    def __init__(self, nqubit, ngate, wires=None, init_state='zeros', den_mat=False, mps=False, chi=None):
        super().__init__(nqubit=nqubit, init_state=init_state, name='RandomCircuitG3', den_mat=den_mat,
                         mps=mps, chi=chi)
        self.ngate = ngate
        self.gate_set = ['CNOT', 'H', 'T']
        if wires == None:
            wires = list(range(nqubit))
        assert type(wires) == list, 'Invalid input type'
        assert all(isinstance(i, int) for i in wires), 'Invalid input type'
        assert min(wires) > -1 and max(wires) < nqubit, 'Invalid input'
        assert len(set(wires)) == len(wires), 'Invalid input'
        self.wires = wires
        for _ in range(ngate):
            gate = random.sample(self.gate_set, 1)[0]
            if gate == 'CNOT':
                wire = random.sample(wires, 2)
            else:
                wire = random.sample(wires, 1)
            if gate == 'CNOT':
                self.cnot(wire[0], wire[1])
            elif gate == 'H':
                self.h(wire)
            elif gate == 'T':
                self.t(wire)