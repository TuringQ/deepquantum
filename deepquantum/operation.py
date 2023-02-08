import torch
import torch.nn as nn


class Operation(nn.Module):
    def __init__(self, name=None, nqubit=1, wires=0, den_mat=False, tsr_mode=False):
        super().__init__()
        self.name = name
        self.nqubit = nqubit
        self.wires = wires
        self.den_mat = den_mat
        self.tsr_mode = tsr_mode
        self.npara = 0

    def tensor_rep(self, x):
        if self.den_mat:
            return x.reshape([-1] + [2] * 2 * self.nqubit + [1])
        else:
            return x.reshape([-1] + [2] * self.nqubit + [1])

    def vector_rep(self, x):
        return x.reshape([-1, 2 ** self.nqubit, 1])

    def matrix_rep(self, x):
        return x.reshape([-1, 2 ** self.nqubit, 2 ** self.nqubit])

    def get_unitary(self):
        raise NotImplementedError
        
    def init_para(self):
        pass

    def forward(self, x):
        if self.tsr_mode:
            return self.tensor_rep(x)
        else:
            if self.den_mat:
                return self.matrix_rep(x)
            else:
                return self.vector_rep(x)


class Gate(Operation):
    def __init__(self, name=None, nqubit=1, wires=0, den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        if type(wires) == int:
            assert wires < nqubit
            self.n = 1
        if type(wires) == list:
            for wire in wires:
                assert type(wire) == int
                assert wire < nqubit
            self.n = len(wires)
        self.register_buffer('identity', torch.eye(2, dtype=torch.cfloat))
        self.register_buffer('paulix', torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat))
        self.register_buffer('pauliy', torch.tensor([[0, -1j], [1j, 0]]))
        self.register_buffer('pauliz', torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat))
        self.register_buffer('hadamard', torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat) / 2 ** 0.5)
        
        self.register_buffer('matrix', torch.empty((2 ** self.n, 2 ** self.n), dtype=torch.cfloat))

    def update_matrix(self):
        return self.matrix

    def op_state(self, x):
        x = self.get_unitary() @ self.vector_rep(x)
        if self.tsr_mode:
            return self.tensor_rep(x)
        return x.squeeze(0)

    def op_den_mat(self, x):
        u = self.get_unitary()
        x = u @ self.matrix_rep(x) @ u.conj().transpose(-1, -2)
        if self.tsr_mode:
            return self.tensor_rep(x)
        return x.squeeze(0)

    def forward(self, x):
        if not self.tsr_mode:
            x = self.tensor_rep(x)
        if self.den_mat:
            return self.op_den_mat(x)
        else:
            return self.op_state(x)


class Layer(Operation):
    def __init__(self, name=None, nqubit=1, wires=None, den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.gates = nn.ModuleList([])

    def get_unitary(self):
        u = torch.eye(2 ** self.nqubit, dtype=torch.cfloat)
        for gate in self.gates:
            u = gate.get_unitary() @ u
        return u

    def init_para(self, inputs=None):
        count = 0
        for gate in self.gates:
            gate.init_para(inputs[count:count+gate.npara])
            count += gate.npara
    
    def update_npara(self):
        self.npara = 0
        for gate in self.gates:
            self.npara += gate.npara

    def forward(self, x):
        if not self.tsr_mode:
            x = self.tensor_rep(x)
        for gate in self.gates:
            x = gate(x)
        if not self.tsr_mode:
            if self.den_mat:
                return self.matrix_rep(x)
            else:
                return self.vector_rep(x)
        return x