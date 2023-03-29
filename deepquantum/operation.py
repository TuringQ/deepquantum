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
            assert x.shape[-1] == 2 ** self.nqubit and x.shape[-2] == 2 ** self.nqubit
            return x.reshape([-1] + [2] * 2 * self.nqubit + [1])
        else:
            if x.ndim == 1:
                assert x.shape[-1] == 2 ** self.nqubit
            else:
                assert x.shape[-1] == 2 ** self.nqubit or x.shape[-2] == 2 ** self.nqubit
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
            self.nwire = 1
        if type(wires) == list:
            for wire in wires:
                assert type(wire) == int
                assert wire < nqubit
            self.nwire = len(wires)

    def update_matrix(self):
        raise NotImplementedError

    def op_state(self, x):
        x = self.get_unitary() @ self.vector_rep(x)
        if self.tsr_mode:
            return self.tensor_rep(x)
        return x.squeeze(0)

    def op_den_mat(self, x):
        u = self.get_unitary()
        x = u @ self.matrix_rep(x) @ u.mH
        if self.tsr_mode:
            return self.tensor_rep(x)
        return x.squeeze(0)

    def forward(self, x):    
        if not self.tsr_mode:
            x = self.tensor_rep(x)
        if self.den_mat:
            assert x.ndim == 2 * self.nqubit + 2
            return self.op_den_mat(x)
        else:
            assert x.ndim == self.nqubit + 2
            return self.op_state(x)


class Layer(Operation):
    def __init__(self, name=None, nqubit=1, wires=None, den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.gates = nn.Sequential()

    def get_unitary(self):
        u = torch.eye(2 ** self.nqubit, dtype=torch.cfloat)
        for gate in self.gates:
            u = gate.get_unitary() @ u
        return u

    def init_para(self, inputs=None):
        count = 0
        for gate in self.gates:
            if inputs == None:
                gate.init_para(inputs)
            else:
                gate.init_para(inputs[count:count+gate.npara])
            count += gate.npara
    
    def update_npara(self):
        self.npara = 0
        for gate in self.gates:
            self.npara += gate.npara

    def forward(self, x):
        if not self.tsr_mode:
            x = self.tensor_rep(x)
        x = self.gates(x)
        if not self.tsr_mode:
            if self.den_mat:
                return self.matrix_rep(x).squeeze(0)
            else:
                return self.vector_rep(x).squeeze(0)
        return x