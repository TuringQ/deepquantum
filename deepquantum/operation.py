import torch
import torch.nn as nn
from deepquantum.qmath import inverse_permutation


class Operation(nn.Module):
    def __init__(self, name=None, nqubit=1, wires=None, den_mat=False, tsr_mode=False):
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
            return x.reshape([-1] + [2] * 2 * self.nqubit)
        else:
            if x.ndim == 1:
                assert x.shape[-1] == 2 ** self.nqubit
            else:
                assert x.shape[-1] == 2 ** self.nqubit or x.shape[-2] == 2 ** self.nqubit
            return x.reshape([-1] + [2] * self.nqubit)

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
    def __init__(self, name=None, nqubit=1, wires=[0], controls=None, den_mat=False, tsr_mode=False):
        if type(wires) == int:
            wires = [wires]
        if type(controls) == int:
            controls = [controls]
        if controls == None:
            controls = []
        assert type(wires) == list and type(controls) == list, 'Invalid input type'
        assert all(isinstance(i, int) for i in wires), 'Invalid input type'
        assert all(isinstance(i, int) for i in controls), 'Invalid input type'
        assert min(wires) > -1 and max(wires) < nqubit, 'Invalid input'
        if len(controls) > 0:
            assert min(controls) > -1 and max(controls) < nqubit, 'Invalid input'
        assert len(set(wires)) == len(wires) and len(set(controls)) == len(controls), 'Invalid input'
        for wire in wires:
            assert wire not in controls, 'Use repeated wires'
        self.nwire = len(wires) + len(controls)
        self.controls = controls
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)

    def update_matrix(self):
        return self.matrix

    def op_state(self, x):
        matrix = self.update_matrix()
        if self.controls == []:
            x = self.op_state_base(x=x, matrix=matrix)
        else:
            x = self.op_state_control(x=x, matrix=matrix)
        if not self.tsr_mode:
            x = self.vector_rep(x).squeeze(0)
        return x
    
    def op_state_base(self, x, matrix):
        nt = len(self.wires)
        wires = [i + 1 for i in self.wires]
        pm_shape = list(range(self.nqubit + 1))
        for i in wires:
            pm_shape.remove(i)
        pm_shape = wires + pm_shape
        x = x.permute(pm_shape).reshape(2 ** nt, -1)
        x = (matrix @ x).reshape([2] * nt + [-1] + [2] * (self.nqubit - nt))
        x = x.permute(inverse_permutation(pm_shape))
        return x
    
    def op_state_control(self, x, matrix):
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
        state1 = x.permute(pm_shape).reshape(2 ** nt, -1, 2 ** nc)
        state2 = (matrix @ state1[:, :, -1]).unsqueeze(-1)
        state1 = torch.cat([state1[:, :, :-1], state2], dim=-1)
        state1 = state1.reshape([2] * nt + [-1] + [2] * (self.nqubit - nt - nc) + [2] * nc)
        x = state1.permute(inverse_permutation(pm_shape))
        return x
    
    def op_den_mat(self, x):
        matrix = self.update_matrix()
        if self.controls == []:
            x = self.op_den_mat_base(x=x, matrix=matrix)
        else:
            x = self.op_den_mat_control(x=x, matrix=matrix)
        if not self.tsr_mode:
            x = self.matrix_rep(x).squeeze(0)
        return x
        
    def op_den_mat_base(self, x, matrix):
        nt = len(self.wires)
        # left multiply
        wires = [i + 1 for i in self.wires]
        pm_shape = list(range(2 * self.nqubit + 1))
        for i in wires:
            pm_shape.remove(i)
        pm_shape = wires + pm_shape
        x = x.permute(pm_shape).reshape(2 ** nt, -1)
        x = (matrix @ x).reshape([2] * nt + [-1] + [2] * (2 * self.nqubit - nt))
        x = x.permute(inverse_permutation(pm_shape))
        # right multiply
        wires = [i + 1 + self.nqubit for i in self.wires]
        pm_shape = list(range(2 * self.nqubit + 1))
        for i in wires:
            pm_shape.remove(i)
        pm_shape = wires + pm_shape
        x = x.permute(pm_shape).reshape(2 ** nt, -1)
        x = (matrix.conj() @ x).reshape([2] * nt + [-1] + [2] * (2 * self.nqubit - nt))
        x = x.permute(inverse_permutation(pm_shape))
        return x
    
    def op_den_mat_control(self, x, matrix):
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
        state1 = x.permute(pm_shape).reshape(2 ** nt, -1, 2 ** nc)
        state2 = (matrix @ state1[:, :, -1]).unsqueeze(-1)
        state1 = torch.cat([state1[:, :, :-1], state2], dim=-1)
        state1 = state1.reshape([2] * nt + [-1] + [2] * (2 * self.nqubit - nt - nc) + [2] * nc)
        x = state1.permute(inverse_permutation(pm_shape))
        # right multiply
        wires = [i + 1 + self.nqubit for i in self.wires]
        controls = [i + 1 + self.nqubit for i in self.controls]
        pm_shape = list(range(2 * self.nqubit + 1))
        for i in wires:
            pm_shape.remove(i)
        for i in controls:
            pm_shape.remove(i)
        pm_shape = wires + pm_shape + controls
        state1 = x.permute(pm_shape).reshape(2 ** nt, -1, 2 ** nc)
        state2 = (matrix.conj() @ state1[:, :, -1]).unsqueeze(-1)
        state1 = torch.cat([state1[:, :, :-1], state2], dim=-1)
        state1 = state1.reshape([2] * nt + [-1] + [2] * (2 * self.nqubit - nt - nc) + [2] * nc)
        x = state1.permute(inverse_permutation(pm_shape))
        return x

    def forward(self, x):    
        if not self.tsr_mode:
            x = self.tensor_rep(x)
        if self.den_mat:
            assert x.ndim == 2 * self.nqubit + 1
            return self.op_den_mat(x)
        else:
            assert x.ndim == self.nqubit + 1
            return self.op_state(x)

    def extra_repr(self):
        s = f'wires={self.wires}'
        if self.controls == []:
            return s
        else:
            return s + f', controls={self.controls}'


class Layer(Operation):
    def __init__(self, name=None, nqubit=1, wires=[[0]], den_mat=False, tsr_mode=False):
        if type(wires) == int:
            wires = [[wires]]
        assert type(wires) == list, 'Invalid input type'
        if all(isinstance(i, int) for i in wires):
            wires = [[i] for i in wires]
        assert all(isinstance(i, list) for i in wires), 'Invalid input type'
        for wire in wires:
            assert all(isinstance(i, int) for i in wire), 'Invalid input type'
            assert min(wire) > -1 and max(wire) < nqubit, 'Invalid input'
            assert len(set(wire)) == len(wire), 'Invalid input'
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.gates = nn.Sequential()

    def get_unitary(self):
        u = None
        for gate in self.gates:
            if u == None:
                u = gate.get_unitary()
            else:
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