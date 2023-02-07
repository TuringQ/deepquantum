import torch
import torch.nn as nn
from operation import Gate
from qmath import multi_kron


class SingleGate(Gate):
    def __init__(self, name=None, nqubit=1, wires=0, den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
    
    def op_state(self, x):
        matrix = self.update_matrix()
        permute_shape = list(range(self.nqubit + 2))
        permute_shape[self.wires + 1] = self.nqubit
        permute_shape[self.nqubit] = self.wires + 1 
        rst = (matrix @ x.permute(permute_shape)).permute(permute_shape)
        if not self.tsr_mode:
            rst = self.vector_rep(rst)
        return rst
        
    def op_den_mat(self, x):
        pass

    def get_unitary(self):
        matrix = self.update_matrix()
        lst = [self.identity] * self.nqubit
        lst[self.wires] = matrix
        return multi_kron(lst)


class DoubleGate(Gate):
    def __init__(self, name=None, nqubit=2, wires=[0,1], den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('zerozero', torch.tensor([[1, 0], [0, 0]], dtype=torch.cfloat))
        self.register_buffer('zeroone', torch.tensor([[0, 1], [0, 0]], dtype=torch.cfloat))
        self.register_buffer('onezero', torch.tensor([[0, 0], [1, 0]], dtype=torch.cfloat))
        self.register_buffer('oneone', torch.tensor([[0, 0], [0, 1]], dtype=torch.cfloat))
    
    def op_state(self, x):
        matrix = self.update_matrix()
        cqbit = self.wires[0]
        tqbit = self.wires[1]
        permute_shape = list(range(self.nqubit + 1))
        permute_shape.remove(cqbit + 1)
        permute_shape.remove(tqbit + 1)
        permute_shape = permute_shape + [cqbit + 1, tqbit + 1] + [self.nqubit + 1]
        x = (matrix @ x.permute(permute_shape).reshape(-1, 4, 1)).reshape([-1] + [2] * self.nqubit)
        permute_shape = list(range(self.nqubit + 1))
        permute_shape.pop()
        permute_shape.pop()
        if cqbit < tqbit:
            permute_shape.insert(cqbit + 1, self.nqubit - 1)
            permute_shape.insert(tqbit + 1, self.nqubit)
        else:
            permute_shape.insert(tqbit + 1, self.nqubit)
            permute_shape.insert(cqbit + 1, self.nqubit - 1)
        rst = x.permute(permute_shape).unsqueeze(-1)
        if not self.tsr_mode:
            rst = self.vector_rep(rst)
        return rst
        
    def op_den_mat(self, x):
        pass
    
    def get_unitary(self):
        matrix = self.update_matrix()
        lst1 = [self.identity] * self.nqubit
        lst1[self.wires[0]] = self.zerozero
        lst1[self.wires[1]] = matrix[0:2, 0:2]

        lst2 = [self.identity] * self.nqubit
        lst2[self.wires[0]] = self.zeroone
        lst2[self.wires[1]] = matrix[0:2, 2:4]

        lst3 = [self.identity] * self.nqubit
        lst3[self.wires[0]] = self.onezero
        lst3[self.wires[1]] = matrix[2:4, 0:2]

        lst4 = [self.identity] * self.nqubit
        lst4[self.wires[0]] = self.oneone
        lst4[self.wires[1]] = matrix[2:4, 2:4]
        return multi_kron(lst1) + multi_kron(lst2) + multi_kron(lst3) + multi_kron(lst4)


class DoubleControlGate(DoubleGate):
    def __init__(self, name=None, nqubit=2, wires=[0,1], den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        
    def get_unitary(self):
        matrix = self.update_matrix()
        lst1 = [self.identity] * self.nqubit
        lst1[self.wires[0]] = self.zerozero

        lst2 = [self.identity] * self.nqubit
        lst2[self.wires[0]] = self.oneone
        lst2[self.wires[1]] = matrix[2:4, 2:4]
        return multi_kron(lst1) + multi_kron(lst2)


class Identity(Gate):
    def __init__(self, nqubit=1, wires=0, den_mat=False, tsr_mode=False):
        super().__init__(name='Identity', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.matrix = torch.eye(2 ** self.nqubit, dtype=torch.cfloat)
        
    def get_unitary(self):
        return self.matrix
        
    def forward(self, x):
        return x


class PauliX(SingleGate):
    def __init__(self, nqubit=1, wires=0, den_mat=False, tsr_mode=False):
        super().__init__(name='PauliX', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.matrix = self.paulix


class Rx(SingleGate):
    def __init__(self, inputs=None, nqubit=1, wires=0, den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name='Rx', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.npara = 1
        self.requires_grad = requires_grad
        while type(inputs) == list:
            inputs = inputs[0]
        if inputs == None:
            inputs = torch.rand(1) * torch.pi
        elif type(inputs) != torch.Tensor:
            inputs = torch.tensor(inputs)
        if requires_grad:
            self.theta = nn.Parameter(inputs)
        else:
            self.register_buffer('theta', inputs)
        self.update_matrix()
        
    def update_matrix(self):
        matrix = torch.cos(self.theta / 2.0) * self.identity \
               - torch.sin(self.theta / 2.0) * self.paulix * 1j
        self.matrix = matrix.detach()
        return matrix
        
    def init_para(self, inputs=None):
        while type(inputs) == list:
            inputs = inputs[0]
        if inputs == None:
            inputs = torch.rand(1) * torch.pi
        elif type(inputs) != torch.Tensor:
            inputs = torch.tensor(inputs)
        if self.requires_grad:
            self.theta = nn.Parameter(inputs)
        else:
            self.theta = inputs
        self.update_matrix()


class CombinedSingleGate(SingleGate):
    def __init__(self, gatelist, name=None, nqubit=1, wires=0, den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.gatelist = nn.ModuleList(gatelist)
        self.update_npara()
        self.update_matrix()
        
    def update_matrix(self):
        matrix = self.identity
        for gate in self.gatelist:
            matrix_i = gate.update_matrix()
            matrix = matrix_i @ matrix
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