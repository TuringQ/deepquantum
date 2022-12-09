import torch
import torch.nn as nn
from operation import Gate
from qmath import multi_kron


class SingleGate(Gate):
    def __init__(self, name=None, nqubit=1, wires=0, MPS=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, MPS=MPS)
    
    def single_gate_TN(self, MPS):
        permute_shape = list(range(self.nqubit + 2))
        permute_shape[self.wires + 1] = self.nqubit
        permute_shape[self.nqubit] = self.wires + 1 
        return (self.matrix() @ MPS.permute(permute_shape)).permute(permute_shape)
        
    def U(self):
        lst = [self.identity] * self.nqubit
        lst[self.wires] = self.matrix
        return multi_kron(lst)
            
    def forward(self, x):
        if not self.MPS:
            x = self.to_MPS(x)
        x = self.single_gate_TN(x)
        if self.MPS:
            return x
        else:
            return self.to_SV(x)


class DoubleGate(Gate):
    def __init__(self, name=None, nqubit=2, wires=[0,1], MPS=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, MPS=MPS)
        self.register_buffer('zerozero', torch.tensor([[1, 0], [0, 0]], dtype=torch.cfloat))
        self.register_buffer('zeroone', torch.tensor([[0, 1], [0, 0]], dtype=torch.cfloat))
        self.register_buffer('onezero', torch.tensor([[0, 0], [1, 0]], dtype=torch.cfloat))
        self.register_buffer('oneone', torch.tensor([[0, 0], [0, 1]], dtype=torch.cfloat))
    
    def double_gate_TN(self, MPS):
        cqbit = self.wires[0]
        tqbit = self.wires[1]
        permute_shape = list(range(self.nqubit + 1))
        permute_shape.remove(cqbit + 1)
        permute_shape.remove(tqbit + 1)
        permute_shape = permute_shape + [cqbit + 1, tqbit + 1] + [self.nqubit + 1]
        MPS = (self.matrix() @ MPS.permute(permute_shape).reshape(-1, 4, 1)).reshape([-1] + [2] * self.nqubit)
        permute_shape = list(range(self.nqubit + 1))
        permute_shape.pop()
        permute_shape.pop()
        if cqbit < tqbit:
            permute_shape.insert(cqbit + 1, self.nqubit - 1)
            permute_shape.insert(tqbit + 1, self.nqubit)
        else:
            permute_shape.insert(tqbit + 1, self.nqubit)
            permute_shape.insert(cqbit + 1, self.nqubit - 1)
        return MPS.permute(permute_shape).unsqueeze(-1)
        
    def U(self):
        lst1 = [self.identity] * self.nqubit
        lst1[self.wires[0]] = self.zerozero
        lst1[self.wires[1]] = self.matrix[0:2, 0:2]

        lst2 = [self.identity] * self.nqubit
        lst2[self.wires[0]] = self.zeroone
        lst2[self.wires[1]] = self.matrix[0:2, 2:4]

        lst3 = [self.identity] * self.nqubit
        lst3[self.wires[0]] = self.onezero
        lst3[self.wires[1]] = self.matrix[2:4, 0:2]

        lst4 = [self.identity] * self.nqubit
        lst4[self.wires[0]] = self.oneone
        lst4[self.wires[1]] = self.matrix[2:4, 2:4]
        return multi_kron(lst1) + multi_kron(lst2) + multi_kron(lst3) + multi_kron(lst4)

    def forward(self, x):
        if not self.MPS:
            x = self.to_MPS(x)
        x = self.double_gate_TN(x)
        if self.MPS:
            return x
        else:
            return self.to_SV(x)


class DoubleControlGate(DoubleGate):
    def __init__(self, name=None, nqubit=2, wires=[0,1], MPS=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, MPS=MPS)
        
    def U(self):
        lst1 = [self.identity] * self.nqubit
        lst1[self.wires[0]] = self.zerozero

        lst2 = [self.identity] * self.nqubit
        lst2[self.wires[0]] = self.oneone
        lst2[self.wires[1]] = self.matrix[2:4, 2:4]
        return multi_kron(lst1) + multi_kron(lst2)


class Identity(Gate):
    def __init__(self, nqubit=1, wires=0, MPS=False):
        super().__init__(name='Identity', nqubit=nqubit, wires=wires, MPS=MPS)
        
    def U(self):
        return torch.eye(2 ** self.nqubit, dtype=torch.cfloat).to(self.matrix.device)
        
    def forward(self, x):
        return x


class PauliX(SingleGate):
    def __init__(self, nqubit=1, wires=0, MPS=False):
        super().__init__(name='PauliX', nqubit=nqubit, wires=wires, MPS=MPS)
        self.matrix = self.paulix


class Rx(SingleGate):
    def __init__(self, theta=None, nqubit=1, wires=0, MPS=False, requires_grad=False):
        super().__init__(name='Rx', nqubit=nqubit, wires=wires, MPS=MPS)
        self.requires_grad = requires_grad
        if theta == None:
            theta = torch.rand(1) * torch.pi
        elif type(theta) != torch.Tensor:
            theta = torch.tensor(theta)
        if requires_grad:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('theta', theta)
        self.matrix()
        
    def matrix(self):
        self.matrix = torch.cos(self.theta / 2.0) * self.identity - 1j * torch.sin(self.theta / 2.0) * self.paulix
        return self.matrix
        
    def reinit_para(self):
        self.theta = torch.rand(1) * torch.pi
        if self.requires_grad:
            self.theta = nn.Parameter(self.theta)
        self.matrix()


class CombinedSingleGate(SingleGate):
    def __init__(self, gatelist, name=None, nqubit=1, wires=0, MPS=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, MPS=MPS)
        self.gatelist = nn.ModuleList(gatelist)
        self.matrix()
        
    def matrix(self):
        self.matrix = self.identity
        for gate in self.gatelist:
            self.matrix = gate.matrix() @ self.matrix
        return self.matrix
        
    def add(self, gate):
        self.gatelist.append(gate)
        self.matrix = gate.matrix() @ self.matrix