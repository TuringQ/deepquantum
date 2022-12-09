import torch
import torch.nn as nn


class Operation(nn.Module):
    def __init__(self, name=None, nqubit=1, wires=0, MPS=False):
        super().__init__()
        self.name = name
        self.nqubit = nqubit
        self.wires = wires
        self.MPS = MPS
        
    def to_MPS(self, x):
        return x.reshape([-1] + [2] * self.nqubit + [1])
        
    def to_SV(self, x):
        return x.reshape([-1, 2 ** self.nqubit, 1])
        
    def reinit_para(self):
        pass
        
    def forward(self, x):
        if self.MPS:
            return self.to_MPS(x)
        else:
            return self.to_SV(x)


class Gate(Operation):
    def __init__(self, name=None, nqubit=1, wires=0, MPS=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, MPS=MPS)
        if type(wires) == int:
            self.n = 1
        if type(wires) == list:
            self.n = len(wires)
        self.register_buffer('identity', torch.eye(2, dtype=torch.cfloat))
        self.register_buffer('paulix', torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat))
        self.register_buffer('pauliy', torch.tensor([[0, -1j], [1j, 0]]))
        self.register_buffer('pauliz', torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat))
        self.register_buffer('hadamard', torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat) / 2 ** 0.5)
        
        self.register_buffer('matrix', torch.eye(2 ** self.n, dtype=torch.cfloat))
        
    def matrix(self):
        return self.matrix
        
    def U(self):
        raise NotImplementedError
        
    def left_multiply(self, x):
        return self.U() @ self.to_SV(x)
        
    def forward(self, x):
        x = self.left_multiply(x)
        if self.MPS:
            return self.to_MPS(x)
        else:
            return x


class Layer(Operation):
    def __init__(self, name=None, nqubit=1, wires=None, first_layer=False, last_layer=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, MPS=True)
        if wires == None:
            self.wires = list(range(nqubit))
        else:
            self.wires = wires    
        self.first_layer=first_layer
        self.last_layer=last_layer
        self.gates = nn.ModuleList([])
        
    def reinit_para(self):
        for gate in self.gates:
            gate.reinit_para()

    def forward(self, x):
        if self.first_layer:
            x = self.to_MPS(x)
        for gate in self.gates:
            x = gate(x)
        if self.last_layer:
            x = self.to_SV(x)
        return x