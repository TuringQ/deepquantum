import torch
import torch.nn as nn
from operation import *
from gate import *
from layer import *
from functorch import vmap
from qmath import *


class Circuit(Operation):
    def __init__(self, nqubit, init_state='zeros', name=None, den_mat=False):
        super().__init__(name=name, nqubit=nqubit, wires=None, den_mat=den_mat)
        if init_state == 'zeros':
            init_state = torch.zeros((2 ** self.nqubit, 1), dtype=torch.cfloat)
            init_state[0] = 1
            if den_mat:
                init_state = init_state @ init_state.T
        self.operators = nn.ModuleList([])
        self.gates = nn.ModuleList([])
        self.encoders = nn.ModuleList([])
        self.measurement = None
        self.register_buffer('init_state', init_state)
        self.state = None

    def forward(self, data=None):
        if self.init_state.ndim == 2:
            return vmap(self.forward_helper)(data)
        else:
            return self.forward_helper(data)

    def forward_helper(self, data=None):
        self.encode(data)
        x = self.tensor_rep(self.init_state)
        for op in self.operators:
            x = op(x)
        if self.den_mat:
            x = self.matrix_rep(x)
        else:
            x = self.vector_rep(x)
        self.state = x.squeeze(0)
        return self.state

    def encode(self, data):
        if data == None:
            return
        count = 0
        for op in self.encoders:
            op.init_para(data[count:count+op.npara])
            count += op.npara

    def amplitude_encoding(self, data):
        self.init_state = amplitude_encoding(data, self.nqubit)
    
    def measure(self):
        pass

    def sample(self):
        pass

    def expectation(self):
        pass

    def get_unitary(self):
        u = torch.eye(2 ** self.nqubit, dtype=torch.cfloat)
        for op in self.operators:
            u = op.get_unitary() @ u
        return u
    
    def init_para(self):
        for op in self.operators:
            op.init_para()
            
    def add(self, op):
        self.operators.append(op)
        if isinstance(op, Gate):
            self.gates.append(op)
        else:
            self.gates += op.gates

    def print(self):
        pass
        
    def draw(self):
        pass

    def rxlayer(self, inputs=None, wires=None, encode=False):
        requires_grad = not encode
        rxl = RxLayer(inputs=inputs, nqubit=self.nqubit, wires=wires, den_mat=self.den_mat,
                      tsr_mode=True, requires_grad=requires_grad)
        self.add(rxl)
        if encode:
            self.encoders.append(rxl)