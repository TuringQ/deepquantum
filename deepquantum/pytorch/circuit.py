import torch.nn as nn
from state import Qubits
from operation import Gate, Layer


class Circuit(nn.Module):
    def __init__(self, nqubit, init_state='zeros'):
        super().__init__()
        self.nqubit = nqubit
        if init_state == 'zeros':
            self.init_state = Qubits(nqubit=nqubit)
        self.layers = nn.ModuleList([])
        self.gates = nn.ModuleList([])
        self.state_f = None

    def forward(self, x):
        state = self.encoding(x)
        for layer in self.layers:
            state = layer(state)
        self.state_f = state
        return self.state_f

    def evolve(self):
        pass
    
    def measure(self):
        pass

    def sample(self):
        pass

    def expectation(self):
        pass

    def get_unitary(self):
        pass
    
    def reinit_para(self):
        for layer in self.layers:
            layer.reinit_para()
            
    def add(self, op):
        if isinstance(op, Gate):
            self.gates.append(op)
        if isinstance(op, Layer):
            self.layers.append(op)
        
    def encoding(self, x):
        pass

    def print(self):
        pass
        
    def draw(self):
        pass