import torch
import torch.nn as nn
from deepquantum.operation import *
from deepquantum.gate import *
from deepquantum.layer import *
from torch import vmap
from deepquantum.qmath import amplitude_encoding, measure, expectation


class QubitCircuit(Operation):
    def __init__(self, nqubit, init_state='zeros', name=None, den_mat=False, reupload=False):
        super().__init__(name=name, nqubit=nqubit, wires=None, den_mat=den_mat)
        if init_state == 'zeros':
            init_state = torch.zeros((2 ** self.nqubit, 1), dtype=torch.cfloat)
            init_state[0] = 1
        elif init_state == 'entangle':
            init_state = torch.ones((2 ** self.nqubit, 1), dtype=torch.cfloat)
            init_state = nn.functional.normalize(init_state, p=2, dim=-2)
        if den_mat:
            s = init_state.shape
            if s[-1] != 2 ** self.nqubit or s[-2] != 2 ** self.nqubit:
                init_state = init_state.reshape(s[0], -1, 1)
                assert init_state.shape[1] == 2 ** self.nqubit, 'The shape of initial state is not correct'
                init_state = init_state @ init_state.conj().transpose(-1, -2)
        self.reupload = reupload
        self.operators = nn.Sequential()
        self.encoders = nn.ModuleList([])
        self.observables = nn.ModuleList([])
        self.register_buffer('init_state', init_state)
        self.state = None

    def forward(self, data=None):
        if self.init_state.ndim == 2:
            self.state = vmap(self.forward_helper)(data)
            self.init_encoder()
        else:
            self.state = self.forward_helper(data)
        return self.state

    def forward_helper(self, data=None):
        self.encode(data)
        x = self.operators(self.tensor_rep(self.init_state))
        if self.den_mat:
            x = self.matrix_rep(x)
        else:
            x = self.vector_rep(x)
        return x.squeeze(0)

    def encode(self, data):
        if data == None:
            return
        count = 0
        for op in self.encoders:
            count_up = count + op.npara
            if count_up > len(data) and self.reupload:
                count = 0
                count_up = count + op.npara
            op.init_para(data[count:count_up])
            count = count_up

    def init_encoder(self): # deal with the problem of state_dict() with vmap
        for op in self.encoders:
            op.init_para()

    def amplitude_encoding(self, data):
        self.init_state = amplitude_encoding(data, self.nqubit)
    
    def observable(self, wires=None, basis='z'):
        observable = Observable(nqubit=self.nqubit, wires=wires, basis=basis,
                                den_mat=self.den_mat, tsr_mode=False)
        self.observables.append(observable)

    def reset_observable(self):
        self.observables = nn.ModuleList([])

    def measure(self, shots=1024, with_prob=False):
        return measure(self.state, shots=shots, with_prob=with_prob)

    def expectation(self):
        if self.observables and self.state != None:
            out = []
            for observable in self.observables:
                expval = expectation(self.state, observable=observable, den_mat=self.den_mat)
                out.append(expval)
            out = torch.stack(out, dim=-1)
            return out

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

    def print(self):
        pass
        
    def draw(self):
        pass

    def u3(self, wires, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        u3 = U3Gate(inputs=inputs, nqubit=self.nqubit, wires=wires, den_mat=self.den_mat,
                    tsr_mode=True, requires_grad=requires_grad)
        self.add(u3)
        if encode:
            self.encoders.append(u3)

    def x(self, wires):
        x = PauliX(nqubit=self.nqubit, wires=wires, den_mat=self.den_mat, tsr_mode=True)
        self.add(x)

    def y(self, wires):
        y = PauliY(nqubit=self.nqubit, wires=wires, den_mat=self.den_mat, tsr_mode=True)
        self.add(y)

    def z(self, wires):
        z = PauliZ(nqubit=self.nqubit, wires=wires, den_mat=self.den_mat, tsr_mode=True)
        self.add(z)

    def h(self, wires):
        h = Hadamard(nqubit=self.nqubit, wires=wires, den_mat=self.den_mat, tsr_mode=True)
        self.add(h)

    def rx(self, wires, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        rx = Rx(inputs=inputs, nqubit=self.nqubit, wires=wires, den_mat=self.den_mat,
                tsr_mode=True, requires_grad=requires_grad)
        self.add(rx)
        if encode:
            self.encoders.append(rx)

    def ry(self, wires, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        ry = Ry(inputs=inputs, nqubit=self.nqubit, wires=wires, den_mat=self.den_mat,
                tsr_mode=True, requires_grad=requires_grad)
        self.add(ry)
        if encode:
            self.encoders.append(ry)

    def rz(self, wires, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        rz = Rz(inputs=inputs, nqubit=self.nqubit, wires=wires, den_mat=self.den_mat,
                tsr_mode=True, requires_grad=requires_grad)
        self.add(rz)
        if encode:
            self.encoders.append(rz)

    def cnot(self, wires):
        cnot = CNOT(nqubit=self.nqubit, wires=wires, den_mat=self.den_mat, tsr_mode=True)
        self.add(cnot)

    def hlayer(self, wires=None):
        hl = HLayer(nqubit=self.nqubit, wires=wires, den_mat=self.den_mat, tsr_mode=True)
        self.add(hl)

    def rxlayer(self, inputs=None, wires=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        rxl = RxLayer(inputs=inputs, nqubit=self.nqubit, wires=wires, den_mat=self.den_mat,
                      tsr_mode=True, requires_grad=requires_grad)
        self.add(rxl)
        if encode:
            self.encoders.append(rxl)
    
    def rylayer(self, inputs=None, wires=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        ryl = RyLayer(inputs=inputs, nqubit=self.nqubit, wires=wires, den_mat=self.den_mat,
                      tsr_mode=True, requires_grad=requires_grad)
        self.add(ryl)
        if encode:
            self.encoders.append(ryl)

    def rzlayer(self, inputs=None, wires=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        rzl = RzLayer(inputs=inputs, nqubit=self.nqubit, wires=wires, den_mat=self.den_mat,
                      tsr_mode=True, requires_grad=requires_grad)
        self.add(rzl)
        if encode:
            self.encoders.append(rzl)

    def cnot_ring(self, minmax=None, step=1, reverse=False):
        cxr = CnotRing(nqubit=self.nqubit, minmax=minmax, den_mat=self.den_mat,
                       tsr_mode=True, step=step, reverse=reverse)
        self.add(cxr)