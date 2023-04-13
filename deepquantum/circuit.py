import numpy as np
import torch
import torch.nn as nn
from deepquantum.state import QubitState
from deepquantum.operation import Operation
from deepquantum.gate import *
from deepquantum.layer import *
from torch import vmap
from deepquantum.qmath import amplitude_encoding, measure, expectation
from qiskit import QuantumCircuit


class QubitCircuit(Operation):
    def __init__(self, nqubit, init_state='zeros', name=None, den_mat=False, reupload=False):
        super().__init__(name=name, nqubit=nqubit, wires=None, den_mat=den_mat)
        init_state = QubitState(nqubit=nqubit, state=init_state, den_mat=den_mat)
        self.operators = nn.Sequential()
        self.encoders = []
        self.observables = nn.ModuleList([])
        self.register_buffer('init_state', init_state.state)
        self.state = None
        self.npara = 0
        self.ndata = 0
        self.depth = np.array([0] * nqubit)
        self.reupload = reupload
        self.wires_measure = None

    def __add__(self, rhs):
        assert self.nqubit == rhs.nqubit
        cir = QubitCircuit(nqubit=self.nqubit, name=self.name, den_mat=self.den_mat, reupload=self.reupload)
        cir.operators = self.operators + rhs.operators
        cir.encoders = self.encoders + rhs.encoders
        cir.observables = rhs.observables
        cir.init_state = self.init_state
        cir.npara = self.npara + rhs.npara
        cir.ndata = self.ndata + rhs.ndata
        cir.depth = self.depth + rhs.depth
        return cir

    def forward(self, data=None, state=None):
        if state == None:
            state = self.init_state
        if data == None:
            self.state = self.forward_helper(state=state)
            if self.state.ndim == 2:
                self.state = self.state.unsqueeze(0)
            if state.ndim == 2:
                self.state = self.state.squeeze(0)
        else:
            if data.ndim == 1:
                data = data.unsqueeze(0)
            assert data.ndim == 2
            if state.ndim == 2:
                self.state = vmap(self.forward_helper, in_dims=(0, None))(data, state)
            elif state.ndim == 3:
                self.state = vmap(self.forward_helper)(data, state)
            self.init_encoder()
        return self.state

    def forward_helper(self, data=None, state=None):
        self.encode(data)
        if state == None:
            state = self.init_state
        x = self.operators(self.tensor_rep(state))
        if self.den_mat:
            x = self.matrix_rep(x)
        else:
            x = self.vector_rep(x)
        return x.squeeze(0)

    def encode(self, data):
        if data == None:
            return
        if not self.reupload:
            assert len(data) >= self.ndata, 'The circuit needs more data, or consider data re-uploading'
        count = 0
        for op in self.encoders:
            count_up = count + op.npara
            if self.reupload and count_up > len(data):
                n = int(np.ceil(count_up / len(data)))
                data_tmp = torch.cat([data] * n)[count:count_up]
                op.init_para(data_tmp)
            else:
                op.init_para(data[count:count_up])
            count = count_up % len(data)

    def init_para(self):
        for op in self.operators:
            op.init_para()

    def init_encoder(self): # deal with the problem of state_dict() with vmap
        for op in self.encoders:
            op.init_para()

    def reset(self, init_state='zeros'):
        self.operators = nn.Sequential()
        self.encoders = []
        self.observables = nn.ModuleList([])
        self.init_state = QubitState(nqubit=self.nqubit, state=init_state, den_mat=self.den_mat).state
        self.state = None
        self.npara = 0
        self.ndata = 0
        self.depth = np.array([0] * self.nqubit)

    def amplitude_encoding(self, data):
        return amplitude_encoding(data, self.nqubit)
    
    def observable(self, wires=None, basis='z'):
        observable = Observable(nqubit=self.nqubit, wires=wires, basis=basis,
                                den_mat=self.den_mat, tsr_mode=False)
        self.observables.append(observable)

    def reset_observable(self):
        self.observables = nn.ModuleList([])

    def measure(self, shots=1024, with_prob=False, wires=None):
        if self.state == None:
            self.forward()
        if wires == None:
            self.wires_measure = list(range(self.nqubit))
        else:
            assert type(wires) in (int, list)
            if type(wires) == int:
                wires = [wires]
            self.wires_measure = wires
        return measure(self.state, shots=shots, with_prob=with_prob, wires=wires)

    def expectation(self):
        assert len(self.observables) > 0, 'There is no observable'
        assert type(self.state) == torch.Tensor, 'There is no final state'
        out = []
        for observable in self.observables:
            expval = expectation(self.state, observable=observable, den_mat=self.den_mat)
            out.append(expval)
        out = torch.stack(out, dim=-1)
        return out

    def get_unitary(self):
        u = None
        for op in self.operators:
            if u == None:
                u = op.get_unitary()
            else:
                u = op.get_unitary() @ u
        return u

    def add(self, op, encode=False):
        assert isinstance(op, Operation)
        if isinstance(op, QubitCircuit):
            assert self.nqubit == op.nqubit
            self.operators += op.operators
            self.encoders  += op.encoders
            self.observables = op.observables
            self.npara += op.npara
            self.ndata += op.ndata
            self.depth += op.depth
        else:
            self.operators.append(op)
            if isinstance(op, Gate):
                for i in op.wires:
                    self.depth[i] += 1
                for i in op.controls:
                    self.depth[i] += 1
            elif isinstance(op, Layer):
                for wire in op.wires:
                    for i in wire:
                        self.depth[i] += 1
            if encode:
                self.encoders.append(op)
                self.ndata += op.npara
            else:
                self.npara += op.npara

    def max_depth(self):
        return max(self.depth)
    
    def qasm(self):
        qasm_str = 'OPENQASM 2.0;\n' + 'include "qelib1.inc";\n'
        qasm_str += f'qreg q[{self.nqubit}];\n' + f'creg c[{self.nqubit}];\n'
        for op in self.operators:
            qasm_str += op.qasm()
        if self.wires_measure != None:
            for wire in self.wires_measure:
                qasm_str += f'measure q[{wire}] -> c[{wire}];\n'
        Gate.qasm_new_gate = []
        return qasm_str

    def print(self):
        pass
        
    def draw(self, output='mpl', **kwargs):
        qc = QuantumCircuit.from_qasm_str(self.qasm())
        return qc.draw(output=output, **kwargs)

    def u3(self, wires, inputs=None, controls=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        u3 = U3Gate(inputs=inputs, nqubit=self.nqubit, wires=wires, controls=controls,
                    den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(u3, encode=encode)

    def cu(self, wires, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        cu = U3Gate(inputs=inputs, nqubit=self.nqubit, wires=[wires[1]], controls=[wires[0]],
                    den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(cu, encode=encode)

    def ps(self, wires, inputs=None, controls=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        ps = PhaseShift(inputs=inputs, nqubit=self.nqubit, wires=wires, controls=controls,
                        den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(ps, encode=encode)

    def cphase(self, wires, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        cphase = PhaseShift(inputs=inputs, nqubit=self.nqubit, wires=[wires[1]], controls=[wires[0]],
                            den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(cphase, encode=encode)

    def x(self, wires, controls=None):
        x = PauliX(nqubit=self.nqubit, wires=wires, controls=controls,
                   den_mat=self.den_mat, tsr_mode=True)
        self.add(x)

    def y(self, wires, controls=None):
        y = PauliY(nqubit=self.nqubit, wires=wires, controls=controls,
                   den_mat=self.den_mat, tsr_mode=True)
        self.add(y)

    def z(self, wires, controls=None):
        z = PauliZ(nqubit=self.nqubit, wires=wires, controls=controls,
                   den_mat=self.den_mat, tsr_mode=True)
        self.add(z)

    def h(self, wires, controls=None):
        h = Hadamard(nqubit=self.nqubit, wires=wires, controls=controls,
                     den_mat=self.den_mat, tsr_mode=True)
        self.add(h)

    def s(self, wires, controls=None):
        s = SGate(nqubit=self.nqubit, wires=wires, controls=controls,
                  den_mat=self.den_mat, tsr_mode=True)
        self.add(s)

    def sdg(self, wires, controls=None):
        sdg = SDaggerGate(nqubit=self.nqubit, wires=wires, controls=controls,
                          den_mat=self.den_mat, tsr_mode=True)
        self.add(sdg)

    def t(self, wires, controls=None):
        t = TGate(nqubit=self.nqubit, wires=wires, controls=controls,
                  den_mat=self.den_mat, tsr_mode=True)
        self.add(t)

    def tdg(self, wires, controls=None):
        tdg = TDaggerGate(nqubit=self.nqubit, wires=wires, controls=controls,
                          den_mat=self.den_mat, tsr_mode=True)
        self.add(tdg)

    def ch(self, wires):
        ch = Hadamard(nqubit=self.nqubit, wires=[wires[1]], controls=[wires[0]],
                      den_mat=self.den_mat, tsr_mode=True)
        self.add(ch)

    def cs(self, wires):
        cs = SGate(nqubit=self.nqubit, wires=[wires[1]], controls=[wires[0]],
                   den_mat=self.den_mat, tsr_mode=True)
        self.add(cs)

    def csdg(self, wires):
        csdg = SDaggerGate(nqubit=self.nqubit, wires=[wires[1]], controls=[wires[0]],
                           den_mat=self.den_mat, tsr_mode=True)
        self.add(csdg)

    def ct(self, wires):
        ct = TGate(nqubit=self.nqubit, wires=[wires[1]], controls=[wires[0]],
                   den_mat=self.den_mat, tsr_mode=True)
        self.add(ct)

    def ctdg(self, wires):
        ctdg = TDaggerGate(nqubit=self.nqubit, wires=[wires[1]], controls=[wires[0]],
                           den_mat=self.den_mat, tsr_mode=True)
        self.add(ctdg)

    def rx(self, wires, inputs=None, controls=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        rx = Rx(inputs=inputs, nqubit=self.nqubit, wires=wires, controls=controls,
                den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(rx, encode=encode)

    def ry(self, wires, inputs=None, controls=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        ry = Ry(inputs=inputs, nqubit=self.nqubit, wires=wires, controls=controls,
                den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(ry, encode=encode)

    def rz(self, wires, inputs=None, controls=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        rz = Rz(inputs=inputs, nqubit=self.nqubit, wires=wires, controls=controls,
                den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(rz, encode=encode)

    def crx(self, wires, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        crx = Rx(inputs=inputs, nqubit=self.nqubit, wires=[wires[1]], controls=[wires[0]],
                 den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(crx, encode=encode)

    def cry(self, wires, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        cry = Ry(inputs=inputs, nqubit=self.nqubit, wires=[wires[1]], controls=[wires[0]],
                 den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(cry, encode=encode)

    def crz(self, wires, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        crz = Rz(inputs=inputs, nqubit=self.nqubit, wires=[wires[1]], controls=[wires[0]],
                 den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(crz, encode=encode)

    def cnot(self, wires):
        cnot = CNOT(nqubit=self.nqubit, wires=wires, den_mat=self.den_mat, tsr_mode=True)
        self.add(cnot)

    def cx(self, wires):
        cx = PauliX(nqubit=self.nqubit, wires=[wires[1]], controls=[wires[0]],
                    den_mat=self.den_mat, tsr_mode=True)
        self.add(cx)

    def cy(self, wires):
        cy = PauliY(nqubit=self.nqubit, wires=[wires[1]], controls=[wires[0]],
                    den_mat=self.den_mat, tsr_mode=True)
        self.add(cy)

    def cz(self, wires):
        cz = PauliZ(nqubit=self.nqubit, wires=[wires[1]], controls=[wires[0]],
                    den_mat=self.den_mat, tsr_mode=True)
        self.add(cz)

    def swap(self, wires, controls=None):
        swap = Swap(nqubit=self.nqubit, wires=wires, controls=controls,
                    den_mat=self.den_mat, tsr_mode=True)
        self.add(swap)

    def rxx(self, wires, inputs=None, controls=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        rxx = Rxx(inputs=inputs, nqubit=self.nqubit, wires=wires, controls=controls,
                  den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(rxx, encode=encode)

    def ryy(self, wires, inputs=None, controls=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        ryy = Ryy(inputs=inputs, nqubit=self.nqubit, wires=wires, controls=controls,
                  den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(ryy, encode=encode)

    def rzz(self, wires, inputs=None, controls=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        rzz = Rzz(inputs=inputs, nqubit=self.nqubit, wires=wires, controls=controls,
                  den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(rzz, encode=encode)

    def rxy(self, wires, inputs=None, controls=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        rxy = Rxy(inputs=inputs, nqubit=self.nqubit, wires=wires, controls=controls,
                  den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(rxy, encode=encode)

    def crxx(self, wires, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        crxx = Rxx(inputs=inputs, nqubit=self.nqubit, wires=[wires[1], wires[2]], controls=[wires[0]],
                   den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(crxx, encode=encode)

    def cryy(self, wires, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        cryy = Ryy(inputs=inputs, nqubit=self.nqubit, wires=[wires[1], wires[2]], controls=[wires[0]],
                   den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(cryy, encode=encode)

    def crzz(self, wires, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        crzz = Rzz(inputs=inputs, nqubit=self.nqubit, wires=[wires[1], wires[2]], controls=[wires[0]],
                   den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(crzz, encode=encode)

    def crxy(self, wires, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        crxy = Rxy(inputs=inputs, nqubit=self.nqubit, wires=[wires[1], wires[2]], controls=[wires[0]],
                   den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(crxy, encode=encode)

    def toffoli(self, wires):
        toffoli = Toffoli(nqubit=self.nqubit, wires=wires, den_mat=self.den_mat, tsr_mode=True)
        self.add(toffoli)

    def ccx(self, wires):
        ccx = PauliX(nqubit=self.nqubit, wires=[wires[2]], controls=[wires[0], wires[1]],
                     den_mat=self.den_mat, tsr_mode=True)
        self.add(ccx)

    def fredkin(self, wires):
        fredkin = Fredkin(nqubit=self.nqubit, wires=wires, den_mat=self.den_mat, tsr_mode=True)
        self.add(fredkin)

    def cswap(self, wires):
        cswap = Swap(nqubit=self.nqubit, wires=[wires[1], wires[2]], controls=[wires[0]],
                     den_mat=self.den_mat, tsr_mode=True)
        self.add(cswap)

    def any(self, unitary, minmax=None, name='uany'):
        uany = UAnyGate(unitary=unitary, nqubit=self.nqubit, minmax=minmax, name=name,
                        den_mat=self.den_mat, tsr_mode=True)
        self.add(uany)

    def hlayer(self, wires=None):
        hl = HLayer(nqubit=self.nqubit, wires=wires, den_mat=self.den_mat, tsr_mode=True)
        self.add(hl)

    def rxlayer(self, wires=None, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        rxl = RxLayer(nqubit=self.nqubit, wires=wires, inputs=inputs, den_mat=self.den_mat,
                      tsr_mode=True, requires_grad=requires_grad)
        self.add(rxl, encode=encode)
    
    def rylayer(self, wires=None, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        ryl = RyLayer(nqubit=self.nqubit, wires=wires, inputs=inputs, den_mat=self.den_mat,
                      tsr_mode=True, requires_grad=requires_grad)
        self.add(ryl, encode=encode)

    def rzlayer(self, wires=None, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        rzl = RzLayer(nqubit=self.nqubit, wires=wires, inputs=inputs, den_mat=self.den_mat,
                      tsr_mode=True, requires_grad=requires_grad)
        self.add(rzl, encode=encode)

    def u3layer(self, wires=None, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        u3l = U3Layer(nqubit=self.nqubit, wires=wires, inputs=inputs, den_mat=self.den_mat,
                      tsr_mode=True, requires_grad=requires_grad)
        self.add(u3l, encode=encode)

    def cxlayer(self, wires=None):
        cxl = CnotLayer(nqubit=self.nqubit, wires=wires, den_mat=self.den_mat, tsr_mode=True)
        self.add(cxl)

    def cnot_ring(self, minmax=None, step=1, reverse=False):
        cxr = CnotRing(nqubit=self.nqubit, minmax=minmax, step=step, reverse=reverse,
                       den_mat=self.den_mat, tsr_mode=True)
        self.add(cxr)