import numpy as np
import torch
import torch.nn as nn
from deepquantum.state import QubitState, MatrixProductState
from deepquantum.operation import Operation
from deepquantum.gate import *
from deepquantum.layer import *
from torch import vmap
from deepquantum.qmath import amplitude_encoding, measure, expectation
from qiskit import QuantumCircuit


class QubitCircuit(Operation):
    def __init__(self, nqubit, init_state='zeros', name=None, den_mat=False, reupload=False,
                 mps=False, chi=None):
        super().__init__(name=name, nqubit=nqubit, wires=None, den_mat=den_mat)
        if type(init_state) in (QubitState, MatrixProductState):
            assert nqubit == init_state.nqubit
            if type(init_state) == MatrixProductState:
                assert den_mat == False, 'Currently, DO NOT support MPS for density matrix'
            else:
                assert den_mat == init_state.den_mat
            self.init_state = init_state
        else:
            if mps:
                self.init_state = MatrixProductState(nqubit=nqubit, state=init_state, chi=chi)
            else:
                self.init_state = QubitState(nqubit=nqubit, state=init_state, den_mat=den_mat)
        self.operators = nn.Sequential()
        self.encoders = []
        self.observables = nn.ModuleList()
        self.state = None
        self.npara = 0
        self.ndata = 0
        self.depth = np.array([0] * nqubit)
        self.reupload = reupload
        self.mps = mps
        self.chi = chi
        self.wires_measure = None

    def __add__(self, rhs):
        assert self.nqubit == rhs.nqubit
        cir = QubitCircuit(nqubit=self.nqubit, name=self.name, den_mat=self.den_mat, reupload=self.reupload,
                           mps=self.mps, chi=self.chi)
        cir.init_state = self.init_state
        cir.operators = self.operators + rhs.operators
        cir.encoders = self.encoders + rhs.encoders
        cir.observables = rhs.observables
        cir.npara = self.npara + rhs.npara
        cir.ndata = self.ndata + rhs.ndata
        cir.depth = self.depth + rhs.depth
        cir.wires_measure = rhs.wires_measure
        return cir

    def to(self, arg):
        if arg == torch.float:
            self.init_state.to(torch.cfloat)
            for op in self.operators:
                if op.npara == 0:
                    op.to(torch.cfloat)
                elif op.npara > 0:
                    op.to(torch.float)
            for ob in self.observables:
                if ob.npara == 0:
                    ob.to(torch.cfloat)
                elif ob.npara > 0:
                    ob.to(torch.float)
        elif arg == torch.double:
            self.init_state.to(torch.cdouble)
            for op in self.operators:
                if op.npara == 0:
                    op.to(torch.cdouble)
                elif op.npara > 0:
                    op.to(torch.double)
            for ob in self.observables:
                if ob.npara == 0:
                    ob.to(torch.cdouble)
                elif ob.npara > 0:
                    ob.to(torch.double)
        else:
            self.init_state.to(arg)
            self.operators.to(arg)
            self.observables.to(arg)

    def forward(self, data=None, state=None):
        if state == None:
            state = self.init_state
        if type(state) == MatrixProductState:
            state = state.tensors
        elif type(state) == QubitState:
            state = state.state
        if data == None:
            self.state = self.forward_helper(state=state)
            if not self.mps:
                if self.state.ndim == 2:
                    self.state = self.state.unsqueeze(0)
                if state.ndim == 2:
                    self.state = self.state.squeeze(0)
        else:
            if data.ndim == 1:
                data = data.unsqueeze(0)
            assert data.ndim == 2
            if self.mps:
                assert state[0].ndim in (3, 4)
                if state[0].ndim == 3:
                    self.state = vmap(self.forward_helper, in_dims=(0, None))(data, state)
                elif state[0].ndim == 4:
                    self.state = vmap(self.forward_helper)(data, state)
            else:
                assert state.ndim in (2, 3)
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
        if self.mps:
            if type(state) != MatrixProductState:
                state = MatrixProductState(nqubit=self.nqubit, state=state, chi=self.chi,
                                           normalize=self.init_state.normalize)
            return self.operators(state).tensors
        if type(state) == QubitState:
            state = state.state
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
        if type(init_state) in (QubitState, MatrixProductState):
            assert self.nqubit == init_state.nqubit
            if type(init_state) == MatrixProductState:
                assert self.den_mat == False, 'Currently, DO NOT support MPS for density matrix'
                self.mps = True
                self.chi = init_state.chi
            else:
                assert self.den_mat == init_state.den_mat
            self.init_state = init_state
        else:
            if self.mps:
                self.init_state = MatrixProductState(nqubit=self.nqubit, state=init_state, chi=self.chi)
            else:
                self.init_state = QubitState(nqubit=self.nqubit, state=init_state, den_mat=self.den_mat)
        self.operators = nn.Sequential()
        self.encoders = []
        self.observables = nn.ModuleList()
        self.state = None
        self.npara = 0
        self.ndata = 0
        self.depth = np.array([0] * self.nqubit)
        self.wires_measure = None

    def amplitude_encoding(self, data):
        return amplitude_encoding(data, self.nqubit)
    
    def observable(self, wires=None, basis='z'):
        observable = Observable(nqubit=self.nqubit, wires=wires, basis=basis,
                                den_mat=self.den_mat, tsr_mode=False)
        self.observables.append(observable)

    def reset_observable(self):
        self.observables = nn.ModuleList()

    def measure(self, shots=1024, with_prob=False, wires=None):
        if wires == None:
            self.wires_measure = list(range(self.nqubit))
        else:
            assert type(wires) in (int, list)
            if type(wires) == int:
                wires = [wires]
            self.wires_measure = wires
        if self.state == None:
            return
        else:
            return measure(self.state, shots=shots, with_prob=with_prob, wires=wires)

    def expectation(self):
        assert len(self.observables) > 0, 'There is no observable'
        if type(self.state) == list:
            assert all(isinstance(i, torch.Tensor) for i in self.state), 'Invalid final state'
            assert len(self.state) == self.nqubit, 'Invalid final state'
        else:
            assert type(self.state) == torch.Tensor, 'There is no final state'
        out = []
        for observable in self.observables:
            expval = expectation(self.state, observable=observable, den_mat=self.den_mat, chi=self.chi)
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
    
    def inverse(self):
        # ATTENTION: Only the circuit structure is guaranteed.
        # You must encode data manually.
        cir = QubitCircuit(nqubit=self.nqubit, name=self.name, den_mat=self.den_mat, reupload=self.reupload,
                           mps=self.mps, chi=self.chi)
        for op in reversed(self.operators):
            op_inv = op.inverse()
            cir.operators.append(op_inv)
            if op in self.encoders:
                cir.encoders.append(op_inv)
        cir.depth = self.depth
        cir.npara = self.npara
        cir.ndata = self.ndata
        return cir
    
    @property
    def max_depth(self):
        return max(self.depth)
    
    def qasm(self):
        qasm_str = 'OPENQASM 2.0;\n' + 'include "qelib1.inc";\n'
        if self.wires_measure == None:
            qasm_str += f'qreg q[{self.nqubit}];\n'
        else:
            qasm_str += f'qreg q[{self.nqubit}];\n' + f'creg c[{self.nqubit}];\n'
        for op in self.operators:
            qasm_str += op.qasm()
        if self.wires_measure != None:
            for wire in self.wires_measure:
                qasm_str += f'measure q[{wire}] -> c[{wire}];\n'
        Gate.qasm_new_gate = []
        return qasm_str
        
    def draw(self, output='mpl', **kwargs):
        qc = QuantumCircuit.from_qasm_str(self.qasm())
        return qc.draw(output=output, **kwargs)

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
            self.wires_measure = op.wires_measure
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

    def u3(self, wires, inputs=None, controls=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        u3 = U3Gate(inputs=inputs, nqubit=self.nqubit, wires=wires, controls=controls,
                    den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(u3, encode=encode)

    def cu(self, control, target, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        cu = U3Gate(inputs=inputs, nqubit=self.nqubit, wires=[target], controls=[control],
                    den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(cu, encode=encode)

    def p(self, wires, inputs=None, controls=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        p = PhaseShift(inputs=inputs, nqubit=self.nqubit, wires=wires, controls=controls,
                       den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(p, encode=encode)

    def cp(self, control, target, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        cp = PhaseShift(inputs=inputs, nqubit=self.nqubit, wires=[target], controls=[control],
                        den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(cp, encode=encode)

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

    def ch(self, control, target):
        ch = Hadamard(nqubit=self.nqubit, wires=[target], controls=[control],
                      den_mat=self.den_mat, tsr_mode=True)
        self.add(ch)

    def cs(self, control, target):
        cs = SGate(nqubit=self.nqubit, wires=[target], controls=[control],
                   den_mat=self.den_mat, tsr_mode=True)
        self.add(cs)

    def csdg(self, control, target):
        csdg = SDaggerGate(nqubit=self.nqubit, wires=[target], controls=[control],
                           den_mat=self.den_mat, tsr_mode=True)
        self.add(csdg)

    def ct(self, control, target):
        ct = TGate(nqubit=self.nqubit, wires=[target], controls=[control],
                   den_mat=self.den_mat, tsr_mode=True)
        self.add(ct)

    def ctdg(self, control, target):
        ctdg = TDaggerGate(nqubit=self.nqubit, wires=[target], controls=[control],
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

    def crx(self, control, target, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        crx = Rx(inputs=inputs, nqubit=self.nqubit, wires=[target], controls=[control],
                 den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(crx, encode=encode)

    def cry(self, control, target, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        cry = Ry(inputs=inputs, nqubit=self.nqubit, wires=[target], controls=[control],
                 den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(cry, encode=encode)

    def crz(self, control, target, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        crz = Rz(inputs=inputs, nqubit=self.nqubit, wires=[target], controls=[control],
                 den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(crz, encode=encode)

    def cnot(self, control, target):
        cnot = CNOT(nqubit=self.nqubit, wires=[control, target], den_mat=self.den_mat, tsr_mode=True)
        self.add(cnot)

    def cx(self, control, target):
        cx = PauliX(nqubit=self.nqubit, wires=[target], controls=[control],
                    den_mat=self.den_mat, tsr_mode=True)
        self.add(cx)

    def cy(self, control, target):
        cy = PauliY(nqubit=self.nqubit, wires=[target], controls=[control],
                    den_mat=self.den_mat, tsr_mode=True)
        self.add(cy)

    def cz(self, control, target):
        cz = PauliZ(nqubit=self.nqubit, wires=[target], controls=[control],
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

    def crxx(self, control, target1, target2, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        crxx = Rxx(inputs=inputs, nqubit=self.nqubit, wires=[target1, target2], controls=[control],
                   den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(crxx, encode=encode)

    def cryy(self, control, target1, target2, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        cryy = Ryy(inputs=inputs, nqubit=self.nqubit, wires=[target1, target2], controls=[control],
                   den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(cryy, encode=encode)

    def crzz(self, control, target1, target2, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        crzz = Rzz(inputs=inputs, nqubit=self.nqubit, wires=[target1, target2], controls=[control],
                   den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(crzz, encode=encode)

    def crxy(self, control, target1, target2, inputs=None, encode=False):
        requires_grad = not encode
        if inputs != None:
            requires_grad = False
        crxy = Rxy(inputs=inputs, nqubit=self.nqubit, wires=[target1, target2], controls=[control],
                   den_mat=self.den_mat, tsr_mode=True, requires_grad=requires_grad)
        self.add(crxy, encode=encode)

    def toffoli(self, control1, control2, target):
        toffoli = Toffoli(nqubit=self.nqubit, wires=[control1, control2, target],
                          den_mat=self.den_mat, tsr_mode=True)
        self.add(toffoli)

    def ccx(self, control1, control2, target):
        ccx = PauliX(nqubit=self.nqubit, wires=[target], controls=[control1, control2],
                     den_mat=self.den_mat, tsr_mode=True)
        self.add(ccx)

    def fredkin(self, control, target1, target2):
        fredkin = Fredkin(nqubit=self.nqubit, wires=[control, target1, target2],
                          den_mat=self.den_mat, tsr_mode=True)
        self.add(fredkin)

    def cswap(self, control, target1, target2):
        cswap = Swap(nqubit=self.nqubit, wires=[target1, target2], controls=[control],
                     den_mat=self.den_mat, tsr_mode=True)
        self.add(cswap)

    def any(self, unitary, minmax=None, name='uany'):
        uany = UAnyGate(unitary=unitary, nqubit=self.nqubit, minmax=minmax, name=name,
                        den_mat=self.den_mat, tsr_mode=True)
        self.add(uany)

    def xlayer(self, wires=None):
        xl = XLayer(nqubit=self.nqubit, wires=wires, den_mat=self.den_mat, tsr_mode=True)
        self.add(xl)

    def ylayer(self, wires=None):
        yl = YLayer(nqubit=self.nqubit, wires=wires, den_mat=self.den_mat, tsr_mode=True)
        self.add(yl)

    def zlayer(self, wires=None):
        zl = ZLayer(nqubit=self.nqubit, wires=wires, den_mat=self.den_mat, tsr_mode=True)
        self.add(zl)

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

    def barrier(self, wires=None):
        if wires == None:
            wires = list(range(self.nqubit))
        br = Barrier(nqubit=self.nqubit, wires=wires)
        self.add(br)