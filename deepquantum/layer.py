from deepquantum.operation import Layer
from deepquantum.gate import *


class SingleLayer(Layer):
    def __init__(self, name=None, nqubit=1, wires=None, den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        if wires == None:
            self.wires = list(range(nqubit))
        elif type(wires) == int:
            self.wires = [wires]
        else:
            self.wires = wires
        for wire in self.wires:
            assert wire < nqubit


class DoubleLayer(Layer):
    def __init__(self, name=None, nqubit=2, wires=None, den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        if wires != None:
            for wire in wires:
                assert len(wire) == 2
                assert type(wire[0]) == int and type(wire[1]) == int
                assert wire[0] < nqubit and wire[1] < nqubit
                assert wire[0] != wire[1]
        self.wires = wires


class Observable(SingleLayer):
    def __init__(self, nqubit=1, wires=None, basis='z', den_mat=False, tsr_mode=False):
        super().__init__(name='Observable', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        if len(basis) == 1:
            self.basis = basis * len(self.wires)
        assert len(self.wires) == len(self.basis), 'The number of wires is not equal to the number of bases'
        for i, wire in enumerate(self.wires):
            if self.basis[i] == 'x':
                gate = PauliX(nqubit=nqubit, wires=wire, den_mat=den_mat, tsr_mode=True)
            elif self.basis[i] == 'y':
                gate = PauliY(nqubit=nqubit, wires=wire, den_mat=den_mat, tsr_mode=True)
            elif self.basis[i] == 'z':
                gate = PauliZ(nqubit=nqubit, wires=wire, den_mat=den_mat, tsr_mode=True)
            else:
                raise ValueError('Use illegal measurement basis')
            self.gates.append(gate)


class HLayer(SingleLayer):
    def __init__(self, nqubit=1, wires=None, den_mat=False, tsr_mode=False):
        super().__init__(name='HLayer', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        for wire in self.wires:
            h = Hadamard(nqubit=nqubit, wires=wire, den_mat=den_mat, tsr_mode=True)
            self.gates.append(h)


class RxLayer(SingleLayer):
    def __init__(self, inputs=None, nqubit=1, wires=None, den_mat=False, tsr_mode=False, requires_grad=True):
        super().__init__(name='RxLayer', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        for i, wire in enumerate(self.wires):
            if inputs == None:
                theta = None
            else:
                theta = inputs[i]
            rx = Rx(inputs=theta, nqubit=nqubit, wires=wire, den_mat=den_mat,
                    tsr_mode=True, requires_grad=requires_grad)
            self.gates.append(rx)
            self.npara += rx.npara


class RyLayer(SingleLayer):
    def __init__(self, inputs=None, nqubit=1, wires=None, den_mat=False, tsr_mode=False, requires_grad=True):
        super().__init__(name='RyLayer', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        for i, wire in enumerate(self.wires):
            if inputs == None:
                theta = None
            else:
                theta = inputs[i]
            ry = Ry(inputs=theta, nqubit=nqubit, wires=wire, den_mat=den_mat,
                    tsr_mode=True, requires_grad=requires_grad)
            self.gates.append(ry)
            self.npara += ry.npara


class RzLayer(SingleLayer):
    def __init__(self, inputs=None, nqubit=1, wires=None, den_mat=False, tsr_mode=False, requires_grad=True):
        super().__init__(name='RzLayer', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        for i, wire in enumerate(self.wires):
            if inputs == None:
                theta = None
            else:
                theta = inputs[i]
            rz = Rz(inputs=theta, nqubit=nqubit, wires=wire, den_mat=den_mat,
                    tsr_mode=True, requires_grad=requires_grad)
            self.gates.append(rz)
            self.npara += rz.npara


class CnotLayer(DoubleLayer):
    def __init__(self, name=None, nqubit=2, wires=[[0,1]], den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        for wire in self.wires:
            cnot = CNOT(nqubit=nqubit, wires=wire, den_mat=den_mat, tsr_mode=True)
            self.gates.append(cnot)


class CnotRing(CnotLayer):
    def __init__(self, nqubit=2, minmax=None, den_mat=False, tsr_mode=False, step=1, reverse=False):
        if minmax == None:
            minmax = [0, nqubit-1]
        self.minmax = minmax
        self.step = step
        self.reverse = reverse
        nwires = minmax[1] - minmax[0] + 1
        if reverse: # from minmax[1] to minmax[0]
            wires = [[minmax[0] + i, minmax[0] + (i-step) % nwires] for i in range(minmax[1] - minmax[0], -1, -1)]
        else:
            wires = [[minmax[0] + i, minmax[0] + (i+step) % nwires] for i in range(minmax[1] - minmax[0] + 1)]
        super().__init__(name='CnotRing', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)