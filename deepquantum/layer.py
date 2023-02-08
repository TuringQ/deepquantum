from operation import Layer
from gate import *


class SingleLayer(Layer):
    def __init__(self, name=None, nqubit=1, wires=None, den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        if wires == None:
            self.wires = list(range(nqubit))
        elif type(wires) == int:
            assert wires < nqubit
            self.wires = [wires]
        else:
            self.wires = wires


class DoubleLayer(Layer):
    def __init__(self, name=None, nqubit=2, wires=None, den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        if wires != None:
            for wire in wires:
                assert len(wire) == 2
                assert type(wire[0]) == int and type(wire[1]) == int
                assert wire[0] != wire[1]
        self.wires = wires


class Measurement(SingleLayer):
    def __init__(self, nqubit=1, wires=None, observables='z', den_mat=False, tsr_mode=False):
        super().__init__(name='Measurement', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        if len(observables) == 1:
            self.observables = observables * len(self.wires)
        assert len(self.wires) == len(self.observables), 'The number of wires is not equal to the number of observables'
        for i, wire in enumerate(self.wires):
            if self.observables[i] == 'x':
                gate = PauliX(nqubit=nqubit, wires=wire, den_mat=den_mat, tsr_mode=True)
            elif self.observables[i] == 'y':
                gate = PauliY(nqubit=nqubit, wires=wire, den_mat=den_mat, tsr_mode=True)
            elif self.observables[i] == 'z':
                gate = PauliZ(nqubit=nqubit, wires=wire, den_mat=den_mat, tsr_mode=True)
            else:
                raise ValueError('Use illegal observables')
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
        for wire in self.wires:
            cnot = CNOT(nqubit=nqubit, wires=wire, den_mat=den_mat, tsr_mode=True)
            self.gates.append(cnot)