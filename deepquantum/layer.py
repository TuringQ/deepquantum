from deepquantum.operation import Layer
from deepquantum.gate import *
from deepquantum.qmath import multi_kron


class SingleLayer(Layer):
    def __init__(self, name=None, nqubit=1, wires=None, den_mat=False, tsr_mode=False):
        if wires == None:
            wires = [[i] for i in range(nqubit)]
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        for wire in self.wires:
            assert len(wire) == 1

    def get_unitary(self):
        assert len(self.gates) > 0, 'There is no quantum gate'
        identity = torch.eye(2, dtype=torch.cfloat, device=self.gates[0].matrix.device)
        lst = [identity] * self.nqubit
        for gate in self.gates:
            lst[gate.wires[0]] = gate.update_matrix()
        return multi_kron(lst)


class DoubleLayer(Layer):
    def __init__(self, name=None, nqubit=2, wires=None, den_mat=False, tsr_mode=False):
        if wires == None:
            wires = [[i, i + 1] for i in range(0, nqubit - 1, 2)]
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        for wire in self.wires:
            assert len(wire) == 2


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


class U3Layer(SingleLayer):
    def __init__(self, nqubit=1, wires=None, inputs=None, den_mat=False, tsr_mode=False, requires_grad=True):
        super().__init__(name='U3Layer', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        for i, wire in enumerate(self.wires):
            if inputs == None:
                thetas = None
            else:
                thetas = inputs[3*i:3*i+3]
            u3 = U3Gate(inputs=thetas, nqubit=nqubit, wires=wire, den_mat=den_mat,
                        tsr_mode=True, requires_grad=requires_grad)
            self.gates.append(u3)
            self.npara += u3.npara


class HLayer(SingleLayer):
    def __init__(self, nqubit=1, wires=None, den_mat=False, tsr_mode=False):
        super().__init__(name='HLayer', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        for wire in self.wires:
            h = Hadamard(nqubit=nqubit, wires=wire, den_mat=den_mat, tsr_mode=True)
            self.gates.append(h)


class RxLayer(SingleLayer):
    def __init__(self, nqubit=1, wires=None, inputs=None, den_mat=False, tsr_mode=False, requires_grad=True):
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
    def __init__(self, nqubit=1, wires=None, inputs=None, den_mat=False, tsr_mode=False, requires_grad=True):
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
    def __init__(self, nqubit=1, wires=None, inputs=None, den_mat=False, tsr_mode=False, requires_grad=True):
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
    def __init__(self, nqubit=2, wires=None, name='CnotLayer', den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        for wire in self.wires:
            cnot = CNOT(nqubit=nqubit, wires=wire, den_mat=den_mat, tsr_mode=True)
            self.gates.append(cnot)


class CnotRing(CnotLayer):
    def __init__(self, nqubit=2, minmax=None, step=1, reverse=False, den_mat=False, tsr_mode=False):
        if minmax == None:
            minmax = [0, nqubit-1]
        assert type(minmax) == list
        assert len(minmax) == 2
        assert all(isinstance(i, int) for i in minmax)
        assert minmax[0] > -1 and minmax[0] < minmax[1] and minmax[1] < nqubit
        self.minmax = minmax
        self.step = step
        self.reverse = reverse
        nwires = minmax[1] - minmax[0] + 1
        if reverse: # from minmax[1] to minmax[0]
            wires = [[minmax[0] + i, minmax[0] + (i-step) % nwires] for i in range(minmax[1] - minmax[0], -1, -1)]
        else:
            wires = [[minmax[0] + i, minmax[0] + (i+step) % nwires] for i in range(minmax[1] - minmax[0] + 1)]
        super().__init__(nqubit=nqubit, wires=wires, name='CnotRing', den_mat=den_mat, tsr_mode=tsr_mode)