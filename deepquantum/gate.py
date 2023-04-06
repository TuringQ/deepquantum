import torch
import torch.nn as nn
from deepquantum.operation import Gate
from deepquantum.qmath import multi_kron, is_unitary


class SingleGate(Gate):
    def __init__(self, name=None, nqubit=1, wires=[0], controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        assert len(self.wires) == 1

    def get_unitary(self):
        matrix = self.update_matrix()
        identity = torch.eye(2, dtype=torch.cfloat, device=matrix.device)
        if self.controls == []:
            lst = [identity] * self.nqubit
            lst[self.wires[0]] = matrix
            return multi_kron(lst)
        else:
            oneone = torch.tensor([[0, 0], [0, 1]], dtype=torch.cfloat, device=matrix.device)
            lst1 = [identity] * self.nqubit
            lst2 = [identity] * self.nqubit
            lst3 = [identity] * self.nqubit
            for i in self.controls:
                lst2[i] = oneone
                lst3[i] = oneone
            lst3[self.wires[0]] = matrix
            return multi_kron(lst1) - multi_kron(lst2) + multi_kron(lst3)


class DoubleGate(Gate):
    def __init__(self, name=None, nqubit=2, wires=[0,1], controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        assert len(self.wires) == 2
    
    def get_unitary(self):
        matrix = self.update_matrix()
        identity = torch.eye(2, dtype=torch.cfloat, device=matrix.device)
        zerozero = torch.tensor([[1, 0], [0, 0]], dtype=torch.cfloat, device=matrix.device)
        zeroone  = torch.tensor([[0, 1], [0, 0]], dtype=torch.cfloat, device=matrix.device)
        onezero  = torch.tensor([[0, 0], [1, 0]], dtype=torch.cfloat, device=matrix.device)
        oneone   = torch.tensor([[0, 0], [0, 1]], dtype=torch.cfloat, device=matrix.device)
        if self.controls == []:
            lst1 = [identity] * self.nqubit
            lst2 = [identity] * self.nqubit
            lst3 = [identity] * self.nqubit
            lst4 = [identity] * self.nqubit

            lst1[self.wires[0]] = zerozero
            lst1[self.wires[1]] = matrix[0:2, 0:2]

            lst2[self.wires[0]] = zeroone
            lst2[self.wires[1]] = matrix[0:2, 2:4]

            lst3[self.wires[0]] = onezero
            lst3[self.wires[1]] = matrix[2:4, 0:2]

            lst4[self.wires[0]] = oneone
            lst4[self.wires[1]] = matrix[2:4, 2:4]
            return multi_kron(lst1) + multi_kron(lst2) + multi_kron(lst3) + multi_kron(lst4)
        else:
            lst1 = [identity] * self.nqubit
            lst2 = [identity] * self.nqubit
            lst3 = [identity] * self.nqubit
            lst4 = [identity] * self.nqubit
            lst5 = [identity] * self.nqubit
            lst6 = [identity] * self.nqubit
            for i in self.controls:
                lst2[i] = oneone

                lst3[i] = oneone
                lst4[i] = oneone
                lst5[i] = oneone
                lst6[i] = oneone

            lst3[self.wires[0]] = zerozero
            lst3[self.wires[1]] = matrix[0:2, 0:2]

            lst4[self.wires[0]] = zeroone
            lst4[self.wires[1]] = matrix[0:2, 2:4]

            lst5[self.wires[0]] = onezero
            lst5[self.wires[1]] = matrix[2:4, 0:2]

            lst6[self.wires[0]] = oneone
            lst6[self.wires[1]] = matrix[2:4, 2:4]
            return multi_kron(lst1) - multi_kron(lst2) + \
                   multi_kron(lst3) + multi_kron(lst4) + multi_kron(lst5) + multi_kron(lst6)


class DoubleControlGate(DoubleGate):
    def __init__(self, name=None, nqubit=2, wires=[0,1], den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=None,
                         den_mat=den_mat, tsr_mode=tsr_mode)

    def get_unitary(self):
        matrix = self.update_matrix()
        identity = torch.eye(2, dtype=torch.cfloat, device=matrix.device)
        zerozero = torch.tensor([[1, 0], [0, 0]], dtype=torch.cfloat, device=matrix.device)
        oneone   = torch.tensor([[0, 0], [0, 1]], dtype=torch.cfloat, device=matrix.device)
        lst1 = [identity] * self.nqubit
        lst2 = [identity] * self.nqubit
        
        lst1[self.wires[0]] = zerozero
        
        lst2[self.wires[0]] = oneone
        lst2[self.wires[1]] = matrix[2:4, 2:4]
        return multi_kron(lst1) + multi_kron(lst2)


class TripleGate(Gate):
    def __init__(self, name=None, nqubit=3, wires=[0,1,2], controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        assert len(self.wires) == 3


class ParametricSingleGate(SingleGate):
    def __init__(self, name=None, inputs=None, nqubit=1, wires=[0], controls=None,
                 den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.npara = 1
        self.requires_grad = requires_grad
        self.init_para(inputs=inputs)

    def inputs_to_tensor(self, inputs=None):
        while type(inputs) == list:
            inputs = inputs[0]
        if inputs == None:
            inputs = torch.rand(1)[0] * torch.pi
        elif type(inputs) != torch.Tensor and type(inputs) != torch.nn.parameter.Parameter:
            inputs = torch.tensor(inputs, dtype=torch.float)
        return inputs

    def update_matrix(self):
        matrix = self.get_matrix(self.theta)
        self.matrix = matrix.detach()
        return matrix
        
    def init_para(self, inputs=None):
        theta = self.inputs_to_tensor(inputs=inputs)
        if self.requires_grad:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('theta', theta)
        self.update_matrix()


class U3Gate(ParametricSingleGate):
    def __init__(self, inputs=None, nqubit=1, wires=[0], controls=None,
                 den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name='U3Gate', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)
        self.npara = 3

    def inputs_to_tensor(self, inputs=None):
        if inputs == None:
            theta = torch.rand(1)[0] * torch.pi
            phi   = torch.rand(1)[0] * 2 * torch.pi
            lambd = torch.rand(1)[0] * 2 * torch.pi
        else:
            theta = inputs[0]
            phi   = inputs[1]
            lambd = inputs[2]
        if type(theta) != torch.Tensor and type(theta) != torch.nn.parameter.Parameter:
            theta = torch.tensor(theta, dtype=torch.float)
        if type(phi) != torch.Tensor and type(phi) != torch.nn.parameter.Parameter:
            phi = torch.tensor(phi, dtype=torch.float)
        if type(lambd) != torch.Tensor and type(lambd) != torch.nn.parameter.Parameter:
            lambd = torch.tensor(lambd, dtype=torch.float)
        return theta, phi, lambd

    def get_matrix(self, theta, phi, lambd):
        theta, phi, lambd = self.inputs_to_tensor([theta, phi, lambd])
        cos_t = torch.cos(theta / 2)
        sin_t = torch.sin(theta / 2)
        e_il  = torch.exp(1j * lambd)
        e_ip  = torch.exp(1j * phi)
        e_ipl = torch.exp(1j * (phi + lambd))
        return torch.stack([cos_t, -e_il * sin_t, e_ip * sin_t, e_ipl * cos_t]).reshape(2, 2)

    def update_matrix(self):
        matrix = self.get_matrix(self.theta, self.phi, self.lambd)
        self.matrix = matrix.detach()
        return matrix

    def init_para(self, inputs=None):
        theta, phi, lambd = self.inputs_to_tensor(inputs=inputs)
        if self.requires_grad:
            self.theta = nn.Parameter(theta)
            self.phi   = nn.Parameter(phi)
            self.lambd = nn.Parameter(lambd)
        else:
            self.register_buffer('theta', theta)
            self.register_buffer('phi', phi)
            self.register_buffer('lambd', lambd)
        self.update_matrix()


class PhaseShift(ParametricSingleGate):
    def __init__(self, inputs=None, nqubit=1, wires=[0], controls=None,
                 den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name='PhaseShift', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)
    
    def inputs_to_tensor(self, inputs=None):
        while type(inputs) == list:
            inputs = inputs[0]
        if inputs == None:
            inputs = torch.rand(1)[0] * 2 * torch.pi
        elif type(inputs) != torch.Tensor and type(inputs) != torch.nn.parameter.Parameter:
            inputs = torch.tensor(inputs, dtype=torch.float)
        return inputs
    
    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        e_it = torch.exp(1j * theta)
        return torch.stack([1, e_it]).reshape(-1).diag_embed()


class Identity(Gate):
    def __init__(self, nqubit=1, wires=[0], den_mat=False, tsr_mode=False):
        super().__init__(name='Identity', nqubit=nqubit, wires=wires, controls=None,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.matrix = torch.eye(2 ** self.nqubit, dtype=torch.cfloat)
        
    def get_unitary(self):
        return self.matrix
        
    def forward(self, x):
        return x


class PauliX(SingleGate):
    def __init__(self, nqubit=1, wires=[0], controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name='PauliX', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat))


class PauliY(SingleGate):
    def __init__(self, nqubit=1, wires=[0], controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name='PauliY', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[0, -1j], [1j, 0]]))


class PauliZ(SingleGate):
    def __init__(self, nqubit=1, wires=[0], controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name='PauliZ', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat))


class Hadamard(SingleGate):
    def __init__(self, nqubit=1, wires=[0], controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name='Hadamard', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat) / 2 ** 0.5)


class SGate(SingleGate):
    def __init__(self, nqubit=1, wires=[0], controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name='SGate', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0], [0, 1j]]))


class SDaggerGate(SingleGate):
    def __init__(self, nqubit=1, wires=[0], controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name='SDaggerGate', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0], [0, -1j]]))


class TGate(SingleGate):
    def __init__(self, nqubit=1, wires=[0], controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name='TGate', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0], [0, (1 + 1j) / 2 ** 0.5]]))


class TDaggerGate(SingleGate):
    def __init__(self, nqubit=1, wires=[0], controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name='TDaggerGate', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0], [0, (1 - 1j) / 2 ** 0.5]]))


class Rx(ParametricSingleGate):
    def __init__(self, inputs=None, nqubit=1, wires=[0], controls=None,
                 den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name='Rx', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)
    
    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        cos = torch.cos(theta / 2)
        sin = torch.sin(theta / 2)
        return torch.stack([cos, -1j * sin, -1j * sin, cos]).reshape(2, 2)


class Ry(ParametricSingleGate):
    def __init__(self, inputs=None, nqubit=1, wires=0, controls=None,
                 den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name='Ry', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)

    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        cos = torch.cos(theta / 2)
        sin = torch.sin(theta / 2)
        return torch.stack([cos, -sin, sin, cos]).reshape(2, 2) + 0j


class Rz(ParametricSingleGate):
    def __init__(self, inputs=None, nqubit=1, wires=[0], controls=None,
                 den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name='Rz', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)
    
    def inputs_to_tensor(self, inputs=None):
        while type(inputs) == list:
            inputs = inputs[0]
        if inputs == None:
            inputs = torch.rand(1)[0] * 2 * torch.pi
        elif type(inputs) != torch.Tensor and type(inputs) != torch.nn.parameter.Parameter:
            inputs = torch.tensor(inputs, dtype=torch.float)
        return inputs

    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        e_m_it = torch.exp(-1j * theta / 2)
        e_it = torch.exp(1j * theta / 2)
        return torch.stack([e_m_it, e_it]).reshape(-1).diag_embed()


class CombinedSingleGate(SingleGate):
    def __init__(self, gatelist, name=None, nqubit=1, wires=[0], controls=None,
                 den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.gatelist = nn.ModuleList(gatelist)
        self.update_npara()
        self.update_matrix()

    def get_matrix(self):
        matrix = None
        for gate in self.gatelist:
            if matrix == None:
                matrix = gate.update_matrix()
            else:
                matrix = gate.update_matrix() @ matrix
        return matrix

    def update_matrix(self):
        matrix = self.get_matrix()
        self.matrix = matrix.detach()
        return matrix

    def update_npara(self):
        self.npara = 0
        for gate in self.gatelist:
            self.npara += gate.npara
        
    def add(self, gate: SingleGate):
        self.gatelist.append(gate)
        self.matrix = gate.matrix @ self.matrix
        self.npara += gate.npara


class CNOT(DoubleControlGate):
    def __init__(self, nqubit=2, wires=[0,1], den_mat=False, tsr_mode=False):
        super().__init__(name='CNOT', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0, 0, 0],
                                                     [0, 1, 0, 0],
                                                     [0, 0, 0, 1],
                                                     [0, 0, 1, 0]]) + 0j)


class Swap(DoubleGate):
    def __init__(self, nqubit=2, wires=[0,1], controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name='Swap', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0, 0, 0],
                                                     [0, 0, 1, 0],
                                                     [0, 1, 0, 0],
                                                     [0, 0, 0, 1]]) + 0j)


class Toffoli(TripleGate):
    def __init__(self, nqubit=3, wires=[0,1,2], den_mat=False, tsr_mode=False):
        super().__init__(name='Toffoli', nqubit=nqubit, wires=wires, controls=None,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0],
                                                     [0, 1, 0, 0, 0, 0, 0, 0],
                                                     [0, 0, 1, 0, 0, 0, 0, 0],
                                                     [0, 0, 0, 1, 0, 0, 0, 0],
                                                     [0, 0, 0, 0, 1, 0, 0, 0],
                                                     [0, 0, 0, 0, 0, 1, 0, 0],
                                                     [0, 0, 0, 0, 0, 0, 0, 1],
                                                     [0, 0, 0, 0, 0, 0, 1, 0]]) + 0j)

    def get_unitary(self):
        matrix = self.update_matrix()
        identity = torch.eye(2, dtype=torch.cfloat, device=matrix.device)
        oneone = torch.tensor([[0, 0], [0, 1]], dtype=torch.cfloat, device=matrix.device)
        lst1 = [identity] * self.nqubit
        lst2 = [identity] * self.nqubit
        lst3 = [identity] * self.nqubit

        lst2[self.wires[0]] = oneone
        lst2[self.wires[1]] = oneone

        lst3[self.wires[0]] = oneone
        lst3[self.wires[1]] = oneone
        lst3[self.wires[2]] = matrix[-2:, -2:]
        return multi_kron(lst1) - multi_kron(lst2) + multi_kron(lst3)


class Fredkin(TripleGate):
    def __init__(self, nqubit=3, wires=[0,1,2], den_mat=False, tsr_mode=False):
        super().__init__(name='Fredkin', nqubit=nqubit, wires=wires, controls=None,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0],
                                                     [0, 1, 0, 0, 0, 0, 0, 0],
                                                     [0, 0, 1, 0, 0, 0, 0, 0],
                                                     [0, 0, 0, 1, 0, 0, 0, 0],
                                                     [0, 0, 0, 0, 1, 0, 0, 0],
                                                     [0, 0, 0, 0, 0, 0, 1, 0],
                                                     [0, 0, 0, 0, 0, 1, 0, 0],
                                                     [0, 0, 0, 0, 0, 0, 0, 1]]) + 0j)

    def get_unitary(self):
        matrix = self.update_matrix()
        identity = torch.eye(2, dtype=torch.cfloat, device=matrix.device)
        zerozero = torch.tensor([[1, 0], [0, 0]], dtype=torch.cfloat, device=matrix.device)
        zeroone  = torch.tensor([[0, 1], [0, 0]], dtype=torch.cfloat, device=matrix.device)
        onezero  = torch.tensor([[0, 0], [1, 0]], dtype=torch.cfloat, device=matrix.device)
        oneone   = torch.tensor([[0, 0], [0, 1]], dtype=torch.cfloat, device=matrix.device)
        lst1 = [identity] * self.nqubit
        lst2 = [identity] * self.nqubit
        lst3 = [identity] * self.nqubit
        lst4 = [identity] * self.nqubit
        lst5 = [identity] * self.nqubit
        
        lst1[self.wires[0]] = zerozero

        lst2[self.wires[0]] = oneone
        lst2[self.wires[1]] = zerozero
        lst2[self.wires[2]] = matrix[-4:-2, -4:-2]

        lst3[self.wires[0]] = oneone
        lst3[self.wires[1]] = zeroone
        lst3[self.wires[2]] = matrix[-4:-2, -2:]

        lst4[self.wires[0]] = oneone
        lst4[self.wires[1]] = onezero
        lst4[self.wires[2]] = matrix[-2:, -4:-2]

        lst5[self.wires[0]] = oneone
        lst5[self.wires[1]] = oneone
        lst5[self.wires[2]] = matrix[-2:, -2:]
        return multi_kron(lst1) + multi_kron(lst2) + multi_kron(lst3) + multi_kron(lst4) + multi_kron(lst5)


class UAnyGate(Gate):
    def __init__(self, unitary, nqubit=1, minmax=None, den_mat=False, tsr_mode=False):
        if minmax == None:
            minmax = [0, nqubit-1]
        assert type(minmax) == list
        assert len(minmax) == 2
        assert all(isinstance(i, int) for i in minmax)
        assert minmax[0] > -1 and minmax[0] <= minmax[1] and minmax[1] < nqubit
        self.minmax = minmax
        wires = [i for i in range(minmax[0], minmax[1] + 1)]
        if type(unitary) != torch.Tensor:
            unitary = torch.tensor(unitary, dtype=torch.cfloat).reshape(-1, 2 ** len(wires))
        assert unitary.dtype == torch.cfloat
        assert unitary.shape[-1] == 2 ** len(wires) and unitary.shape[-2] == 2 ** len(wires)
        assert is_unitary(unitary)
        super().__init__(name='UAnyGate', nqubit=nqubit, wires=wires, controls=None,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', unitary)

    def get_unitary(self):
        identity = torch.eye(2, dtype=torch.cfloat, device=self.matrix.device)
        lst = [identity] * (self.nqubit - len(self.wires) + 1)
        lst[self.wires[0]] = self.matrix
        return multi_kron(lst)