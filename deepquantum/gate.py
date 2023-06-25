import torch
import torch.nn as nn
from .operation import Gate
from .qmath import multi_kron, is_unitary, SVD


svd = SVD.apply


class SingleGate(Gate):
    def __init__(self, name=None, nqubit=1, wires=[0], controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        assert len(self.wires) == 1

    def get_unitary(self):
        matrix = self.update_matrix()
        identity = torch.eye(2, dtype=matrix.dtype, device=matrix.device)
        if self.controls == []:
            lst = [identity] * self.nqubit
            lst[self.wires[0]] = matrix
            return multi_kron(lst)
        else:
            oneone = torch.tensor([[0, 0], [0, 1]], dtype=matrix.dtype, device=matrix.device)
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
        identity = torch.eye(2, dtype=matrix.dtype, device=matrix.device)
        zerozero = torch.tensor([[1, 0], [0, 0]], dtype=matrix.dtype, device=matrix.device)
        zeroone  = torch.tensor([[0, 1], [0, 0]], dtype=matrix.dtype, device=matrix.device)
        onezero  = torch.tensor([[0, 0], [1, 0]], dtype=matrix.dtype, device=matrix.device)
        oneone   = torch.tensor([[0, 0], [0, 1]], dtype=matrix.dtype, device=matrix.device)
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
        identity = torch.eye(2, dtype=matrix.dtype, device=matrix.device)
        zerozero = torch.tensor([[1, 0], [0, 0]], dtype=matrix.dtype, device=matrix.device)
        oneone   = torch.tensor([[0, 0], [0, 1]], dtype=matrix.dtype, device=matrix.device)
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


class ArbitraryGate(Gate):
    def __init__(self, name=None, nqubit=1, wires=None, minmax=None, den_mat=False, tsr_mode=False):
        if type(wires) == int:
            wires = [wires]
        if wires == None:
            if minmax == None:
                minmax = [0, nqubit - 1]
            assert type(minmax) == list
            assert len(minmax) == 2
            assert all(isinstance(i, int) for i in minmax)
            assert minmax[0] > -1 and minmax[0] <= minmax[1] and minmax[1] < nqubit
            wires = list(range(minmax[0], minmax[1] + 1))
        self.minmax = minmax
        super().__init__(name=name, nqubit=nqubit, wires=wires, controls=None,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        
    def get_unitary(self):
        if self.minmax != None:
            matrix = self.update_matrix()
            identity = torch.eye(2, dtype=matrix.dtype, device=matrix.device)
            lst = [identity] * (self.nqubit - len(self.wires) + 1)
            lst[self.wires[0]] = matrix
            return multi_kron(lst)
        else:
            matrix = self.update_matrix()
            identity = torch.eye(2 ** self.nqubit, dtype=matrix.dtype, device=matrix.device)
            identity = identity.reshape([2 ** self.nqubit] + [2] * self.nqubit)
            return self.op_state_base(identity, matrix).reshape(2 ** self.nqubit, 2 ** self.nqubit).T
        
    def qasm(self):
        return self.qasm_customized(self.name)


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
            inputs = torch.rand(1)[0] * 4 * torch.pi
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

    def extra_repr(self):
        s = f'wires={self.wires}, theta={self.theta.item()}'
        if self.controls == []:
            return s
        else:
            return s + f', controls={self.controls}'


class ParametricDoubleGate(DoubleGate):
    def __init__(self, name=None, inputs=None, nqubit=2, wires=[0,1], controls=None,
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
            inputs = torch.rand(1)[0] * 4 * torch.pi
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

    def extra_repr(self):
        s = f'wires={self.wires}, theta={self.theta.item()}'
        if self.controls == []:
            return s
        else:
            return s + f', controls={self.controls}'


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

    def inverse(self):
        return U3Gate(inputs=[-self.theta, -self.lambd, -self.phi], nqubit=self.nqubit,
                      wires=self.wires, controls=self.controls, den_mat=self.den_mat,
                      tsr_mode=self.tsr_mode, requires_grad=False)

    def extra_repr(self):
        s = f'wires={self.wires}, theta={self.theta.item()}, phi={self.phi.item()}, lambda={self.lambd.item()}'
        if self.controls == []:
            return s
        else:
            return s + f', controls={self.controls}'

    def qasm(self):
        if self.controls == []:
            return f'u({self.theta.item()},{self.phi.item()},{self.lambd.item()}) q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'cu({self.theta.item()},{self.phi.item()},{self.lambd.item()},0.0) q{self.controls},q{self.wires};\n'
        else:
            return self.qasm_customized('u')


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
        m1 = torch.eye(1, dtype=theta.dtype, device=theta.device)
        e_it = torch.exp(1j * theta)
        return torch.block_diag(m1, e_it)

    def inverse(self):
        return PhaseShift(inputs=-self.theta, nqubit=self.nqubit, wires=self.wires, controls=self.controls,
                          den_mat=self.den_mat, tsr_mode=self.tsr_mode, requires_grad=False)
    
    def qasm(self):
        if self.controls == []:
            return f'p({self.theta.item()}) q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'cp({self.theta.item()}) q{self.controls},q{self.wires};\n'
        else:
            return self.qasm_customized('p')


class Identity(Gate):
    def __init__(self, nqubit=1, wires=[0], den_mat=False, tsr_mode=False):
        super().__init__(name='Identity', nqubit=nqubit, wires=wires, controls=None,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.eye(2 ** self.nqubit, dtype=torch.cfloat))
        
    def get_unitary(self):
        return self.matrix
        
    def forward(self, x):
        return x


class PauliX(SingleGate):
    def __init__(self, nqubit=1, wires=[0], controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name='PauliX', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat))

    def qasm(self):
        if self.controls == []:
            return f'x q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'cx q{self.controls},q{self.wires};\n'
        elif len(self.controls) == 2:
            return f'ccx q[{self.controls[0]}],q[{self.controls[1]}],q{self.wires};\n'
        else:
            return self.qasm_customized('x')


class PauliY(SingleGate):
    def __init__(self, nqubit=1, wires=[0], controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name='PauliY', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[0, -1j], [1j, 0]]))

    def qasm(self):
        if self.controls == []:
            return f'y q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'cy q{self.controls},q{self.wires};\n'
        else:
            return self.qasm_customized('y')

class PauliZ(SingleGate):
    def __init__(self, nqubit=1, wires=[0], controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name='PauliZ', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat))

    def qasm(self):
        if self.controls == []:
            return f'z q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'cz q{self.controls},q{self.wires};\n'
        else:
            return self.qasm_customized('z')


class Hadamard(SingleGate):
    def __init__(self, nqubit=1, wires=[0], controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name='Hadamard', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat) / 2 ** 0.5)

    def qasm(self):
        if self.controls == []:
            return f'h q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'ch q{self.controls},q{self.wires};\n'
        else:
            return self.qasm_customized('h')


class SGate(SingleGate):
    def __init__(self, nqubit=1, wires=[0], controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name='SGate', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0], [0, 1j]]))

    def inverse(self):
        return SDaggerGate(nqubit=self.nqubit, wires=self.wires, controls=self.controls,
                           den_mat=self.den_mat, tsr_mode=self.tsr_mode)

    def qasm(self):
        if self.controls == []:
            return f's q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'cs q{self.controls},q{self.wires};\n'
        else:
            return self.qasm_customized('s')


class SDaggerGate(SingleGate):
    def __init__(self, nqubit=1, wires=[0], controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name='SDaggerGate', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0], [0, -1j]]))

    def inverse(self):
        return SGate(nqubit=self.nqubit, wires=self.wires, controls=self.controls,
                     den_mat=self.den_mat, tsr_mode=self.tsr_mode)

    def qasm(self):
        if self.controls == []:
            return f'sdg q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'csdg q{self.controls},q{self.wires};\n'
        else:
            return self.qasm_customized('sdg')


class TGate(SingleGate):
    def __init__(self, nqubit=1, wires=[0], controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name='TGate', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0], [0, (1 + 1j) / 2 ** 0.5]]))

    def inverse(self):
        return TDaggerGate(nqubit=self.nqubit, wires=self.wires, controls=self.controls,
                           den_mat=self.den_mat, tsr_mode=self.tsr_mode)

    def qasm(self):
        if self.controls == []:
            return f't q{self.wires};\n'
        else:
            return self.qasm_customized('t')


class TDaggerGate(SingleGate):
    def __init__(self, nqubit=1, wires=[0], controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name='TDaggerGate', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0], [0, (1 - 1j) / 2 ** 0.5]]))

    def inverse(self):
        return TGate(nqubit=self.nqubit, wires=self.wires, controls=self.controls,
                     den_mat=self.den_mat, tsr_mode=self.tsr_mode)

    def qasm(self):
        if self.controls == []:
            return f'tdg q{self.wires};\n'
        else:
            return self.qasm_customized('tdg')


class Rx(ParametricSingleGate):
    def __init__(self, inputs=None, nqubit=1, wires=[0], controls=None,
                 den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name='Rx', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)
    
    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        cos  = torch.cos(theta / 2)
        isin = torch.sin(theta / 2) * 1j
        return torch.stack([cos, -isin, -isin, cos]).reshape(2, 2)
    
    def inverse(self):
        return Rx(inputs=-self.theta, nqubit=self.nqubit, wires=self.wires, controls=self.controls,
                  den_mat=self.den_mat, tsr_mode=self.tsr_mode, requires_grad=False)

    def qasm(self):
        if self.controls == []:
            return f'rx({self.theta.item()}) q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'crx({self.theta.item()}) q{self.controls},q{self.wires};\n'
        else:
            return self.qasm_customized('rx')


class Ry(ParametricSingleGate):
    def __init__(self, inputs=None, nqubit=1, wires=[0], controls=None,
                 den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name='Ry', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)

    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        cos = torch.cos(theta / 2)
        sin = torch.sin(theta / 2)
        return torch.stack([cos, -sin, sin, cos]).reshape(2, 2) + 0j
    
    def inverse(self):
        return Ry(inputs=-self.theta, nqubit=self.nqubit, wires=self.wires, controls=self.controls,
                  den_mat=self.den_mat, tsr_mode=self.tsr_mode, requires_grad=False)
    
    def qasm(self):
        if self.controls == []:
            return f'ry({self.theta.item()}) q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'cry({self.theta.item()}) q{self.controls},q{self.wires};\n'
        else:
            return self.qasm_customized('ry')


class Rz(ParametricSingleGate):
    def __init__(self, inputs=None, nqubit=1, wires=[0], controls=None,
                 den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name='Rz', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)

    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        e_m_it = torch.exp(-1j * theta / 2)
        e_it = torch.exp(1j * theta / 2)
        return torch.stack([e_m_it, e_it]).reshape(-1).diag_embed()
    
    def inverse(self):
        return Rz(inputs=-self.theta, nqubit=self.nqubit, wires=self.wires, controls=self.controls,
                  den_mat=self.den_mat, tsr_mode=self.tsr_mode, requires_grad=False)
    
    def qasm(self):
        if self.controls == []:
            return f'rz({self.theta.item()}) q{self.wires};\n'
        elif len(self.controls) == 1:
            return f'crz({self.theta.item()}) q{self.controls},q{self.wires};\n'
        else:
            return self.qasm_customized('rz')


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

    def inverse(self):
        gatelist = nn.ModuleList()
        for gate in reversed(self.gatelist):
            gatelist.append(gate.inverse())
        return CombinedSingleGate(gatelist=gatelist, name=self.name, nqubit=self.nqubit, wires=self.wires,
                                  controls=self.controls, den_mat=self.den_mat, tsr_mode=self.tsr_mode)

    def qasm(self):
        s = ''
        for gate in self.gatelist:
            s += gate.qasm()
        return s


class CNOT(DoubleControlGate):
    def __init__(self, nqubit=2, wires=[0,1], den_mat=False, tsr_mode=False):
        super().__init__(name='CNOT', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0, 0, 0],
                                                     [0, 1, 0, 0],
                                                     [0, 0, 0, 1],
                                                     [0, 0, 1, 0]]) + 0j)

    def qasm(self):
        return f'cx q[{self.wires[0]}],q[{self.wires[1]}];\n'


class Swap(DoubleGate):
    def __init__(self, nqubit=2, wires=[0,1], controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name='Swap', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0, 0, 0],
                                                     [0, 0, 1, 0],
                                                     [0, 1, 0, 0],
                                                     [0, 0, 0, 1]]) + 0j)

    def qasm(self):
        if self.controls == []:
            return f'swap q[{self.wires[0]}],q[{self.wires[1]}];\n'
        elif len(self.controls) == 1:
            return f'cswap q{self.controls},q[{self.wires[0]}],q[{self.wires[1]}];\n'
        else:
            return self.qasm_customized('swap')


class Rxx(ParametricDoubleGate):
    def __init__(self, inputs=None, nqubit=2, wires=[0,1], controls=None,
                 den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name='Rxx', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)
    
    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        cos  = torch.cos(theta / 2)
        isin = torch.sin(theta / 2) * 1j
        m1 = torch.stack([cos, cos, cos, cos]).reshape(-1).diag_embed()
        m2 = torch.stack([-isin, -isin, -isin, -isin]).reshape(-1).diag_embed().fliplr()
        return m1 + m2
    
    def inverse(self):
        return Rxx(inputs=-self.theta, nqubit=self.nqubit, wires=self.wires, controls=self.controls,
                   den_mat=self.den_mat, tsr_mode=self.tsr_mode, requires_grad=False)
    
    def qasm(self):
        if self.controls == []:
            return f'rxx({self.theta.item()}) q[{self.wires[0]}],q[{self.wires[1]}];\n'
        else:
            return self.qasm_customized('rxx')


class Ryy(ParametricDoubleGate):
    def __init__(self, inputs=None, nqubit=2, wires=[0,1], controls=None,
                 den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name='Ryy', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)

    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        cos  = torch.cos(theta / 2)
        isin = torch.sin(theta / 2) * 1j
        m1 = torch.stack([cos, cos, cos, cos]).reshape(-1).diag_embed()
        m2 = torch.stack([isin, -isin, -isin, isin]).reshape(-1).diag_embed().fliplr()
        return m1 + m2
    
    def inverse(self):
        return Ryy(inputs=-self.theta, nqubit=self.nqubit, wires=self.wires, controls=self.controls,
                   den_mat=self.den_mat, tsr_mode=self.tsr_mode, requires_grad=False)
    
    def qasm(self):
        if self.controls == []:
            qasm_str1 = ''
            qasm_str2 = f'ryy({self.theta.item()}) q[{self.wires[0]}],q[{self.wires[1]}];\n'
            if 'ryy' not in Gate.qasm_new_gate:
                qasm_str1 += 'gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(param0) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }\n'
                Gate.qasm_new_gate.append('ryy')
            return qasm_str1 + qasm_str2
        else:
            return self.qasm_customized('ryy')


class Rzz(ParametricDoubleGate):
    def __init__(self, inputs=None, nqubit=2, wires=[0,1], controls=None,
                 den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name='Rzz', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)

    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        e_m_it = torch.exp(-1j * theta / 2)
        e_it = torch.exp(1j * theta / 2)
        return torch.stack([e_m_it, e_it, e_it, e_m_it]).reshape(-1).diag_embed()
    
    def inverse(self):
        return Rzz(inputs=-self.theta, nqubit=self.nqubit, wires=self.wires, controls=self.controls,
                   den_mat=self.den_mat, tsr_mode=self.tsr_mode, requires_grad=False)
    
    def qasm(self):
        if self.controls == []:
            return f'rzz({self.theta.item()}) q[{self.wires[0]}],q[{self.wires[1]}];\n'
        else:
            return self.qasm_customized('rzz')


class Rxy(ParametricDoubleGate):
    def __init__(self, inputs=None, nqubit=2, wires=[0,1], controls=None,
                 den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name='Rxy', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)

    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        cos  = torch.cos(theta / 2)
        isin = torch.sin(theta / 2) * 1j
        m1 = torch.eye(1, dtype=theta.dtype, device=theta.device)
        m2 = torch.stack([cos, -isin, -isin, cos]).reshape(2, 2)
        return torch.block_diag(m1, m2, m1)
    
    def inverse(self):
        return Rxy(inputs=-self.theta, nqubit=self.nqubit, wires=self.wires, controls=self.controls,
                   den_mat=self.den_mat, tsr_mode=self.tsr_mode, requires_grad=False)
    
    def qasm(self):
        return self.qasm_customized('rxy')


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
        identity = torch.eye(2, dtype=matrix.dtype, device=matrix.device)
        oneone = torch.tensor([[0, 0], [0, 1]], dtype=matrix.dtype, device=matrix.device)
        lst1 = [identity] * self.nqubit
        lst2 = [identity] * self.nqubit
        lst3 = [identity] * self.nqubit

        lst2[self.wires[0]] = oneone
        lst2[self.wires[1]] = oneone

        lst3[self.wires[0]] = oneone
        lst3[self.wires[1]] = oneone
        lst3[self.wires[2]] = matrix[-2:, -2:]
        return multi_kron(lst1) - multi_kron(lst2) + multi_kron(lst3)
    
    def qasm(self):
        return f'ccx q[{self.wires[0]}],q[{self.wires[1]}],q[{self.wires[2]}];\n'


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
        identity = torch.eye(2, dtype=matrix.dtype, device=matrix.device)
        zerozero = torch.tensor([[1, 0], [0, 0]], dtype=matrix.dtype, device=matrix.device)
        zeroone  = torch.tensor([[0, 1], [0, 0]], dtype=matrix.dtype, device=matrix.device)
        onezero  = torch.tensor([[0, 0], [1, 0]], dtype=matrix.dtype, device=matrix.device)
        oneone   = torch.tensor([[0, 0], [0, 1]], dtype=matrix.dtype, device=matrix.device)
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
    
    def qasm(self):
        return f'cswap q[{self.wires[0]}],q[{self.wires[1]}],q[{self.wires[2]}];\n'


class UAnyGate(ArbitraryGate):
    def __init__(self, unitary, nqubit=1, wires=None, minmax=None, name='UAnyGate', den_mat=False,
                 tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, minmax=minmax, den_mat=den_mat,
                         tsr_mode=tsr_mode)
        if type(unitary) != torch.Tensor:
            unitary = torch.tensor(unitary, dtype=torch.cfloat).reshape(-1, 2 ** len(self.wires))
        assert unitary.dtype in (torch.cfloat, torch.cdouble)
        assert unitary.shape[-1] == 2 ** len(self.wires) and unitary.shape[-2] == 2 ** len(self.wires)
        assert is_unitary(unitary)
        self.register_buffer('matrix', unitary)
    
    def inverse(self):
        name = self.name + '_dagger'
        return UAnyGate(unitary=self.matrix.mH, nqubit=self.nqubit, wires=self.wires, minmax=self.minmax,
                        name=name, den_mat=self.den_mat, tsr_mode=self.tsr_mode)


class LatentGate(ArbitraryGate):
    def __init__(self, inputs=None, nqubit=1, wires=None, minmax=None, name='LatentGate',
                 den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, minmax=minmax, den_mat=den_mat,
                         tsr_mode=tsr_mode)
        self.requires_grad = requires_grad
        self.init_para(inputs=inputs)

    def inputs_to_tensor(self, inputs=None):
        if inputs == None:
            inputs = torch.randn(2 ** len(self.wires), 2 ** len(self.wires))
        elif type(inputs) != torch.Tensor and type(inputs) != torch.nn.parameter.Parameter:
            inputs = torch.tensor(inputs, dtype=torch.float)
        assert inputs.shape[-1] == 2 ** len(self.wires) and inputs.shape[-2] == 2 ** len(self.wires)
        return inputs
    
    def get_matrix(self, inputs):
        latent = self.inputs_to_tensor(inputs) + 0j
        u, _, vh = svd(latent)
        return u @ vh

    def update_matrix(self):
        matrix = self.get_matrix(self.latent)
        self.matrix = matrix.detach()
        return matrix
        
    def init_para(self, inputs=None):
        latent = self.inputs_to_tensor(inputs=inputs)
        if self.requires_grad:
            self.latent = nn.Parameter(latent)
        else:
            self.register_buffer('latent', latent)
        self.update_matrix()
        self.npara = self.latent.numel()

    def inverse(self):
        name = self.name + '_dagger'
        return LatentGate(inputs=self.latent.mH, nqubit=self.nqubit, wires=self.wires, minmax=self.minmax,
                          name=name, den_mat=self.den_mat, tsr_mode=self.tsr_mode, requires_grad=False)


class Barrier(Gate):
    def __init__(self, nqubit=1, wires=[0]):
        super().__init__(name='Barrier', nqubit=nqubit, wires=wires)

    def forward(self, x):
        return x
    
    def qasm(self):
        qasm_str = 'barrier '
        for wire in self.wires:
            qasm_str += f'q[{wire}],'
        return qasm_str[:-1] + ';\n'