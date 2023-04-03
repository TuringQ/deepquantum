import torch
import torch.nn as nn
from deepquantum.operation import Gate
from deepquantum.qmath import multi_kron, inverse_permutation, is_unitary


class SingleGate(Gate):
    def __init__(self, name=None, nqubit=1, wires=0, controls=None, den_mat=False, tsr_mode=False):
        while type(wires) == list:
            assert len(wires) == 1, 'Please input one and only one wire'
            wires = wires[0]
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        if type(controls) == int:
            controls = [controls]
        if controls == None:
            controls = []
        assert type(controls) == list
        assert self.wires not in controls, 'Use repeated wires'
        self.nwire += len(controls)
        self.controls = controls
    
    def op_state(self, x):
        matrix = self.update_matrix()
        if self.controls == []:
            x = self.op_state_base(x=x, matrix=matrix)
        else:
            x = self.op_state_control(x=x, matrix=matrix)
        if not self.tsr_mode:
            x = self.vector_rep(x).squeeze(0)
        return x
    
    def op_state_base(self, x, matrix):
        pm_shape = list(range(self.nqubit + 1))
        pm_shape.remove(self.wires + 1)
        pm_shape = [self.wires + 1] + pm_shape
        x = x.permute(pm_shape).reshape(2, -1)
        x = (matrix @ x).reshape([2] + [-1] + [2] * (self.nqubit - 1))
        x = x.permute(inverse_permutation(pm_shape))
        return x
    
    def op_state_control(self, x, matrix):
        nc = len(self.controls)
        target = self.wires + 1
        controls = [i + 1 for i in self.controls]
        pm_shape = list(range(self.nqubit + 1))
        pm_shape.remove(target)
        for i in controls:
            pm_shape.remove(i)
        pm_shape = [target] + pm_shape + controls
        state1 = x.permute(pm_shape).reshape(2, -1, 2 ** nc)
        state2 = (matrix @ state1[:, :, -1]).unsqueeze(-1)
        state1 = torch.cat([state1[:, :, :-1], state2], dim=-1)
        state1 = state1.reshape([2] + [-1] + [2] * (self.nqubit - nc - 1) + [2] * nc)
        x = state1.permute(inverse_permutation(pm_shape))
        return x
    
    def op_den_mat(self, x):
        matrix = self.update_matrix()
        if self.controls == []:
            x = self.op_den_mat_base(x=x, matrix=matrix)
        else:
            x = self.op_den_mat_control(x=x, matrix=matrix)
        if not self.tsr_mode:
            x = self.matrix_rep(x).squeeze(0)
        return x
        
    def op_den_mat_base(self, x, matrix):
        # left multiply
        pm_shape = list(range(2 * self.nqubit + 1))
        pm_shape.remove(self.wires + 1)
        pm_shape = [self.wires + 1] + pm_shape
        x = x.permute(pm_shape).reshape(2, -1)
        x = (matrix @ x).reshape([2] + [-1] + [2] * (2 * self.nqubit - 1))
        x = x.permute(inverse_permutation(pm_shape))
        # right multiply
        pm_shape = list(range(2 * self.nqubit + 1))
        pm_shape.remove(self.wires + 1 + self.nqubit)
        pm_shape = [self.wires + 1 + self.nqubit] + pm_shape
        x = x.permute(pm_shape).reshape(2, -1)
        x = (matrix.conj() @ x).reshape([2] + [-1] + [2] * (2 * self.nqubit - 1))
        x = x.permute(inverse_permutation(pm_shape))
        return x
    
    def op_den_mat_control(self, x, matrix):
        nc = len(self.controls)
        # left multiply
        target = self.wires + 1
        controls = [i + 1 for i in self.controls]
        pm_shape = list(range(2 * self.nqubit + 1))
        pm_shape.remove(target)
        for i in controls:
            pm_shape.remove(i)
        pm_shape = [target] + pm_shape + controls
        state1 = x.permute(pm_shape).reshape(2, -1, 2 ** nc)
        state2 = (matrix @ state1[:, :, -1]).unsqueeze(-1)
        state1 = torch.cat([state1[:, :, :-1], state2], dim=-1)
        state1 = state1.reshape([2] + [-1] + [2] * (2 * self.nqubit - nc - 1) + [2] * nc)
        x = state1.permute(inverse_permutation(pm_shape))
        # right multiply
        target = self.wires + 1 + self.nqubit
        controls = [i + 1 + self.nqubit for i in self.controls]
        pm_shape = list(range(2 * self.nqubit + 1))
        pm_shape.remove(target)
        for i in controls:
            pm_shape.remove(i)
        pm_shape = [target] + pm_shape + controls
        state1 = x.permute(pm_shape).reshape(2, -1, 2 ** nc)
        state2 = (matrix.conj() @ state1[:, :, -1]).unsqueeze(-1)
        state1 = torch.cat([state1[:, :, :-1], state2], dim=-1)
        state1 = state1.reshape([2] + [-1] + [2] * (2 * self.nqubit - nc - 1) + [2] * nc)
        x = state1.permute(inverse_permutation(pm_shape))
        return x

    def get_unitary(self):
        matrix = self.update_matrix()
        identity = torch.eye(2, dtype=torch.cfloat, device=matrix.device)
        if self.controls == None:
            lst = [identity] * self.nqubit
            lst[self.wires] = matrix
            return multi_kron(lst)
        else:
            oneone = torch.tensor([[0, 0], [0, 1]], dtype=torch.cfloat, device=matrix.device)
            lst1 = [identity] * self.nqubit
            lst2 = [identity] * self.nqubit
            lst3 = [identity] * self.nqubit
            for i in self.controls:
                lst2[i] = oneone
                lst3[i] = oneone
            lst3[self.wires] = matrix
            return multi_kron(lst1) - multi_kron(lst2) + multi_kron(lst3)


class DoubleGate(Gate):
    def __init__(self, name=None, nqubit=2, wires=[0,1], den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        assert len(wires) == 2
        assert wires[0] != wires[1]
    
    def op_state(self, x):
        matrix = self.update_matrix()
        cqbit = self.wires[0] + 1
        tqbit = self.wires[1] + 1
        pm_shape = list(range(self.nqubit + 1))
        pm_shape.remove(cqbit)
        pm_shape.remove(tqbit)
        pm_shape = [cqbit, tqbit] + pm_shape
        x = x.permute(pm_shape).reshape(4, -1)
        x = (matrix @ x).reshape([2, 2] + [-1] + [2] * (self.nqubit - 2))
        x = x.permute(inverse_permutation(pm_shape))
        if not self.tsr_mode:
            x = self.vector_rep(x).squeeze(0)
        return x
        
    def op_den_mat(self, x):
        matrix = self.update_matrix()
        # left multiply
        cqbit = self.wires[0] + 1
        tqbit = self.wires[1] + 1
        pm_shape = list(range(2 * self.nqubit + 1))
        pm_shape.remove(cqbit)
        pm_shape.remove(tqbit)
        pm_shape = [cqbit, tqbit] + pm_shape
        x = x.permute(pm_shape).reshape(4, -1)
        x = (matrix @ x).reshape([2, 2] + [-1] + [2] * (2 * self.nqubit - 2))
        x = x.permute(inverse_permutation(pm_shape))
        # right multiply
        cqbit = self.wires[0] + 1 + self.nqubit
        tqbit = self.wires[1] + 1 + self.nqubit
        pm_shape = list(range(2 * self.nqubit + 1))
        pm_shape.remove(cqbit)
        pm_shape.remove(tqbit)
        pm_shape = [cqbit, tqbit] + pm_shape
        x = x.permute(pm_shape).reshape(4, -1)
        x = (matrix.conj() @ x).reshape([2, 2] + [-1] + [2] * (2 * self.nqubit - 2))
        x = x.permute(inverse_permutation(pm_shape))
        if not self.tsr_mode:
            x = self.matrix_rep(x).squeeze(0)
        return x
    
    def get_unitary(self):
        matrix = self.update_matrix()
        identity = torch.eye(2, dtype=torch.cfloat, device=matrix.device)
        zerozero = torch.tensor([[1, 0], [0, 0]], dtype=torch.cfloat, device=matrix.device)
        zeroone  = torch.tensor([[0, 1], [0, 0]], dtype=torch.cfloat, device=matrix.device)
        onezero  = torch.tensor([[0, 0], [1, 0]], dtype=torch.cfloat, device=matrix.device)
        oneone   = torch.tensor([[0, 0], [0, 1]], dtype=torch.cfloat, device=matrix.device)
        lst1 = [identity] * self.nqubit
        lst1[self.wires[0]] = zerozero
        lst1[self.wires[1]] = matrix[0:2, 0:2]

        lst2 = [identity] * self.nqubit
        lst2[self.wires[0]] = zeroone
        lst2[self.wires[1]] = matrix[0:2, 2:4]

        lst3 = [identity] * self.nqubit
        lst3[self.wires[0]] = onezero
        lst3[self.wires[1]] = matrix[2:4, 0:2]

        lst4 = [identity] * self.nqubit
        lst4[self.wires[0]] = oneone
        lst4[self.wires[1]] = matrix[2:4, 2:4]
        return multi_kron(lst1) + multi_kron(lst2) + multi_kron(lst3) + multi_kron(lst4)


class DoubleControlGate(DoubleGate):
    def __init__(self, name=None, nqubit=2, wires=[0,1], den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        
    def get_unitary(self):
        matrix = self.update_matrix()
        identity = torch.eye(2, dtype=torch.cfloat, device=matrix.device)
        zerozero = torch.tensor([[1, 0], [0, 0]], dtype=torch.cfloat, device=matrix.device)
        oneone   = torch.tensor([[0, 0], [0, 1]], dtype=torch.cfloat, device=matrix.device)
        lst1 = [identity] * self.nqubit
        lst1[self.wires[0]] = zerozero

        lst2 = [identity] * self.nqubit
        lst2[self.wires[0]] = oneone
        lst2[self.wires[1]] = matrix[2:4, 2:4]
        return multi_kron(lst1) + multi_kron(lst2)


class U3Gate(SingleGate):
    def __init__(self, inputs=None, nqubit=1, wires=0, controls=None,
                 den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name='U3Gate', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.npara = 3
        self.requires_grad = requires_grad
        self.init_para(inputs=inputs)

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


class Identity(Gate):
    def __init__(self, nqubit=1, wires=0, den_mat=False, tsr_mode=False):
        super().__init__(name='Identity', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.matrix = torch.eye(2 ** self.nqubit, dtype=torch.cfloat)
        
    def get_unitary(self):
        return self.matrix
        
    def forward(self, x):
        return x


class PauliX(SingleGate):
    def __init__(self, nqubit=1, wires=0, controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name='PauliX', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat))


class PauliY(SingleGate):
    def __init__(self, nqubit=1, wires=0, controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name='PauliY', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[0, -1j], [1j, 0]]))


class PauliZ(SingleGate):
    def __init__(self, nqubit=1, wires=0, controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name='PauliZ', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat))


class Hadamard(SingleGate):
    def __init__(self, nqubit=1, wires=0, controls=None, den_mat=False, tsr_mode=False):
        super().__init__(name='Hadamard', nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat) / 2 ** 0.5)


class SingleParametricGate(SingleGate):
    def __init__(self, name=None, inputs=None, nqubit=1, wires=0, controls=None,
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
            inputs = torch.rand(1)[0] * torch.pi # from 0 to 2pi for Rz
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


class Rx(SingleParametricGate):
    def __init__(self, inputs=None, nqubit=1, wires=0, controls=None,
                 den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name='Rx', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)
    
    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        cos = torch.cos(theta / 2)
        sin = torch.sin(theta / 2)
        return torch.stack([cos, -1j * sin, -1j * sin, cos]).reshape(2, 2)


class Ry(SingleParametricGate):
    def __init__(self, inputs=None, nqubit=1, wires=0, controls=None,
                 den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name='Ry', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)

    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        cos = torch.cos(theta / 2)
        sin = torch.sin(theta / 2)
        return torch.stack([cos, -sin, sin, cos]).reshape(2, 2) + 0j


class Rz(SingleParametricGate):
    def __init__(self, inputs=None, nqubit=1, wires=0, controls=None,
                 den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name='Rz', inputs=inputs, nqubit=nqubit, wires=wires, controls=controls,
                         den_mat=den_mat, tsr_mode=tsr_mode, requires_grad=requires_grad)

    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        e_m_it = torch.exp(-1j * theta / 2)
        e_it = torch.exp(1j * theta / 2)
        return torch.stack([e_m_it, e_it]).reshape(-1).diag_embed()


class CombinedSingleGate(SingleGate):
    def __init__(self, gatelist, name=None, nqubit=1, wires=0, den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.gatelist = nn.ModuleList(gatelist)
        self.update_npara()
        self.update_matrix()
        
    def update_matrix(self):
        matrix = None
        for gate in self.gatelist:
            matrix_i = gate.update_matrix()
            if matrix == None:
                matrix = matrix_i
            else:
                matrix = matrix_i @ matrix
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


class UAnyGate(Gate):
    def __init__(self, unitary, nqubit=1, minmax=None, den_mat=False, tsr_mode=False):
        if minmax == None:
            minmax = [0, nqubit-1]
        self.minmax = minmax
        wires = [i for i in range(minmax[0], minmax[1] + 1)]
        if type(unitary) != torch.Tensor:
            unitary = torch.tensor(unitary, dtype=torch.cfloat).reshape(-1, 2 ** len(wires))
        assert unitary.dtype == torch.cfloat
        assert unitary.shape[-1] == 2 ** len(wires) and unitary.shape[-2] == 2 ** len(wires)
        assert is_unitary(unitary)
        super().__init__(name='UAnyGate', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('matrix', unitary)

    def get_unitary(self):
        identity = torch.eye(2, dtype=torch.cfloat, device=self.matrix.device)
        lst = [identity] * (self.nqubit - len(self.wires) + 1)
        lst[self.wires[0]] = self.matrix
        return multi_kron(lst)