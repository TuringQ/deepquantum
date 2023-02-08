import torch
import torch.nn as nn
from deepquantum.operation import Gate
from deepquantum.qmath import multi_kron


class SingleGate(Gate):
    def __init__(self, name=None, nqubit=1, wires=0, den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
    
    def op_state(self, x):
        matrix = self.update_matrix()
        permute_shape = list(range(self.nqubit + 2))
        permute_shape[self.wires + 1] = self.nqubit
        permute_shape[self.nqubit] = self.wires + 1 
        rst = (matrix @ x.permute(permute_shape)).permute(permute_shape)
        if not self.tsr_mode:
            rst = self.vector_rep(rst).squeeze(0)
        return rst
        
    def op_den_mat(self, x):
        pass

    def get_unitary(self):
        matrix = self.update_matrix()
        lst = [self.identity] * self.nqubit
        lst[self.wires] = matrix
        return multi_kron(lst)


class DoubleGate(Gate):
    def __init__(self, name=None, nqubit=2, wires=[0,1], den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.register_buffer('zerozero', torch.tensor([[1, 0], [0, 0]], dtype=torch.cfloat))
        self.register_buffer('zeroone', torch.tensor([[0, 1], [0, 0]], dtype=torch.cfloat))
        self.register_buffer('onezero', torch.tensor([[0, 0], [1, 0]], dtype=torch.cfloat))
        self.register_buffer('oneone', torch.tensor([[0, 0], [0, 1]], dtype=torch.cfloat))
    
    def op_state(self, x):
        matrix = self.update_matrix()
        cqbit = self.wires[0]
        tqbit = self.wires[1]
        permute_shape = list(range(self.nqubit + 1))
        permute_shape.remove(cqbit + 1)
        permute_shape.remove(tqbit + 1)
        permute_shape = permute_shape + [cqbit + 1, tqbit + 1] + [self.nqubit + 1]
        x = (matrix @ x.permute(permute_shape).reshape(-1, 4, 1)).reshape([-1] + [2] * self.nqubit)
        permute_shape = list(range(self.nqubit + 1))
        permute_shape.pop()
        permute_shape.pop()
        if cqbit < tqbit:
            permute_shape.insert(cqbit + 1, self.nqubit - 1)
            permute_shape.insert(tqbit + 1, self.nqubit)
        else:
            permute_shape.insert(tqbit + 1, self.nqubit)
            permute_shape.insert(cqbit + 1, self.nqubit - 1)
        rst = x.permute(permute_shape).unsqueeze(-1)
        if not self.tsr_mode:
            rst = self.vector_rep(rst).squeeze(0)
        return rst
        
    def op_den_mat(self, x):
        pass
    
    def get_unitary(self):
        matrix = self.update_matrix()
        lst1 = [self.identity] * self.nqubit
        lst1[self.wires[0]] = self.zerozero
        lst1[self.wires[1]] = matrix[0:2, 0:2]

        lst2 = [self.identity] * self.nqubit
        lst2[self.wires[0]] = self.zeroone
        lst2[self.wires[1]] = matrix[0:2, 2:4]

        lst3 = [self.identity] * self.nqubit
        lst3[self.wires[0]] = self.onezero
        lst3[self.wires[1]] = matrix[2:4, 0:2]

        lst4 = [self.identity] * self.nqubit
        lst4[self.wires[0]] = self.oneone
        lst4[self.wires[1]] = matrix[2:4, 2:4]
        return multi_kron(lst1) + multi_kron(lst2) + multi_kron(lst3) + multi_kron(lst4)


class DoubleControlGate(DoubleGate):
    def __init__(self, name=None, nqubit=2, wires=[0,1], den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        
    def get_unitary(self):
        matrix = self.update_matrix()
        lst1 = [self.identity] * self.nqubit
        lst1[self.wires[0]] = self.zerozero

        lst2 = [self.identity] * self.nqubit
        lst2[self.wires[0]] = self.oneone
        lst2[self.wires[1]] = matrix[2:4, 2:4]
        return multi_kron(lst1) + multi_kron(lst2)


class U3Gate(SingleGate):
    def __init__(self, inputs=None, nqubit=1, wires=0, den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name='U3Gate', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.npara = 3
        self.requires_grad = requires_grad
        self.register_buffer('rx_m_pi_2', torch.tensor([[1, 1j], [1j, 1]]) / 2 ** 0.5) # rx(-pi/2)
        self.register_buffer('rx_p_pi_2', torch.tensor([[1, -1j], [-1j, 1]]) / 2 ** 0.5) # rx(pi/2)
        self.init_para(inputs=inputs)

    def inputs_to_tensor(self, inputs=None):
        if inputs == None:
            theta = torch.rand(1)[0] * torch.pi
            phi   = torch.rand(1)[0] * 2 * torch.pi
            lambd = torch.rand(1)[0] * 2 * torch.pi
        elif type(inputs) != torch.Tensor and type(inputs) != torch.nn.parameter.Parameter:
            inputs = torch.tensor(inputs, dtype=torch.float)
            theta = inputs[0]
            phi   = inputs[1]
            lambd = inputs[2]
        return theta, phi, lambd
    
    def get_matrix(self, theta, phi, lambd):
        rz1 = Rz().get_matrix(phi)
        rz2 = Rz().get_matrix(theta)
        rz3 = Rz().get_matrix(lambd)
        matrix = rz1 @ self.rx_m_pi_2 @ rz2 @ self.rx_p_pi_2 @ rz3
        # cos_t = torch.cos(theta / 2.0)
        # sin_t = torch.sin(theta / 2.0)
        # e_il = torch.cos(lambd) + 1j * torch.sin(lambd)
        # e_ip = torch.cos(phi) + 1j * torch.sin(phi)
        # e_ipl = torch.cos(lambd + phi) + 1j * torch.sin(lambd + phi)
        # matrix = torch.tensor([[cos_t, -e_il * sin_t],
        #                        [e_ip * sin_t, e_ipl * cos_t]])
        return matrix

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
    def __init__(self, nqubit=1, wires=0, den_mat=False, tsr_mode=False):
        super().__init__(name='PauliX', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.matrix = self.paulix


class PauliY(SingleGate):
    def __init__(self, nqubit=1, wires=0, den_mat=False, tsr_mode=False):
        super().__init__(name='PauliY', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.matrix = self.pauliy


class PauliZ(SingleGate):
    def __init__(self, nqubit=1, wires=0, den_mat=False, tsr_mode=False):
        super().__init__(name='PauliZ', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.matrix = self.pauliz


class Hadamard(SingleGate):
    def __init__(self, nqubit=1, wires=0, den_mat=False, tsr_mode=False):
        super().__init__(name='Hadamard', nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.matrix = self.hadamard


class SingleRotationGate(SingleGate):
    def __init__(self, name=None, inputs=None, nqubit=1, wires=0, den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
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


class Rx(SingleRotationGate):
    def __init__(self, inputs=None, nqubit=1, wires=0, den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name='Rx', inputs=inputs, nqubit=nqubit, wires=wires, den_mat=den_mat,
                         tsr_mode=tsr_mode, requires_grad=requires_grad)
    
    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        matrix = torch.cos(theta / 2.0) * self.identity \
               - torch.sin(theta / 2.0) * self.paulix * 1j
        # cos = torch.cos(theta / 2.0)
        # sin = torch.sin(theta / 2.0)
        # matrix = torch.tensor([[cos, -1j * sin],    # conflict with ad and vmap
        #                        [-1j * sin, cos]])
        return matrix


class Ry(SingleRotationGate):
    def __init__(self, inputs=None, nqubit=1, wires=0, den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name='Ry', inputs=inputs, nqubit=nqubit, wires=wires, den_mat=den_mat,
                         tsr_mode=tsr_mode, requires_grad=requires_grad)

    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        matrix = torch.cos(theta / 2.0) * self.identity \
               - torch.sin(theta / 2.0) * self.pauliy * 1j
        # cos = torch.cos(theta / 2.0)
        # sin = torch.sin(theta / 2.0)
        # matrix = torch.tensor([[cos, -sin],
        #                        [sin,  cos]]) + 0j
        return matrix


class Rz(SingleRotationGate):
    def __init__(self, inputs=None, nqubit=1, wires=0, den_mat=False, tsr_mode=False, requires_grad=False):
        super().__init__(name='Rz', inputs=inputs, nqubit=nqubit, wires=wires, den_mat=den_mat,
                         tsr_mode=tsr_mode, requires_grad=requires_grad)

    def get_matrix(self, theta):
        theta = self.inputs_to_tensor(theta)
        matrix = torch.cos(theta / 2.0) * self.identity \
               - torch.sin(theta / 2.0) * self.pauliz * 1j
        # cos = torch.cos(theta / 2.0)
        # sin = torch.sin(theta / 2.0)
        # matrix = torch.tensor([[cos - 1j * sin, 0],
        #                        [0, cos + 1j * sin]])
        return matrix


class CombinedSingleGate(SingleGate):
    def __init__(self, gatelist, name=None, nqubit=1, wires=0, den_mat=False, tsr_mode=False):
        super().__init__(name=name, nqubit=nqubit, wires=wires, den_mat=den_mat, tsr_mode=tsr_mode)
        self.gatelist = nn.ModuleList(gatelist)
        self.update_npara()
        self.update_matrix()
        
    def update_matrix(self):
        matrix = self.identity
        for gate in self.gatelist:
            matrix_i = gate.update_matrix()
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
        self.matrix = torch.tensor([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, 1, 0]]) + 0j