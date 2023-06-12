import torch
import torch.nn as nn
from deepquantum.qmath import is_density_matrix, amplitude_encoding, inner_product_mps, split_tensor, SVD


svd = SVD.apply


class QubitState(nn.Module):
    def __init__(self, nqubit=1, state='zeros', den_mat=False) -> None:
        super().__init__()
        self.nqubit = nqubit
        self.den_mat = den_mat
        if state == 'zeros':
            state = torch.zeros((2 ** nqubit, 1), dtype=torch.cfloat)
            state[0] = 1
            if den_mat:
                state = state @ state.mH
            self.register_buffer('state', state)
        elif state == 'equal':
            state = torch.ones((2 ** nqubit, 1), dtype=torch.cfloat)
            state = nn.functional.normalize(state, p=2, dim=-2)
            if den_mat:
                state = state @ state.mH
            self.register_buffer('state', state)
        elif state in ('entangle', 'GHZ', 'ghz'):
            state = torch.zeros((2 ** nqubit, 1), dtype=torch.cfloat)
            state[0] = 1 / 2 ** 0.5
            state[-1] = 1 / 2 ** 0.5
            if den_mat:
                state = state @ state.mH
            self.register_buffer('state', state)
        else:
            if type(state) != torch.Tensor:
                state = torch.tensor(state, dtype=torch.cfloat)
            ndim = state.ndim
            s = state.shape
            if den_mat and s[-1] == 2 ** nqubit and is_density_matrix(state):
                self.register_buffer('state', state)
            else:
                state = amplitude_encoding(data=state, nqubit=nqubit)
                if state.ndim > ndim:
                    state = state.squeeze(0)
                if den_mat:
                    state = state @ state.mH
                self.register_buffer('state', state)

    def forward(self):
        pass


class MatrixProductState(nn.Module):
    def __init__(self, nqubit=1, state='zeros', chi=None, qudit=2, normalize=True) -> None:
        super().__init__()
        if chi == None:
            chi = 10 * nqubit
        self.nqubit = nqubit
        self.chi = chi
        self.qudit = qudit
        self.normalize = normalize
        self.center = -1
        tensors = []
        if state == 'zeros':
            if nqubit == 1:
                tensor = torch.zeros(qudit, dtype=torch.cfloat)
                tensor[0] = 1.
                tensor = tensor.reshape(1, qudit, 1)
                tensors.append(tensor)
            else:
                tensor = torch.zeros(qudit * chi, dtype=torch.cfloat)
                tensor[0] = 1.
                tensor = tensor.reshape(1, qudit, chi)
                tensors.append(tensor)
                for i in range(1, nqubit - 1):
                    tensor = torch.zeros(chi * qudit * chi, dtype=torch.cfloat)
                    tensor[0] = 1.
                    tensor = tensor.reshape(chi, qudit, chi)
                    tensors.append(tensor)
                tensor = torch.zeros(qudit * chi, dtype=torch.cfloat)
                tensor[0] = 1.
                tensor = tensor.reshape(chi, qudit, 1)
                tensors.append(tensor)
        else:
            assert type(state) == list, 'Invalid input type'
            assert all(isinstance(i, torch.Tensor) for i in state), 'Invalid input type'
            assert len(state) == nqubit
            tensors = state
        for i in range(nqubit):
            self.register_buffer(f'tensor{i}', tensors[i])

    @property
    def tensors(self):
        # ATTENTION: This output is provided for reading only.
        # Please modify the tensors through buffers.
        tensors = []
        for j in range(self.nqubit):
            tensors.append(self.__getattr__(f'tensor{j}'))
        return tensors

    def set_tensors(self, tensors):
        assert type(tensors) == list, 'Invalid input type'
        assert all(isinstance(i, torch.Tensor) for i in tensors), 'Invalid input type'
        assert len(tensors) == self.nqubit
        for i in range(self.nqubit):
            self.register_buffer(f'tensor{i}', tensors[i])

    def center_orthogonalization(self, c, dc=-1, normalize=False):
        if c == -1:
            c = self.nqubit - 1
        if self.center < -0.5:
            self.orthogonalize_n1_n2(0, c, dc, normalize)
            self.orthogonalize_n1_n2(self.nqubit - 1, c, dc, normalize)
        elif self.center != c:
            self.orthogonalize_n1_n2(self.center, c, dc, normalize)
        self.center = c
        if normalize:
            self.normalize_central_tensor()

    def check_center_orthogonality(self, prt=False):
        tensors = self.tensors
        assert tensors[0].ndim == 3
        if self.center < -0.5:
            if prt:
                print('MPS NOT in center-orthogonal form!')
        else:
            err = [None] * self.nqubit
            for i in range(self.center):
                s = tensors[i].shape
                tmp = tensors[i].reshape(-1, s[-1])
                tmp = tmp.mH @ tmp
                err[i] = (tmp - torch.eye(tmp.shape[0], device=tmp.device,
                                          dtype=tmp.dtype)).norm(p=1).item()
            for i in range(self.nqubit - 1, self.center, -1):
                s = tensors[i].shape
                tmp = tensors[i].reshape(s[0], -1)
                tmp = tmp @ tmp.mH
                err[i] = (tmp - torch.eye(tmp.shape[0], device=tmp.device,
                                          dtype=tmp.dtype)).norm(p=1).item()
            if prt:
                print('Orthogonality check:')
                print('=' * 35)
                err_av = 0.0
                for i in range(self.nqubit):
                    if err[i] is None:
                        print('Site ' + str(i) + ':  center')
                    else:
                        print('Site ' + str(i) + ': ', err[i])
                        err_av += err[i]
                print('-' * 35)
                print('Average error = %g' % (err_av / (self.nqubit - 1)))
                print('=' * 35)
            return err
        
    def full_tensor(self):
        assert self.nqubit < 24
        tensors = self.tensors
        psi = tensors[0]
        for i in range(1, self.nqubit):
            psi = torch.einsum('...abc,...cde->...abde', psi, tensors[i])
            s = psi.shape
            psi = psi.reshape(-1, s[-4], s[-3]*s[-2], s[-1])
        return psi.squeeze()
    
    def inner(self, tensors, form='norm'):
        # form: 'log' or 'list'
        if type(tensors) is list:
            return inner_product_mps(self.tensors, tensors, form=form)
        else:
            return inner_product_mps(self.tensors, tensors.tensors, form=form)

    def normalize_central_tensor(self):
        assert self.center in list(range(self.nqubit))
        tensors = self.tensors
        if tensors[self.center].ndim == 3:
            norm = tensors[self.center].norm()
        elif tensors[self.center].ndim == 4:
            norm = tensors[self.center].norm(p=2, dim=[1,2,3], keepdim=True)
        self._buffers[f'tensor{self.center}'] = self._buffers[f'tensor{self.center}'] / norm

    def orthogonalize_left2right(self, site, dc=-1, normalize=False):
        # no truncation if dc=-1
        assert site < self.nqubit - 1
        tensors = self.tensors
        shape = tensors[site].shape
        if len(shape) == 3:
            batch = 1
        else:
            batch = shape[0]
        if 0 < dc < shape[-1]:
            if_trun = True
        else:
            if_trun = False
        u, s, vh = svd(tensors[site].reshape(batch, -1, shape[-1]))
        if if_trun:
            u = u[:, :, :dc]
            r = s[:, :dc].diag_embed() @ vh[:, :dc, :]
        else:
            r = s.diag_embed() @ vh
        self._buffers[f'tensor{site}'] = u.reshape(batch, shape[-3], shape[-2], -1)
        if normalize:
            r = r / r.norm(dim=[-2,-1], keepdim=True)
        self._buffers[f'tensor{site + 1}'] = torch.einsum('...ab,...bcd->...acd', r, tensors[site + 1])
        if len(shape) == 3:
            tensors = self.tensors
            self._buffers[f'tensor{site}'] = tensors[site].squeeze(0)
            self._buffers[f'tensor{site + 1}'] = tensors[site + 1].squeeze(0)
        return s
    
    def orthogonalize_right2left(self, site, dc=-1, normalize=False):
        # no truncation if dc=-1
        assert site > 0
        tensors = self.tensors
        shape = tensors[site].shape
        if len(shape) == 3:
            batch = 1
        else:
            batch = shape[0]
        if 0 < dc < shape[-3]:
            if_trun = True
        else:
            if_trun = False
        u, s, vh = svd(tensors[site].reshape(batch, shape[-3], -1))
        if if_trun:
            vh = vh[:, :dc, :]
            l = u[:, :, :dc] @ s[:, :dc].diag_embed()
        else:
            l = u @ s.diag_embed()
        self._buffers[f'tensor{site}'] = vh.reshape(batch, -1, shape[-2], shape[-1])
        if normalize:
            l = l / l.norm(dim=[-2,-1], keepdim=True)
        self._buffers[f'tensor{site - 1}'] = torch.einsum('...abc,...cd->...abd', tensors[site - 1], l)
        if len(shape) == 3:
            tensors = self.tensors
            self._buffers[f'tensor{site}'] = tensors[site].squeeze(0)
            self._buffers[f'tensor{site - 1}'] = tensors[site - 1].squeeze(0)
        return s
    
    def orthogonalize_n1_n2(self, n1, n2, dc, normalize):
        if n1 < n2:
            for site in range(n1, n2, 1):
                self.orthogonalize_left2right(site, dc, normalize)
        else:
            for site in range(n1, n2, -1):
                self.orthogonalize_right2left(site, dc, normalize)

    # def reduce_dimension(self, index_left, center_left):
    #     index_right = index_left + 1
    #     assert self.center in (index_left, index_right)
    #     tensor_left = self.tensors[index_left]
    #     tensor_right = self.tensors[index_right]
    #     ni = tensor_left.shape[0]
    #     nk = tensor_right.shape[-1]
    #     T = torch.einsum("iaj,jbk->iabk", tensor_left, tensor_right)
    #     T = T.reshape(ni * 2, nk * 2)
    #     new_tensor_left, new_tensor_right = split_tensor(T, center_left=center_left)
    #     new_tensor_left = new_tensor_left.reshape(ni, 2, -1)
    #     new_tensor_right = new_tensor_right.reshape(-1, 2, nk)
    #     self.tensors[index_left] = new_tensor_left
    #     self.tensors[index_right] = new_tensor_right
    #     if center_left:
    #         self.center = index_left
    #     else:
    #         self.center = index_right

    def forward(self):
        pass