import torch
import torch.nn as nn
from deepquantum.qmath import is_density_matrix, amplitude_encoding


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
        elif state == 'entangle':
            state = torch.ones((2 ** nqubit, 1), dtype=torch.cfloat)
            state = nn.functional.normalize(state, p=2, dim=-2)
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