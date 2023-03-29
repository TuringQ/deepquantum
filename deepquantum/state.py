import torch
import torch.nn as nn
from deepquantum.qmath import amplitude_encoding

class QubitState(nn.Module):
    def __init__(self, nqubit=1, state='zeros', den_mat=False) -> None:
        super().__init__()
        if state == 'zeros':
            state = torch.zeros((2 ** nqubit, 1), dtype=torch.cfloat)
            state[0] = 1
        elif state == 'entangle':
            state = torch.ones((2 ** nqubit, 1), dtype=torch.cfloat)
            state = nn.functional.normalize(state, p=2, dim=-2)
        else:
            state = amplitude_encoding(data=state, nqubit=nqubit)
        if den_mat:
            if state.ndim == 1 or (state.ndim == 2 and state.shape[-1] == 1):
                state = state.reshape(1, -1, 1)
            s = state.shape
            if s[-1] != 2 ** nqubit or s[-2] != 2 ** nqubit:
                state = state.reshape(s[0], -1, 1).squeeze(0)
                assert state.shape[-2] == 2 ** nqubit, 'The shape of the state is not correct'
                state = state @ state.mH

        self.register_buffer('state', state)

    def forward(self):
        pass