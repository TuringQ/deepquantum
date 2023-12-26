from typing import Any, List, Optional, Union

import torch
from torch import nn

from .qmath import dirac_ket


class FockState(nn.Module):
    """A state of n modes, including fock state basis or full fock state.

    Args:
        state (Any): The Fock state. It can be a Fock basis state, e.g., [1,0,0],
            or a Fock state tensor, e.g., [(sqrt(2)/2, [1,0]), (sqrt(2)/2, [0,1])].
            Alternatively, it can be a tensor representation.
        nmode (int, optional): The number of modes in the state. Default: None
    """
    def __init__(self, state: Any, nmode: int = None, cutoff = None, basis: bool = True) -> None:
        super().__init__()
        self.basis = basis
        if self.basis:
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.int).reshape(-1)
            else:
                state = state.int().reshape(-1)
            if nmode is None:
                nmode = state.numel()
            if cutoff is None:
                cutoff = sum(state) + 1
            self.nmode = nmode
            self.cutoff = cutoff
            state_ts = torch.zeros(self.nmode, dtype=torch.int, device=state.device)
            size = len(state)
            if self.nmode > size:
                state_ts[:size] = state[:]
            else:
                state_ts[:] = state[:self.nmode]
            assert len(state_ts) == self.nmode
            assert state_ts.max() < self.cutoff
        else:
            if isinstance(state, torch.Tensor):  # need additional batch size
                if nmode is None:
                    nmode = state.ndim - 1
                if cutoff is None:
                    cutoff = state.shape[-1]
                self.nmode = nmode
                self.cutoff = cutoff
                state_ts = state
            else:
                assert isinstance(state, list) and all(isinstance(i, tuple) for i in state)
                nphoton = 0
                for s in state:
                    nphoton = max(nphoton, sum(s[1]))
                    if nmode is None:
                        nmode = len(s[1])
                if cutoff is None:
                    cutoff = nphoton + 1
                self.nmode = nmode
                self.cutoff = cutoff
                state_ts = torch.zeros([self.cutoff] * self.nmode, dtype=torch.cfloat)
                for s in state:
                    amp = s[0]
                    fock_basis = tuple(s[1])
                    state_ts[fock_basis] = amp
                state_ts = state_ts.unsqueeze(0)   # add additional batch size
            assert state_ts.ndim == self.nmode + 1
            assert all(i == self.cutoff for i in state_ts.shape[1:])
        self.register_buffer('state', state_ts)

    def __repr__(self):
        if self.basis:
            lst = list(map(lambda x:str(x),self.state.tolist()))
            state_str = ''.join(lst)
            return ('|' + state_str + '>')
        else:
            ket_dict = dirac_ket(self.state)
            temp = ''
            for key in ket_dict.keys():
                temp = temp + key + ': ' + ket_dict[key] + '\n'
            return temp

    def __eq__(self, other):
        return all([self.nmode == other.nmode, self.state.equal(other.state)])

    def __hash__(self):
        # 注意__hash__需要返回一个整数
        if self.basis:
            lst1 = list(map(lambda x:str(x),self.state.tolist()))
            state_str = ''.join(lst1)
            return hash(state_str)
        else:
            ket_dict = dirac_ket(self.state)
            temp = ''
            for key in ket_dict.keys():
                temp = temp + key + ': ' + ket_dict[key] + '\n'
            return hash(temp)
