from typing import Any, List, Optional, Union

import numpy as np
import torch
from torch import nn

from .photonic_qmath import dirac_ket


class FockState(nn.Module):
    """A state of n modes, including fock state basis or full fock state.
    Args:
        nmode (int, optional): The number of mode in the state. Default: 2
        state (Any, optional): The Fock state. the input can be list or torch.tensor
        state: input can be torch.tensor or List[Tuple],
        example: [(sqrt(2)/2, [1,0]), (sqrt(2)/2, [0,1])]
    """
    def __init__(self, nmode: int = None, state: Any = None, cutoff = None, basis: bool = True) -> None:
        super().__init__()
        self.nmode = nmode
        self.cutoff = cutoff
        self.basis = basis
        if self.basis:
            if not isinstance(state, torch.Tensor):
                state_ts = torch.tensor(state, dtype=torch.int)
            else:
                state_ts = state
            if nmode is None:
                self.nmode = state_ts.numel()
            if cutoff is None:
                self.cutoff = sum(state) + 1
        else:
            if isinstance(state, list) and all(isinstance(i, tuple) for i in state):
                nphoton = 0
                for s in state:
                    nphoton = max(nphoton, sum(s[1]))
                    self.nmode = len(s[1])
                if cutoff is None:
                    self.cutoff = nphoton + 1
                state_ts = torch.zeros([self.cutoff] * self.nmode, dtype=torch.cfloat)
                for s in state:
                    amp = s[0]
                    fock_basis = tuple(s[1])
                    state_ts[fock_basis] = amp
                state_ts = state_ts.unsqueeze(0)   # add additional batch size
            else:
                assert isinstance(state, torch.Tensor)  # need additional batch size
                self.cutoff = state.shape[1]
                self.nmode = len(state.shape) - 1
                state_ts = state
        self.register_buffer('state', state_ts)

    def __repr__(self):
        if self.basis:
            lst = list(map(lambda x:str(x),self.state.tolist()))
            state_str = ''.join(lst)
            return ('|' + state_str + '>')
        else:
            ket_dict = dirac_ket(self.state) # for the torch.tensor case
            temp = ''
            for key in ket_dict.keys():
                temp = temp + key + ': ' + ket_dict[key] + '\n'
            return temp

    # for compare the class
    def __eq__(self, other):
        return all([self.nmode == other.nmode, self.state.equal(other.state)])

    def __hash__(self):
        # 注意__hash__需要返回一个整数
        if self.basis:
            lst1 = list(map(lambda x:str(x),self.state.tolist()))
            state_str = ''.join(lst1)
            return hash(state_str)
        else:
            ket_dict = dirac_ket(self.state) # for the torch.tensor case
            temp = ''
            for key in ket_dict.keys():
                temp = temp + key + ': ' + ket_dict[key] + '\n'
            return hash(temp)
