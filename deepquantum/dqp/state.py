import numpy as np
import torch
from torch import nn
from typing import Any, List, Optional, Union
# from photonic_qmath import dirac_ket
# from gate import PhaseShift, BeamSplitter, BeamSplitter_1, BeamSplitter_2
# from collections import defaultdict


class FockState(nn.Module):
    """A state of n modes, including fock state basis or full fock state.
    Args:
        nmode (int, optional): The number of mode in the state. Default: 2
        state (Any, optional): The initial fock state. the input can be list or torch.tensor
        state: input can be torch.tensor or List[Tuple],
        example: [(sqrt(2)/2, [1,0]), (sqrt(2)/2, [0,1])]
    """
    # def __init__(self, nmode: int = None, state:  Any = None, den_mat: bool = False) -> None:
    def __init__(self, nmode: int = None, state: Any = None, cutoff=None, basis: bool = True) -> None:
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
                state_ts = torch.zeros([self.cutoff]*self.nmode, dtype=torch.cfloat)
                for s in state:
                    amp = s[0]
                    fock_basis = tuple(s[1])
                    state_ts[fock_basis] = amp
                state_ts = state_ts.unsqueeze(0)   # add additional  batch size
            else:
                assert isinstance(state, torch.Tensor)  # need additional  batch size
                self.cutoff = state.shape[1]
                self.nmode = len(state.shape) - 1
                state_ts = state

        # ndim = state.ndim
        # s = state.shape
        self.register_buffer('state', state_ts)
    

    def __repr__(self):
        if self.basis:
            lst1=list(map(lambda x:str(x),self.state.tolist()))
            state_str =  ''.join(lst1)
            return ('|' + state_str + '>')
        else:
            ket_dict = FockState.dirac_ket(self.state) # for the torch.tensor case
            temp = ""
            for key in ket_dict.keys():
                temp = temp + key + ": "+ ket_dict[key] + "\n"
            return temp
    
    # for compare the class
    def __eq__(self, other):
        return all([self.nmode == other.nmode, self.state.equal(other.state)])
    
    def __hash__(self):
        # 注意__hash__需要返回一个整数
        if self.basis:
            lst1=list(map(lambda x:str(x),self.state.tolist()))
            state_str =  ''.join(lst1)
            return hash(state_str)
        else:
            ket_dict = dirac_ket(self.state) # for the torch.tensor case
            temp = ""
            for key in ket_dict.keys():
                temp = temp + key + ": "+ ket_dict[key] + "\n"
            return hash(temp)
        
    @staticmethod 
    def dirac_ket(matrix: torch.tensor) ->str:
        """
        the dirac state output with batch
        """
        ket_dict = {}
        for i in range(matrix.shape[0]): # consider batch i
            state_i = matrix[i]
            abs_state = abs(state_i)
            top_k = torch.topk(abs_state.flatten(), k=5, largest=True).values # get largest k values with abs(amplitudes)
            idx_all = []
            ket_repr_i = ""  
            for amp in top_k:
                idx = torch.nonzero(abs_state==amp)[0].tolist()
                idx_all.append(idx)
                abs_state[tuple(idx)] = 0 # after finding the indx, set the value to 0, avoid the same abs values
            
                lst1=list(map(lambda x:str(x), idx))
                # print(amp, idx, state_i[tuple(idx)], lst1)
                if amp>0:
                    state_str =  f"({state_i[tuple(idx)]:8.3f})" + "|" + "".join(lst1)+">"
                    ket_repr_i = ket_repr_i + "+" + state_str

            batch_i = "State" + f"{i}"

            ket_dict[ batch_i] = ket_repr_i[1:]
        return ket_dict