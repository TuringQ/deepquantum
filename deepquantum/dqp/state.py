import numpy as np
import torch
from torch import nn
from typing import Any, List, Optional, Union
# from gate import PhaseShift, BeamSplitter, BeamSplitter_1, BeamSplitter_2
# from collections import defaultdict


class FockState(nn.Module):
    """A state of n modes, including both fock state.

    Args:
        nmode (int, optional): The number of mode in the state. Default: 2
        state (Any, optional): The initial fock state. the input can be list or torch.tensor
        den_mat (bool, optional): Whether the state is a density matrix or not. Default: ``False``
    """
    def __init__(self, nmode: int = None, state:  Any = None, den_mat: bool = False) -> None:
        super().__init__()
        self.nmode = nmode
        # self.cutoff = cutoff
        self.photons = sum(state)
        self.den_mat = den_mat
        # self.
    
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.int)
        # ndim = state.ndim
        # s = state.shape
        self.register_buffer('state', state)
    

    def __repr__(self):
        lst1=list(map(lambda x:str(x),self.state.tolist()))
        state_str =  ''.join(lst1)
        return ('|' + state_str + '>')
    
    # for compare the class
    def __eq__(self, other):
        return all([self.nmode == other.nmode, self.photons == other.photons, self.state.equal(other.state)])
    
    def __hash__(self):
        # 注意__hash__需要返回一个整数
        lst1=list(map(lambda x:str(x),self.state.tolist()))
        state_str =  ''.join(lst1)
        return hash(state_str)
    

class TensorState(nn.Module):
    """
    A state of n modes, represented using tensor
    support superposition state in tensor representation
    Args:
        nmode:
        cutoff:
        state: input can be torch.tensor or List[List],
        example: [[sqrt(2)/2, (1,0)], [sqrt(2)/2, (0,1)]]
    """
    def __init__(
        self, 
        nmode: int = None, 
        cutoff: int = None, 
        state: Union[List[List], None] = None, 
        den_mat: bool = False
    ) -> None:
        super().__init__()
        self.nmode = nmode
        self.cutoff = cutoff 
        # self.photons = sum(state)
        self.den_mat = den_mat
        if not isinstance(state, torch.Tensor):
           state_ts = torch.zeros([self.cutoff]*self.nmode, dtype=torch.complex128)
           for s in state:
               amp = s[0]
               fock_ = tuple(s[1])
               state_ts[fock_] = amp 
        self.register_buffer('state', state_ts)
        

    

