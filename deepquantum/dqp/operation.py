"""
Base classes
"""

from copy import copy
from typing import Any, List, Optional, Tuple, Union
# pylint: disable=unused-import
import warnings
import numpy as np
import torch
import string 
from torch import nn

from state import FockState
from photonic_qmath import FockOutput, inverse_permutation


class Operation(nn.Module):
    def __init__(
        self,
        name: Optional[str] = None,
        nmode: int = 1,
        wires: Union[int, List, None] = None,
        cutoff: int = None
    ) -> None:
        super().__init__()
        if wires is None:
            wires =[]
        if isinstance(wires, int):
            wires = [wires]
        assert isinstance(wires, (int, List)), "wires should be int or List"  ## typing does not support List[int]
        assert all(isinstance(i, int) for i in wires), 'wires should be int or List'
        

        self.name = name
        self.nmode = nmode
        self.wires = wires
        self.cutoff = cutoff
        self.npara = 0

    def tensor_rep(self, x: torch.Tensor) -> torch.Tensor:
        """Get the tensor representation of the state."""
        assert x.shape[-1] == x.shape[-2] == self.cutoff

        if x.ndim == self.nmode+1:
            return x
        if x.ndim == self.nmode:
            return x.unsqueeze(0)
        
    # def update_matrix(self):

class Gate(Operation):
    def __init__(self,
        name: Optional[str] = None,
        nmode: int = None,
        wires: Union[int, List[int], None] = None,
        cutoff: int = None
    ) -> None:
        super().__init__(name = name, nmode=nmode, wires=wires, cutoff=cutoff)


    def forward(self, x: Union[torch.Tensor, FockState]) -> Union[torch.Tensor, FockState]:
        """Perform a forward pass in tensor representation."""
        assert isinstance(x, torch.Tensor), "Only tensor states performs forward pass"
        # wires = self.wires
        x = self.tensor_rep(x)
        wires = [i + 1 for i in self.wires]  # the first dimension for batch here
        len_ = len(wires)
        if len_>6:
            print("too many dimensions of the unitary tensor")
        B = self.update_unitary_state()      # obtain the unitary tensor for ps, bs and u_any

        per_idx = list(range(self.nmode + 1))
        for idx in wires:
            per_idx.remove(idx)
        per_idx = wires + per_idx
        per_idx2 = inverse_permutation(per_idx)
        per_idx = tuple(per_idx)
        per_idx2 = tuple(per_idx2)
        lower = string.ascii_lowercase
        upper = string.ascii_uppercase
        einsum = lower[:len_] + "...," + upper[:len_]+lower[:len_] + "->" + upper[:len_] +"..." 
        
        # print(per_idx)
        state_1 = x.permute(per_idx)
        state_2 = torch.einsum(einsum, [state_1, B])
        state_3 = state_2.permute(per_idx2)
        return state_3



    
    
    

        
    # def get_matrix(self)
        
    # def get_unitary_state():
    #     mu, sigma -> unitary_fock

    # def op_fock_state()

    # # def op_fock_state_list()
    
    # def op_fock_state_tensor(self, x: torch.Tensor) -> torch.Tensor:
    #     """Perform a forward pass for state vectors."""
    #     unitary_fock @ fock_state_tensor
    #     return x