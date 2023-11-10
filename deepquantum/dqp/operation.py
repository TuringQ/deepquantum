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

from state import FockState, TensorState
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
        
    # def update_matrix(self):

class Gate(Operation):
    def __init__(self,
        name: Optional[str] = None,
        nmode: int = None,
        wires: Union[int, List[int], None] = None,
        cutoff: int = None
    ) -> None:
        super().__init__(name = name, nmode=nmode, wires=wires, cutoff=cutoff)





    def forward(self, x: Union[torch.Tensor, TensorState]) -> Union[torch.Tensor, TensorState]:
        """Perform a forward pass in tensor representation."""
        assert isinstance(x, torch.Tensor), "Only tensor states performs forward pass"
        wires = self.wires
        len_ = len(wires)
        if len_>6:
            print("too many dimensions of the unitary tensor")
        B = self.update_unitary_tensor()      # obtain the unitary tensor for ps, bs and u_any

        per_idx = list(range(0, self.nmode))
        for idx in wires:
            per_idx.remove(idx)
        per_idx = wires + per_idx
        per_idx2 = inverse_permutation(per_idx)
        per_idx = tuple(per_idx)
        per_idx2 = tuple(per_idx2)
        lower = string.ascii_lowercase
        upper = string.ascii_uppercase
        einsum = lower[:len_] + "...," + upper[:len_]+lower[:len_] + "->" + upper[:len_] +"..." 
        
        state_1 = x.permute(per_idx)
        state_2 = torch.einsum(einsum, [state_1, B])
        state_3 = state_2.permute(per_idx2)
        return state_3



    
    
    

        
    # def get_matrix(self)
        
    # def get_unitary_fock():
    #     mu, sigma -> unitary_fock

    # def op_fock_state()

    # # def op_fock_state_list()
    
    # def op_fock_state_tensor(self, x: torch.Tensor) -> torch.Tensor:
    #     """Perform a forward pass for state vectors."""
    #     unitary_fock @ fock_state_tensor
    #     return x