"""
some math function
"""
from typing import Any, List, Optional, Tuple, Union
import torch

def kron(ts_list:List[torch.Tensor]) -> torch.Tensor:
    ts_kron = torch.tensor(1)
    for ts in ts_list:
        ts_kron = torch.kron(ts_kron, ts)
    return ts_kron

def list_xor(list1, list2):
    return list(set(list1) ^ set(list2))