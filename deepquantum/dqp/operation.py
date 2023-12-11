"""
Base classes
"""

from copy import copy
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from .state import FockState
from ..qmath import inverse_permutation


class Operation(nn.Module):
    def __init__(
        self,
        name: Optional[str] = None,
        nmode: int = 1,
        wires: Union[int, List, None] = None,
        cutoff: int = 2
    ) -> None:
        super().__init__()
        self.name = name
        self.nmode = nmode
        self.wires = wires
        self.cutoff = cutoff
        self.npara = 0

    def tensor_rep(self, x: torch.Tensor) -> torch.Tensor:
        """Get the tensor representation of the state."""
        return x.reshape([-1] + [self.cutoff] * self.nmode)

    def init_para(self) -> None:
        """Initialize the parameters."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass."""
        return self.tensor_rep(x)

    def _convert_indices(self, indices: Union[int, List[int]]) -> List[int]:
        """Convert and check the indices of the qumodes."""
        if isinstance(indices, int):
            indices = [indices]
        assert isinstance(indices, list), 'Invalid input type'
        assert all(isinstance(i, int) for i in indices), 'Invalid input type'
        if len(indices) > 0:
            assert min(indices) > -1 and max(indices) < self.nmode, 'Invalid input'
        assert len(set(indices)) == len(indices), 'Invalid input'
        return indices

    def _check_minmax(self, minmax: List[int]) -> None:
        """Check the minmum and maximum indices of the qubits."""
        assert isinstance(minmax, list)
        assert len(minmax) == 2
        assert all(isinstance(i, int) for i in minmax)
        assert -1 < minmax[0] <= minmax[1] < self.nqubit


class Gate(Operation):
    def __init__(
        self,
        name: Optional[str] = None,
        nmode: int = None,
        wires: Union[int, List[int], None] = None,
        cutoff: int = 2
    ) -> None:
        self.nmode = nmode
        if wires is None:
            wires = [0]
        wires = self._convert_indices(wires)
        super().__init__(name=name, nmode=nmode, wires=wires, cutoff=cutoff)

    def get_unitary_op(self) -> torch.Tensor:
        """Get the global unitary matrix acting on operators."""
        matrix = self.update_matrix()
        nmode1 = min(self.wires)
        nmode2 = self.nmode - nmode1 - len(self.wires)
        m1 = torch.eye(nmode1, dtype=matrix.dtype, device=matrix.device)
        m2 = torch.eye(nmode2, dtype=matrix.dtype, device=matrix.device)
        return torch.block_diag(m1, matrix, m2)

    def op_state_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass for state tensors."""
        nt = len(self.wires)
        matrix = self.update_unitary_state().reshape(self.cutoff ** nt, self.cutoff ** nt)
        wires = [i + 1 for i in self.wires]
        pm_shape = list(range(self.nmode + 1))
        for i in wires:
            pm_shape.remove(i)
        pm_shape = wires + pm_shape
        x = x.permute(pm_shape).reshape(self.cutoff ** nt, -1)
        x = (matrix @ x).reshape([self.cutoff] * nt + [-1] + [self.cutoff] * (self.nmode - nt))
        x = x.permute(inverse_permutation(pm_shape))
        return x

    def forward(self, x: Union[torch.Tensor, FockState]) -> Union[torch.Tensor, FockState]:
        """Perform a forward pass."""
        if isinstance(x, FockState):
            x = x.state
        return self.op_state_tensor(x)
