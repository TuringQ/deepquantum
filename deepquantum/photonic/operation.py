"""
Base classes
"""

from typing import List, Optional, Union

import torch
from torch import nn

from ..qmath import inverse_permutation


class Operation(nn.Module):
    """A base class for quantum operations.

    Args:
        name (str or None, optional): The name of the quantum operation. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int, optional): The Fock space truncation. Default: 2
    """
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
        """Convert and check the indices of the modes."""
        if isinstance(indices, int):
            indices = [indices]
        assert isinstance(indices, list), 'Invalid input type'
        assert all(isinstance(i, int) for i in indices), 'Invalid input type'
        if len(indices) > 0:
            assert min(indices) > -1 and max(indices) < self.nmode, 'Invalid input'
        assert len(set(indices)) == len(indices), 'Invalid input'
        return indices

    def _check_minmax(self, minmax: List[int]) -> None:
        """Check the minmum and maximum indices of the modes."""
        assert isinstance(minmax, list)
        assert len(minmax) == 2
        assert all(isinstance(i, int) for i in minmax)
        assert -1 < minmax[0] <= minmax[1] < self.nqubit


class Gate(Operation):
    r"""A base class for photonic quantum gates.

    Args:
        name (str or None, optional): The name of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int, optional): The Fock space truncation. Default: 2
    """
    def __init__(
        self,
        name: Optional[str] = None,
        nmode: int = 1,
        wires: Union[int, List[int], None] = None,
        cutoff: int = 2
    ) -> None:
        self.nmode = nmode
        if wires is None:
            wires = [0]
        wires = self._convert_indices(wires)
        super().__init__(name=name, nmode=nmode, wires=wires, cutoff=cutoff)

    def update_matrix(self) -> torch.Tensor:
        """Update the local unitary matrix acting on operators."""
        return self.matrix

    def get_unitary_op(self) -> torch.Tensor:
        """Get the global unitary matrix acting on operators."""
        matrix = self.update_matrix()
        nmode1 = min(self.wires)
        nmode2 = self.nmode - nmode1 - len(self.wires)
        m1 = torch.eye(nmode1, dtype=matrix.dtype, device=matrix.device)
        m2 = torch.eye(nmode2, dtype=matrix.dtype, device=matrix.device)
        return torch.block_diag(m1, matrix, m2)

    def get_unitary_state(self, matrix: torch.Tensor) -> torch.Tensor:
        """Get the local unitary matrix acting on Fock state tensors."""
        raise NotImplementedError

    def update_unitary_state(self) -> torch.Tensor:
        """Update the local unitary matrix acting on Fock state tensors."""
        matrix = self.update_matrix()
        return self.get_unitary_state(matrix)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass."""
        return self.op_state_tensor(x)
