"""
Base classes
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import random
import torch
from torch import nn

class Operation(nn.Module):
    """A base class for quantum operations.

    Args:
        name (str or None, optional): The name of the quantum operation. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int, optional): The Fock space truncation. Default: 2
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """
    def __init__(
        self,
        name: Optional[str] = None,
        nqubit: int = 1,
        wires: Union[int, List, None] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.nqubit = nqubit
        self.wires = wires
        self.npara = 0

    def tensor_rep(self, x: torch.Tensor) -> torch.Tensor:
        """Get the tensor representation of the state."""
        return x.reshape([-1] + [2] * self.nqubit)

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
        assert len(set(indices)) == len(indices), 'Invalid input'
        return indices

    def _check_minmax(self, minmax: List[int]) -> None:
        """Check the minimum and maximum indices of the modes."""
        assert isinstance(minmax, list)
        assert len(minmax) == 2
        assert all(isinstance(i, int) for i in minmax)
        assert -1 < minmax[0] <= minmax[1] < self.nqubit

class Node(Operation):
    """
    Adding a node in MBQC graph
    """
    def __init__(
        self,
        wires: Union[int, List[int]] = None,
        state: Optional[torch.Tensor] = None
    ) -> None:
        wires = self._convert_indices(wires)
        if state is None:
            state = torch.sqrt(torch.tensor(2)) * torch.tensor([1,1]) / 2
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state)
        self.node_state = state
        super().__init__(name='node', nqubit=1, wires=wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.kron(x, self.node_state)

    def extra_repr(self) -> str:
        return f'wires={self.wires}'

class Entanglement(Operation):
    """
    Entangling a pair of qubits via CZ gate
    """
    def __init__(
        self,
        wires: List[int] = None
    ) -> None:
        self.matrix = torch.tensor([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, -1]])
        wires = self._convert_indices(wires)
        super().__init__(name='entanglement', wires=wires)

    def forward(self, wires:List, x: torch.Tensor) -> torch.Tensor:
        i, j = wires
        batch_size = x.size()[0]
        x = x.reshape(batch_size,-1)
        nqubit = int(torch.log2(torch.tensor(x.size(1))))
        perm = [0] + [i+1, j+1] + [k+1 for k in range(nqubit) if k not in (i, j)]
        inv_perm = [0] + [perm[1:].index(k) +1  for k in range(1, nqubit+1)]
        x = x.reshape([batch_size] + [2] * nqubit)
        x = x.permute(*perm)
        self.matrix = self.matrix.to(x.dtype)
        x = torch.matmul(self.matrix, x.reshape(batch_size, 4, -1))
        x = x.view([batch_size] + [2] * nqubit).permute(*inv_perm).reshape(batch_size, -1)
        return x

    def extra_repr(self) -> str:
        return f'wires={self.wires}'

class Measurement(Operation):
    """
    Measurement operator acting on single qubit with certain measurement plane and angle
    """
    def __init__(
        self,
        wires: Union[int, List[int]] = None,
        plane: Optional[str] = 'XY',
        angle: float = 0,
        t_domain: Union[int, List[int]] = [],
        s_domain: Union[int, List[int]] = []
    ) -> None:
        if plane is None:
            plane = 'XY'
        self.plane = plane
        if not isinstance(angle, torch.Tensor):
            angle = torch.tensor(angle)
        self.angle = angle
        self.t_domain = t_domain
        self.s_domain = s_domain
        wires = self._convert_indices(wires)
        super().__init__(name='Measurement', nqubit=2, wires=wires)

    def func_j_alpha(self, alpha):
        """
        Transfer matrix J(alpha) from XY, XZ, YZ plane to Z measurement,
        M(\alpha)\psi = M_z J(\alpha)\psi
        """
        if self.plane in ['XY', 'YX']:
            matrix_j = torch.sqrt(torch.tensor(2))/2 * torch.stack([
                torch.stack([torch.ones_like(alpha), torch.exp(-1j * alpha)], dim=-1),
                torch.stack([torch.ones_like(alpha), -torch.exp(-1j * alpha)], dim=-1)
            ], dim=-2)
        elif self.plane in ['XZ', 'ZX']:
            matrix_j = torch.stack([
                torch.stack([torch.cos(alpha/2), -1j * torch.sin(alpha/2)], dim=-1),
                torch.stack([torch.cos(alpha/2), 1j * torch.sin(alpha/2)], dim=-1)
            ], dim=-2)
        elif self.plane in ['YZ', 'ZY']:
            matrix_j = torch.stack([
                torch.stack([torch.cos(alpha/2), torch.sin(alpha/2)], dim=-1),
                torch.stack([torch.sin(alpha/2), -torch.cos(alpha/2)], dim=-1)
            ], dim=-2)
        else:
            raise ValueError(f"Unsupported measurement plane: {self.plane}")
        return matrix_j

    def forward(self, wires: int, x: torch.Tensor, measured_dic: dict) -> torch.Tensor:
        i = wires
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        nqubit =  int(torch.log2(torch.tensor(x.size(1))))
        perm = [0] + [i+1] + [k+1 for k in range(nqubit) if k != i]
        x = x.reshape([batch_size] + [2] * nqubit)
        x = x.permute(*perm).reshape(batch_size, 2, -1)
        s_signal = [0] * batch_size
        t_signal = [0] * batch_size
        if len(measured_dic) > 0:
            for i in range(batch_size):
                s_signal[i] = sum(measured_dic.get(wire, [0]*batch_size)[i] for wire in self.s_domain) % 2
                t_signal[i] = sum(measured_dic.get(wire, [0]*batch_size)[i] for wire in self.t_domain) % 2
        angle = (-1) ** torch.tensor(s_signal) * self.angle + torch.pi * torch.tensor(t_signal)
        j_alpha = self.func_j_alpha(angle)
        if self.plane in ['YZ', 'ZY']:
            x = torch.matmul(j_alpha.to(x.dtype), x)
        else:
            x = torch.matmul(j_alpha.to(torch.complex64), x.to(torch.complex64))
        samples_tot = []
        x_measured = []
        for i in range(batch_size):
            probs = torch.abs(x[i]) ** 2
            probs = probs.sum(-1)
            sample = random.choices([0, 1], weights=probs, k=1)
            samples_tot.append(sample[0])
            x_measured_i = x[i][sample[0], ...]
            x_measured.append(x_measured_i.reshape([2] * (nqubit-1)))
        self.sample = samples_tot
        x_measured = torch.stack(x_measured)
        x_measured = nn.functional.normalize(x_measured, dim=list(range(1, x_measured.ndim)))
        x_measured = x_measured.reshape(batch_size, -1)
        return x_measured

    def extra_repr(self) -> str:
        return f'wires={self.wires}, plane={self.plane}, angle={self.angle}, t_domain={self.t_domain}, s_domain={self.s_domain}'

class Correction(Operation):
    """
    correction operator acting on single qubit
    """
    def __init__(
        self,
        name: Optional[str] = None,
        wires: Union[int, List[int]] = None,
        signal_domain: Union[int, List[int]] = None,
        matrix: Any = None
    ) -> None:
        wires = self._convert_indices(wires)
        if signal_domain is None:
            signal_domain = [ ]
        signal_domain = self._convert_indices(signal_domain)
        self.signal_domain = signal_domain
        if not isinstance(matrix, torch.Tensor):
            matrix = torch.tensor(matrix)
        self.matrix = matrix
        super().__init__(name=name, nqubit=1, wires=wires)

    def forward(self, wires: int, x: torch.Tensor, measured_dic: dict):
        i = wires
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        nqubit =  int(torch.log2(torch.tensor(x.size(1))))
        perm = [0] + [i+1] + [k+1 for k in range(nqubit) if k != i]
        inv_perm = [0] + [perm[1:].index(k) +1  for k in range(1, nqubit+1)]
        x = x.reshape([batch_size] + [2] * nqubit)
        x = x.permute(*perm).reshape(batch_size, 2, -1)
        correct_mat = []
        for i in range(batch_size):
            parity = sum(measured_dic.get(wire, [0] * batch_size)[i] for wire in self.signal_domain) % 2
            if parity == 1:
                correct_mat.append(self.matrix)
            else:
                correct_mat.append(torch.eye(2))
        correct_mat = torch.stack(correct_mat)
        x = torch.matmul(correct_mat.to(x.dtype), x)
        x = x.reshape([batch_size] + [2] * nqubit)
        x = x.permute(*inv_perm).reshape(batch_size, -1)
        return x

    def extra_repr(self) -> str:
        return f'wires={self.wires}, signal_domain={self.signal_domain}'

class XCorrection(Correction):
    """
    X correction operator acting on single qubit
    """
    def __init__(
        self,
        wires: Union[int, List[int]] = None,
        signal_domain: List[int] = None
    ) -> None:
        matrix = torch.tensor([[0, 1],
                               [1, 0]])
        super().__init__(name='xcorrection', wires=wires, signal_domain=signal_domain, matrix=matrix)

class ZCorrection(Correction):
    """
    X correction operator acting on single qubit
    """
    def __init__(
        self,
        wires: Union[int, List[int]] = None,
        signal_domain: List[int] = None
    ) -> None:
        matrix = torch.tensor([[1, 0],
                               [0, -1]])
        super().__init__(name='zcorrection', wires=wires, signal_domain=signal_domain, matrix=matrix)