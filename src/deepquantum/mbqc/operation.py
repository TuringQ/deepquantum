"""
Base classes
"""

from typing import Any, List, Optional, Union

import random
import torch
from torch import nn
from ..qmath import inverse_permutation

class Operation(nn.Module):
    """A base class for quantum operations.

    Args:
        name (str or None, optional): The name of the quantum operation. Default: ``None``
        n_input_nmodes (int): The number of modes that the quantum operation acts on. Default: 1
        node (int, List or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
    """
    def __init__(
        self,
        name: Optional[str] = None,
        n_input_nodes: int = 1,
        node: Union[int, List, None] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.n_input_nodes = n_input_nodes
        self.node = node
        self.npara = 0

    def _convert_indices(self, indices: Union[int, List[int]]) -> List[int]:
        """Convert and check the indices of the modes."""
        if isinstance(indices, int):
            indices = [indices]
        assert isinstance(indices, list), 'Invalid input type'
        assert all(isinstance(i, int) for i in indices), 'Invalid input type'
        assert len(set(indices)) == len(indices), 'Invalid input'
        return indices

class Node(Operation):
    """
    Adding a node in MBQC graph

    Args:
        node (Union[int, List[int]], optional): The index or indices of the node in the graph.
            Default: ``None``.
        state (Optional[torch.Tensor], optional): The initial quantum state of the node.
            Default: |+⟩ state.
    """
    def __init__(
        self,
        node: Union[int, List[int]] = None,
        state: Optional[torch.Tensor] = None
    ) -> None:
        node = self._convert_indices(node)
        if state is None:
            state = torch.sqrt(torch.tensor(2)) * torch.tensor([1,1]) / 2
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state)
        self.node_state = state
        super().__init__(name='node', node=node)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass by taking the Kronecker product with the node state.

        Args:
            x (torch.Tensor): Input quantum state tensor.
        """
        return torch.kron(x, self.node_state)

    def extra_repr(self) -> str:
        return f'node={self.node}'

class Entanglement(Operation):
    """
    Entangling a pair of qubits via CZ gate

    Args:
    node (List[int], optional): A list containing exactly two integers specifying
        the indices of the qubits to be entangled. Default: ``None``
    """
    def __init__(
        self,
        node: List[int] = None
    ) -> None:
        self.matrix = torch.tensor([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, -1]])
        node = self._convert_indices(node)
        super().__init__(name='entanglement', n_input_nodes=2, node=node)

    def forward(self, node:List, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the CZ entangling operation to the specified qubits.

        Args:
            node (List): A list containing two integers specifying the indices of
                qubits to be entangled.
            x (torch.Tensor): Input quantum state tensor of shape (batch_size, ...)
        """
        i, j = node
        batch_size = x.size()[0]
        x = x.reshape(batch_size,-1)
        nqubit = int(torch.log2(torch.tensor(x.size(1))))
        perm = [0] + [i+1, j+1] + [k+1 for k in range(nqubit) if k not in (i, j)]
        inv_perm = inverse_permutation(perm)
        x = x.reshape([batch_size] + [2] * nqubit)
        x = x.permute(*perm)
        self.matrix = self.matrix.to(x.dtype)
        x = torch.matmul(self.matrix, x.reshape(batch_size, 4, -1))
        x = x.view([batch_size] + [2] * nqubit).permute(*inv_perm).reshape(batch_size, -1)
        return x

    def extra_repr(self) -> str:
        return f'node={self.node}'

class Measurement(Operation):
    """
    A quantum measurement operator that acts on a single qubit
        with a specified measurement plane and angle.

    Args:
        node (Union[int, List[int]], optional): The index or indices of the qubit(s) to measure. Default: ``None``
        plane (Optional[str], optional): The measurement plane ('XY', 'XZ', or 'YZ'). Default: 'XY'
        angle (float, optional): The measurement angle in radians. Default: 0
        t_domain (Union[int, List[int]], optional): List of qubits that contribute to the t signal. Default: []
        s_domain (Union[int, List[int]], optional): List of qubits that contribute to the s signal. Default: []
    """
    def __init__(
        self,
        node: Union[int, List[int]] = None,
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
        node = self._convert_indices(node)
        super().__init__(name='Measurement', node=node)
        self.npara = 1

    def func_j_alpha(self, alpha):
        r"""
        Computes the transfer matrix J(alpha) that transforms measurements in
            the XY, XZ, or YZ plane to Z measurements. The transformation follows:
            M(\alpha)\psi = M_z J(\alpha)\psi, where M_z is the Z-basis measurement.

        Args:
            alpha (torch.Tensor): The measurement angle(s) in radians.
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

    def forward(self, node: int, x: torch.Tensor, measured_dic: dict) -> torch.Tensor:
        """
        Performs the measurement operation on the specified node.

        Args:
            node (int): The index of the qubit to measure.
            x (torch.Tensor): The input quantum state tensor of shape (batch_size, ...).
            measured_dic (dict): Dictionary containing previous measurement results, used for feed-forward corrections.
        """
        i = node
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
        return f'node={self.node}, plane={self.plane}, angle={self.angle}, t_domain={self.t_domain}, s_domain={self.s_domain}'

class Correction(Operation):
    """
    A base class for quantum correction operations.

    Args:
        name (Optional[str]): The name of the correction operation. Default: ``None``
        node (Union[int, List[int]], optional): The index or indices of the qubit(s) to apply
            the correction to. Default: ``None``
        signal_domain (Union[int, List[int]], optional): List of qubits whose measurement results
            determine whether to apply the correction. Default: ``None``
        matrix (Any, optional): The correction operation matrix to apply. Default: ``None``
    """
    def __init__(
        self,
        name: Optional[str] = None,
        node: Union[int, List[int]] = None,
        signal_domain: Union[int, List[int]] = None,
        matrix: Any = None
    ) -> None:
        node = self._convert_indices(node)
        if signal_domain is None:
            signal_domain = [ ]
        signal_domain = self._convert_indices(signal_domain)
        self.signal_domain = signal_domain
        if not isinstance(matrix, torch.Tensor):
            matrix = torch.tensor(matrix)
        self.matrix = matrix
        super().__init__(name=name, node=node)

    def forward(self, node: int, x: torch.Tensor, measured_dic: dict):
        """
        Apply the correction operation based on measurement results.

        Args:
            node (int): The index of the qubit to apply the correction to.
            x (torch.Tensor): Input quantum state tensor of shape (batch_size, ...).
            measured_dic (dict): Dictionary containing previous measurement results,
                used to determine whether to apply the correction.
        """
        i = node
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        nqubit =  int(torch.log2(torch.tensor(x.size(1))))
        perm = [0] + [i+1] + [k+1 for k in range(nqubit) if k != i]
        inv_perm = inverse_permutation(perm)
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
        return f'node={self.node}, signal_domain={self.signal_domain}'

class XCorrection(Correction):
    """
    X correction operator acting on single qubit
    """
    def __init__(
        self,
        node: Union[int, List[int]] = None,
        signal_domain: List[int] = None
    ) -> None:
        matrix = torch.tensor([[0, 1],
                               [1, 0]])
        super().__init__(name='xcorrection', node=node, signal_domain=signal_domain, matrix=matrix)

class ZCorrection(Correction):
    """
    X correction operator acting on single qubit
    """
    def __init__(
        self,
        node: Union[int, List[int]] = None,
        signal_domain: List[int] = None
    ) -> None:
        matrix = torch.tensor([[1, 0],
                               [0, -1]])
        super().__init__(name='zcorrection', node=node, signal_domain=signal_domain, matrix=matrix)
