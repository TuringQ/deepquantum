"""
Quantum states
"""

from typing import Any, List, Optional, Union

import torch
from torch import nn

from .bitmath import power_of_2, is_power_of_2, log_base2
from .communication import comm_get_rank, comm_get_world_size
from .qmath import is_density_matrix, amplitude_encoding, inner_product_mps, svd, qr


class QubitState(nn.Module):
    """A quantum state of n qubits, including both pure states and density matrices.

    Args:
        nqubit (int, optional): The number of qubits in the state. Default: 1
        state (Any, optional): The representation of the state. It can be one of the following strings:
            ``'zeros'``, ``'equal'``, ``'entangle'``, ``'GHZ'``, or ``'ghz'``. Alternatively, it can be
            a tensor that represents a custom state vector or density matrix. Default: ``'zeros'``
        den_mat (bool, optional): Whether the state is a density matrix or not. Default: ``False``
    """
    def __init__(self, nqubit: int = 1, state: Any = 'zeros', den_mat: bool = False) -> None:
        super().__init__()
        self.nqubit = nqubit
        self.den_mat = den_mat
        if state == 'zeros':
            state = torch.zeros((2 ** nqubit, 1), dtype=torch.cfloat)
            state[0] = 1
            if den_mat:
                state = state @ state.mH
            self.register_buffer('state', state)
        elif state == 'equal':
            state = torch.ones((2 ** nqubit, 1), dtype=torch.cfloat)
            state = nn.functional.normalize(state, p=2, dim=-2)
            if den_mat:
                state = state @ state.mH
            self.register_buffer('state', state)
        elif state in ('entangle', 'GHZ', 'ghz'):
            state = torch.zeros((2 ** nqubit, 1), dtype=torch.cfloat)
            state[0] = 1 / 2 ** 0.5
            state[-1] = 1 / 2 ** 0.5
            if den_mat:
                state = state @ state.mH
            self.register_buffer('state', state)
        else:
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.cfloat)
            ndim = state.ndim
            s = state.shape
            if den_mat and s[-1] == 2 ** nqubit and is_density_matrix(state):
                self.register_buffer('state', state)
            else:
                state = amplitude_encoding(data=state, nqubit=nqubit)
                if state.ndim > ndim:
                    state = state.squeeze(0)
                if den_mat:
                    state = state @ state.mH
                self.register_buffer('state', state)

    def to(self, arg: Any) -> 'QubitState':
        """Set dtype or device of the ``QubitState``."""
        if arg == torch.float:
            self.state = self.state.to(torch.cfloat)
        elif arg == torch.double:
            self.state = self.state.to(torch.cdouble)
        else:
            self.state = self.state.to(arg)
        return self

    def forward(self) -> None:
        """Pass."""
        pass


class MatrixProductState(nn.Module):
    r"""A matrix product state (MPS) for quantum systems.

    A matrix product state is a way of representing a quantum state as a product of local tensors.
    Each tensor has one physical index and one or two bond indices. The physical index corresponds to
    the local Hilbert space dimension of the qudit, while the bond indices correspond to the entanglement
    between qudits.

    Args:
        nsite (int, optional): The number of sites of the MPS. Default: 1
        state (str, List[torch.Tensor] or List[int], optional): The representation of the MPS.
            If ``'zeros'`` or ``'vac'``, the MPS is initialized to the all-zero state. If a list of tensors,
            the MPS is initialized to the given tensors. The tensors must have the correct shape and dtype.
            If a list of integers, the MPS is initialized to the corresponding basis state. Default: ``'zeros'``
        chi (int or None, optional): The maximum bond dimension of the MPS. Default: 10 * ``nsite``
        qudit (int, optional): The local Hilbert space dimension of each qudit. Default: 2
        normalize (bool, optional): Whether to normalize the MPS after each operation. Default: ``True``
    """
    def __init__(
        self,
        nsite: int = 1,
        state: Union[str, List[torch.Tensor], List[int]] = 'zeros',
        chi: Optional[int] = None,
        qudit: int = 2,
        normalize: bool = True
    ) -> None:
        super().__init__()
        if chi is None:
            chi = 10 * nsite
        self.nsite = nsite
        self.chi = chi
        self.qudit = qudit
        self.normalize = normalize
        self.center = -1
        self.set_tensors(state)

    def to(self, arg: Any) -> 'MatrixProductState':
        """Set dtype or device of the ``MatrixProductState``."""
        tensors = self.tensors
        for i in range(self.nsite):
            if arg == torch.float:
                self._buffers[f'tensor{i}'] = tensors[i].to(torch.cfloat)
            elif arg == torch.double:
                self._buffers[f'tensor{i}'] = tensors[i].to(torch.cdouble)
            else:
                self._buffers[f'tensor{i}'] = tensors[i].to(arg)
        return self

    @property
    def tensors(self) -> List[torch.Tensor]:
        """Get the tensors of the matrix product state.

        Note:
            This output is provided for reading only.
            Please modify the tensors through buffers.
        """
        tensors = []
        for j in range(self.nsite):
            tensors.append(getattr(self, f'tensor{j}'))
        return tensors

    def set_tensors(self, state: Union[str, List[torch.Tensor], List[int]]) -> None:
        """Set the tensors of the matrix product state."""
        if state in ('zeros', 'vac'):
            state = [0] * self.nsite
        assert isinstance(state, list), 'Invalid input type'
        if len(state) < self.nsite:
            state += [0] * (self.nsite - len(state))
        for i in range(self.nsite):
            assert isinstance(state[i], (torch.Tensor, int)), 'Invalid input type'
            if isinstance(state[i], torch.Tensor):
                self.register_buffer(f'tensor{i}', state[i])
            elif isinstance(state[i], int):
                assert 0 <= state[i] < self.qudit, 'Invalid input'
                tensor = torch.zeros(self.qudit, dtype=torch.cfloat)
                tensor[state[i]] = 1.
                # the bond dimension is 1
                self.register_buffer(f'tensor{i}', tensor.reshape(1, self.qudit, 1))

    def center_orthogonalization(self, c: int, dc: int = -1, normalize: bool = False) -> None:
        """Get the center-orthogonalization form of the MPS with center ``c``."""
        if c == -1:
            c = self.nsite - 1
        if self.center < -0.5:
            self.orthogonalize_n1_n2(0, c, dc, normalize)
            self.orthogonalize_n1_n2(self.nsite - 1, c, dc, normalize)
        elif self.center != c:
            self.orthogonalize_n1_n2(self.center, c, dc, normalize)
        self.center = c
        if normalize:
            self.normalize_central_tensor()

    def check_center_orthogonality(self, prt: bool = False) -> List[torch.Tensor]:
        """Check if the MPS is in center-orthogonal form."""
        tensors = self.tensors
        assert tensors[0].ndim == 3
        if self.center < -0.5:
            if prt:
                print('MPS NOT in center-orthogonal form!')
        else:
            err = [None] * self.nsite
            for i in range(self.center):
                s = tensors[i].shape
                tmp = tensors[i].reshape(-1, s[-1])
                tmp = tmp.mH @ tmp
                err[i] = (tmp - torch.eye(tmp.shape[0], device=tmp.device,
                                          dtype=tmp.dtype)).norm(p=1).item()
            for i in range(self.nsite - 1, self.center, -1):
                s = tensors[i].shape
                tmp = tensors[i].reshape(s[0], -1)
                tmp = tmp @ tmp.mH
                err[i] = (tmp - torch.eye(tmp.shape[0], device=tmp.device,
                                          dtype=tmp.dtype)).norm(p=1).item()
            if prt:
                print('Orthogonality check:')
                print('=' * 35)
                err_av = 0.0
                for i in range(self.nsite):
                    if err[i] is None:
                        print('Site ' + str(i) + ':  center')
                    else:
                        print('Site ' + str(i) + ': ', err[i])
                        err_av += err[i]
                print('-' * 35)
                print(f'Average error = {err_av / (self.nsite - 1)}')
                print('=' * 35)
            return err

    def full_tensor(self) -> torch.Tensor:
        """Get the full tensor product of the state."""
        assert self.nsite < 30
        tensors = self.tensors
        psi = tensors[0]
        for i in range(1, self.nsite):
            psi = torch.einsum('...abc,...cde->...abde', psi, tensors[i])
            s = psi.shape
            psi = psi.reshape(-1, s[-4], s[-3]*s[-2], s[-1])
        return psi.squeeze()

    def inner(
        self,
        tensors: Union[List[torch.Tensor], 'MatrixProductState'],
        form: str = 'norm'
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Get the inner product with another matrix product state."""
        # form: 'log' or 'list'
        if isinstance(tensors, list):
            return inner_product_mps(self.tensors, tensors, form=form)
        else:
            return inner_product_mps(self.tensors, tensors.tensors, form=form)

    def normalize_central_tensor(self) -> None:
        """Normalize the center tensor."""
        assert self.center in list(range(self.nsite))
        tensors = self.tensors
        if tensors[self.center].ndim == 3:
            norm = tensors[self.center].norm()
        elif tensors[self.center].ndim == 4:
            norm = tensors[self.center].norm(p=2, dim=[1,2,3], keepdim=True)
        self._buffers[f'tensor{self.center}'] = self._buffers[f'tensor{self.center}'] / norm

    def orthogonalize_left2right(self, site: int, dc: int = -1, normalize: bool = False) -> None:
        r"""Orthogonalize the tensor at ``site`` and update the next one at ``site`` + 1.

        It uses the QR decomposition or SVD, i.e., :math:`T = UR` for the QR decomposition and
        :math:`T = USV^{\dagger} = UR` for SVD. The tensor at ``site`` is replaced by :math:`U`.
        The tensor at ``site`` + 1 is updated by :math:`R`.

        Args:
            site (int): The site of tensor to be orthogonalized.
            dc (int, optional): Keep the first ``dc`` singular values after truncation.
                Default: -1 (which means no truncation)
            normalize (bool, optional): Whether to normalize the tensor :math:`R`. Default: ``False``
        """
        assert site < self.nsite - 1
        tensors = self.tensors
        shape = tensors[site].shape
        if len(shape) == 3:
            batch = 1
        else:
            batch = shape[0]
        if_trun = 0 < dc < shape[-1]
        if if_trun:
            u, s, vh = svd(tensors[site].reshape(batch, -1, shape[-1]))
            u = u[:, :, :dc]
            r = s[:, :dc].diag_embed() @ vh[:, :dc, :]
        else:
            u, r = qr(tensors[site].reshape(batch, -1, shape[-1]))
        self._buffers[f'tensor{site}'] = u.reshape(batch, shape[-3], shape[-2], -1)
        if normalize:
            norm = r.norm(dim=[-2,-1], keepdim=True)
            r = r / norm
        self._buffers[f'tensor{site + 1}'] = torch.einsum('...ab,...bcd->...acd', r, tensors[site + 1])
        if len(shape) == 3:
            tensors = self.tensors
            self._buffers[f'tensor{site}'] = tensors[site].squeeze(0)
            self._buffers[f'tensor{site + 1}'] = tensors[site + 1].squeeze(0)

    def orthogonalize_right2left(self, site: int, dc: int = -1, normalize: bool = False) -> None:
        r"""Orthogonalize the tensor at ``site`` and update the next one at ``site`` - 1.

        It uses the QR decomposition or SVD, i.e., :math:`T^{\dagger} = QR` for the QR decomposition, which
        gives :math:`T = R^{\dagger}Q^{\dagger} = LV^{\dagger}`, and :math:`T = USV^{\dagger} = LV^{\dagger}`
        for SVD. The tensor at ``site`` is replaced by :math:`V^{\dagger}`. The tensor at ``site`` - 1 is
        updated by :math:`L`.

        Args:
            site (int): The site of tensor to be orthogonalized.
            dc (int, optional): Keep the first ``dc`` singular values after truncation.
                Default: -1 (which means no truncation)
            normalize (bool, optional): Whether to normalize the tensor :math:`L`. Default: ``False``
        """
        assert site > 0
        tensors = self.tensors
        shape = tensors[site].shape
        if len(shape) == 3:
            batch = 1
        else:
            batch = shape[0]
        if_trun = 0 < dc < shape[-3]
        if if_trun:
            u, s, vh = svd(tensors[site].reshape(batch, shape[-3], -1))
            vh = vh[:, :dc, :]
            l = u[:, :, :dc] @ s[:, :dc].diag_embed()
        else:
            q, r = qr(tensors[site].reshape(batch, shape[-3], -1).mH)
            vh = q.mH
            l = r.mH
        self._buffers[f'tensor{site}'] = vh.reshape(batch, -1, shape[-2], shape[-1])
        if normalize:
            norm = l.norm(dim=[-2,-1], keepdim=True)
            l = l / norm
        self._buffers[f'tensor{site - 1}'] = torch.einsum('...abc,...cd->...abd', tensors[site - 1], l)
        if len(shape) == 3:
            tensors = self.tensors
            self._buffers[f'tensor{site}'] = tensors[site].squeeze(0)
            self._buffers[f'tensor{site - 1}'] = tensors[site - 1].squeeze(0)

    def orthogonalize_n1_n2(self, n1: int, n2: int, dc: int, normalize: bool) -> None:
        """Orthogonalize the MPS from site ``n1`` to site ``n2``."""
        if n1 < n2:
            for site in range(n1, n2, 1):
                self.orthogonalize_left2right(site, dc, normalize)
        else:
            for site in range(n1, n2, -1):
                self.orthogonalize_right2left(site, dc, normalize)

    def apply_mpo(self, mpo: List[torch.Tensor], sites: List[int]) -> None:
        """Use TEBD algorithm to contract tensors (contract local states with local operators), i.e.,

            >>>          a
            >>>          |
            >>>    i-----O-----j            a
            >>>          |        ->        |
            >>>          b             ik---X---jl
            >>>          |
            >>>    k-----T-----l
        """
        assert len(mpo) == len(sites)
        for i, site in enumerate(sites):
            tensor = torch.einsum('iabj,...kbl->...ikajl', mpo[i], self.tensors[site])
            s = tensor.shape
            if len(s) == 5:
                self._buffers[f'tensor{site}'] = tensor.reshape(s[-5] * s[-4], s[-3], s[-2] * s[-1])
            else:
                self._buffers[f'tensor{site}'] = tensor.reshape(-1, s[-5] * s[-4], s[-3], s[-2] * s[-1])

    def forward(self) -> None:
        """Pass."""
        pass


class DistributedQubitState(nn.Module):
    """A quantum state of n qubits distributed between w nodes, including both pure states and density matrices.

    Args:
        nqubit (int, optional): The number of qubits in the state.
    """
    def __init__(self, nqubit: int) -> None:
        super().__init__()
        self.world_size = comm_get_world_size()
        self.rank = comm_get_rank()
        assert is_power_of_2(self.world_size)
        assert power_of_2(nqubit) >= self.world_size
        assert 0 <= self.rank < self.world_size
        self.nqubit = nqubit

        self.log_num_nodes = log_base2(self.world_size)
        self.log_num_amps_per_node = nqubit - self.log_num_nodes
        self.num_amps_per_node = power_of_2(self.log_num_amps_per_node)

        # print(f"Rank {rank}: nqubit={nqubit}, log_nodes={self.log_num_nodes}, "
        #       f"log_local_amps={self.log_num_amps_per_node}, local_amps={self.num_amps_per_node}")

        amps = torch.zeros(self.num_amps_per_node) + 0j
        if self.rank == 0:
            amps[0] = 1.0
        buffer = torch.zeros_like(amps)
        self.register_buffer('amps', amps)
        self.register_buffer('buffer', buffer)

    def to(self, arg: Any) -> 'DistributedQubitState':
        """Set dtype or device of the ``DistributedQubitState``."""
        if arg == torch.float:
            self.amps = self.amps.to(torch.cfloat)
            self.buffer = self.buffer.to(torch.cfloat)
        elif arg == torch.double:
            self.amps = self.amps.to(torch.cdouble)
            self.buffer = self.buffer.to(torch.cdouble)
        else:
            self.amps = self.amps.to(arg)
            self.buffer = self.buffer.to(arg)
        return self

    def reset(self):
        self.amps.zero_()
        if self.rank == 0:
            self.amps[0] = 1.0
        self.buffer.zero_()

    # def get_global_qubit_range(self):
    #     """Returns the range of global qubit indices this rank 'controls'."""
    #     # Qubits >= log_num_amps_per_node influence the rank
    #     return range(self.log_num_amps_per_node, self.nqubit)

    # def get_local_qubit_range(self):
    #     """Returns the range of qubit indices local to this rank."""
    #     # Qubits < log_num_amps_per_node are local
    #     return range(0, self.log_num_amps_per_node)

    # def global_to_local(self, global_indices):
    #     """Converts global indices to local indices for this rank (placeholder)."""
    #     # Actual implementation depends on how global indices are handled.
    #     # For a single index: local_idx = global_idx & (self.num_amps_per_node - 1)
    #     # We'll mostly work with local indices directly or generate masks.
    #     raise NotImplementedError

    def local_to_global(self, local_indices: torch.Tensor) -> torch.Tensor:
        """Convert local indices to global indices for this rank."""
        return (self.rank << self.log_num_amps_per_node) | local_indices
