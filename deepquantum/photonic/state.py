"""
Quantum states
"""

from typing import Any, Optional

import torch
from torch import nn

import deepquantum.photonic as dqp
from .qmath import dirac_ket


class FockState(nn.Module):
    """A Fock state of n modes, including Fock basis states and Fock state tensors.

    Args:
        state (Any): The Fock state. It can be a Fock basis state, e.g., ``[1,0,0]``,
            or a Fock state tensor, e.g., ``[(1/2**0.5, [1,0]), (1/2**0.5, [0,1])]``.
            Alternatively, it can be a tensor representation.
        nmode (int or None, optional): The number of modes in the state. Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        basis (bool, optional): Whether the state is a Fock basis state or not. Default: ``True``
    """
    def __init__(
        self,
        state: Any,
        nmode: Optional[int] = None,
        cutoff: Optional[int] = None,
        basis: bool = True
    ) -> None:
        super().__init__()
        self.basis = basis
        if self.basis:
            # Process Fock basis state
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.int).reshape(-1)
            else:
                state = state.int().reshape(-1)
            if nmode is None:
                nmode = state.numel()
            if cutoff is None:
                cutoff = sum(state) + 1
            self.nmode = nmode
            self.cutoff = cutoff
            state_ts = torch.zeros(self.nmode, dtype=torch.int, device=state.device)
            size = len(state)
            if self.nmode > size:
                state_ts[:size] = state[:]
            else:
                state_ts[:] = state[:self.nmode]
            assert len(state_ts) == self.nmode
            assert state_ts.max() < self.cutoff
        else:
            # Process Fock state tensor
            if isinstance(state, torch.Tensor):  # with the dimension of batch size
                if nmode is None:
                    nmode = state.ndim - 1
                if cutoff is None:
                    cutoff = state.shape[-1]
                self.nmode = nmode
                self.cutoff = cutoff
                state_ts = state
            else:
                # Process Fock state tensor from Fock basis states
                assert isinstance(state, list) and all(isinstance(i, tuple) for i in state)
                nphoton = 0
                # Determine the number of photons and modes
                for s in state:
                    nphoton = max(nphoton, sum(s[1]))
                    if nmode is None:
                        nmode = len(s[1])
                if cutoff is None:
                    cutoff = nphoton + 1
                self.nmode = nmode
                self.cutoff = cutoff
                state_ts = torch.zeros([self.cutoff] * self.nmode, dtype=torch.cfloat)
                # Populate Fock state tensor with the input Fock basis states
                for s in state:
                    amp = s[0]
                    fock_basis = tuple(s[1])
                    state_ts[fock_basis] = amp
                state_ts = state_ts.unsqueeze(0)  # add additional batch size
            assert state_ts.ndim == self.nmode + 1
            assert all(i == self.cutoff for i in state_ts.shape[1:])
        self.register_buffer('state', state_ts)

    def __repr__(self) -> str:
        """Return a string representation of the ``FockState``."""
        if self.basis:
            # Represent Fock basis state as a string
            state_str = ''.join(map(str, self.state.tolist()))
            return f'|{state_str}>'
        else:
            # Represent Fock state tensor using Dirac notation
            ket_dict = dirac_ket(self.state)
            temp = ''
            for key, value in ket_dict.items():
                temp += f'{key}: {value}\n'
            return temp

    def __eq__(self, other: 'FockState') -> bool:
        """Check if two ``FockState`` instances are equal."""
        return all([self.nmode == other.nmode, self.state.equal(other.state)])

    def __hash__(self) -> int:
        """Compute the hash value for the ``FockState``."""
        if self.basis:
            # Hash Fock basis state as a string
            state_str = ''.join(map(str, self.state.tolist()))
            return hash(state_str)
        else:
            # Hash Fock state tensor using Dirac notation
            ket_dict = dirac_ket(self.state)
            temp = ''
            for key, value in ket_dict.items():
                temp += f'{key}: {value}\n'
            return hash(temp)


class GaussianState(nn.Module):
    """A Gaussian state of n modes.

    Args:
        state (Any): The Gaussian state, it can be a vacuum state with 'vac', or arbitrary Gaussian states with [``cov``, ``mean``].
            ``cov`` and ``mean`` are the covariance matrix and the displacement vector of the Gaussian state, respectively.
            Use ``xxpp`` convention and :math:`\hbar=2` by default. Default: ``'vac'``
        nmode (int or None, optional): The number of modes in the state. Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
    """
    def __init__(
        self,
        state: Any = 'vac',
        nmode: Optional[int] = None,
        cutoff: Optional[int] = 5
    ) -> None:
        if state == 'vac':
            self.cov = dqp.hbar * torch.eye(2 * nmode) / 2
            self.mean = torch.zeros(2 * nmode)
        elif isinstance(state, list):
            cov = state[0]
            mean = state[1]
            if not isinstance(mean, torch.Tensor):
                mean_ts = torch.tensor(mean)
                self.mean = mean_ts.reshape([1, mean_ts.numel()]).squeeze()
            else:
                self.mean = mean
            if not isinstance(cov, torch.Tensor):
                self.cov = torch.tensor(cov)
            else:
                self.cov = cov

        assert self.cov.size()[0] == self.cov.size()[1] == 2 * nmode, 'The shape of the covariance matrix should be (2*nmode, 2*nmode)'
        assert self.mean.size()[0] == 2 * nmode, 'The length of the mean vector should be 2*nmode'
        self.nmode = nmode
        self.cutoff = cutoff
        self.is_pure = self.check_purity()

    def check_purity(self, rtol = 1e-5, atol = 1e-8):
        """Check if the Gaussian state is pure state

        See https://arxiv.org/pdf/quant-ph/0503237.pdf Eq.(2.5)
        """
        purity = 1 / torch.sqrt(2 * torch.det(self.cov) / dqp.hbar)
        return torch.allclose(purity, torch.tensor(1.0, dtype=self.cov.dtype), rtol=rtol, atol=atol)
