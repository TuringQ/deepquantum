"""
Photonic measurements
"""

import copy
import itertools
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal

import deepquantum.photonic as dqp
from .operation import Operation
from .gate import PhaseShift

class Generaldyne(Operation):
    """Generaldyne measurement.

    Args:
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        covmat (Any): Performs the general-dyne measurement specified in covmat. Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        name (str, optional): The name of the gate. Default: ``'Generaldyne'`
    """
    def __init__(
        self,
        nmode: int = 1,
        wires: Union[int, List[int], None] = None,
        covmat: Any = None,
        cutoff: Optional[int] = None,
        name: str = 'Generaldyne'
    ) -> None:
        self.nmode = nmode
        if wires is None:
            wires = list(range(nmode))
        super().__init__(name=name, nmode=nmode, wires=wires, cutoff=cutoff, noise=False)
        if not isinstance(covmat, torch.Tensor):
            covmat = torch.tensor(covmat).reshape(-1, 2 * len(self.wires))
        assert covmat.shape[-1] == 2 * len(self.wires), "Covariance matrix size does not match wires provided"
        self.covmat = covmat

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Perform a forward pass for Gaussian states.

        See Quantum Continuous Variables: A Primer of Theoretical Methods
        by Alessio Serafini page 121
        """
        cov, mean =  x
        covmat = self.covmat
        covmat = covmat.to(cov.dtype)
        wires = torch.tensor(self.wires)
        size = cov.size()
        indices = torch.cat([wires, wires+self.nmode]) # xxpp order
        all_idx = torch.arange(2*self.nmode)
        mask = ~torch.isin(all_idx, indices)
        idx_rest = all_idx[mask]

        cov_a = cov[:, idx_rest[:, None], idx_rest]
        cov_b = cov[:, indices[:, None], indices]
        cov_ab = cov[:, idx_rest[:, None], indices]
        mean_b_ = mean[:, indices].squeeze(-1)
        mean_a_ = mean[:, idx_rest].squeeze(-1)
        temp = cov_ab @ torch.linalg.inv(cov_b + covmat)
        cov_a2 = cov_a - temp @ cov_ab.mT # update the unmeasured part
        cov_a3 = torch.stack([torch.eye(size[-1])]*size[0])
        cov_a3 = cov_a3.to(cov_a2.dtype)
        cov_a3[:, idx_rest[:, None], idx_rest] = cov_a2 # update the total cov mat

        samples = MultivariateNormal(mean_b_, cov_b + covmat).sample([1]) # xxpp order, shape: (shots, batch, 2 * nwire) shots=1000?
        samples = samples.permute(1, 0, 2)
        temp2  = (samples[0] - mean_b_).unsqueeze(2)
        mean_a2 = mean_a_.unsqueeze(2) + temp @ temp2
        mean_new = torch.zeros_like(mean)
        mean_new[:, idx_rest] = mean_a2
        samples_half = samples[:,:,:len(self.wires)] # xxpp order
        return samples_half, [cov_a3, mean_new]

class Homodyne(Generaldyne):
    """Homodyne measurement.

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        name (str, optional): The name of the gate. Default: ``'UAnyGate'`
    """
    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 1,
        wires: Union[int, List[int], None] = None,
        cutoff: Optional[int] = None,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
        name: str = 'Homodyne'
    ) -> None:
        self.nmode = nmode
        self.noise = noise
        self.mu = mu
        self.sigma = sigma
        wires = sorted(self._convert_indices(wires))
        eps = 2e-4
        covmat = torch.zeros(2 * len(wires))
        covmat[:len(wires)] = eps ** 2
        covmat[len(wires):] = 1/eps ** 2 # xxpp
        covmat = torch.diag(covmat)
        super().__init__(name=name, nmode=nmode, wires=wires, covmat=covmat, cutoff=cutoff)
        assert len(self.wires) == 1, f'{self.name} must act on one mode'
        self.requires_grad = False
        self.init_para(inputs)
        self.npara = 1
    def inputs_to_tensor(self, inputs: Any = None) -> torch.Tensor:
        """Convert inputs to torch.Tensor."""
        if inputs is None:
            inputs = torch.rand(len(self.wires)) * 2 * torch.pi
        elif not isinstance(inputs, (torch.Tensor, nn.Parameter)):
            inputs = torch.tensor(inputs, dtype=torch.float)
        if inputs.dim() == 0:
           inputs = inputs.unsqueeze(0)
        assert len(inputs) == len(self.wires)
        if self.noise:
            inputs = inputs + torch.normal(self.mu, self.sigma, size=(len(self.wires), )).squeeze()
        return inputs
    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        theta = self.inputs_to_tensor(inputs)
        if self.requires_grad:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('theta', theta)

    def extra_repr(self) -> str:
        return f'wires={self.wires}, theta={self.theta.item()}'

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Perform a forward pass for Gaussian states."""
        cov, mean =  x
        theta = self.theta
        r = PhaseShift(inputs=-theta, nmode=self.nmode, wires=self.wires, cutoff=self.cutoff)
        sp_mat = r.get_symplectic()
        sp_mat = sp_mat.to(cov.dtype)
        cov = sp_mat @ cov @ sp_mat.mT
        mean = sp_mat @ mean
        return super().forward([cov, mean])