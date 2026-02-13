"""Photonic measurements"""

from typing import Any

import numpy as np
import torch
from scipy.special import factorial
from torch import nn, vmap
from torch.distributions.multivariate_normal import MultivariateNormal

import deepquantum.photonic as dqp

from .gate import Displacement, PhaseShift
from .operation import Operation, evolve_den_mat, evolve_state
from .qmath import sample_homodyne_fock, sample_reject_bosonic
from .state import FockStateBosonic


class Generaldyne(Operation):
    """General-dyne measurement.

    Args:
        cov_m (Any): The covariance matrix for the general-dyne measurement.
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        name (str, optional): The name of the measurement. Default: ``'Generaldyne'``
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """

    def __init__(
        self,
        cov_m: Any,
        nmode: int = 1,
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        name: str = 'Generaldyne',
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        self.nmode = nmode
        if wires is None:
            wires = list(range(nmode))
        wires = self._convert_indices(wires)
        nwire = len(wires)
        if cutoff is None:
            cutoff = 2
        super().__init__(
            name=name, nmode=nmode, wires=wires, cutoff=cutoff, den_mat=den_mat, noise=noise, mu=mu, sigma=sigma
        )
        if not isinstance(cov_m, torch.Tensor):
            cov_m = torch.tensor(cov_m, dtype=torch.float).reshape(2 * nwire, 2 * nwire)
        assert cov_m.shape[-2] == cov_m.shape[-1] == 2 * nwire, 'The size of cov_m does not match the wires'
        self.register_buffer('cov_m', cov_m)
        self.samples = None

    def forward(self, x: list[torch.Tensor], samples: Any = None) -> list[torch.Tensor]:
        """Perform a forward pass for Gaussian (Bosonic) states.

        See Quantum Continuous Variables: A Primer of Theoretical Methods (2024)
        by Alessio Serafini Eq.(5.143) and Eq.(5.144) in page 121

        For Bosonic state, see https://arxiv.org/abs/2103.05530 Eq.(35-37)
        """
        cov, mean = x[:2]
        size = cov.size()
        wires = torch.tensor(self.wires)
        idx = torch.cat([wires, wires + self.nmode])  # xxpp order
        idx_all = torch.arange(2 * self.nmode)
        mask = ~torch.isin(idx_all, idx)
        idx_rest = idx_all[mask]

        cov_a = cov[..., idx_rest[:, None], idx_rest]
        cov_b = cov[..., idx[:, None], idx]
        cov_ab = cov[..., idx_rest[:, None], idx]
        mean_a = mean[..., idx_rest, :]
        mean_b = mean[..., idx, :]
        cov_t = cov_b + self.cov_m

        cov_a = cov_a - cov_ab @ torch.linalg.solve(cov_t, cov_ab.mT)  # update the unmeasured part
        cov_out = cov.new_ones(size[:-1].numel()).reshape(size[:-1]).diag_embed()
        cov_out[..., idx_rest[:, None], idx_rest] = cov_a  # update the total cov mat

        if len(x) == 2:  # Gaussian
            if samples is None:
                mean_m = MultivariateNormal(mean_b.squeeze(-1), cov_t).sample([1])[0]  # (batch, 2 * nwire)
            else:
                if not isinstance(samples, torch.Tensor):
                    samples = torch.tensor(samples, dtype=cov.dtype, device=cov.device)
                mean_m = samples.reshape(-1, 2 * len(self.wires))
            mean_a = mean_a + cov_ab @ torch.linalg.solve(cov_t, mean_m.unsqueeze(-1) - mean_b)
        elif len(x) == 3:  # Bosonic
            weight = x[2]
            if samples is None:
                mean_m = sample_reject_bosonic(cov_b, mean_b, weight, self.cov_m, 1)[:, 0]  # (batch, 2 * nwire)
            else:
                if not isinstance(samples, torch.Tensor):
                    samples = torch.tensor(samples, dtype=cov.dtype, device=cov.device)
                mean_m = samples.reshape(-1, 2 * len(self.wires))
            # (batch, ncomb)
            exp_real = torch.exp(mean_b.imag.mT @ torch.linalg.solve(cov_t, mean_b.imag) / 2).squeeze(-2, -1)
            gaus_b = MultivariateNormal(mean_b.squeeze(-1).real, cov_t)  # (batch, ncomb, 2 * nwire)
            prob_g = gaus_b.log_prob(mean_m.unsqueeze(-2)).exp()  # (batch, ncomb, 2 * nwire) -> (batch, ncomb)
            rm = mean_m.unsqueeze(-1).unsqueeze(-3)  # (batch, 2 * nwire) -> (batch, 1, 2 * nwire, 1)
            # (batch, ncomb)
            exp_imag = torch.exp((rm - mean_b.real).mT @ torch.linalg.solve(cov_t, mean_b.imag) * 1j).squeeze(-2, -1)
            weight *= exp_real * prob_g * exp_imag
            weight /= weight.sum(dim=-1, keepdim=True)
            mean_a = mean_a + cov_ab.to(mean_b.dtype) @ torch.linalg.solve(cov_t.to(mean_b.dtype), rm - mean_b)

        mean_out = torch.zeros_like(mean)
        mean_out[..., idx_rest, :] = mean_a

        self.samples = mean_m  # xxpp order
        if len(x) == 2:
            return [cov_out, mean_out]
        elif len(x) == 3:
            return [cov_out, mean_out, weight]


class Homodyne(Generaldyne):
    """Homodyne measurement.

    Args:
        phi (Any, optional): The homodyne measurement angle. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        eps (float, optional): The measurement accuracy. Default: 2e-4
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
        name (str, optional): The name of the measurement. Default: ``'Homodyne'``
    """

    def __init__(
        self,
        phi: Any = None,
        nmode: int = 1,
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        eps: float = 2e-4,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
        name: str = 'Homodyne',
    ) -> None:
        self.nmode = nmode
        if wires is None:
            wires = [0]
        wires = self._convert_indices(wires)
        cov_m = torch.diag(torch.tensor([eps**2] * len(wires) + [1 / eps**2] * len(wires)))  # xxpp
        super().__init__(
            cov_m=cov_m,
            nmode=nmode,
            wires=wires,
            cutoff=cutoff,
            den_mat=den_mat,
            name=name,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        assert len(self.wires) == 1, f'{self.name} must act on one mode'
        self.requires_grad = requires_grad
        self.init_para(inputs=phi)
        self.npara = 1

    def inputs_to_tensor(self, inputs: Any = None) -> torch.Tensor:
        """Convert inputs to torch.Tensor."""
        if inputs is None:
            inputs = torch.rand(len(self.wires)) * 2 * torch.pi
        elif not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float)
        inputs = inputs.reshape(-1)
        assert len(inputs) == len(self.wires)
        if self.noise:
            inputs = inputs + torch.normal(self.mu, self.sigma, size=(len(self.wires),)).squeeze()
        return inputs

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        phi = self.inputs_to_tensor(inputs)
        if self.requires_grad:
            self.phi = nn.Parameter(phi)
        else:
            self.register_buffer('phi', phi)

    def op_fock(self, x: torch.Tensor, samples: Any = None) -> torch.Tensor:
        """Perform a forward pass for Fock state tensors."""
        r = PhaseShift(inputs=-self.phi, nmode=self.nmode, wires=self.wires, cutoff=self.cutoff, den_mat=self.den_mat)
        if samples is None:
            # (batch, 1, 1)
            samples = sample_homodyne_fock(r(x), self.wires[0], self.nmode, self.cutoff, 1, self.den_mat)
        else:
            if not isinstance(samples, torch.Tensor):
                samples = torch.tensor(samples, dtype=x.real.dtype, device=x.device)
            samples = samples.reshape(-1, 1, 1)
        self.samples = samples.squeeze(-2)  # with dimension \sqrt{m\omega\hbar}
        # projection operator as single gate
        vac_state = x.new_zeros(self.cutoff)  # (cutoff)
        vac_state[0] = 1
        inf_sqz_vac = torch.zeros_like(vac_state)  # (cutoff)
        orders = torch.arange(np.ceil(self.cutoff / 2), dtype=x.real.dtype, device=x.device)
        fac_2n = torch.tensor(factorial(2 * orders), dtype=orders.dtype, device=orders.device)
        fac_n = torch.tensor(factorial(orders), dtype=orders.dtype, device=orders.device)
        inf_sqz_vac[::2] = (-0.5) ** orders * fac_2n**0.5 / fac_n  # unnormalized
        alpha = self.samples * dqp.kappa / dqp.hbar**0.5
        d = Displacement(cutoff=self.cutoff)
        d_mat = vmap(d.get_matrix_state, in_dims=(0, None))(alpha, 0)  # (batch, cutoff, cutoff)
        r = PhaseShift(inputs=self.phi, nmode=1, wires=0, cutoff=self.cutoff)
        eigenstate = r(evolve_state(inf_sqz_vac.unsqueeze(0), d_mat, 1, [0], self.cutoff))  # (batch, cutoff)
        project_op = vmap(torch.outer, in_dims=(None, 0))(vac_state, eigenstate.conj())  # (batch, cutoff, cutoff)
        if self.den_mat:  # (batch, 1, [cutoff] * 2 * nmode)
            evolve = vmap(evolve_den_mat, in_dims=(0, 0, None, None, None))
        else:  # (batch, 1, [cutoff] * nmode)
            evolve = vmap(evolve_state, in_dims=(0, 0, None, None, None))
        x = evolve(x.unsqueeze(1), project_op, self.nmode, self.wires, self.cutoff).squeeze(1)
        # normalization
        if self.den_mat:
            norm = vmap(torch.trace)(x.reshape(-1, self.cutoff**self.nmode, self.cutoff**self.nmode))  # (batch)
            x = x / norm.reshape([-1] + [1] * 2 * self.nmode)
        else:
            norm = (x.reshape(-1, self.cutoff**self.nmode).abs() ** 2).sum(-1) ** 0.5  # (batch)
            x = x / norm.reshape([-1] + [1] * self.nmode)
        return x

    def op_cv(self, x: list[torch.Tensor], samples: Any = None) -> list[torch.Tensor]:
        """Perform a forward pass for Gaussian (Bosonic) states."""
        r = PhaseShift(inputs=-self.phi, nmode=self.nmode, wires=self.wires, cutoff=self.cutoff)
        cov, mean = x[:2]
        cov, mean = r([cov, mean])
        return super().forward([cov, mean] + x[2:], samples)

    def forward(self, x: torch.Tensor | list[torch.Tensor], samples: Any = None) -> torch.Tensor | list[torch.Tensor]:
        """Perform a forward pass."""
        if isinstance(x, torch.Tensor):
            return self.op_fock(x, samples)
        elif isinstance(x, list):
            return self.op_cv(x, samples)

    def extra_repr(self) -> str:
        return f'wires={self.wires}, phi={self.phi.item()}'


class GeneralBosonic(Operation):
    """General Bosonic measurement.

    Args:
        cov (Any): The covariance matrices for the general Bosonic measurement.
        weight (Any): The weights for the general Bosonic measurement.
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        name (str, optional): The name of the measurement. Default: ``'GeneralBosonic'``
    """

    def __init__(
        self,
        cov: Any,
        weight: Any,
        nmode: int = 1,
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        name: str = 'GeneralBosonic',
    ) -> None:
        self.nmode = nmode
        if wires is None:
            wires = list(range(nmode))
        wires = self._convert_indices(wires)
        nwire = len(wires)
        if cutoff is None:
            cutoff = 2
        super().__init__(name=name, nmode=nmode, wires=wires, cutoff=cutoff)
        if not isinstance(cov, torch.Tensor):
            cov = torch.tensor(cov, dtype=torch.float)
        if not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight, dtype=torch.cfloat)
        cov = cov.reshape(-1, 2 * nwire, 2 * nwire)
        weight = weight.reshape(-1)
        ncomb = weight.shape[-1]
        assert cov.shape[-2] == cov.shape[-1] == 2 * nwire, 'The size does not match the wires'
        assert cov.shape[0] in (1, ncomb)
        self.register_buffer('cov', cov)
        self.register_buffer('weight', weight)
        self.samples = None

    def forward(self, x: list[torch.Tensor], samples: Any = None) -> list[torch.Tensor]:
        """Perform a forward pass for Gaussian (Bosonic) states.

        See https://arxiv.org/abs/2103.05530 Eq.(30-31) and Eq.(35-37)
        """
        cov, mean = x[:2]
        size = cov.size()
        wires = torch.tensor(self.wires)
        idx = torch.cat([wires, wires + self.nmode])  # xxpp order
        idx_all = torch.arange(2 * self.nmode)
        mask = ~torch.isin(idx_all, idx)
        idx_rest = idx_all[mask]

        # for ncomb_j of the measurement
        if len(x) == 2:  # Gaussian
            cov = cov.unsqueeze(-3).unsqueeze(-3)
            mean = mean.unsqueeze(-3).unsqueeze(-3) + 0j
            weight = mean.new_ones(size[0], 1, 1)
        elif len(x) == 3:  # Bosonic
            cov = cov.unsqueeze(-3)
            mean = mean.unsqueeze(-3) + 0j
            weight = x[2].unsqueeze(-1)
        cov_a = cov[..., idx_rest[:, None], idx_rest]
        cov_b = cov[..., idx[:, None], idx]
        cov_ab = cov[..., idx_rest[:, None], idx]
        mean_a = mean[..., idx_rest, :]
        mean_b = mean[..., idx, :]

        ncomb_j = self.weight.shape[-1]
        # (batch, ncomb, ncomb_j, 2 * nwire, 2 * nwire)
        cov_t = cov_b + self.cov.expand(ncomb_j, -1, -1) if self.cov.shape[0] == 1 else cov_b + self.cov
        cov_new = cov_t.flatten(-4, -3)  # (batch, ncomb_new, 2 * nwire, 2 * nwire)
        mean_new = mean_b.expand(-1, -1, ncomb_j, -1, -1).flatten(-4, -3)
        weight_new = (weight * self.weight).flatten(-2, -1)
        size_out = cov_new.size()[:-2] + size[-2:]  # (batch, ncomb_new, 2 * nmode, 2 * nmode)
        # (batch, ncomb, ncomb_j, 2 * nwire_rest, 2 * nwire_rest)
        cov_a = cov_a - cov_ab @ torch.linalg.solve(cov_t, cov_ab.mT)  # update the unmeasured part
        cov_out = cov.new_ones(size_out[:-1].numel()).reshape(size_out[:-1]).diag_embed()
        cov_out[..., idx_rest[:, None], idx_rest] = cov_a.flatten(-4, -3)  # update the total cov mat
        if samples is None:
            # (batch, 2 * nwire)
            mean_m = sample_reject_bosonic(cov_new, mean_new, weight_new, cov_new.new_zeros(1), 1)[:, 0]
        else:
            if not isinstance(samples, torch.Tensor):
                samples = torch.tensor(samples, dtype=cov.dtype, device=cov.device)
            mean_m = samples.reshape(-1, 2 * len(self.wires))
        # (batch, ncomb_new)
        exp_real = torch.exp(mean_new.imag.mT @ torch.linalg.solve(cov_new, mean_new.imag) / 2).squeeze(-2, -1)
        gaus_b = MultivariateNormal(mean_new.squeeze(-1).real, cov_new)  # (batch, ncomb_new, 2 * nwire)
        prob_g = gaus_b.log_prob(mean_m.unsqueeze(-2)).exp()  # (batch, ncomb_new, 2 * nwire) -> (batch, ncomb_new)
        rm = mean_m.unsqueeze(-1).unsqueeze(-3)  # (batch, 2 * nwire) -> (batch, 1, 2 * nwire, 1)
        # (batch, ncomb_new)
        exp_imag = torch.exp((rm - mean_new.real).mT @ torch.linalg.solve(cov_new, mean_new.imag) * 1j).squeeze(-2, -1)
        weight_out = weight_new * exp_real * prob_g * exp_imag
        weight_out /= weight_out.sum(dim=-1, keepdim=True)
        # (batch, ncomb, ncomb_j, 2 * nwire_rest, 1)
        mean_a = mean_a + cov_ab.to(mean.dtype) @ torch.linalg.solve(cov_t.to(mean.dtype), rm.unsqueeze(-3) - mean_b)
        mean_out = mean.new_zeros(size_out[:-1]).unsqueeze(-1)
        mean_out[..., idx_rest, :] = mean_a.flatten(-4, -3)

        self.samples = mean_m  # xxpp order
        return [cov_out, mean_out, weight_out]


class PhotonNumberResolvingBosonic(GeneralBosonic):
    """Photon-number-resolving measurement for Bosonic state.

    Args:
        n (int): Photon number.
        r (Any, optional): The quality parameter for the approximation. Default: 0.05
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        name (str, optional): The name of the measurement. Default: ``'PhotonNumberResolvingBosonic'``
    """

    def __init__(
        self,
        n: int,
        r: Any = 0.05,
        nmode: int = 1,
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        name: str = 'PhotonNumberResolvingBosonic',
    ) -> None:
        self.nmode = nmode
        if wires is None:
            wires = [0]
        state = FockStateBosonic(n, r, cutoff)
        cov = state.cov
        weight = state.weight
        if cutoff is None:
            cutoff = state.cutoff
        super().__init__(cov=cov, weight=weight, nmode=nmode, wires=wires, cutoff=cutoff, name=name)
        assert len(self.wires) == 1, f'{self.name} must act on one mode'

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        cov = x[0]
        batch = cov.shape[0]
        return super().forward(x, cov.new_zeros(batch, 2))
