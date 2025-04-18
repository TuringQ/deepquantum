"""
Quantum states
"""

from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from scipy.special import comb
from torch import nn, vmap
from torch.distributions.multivariate_normal import MultivariateNormal

import deepquantum.photonic as dqp
from ..qmath import multi_kron
from .gate import PhaseShift
from .qmath import dirac_ket, xpxp_to_xxpp, xxpp_to_xpxp


class FockState(nn.Module):
    """A Fock state of n modes, including Fock basis states and Fock state tensors.

    Args:
        state (Any): The Fock state. It can be a vacuum state with ``'vac'`` or ``'zeros'``.
            It can be a Fock basis state, e.g., ``[1,0,0]``, or a Fock state tensor,
            e.g., ``[(1/2**0.5, [1,0]), (1/2**0.5, [0,1])]``. Alternatively, it can be a tensor representation.
        nmode (int or None, optional): The number of modes in the state. Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        basis (bool, optional): Whether the state is a Fock basis state or not. Default: ``True``
        den_mat (bool, optional): Whether to use density matrix representation. Only valid for Fock state tensor.
            Default: ``False``
    """
    def __init__(
        self,
        state: Any,
        nmode: Optional[int] = None,
        cutoff: Optional[int] = None,
        basis: bool = True,
        den_mat: bool = False
    ) -> None:
        super().__init__()
        self.basis = basis
        self.den_mat = den_mat
        if self.basis:
            if state in ('vac', 'zeros'):
                state = [0] * nmode
            # Process Fock basis state
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.long)
            else:
                state = state.long()
            if state.ndim == 1:
                state = state.unsqueeze(0)
            assert state.ndim == 2
            if nmode is None:
                nmode = state.shape[-1]
            if cutoff is None:
                cutoff = torch.max(torch.sum(state, dim=-1)).item() + 1
            self.nmode = nmode
            self.cutoff = cutoff
            batch, size = state.shape
            state_ts = torch.zeros([batch, nmode], dtype=torch.long, device=state.device)
            if nmode > size:
                state_ts[:, :size] = state[:, :]
            else:
                state_ts[:, :] = state[:, :nmode]
            state_ts = state_ts.squeeze(0)
            assert state_ts.max() < self.cutoff
        else:
            if state in ('vac', 'zeros'):
                state = [(1, [0] * nmode)]
            # Process Fock state tensor
            if isinstance(state, torch.Tensor):  # with the dimension of batch size
                if nmode is None:
                    if den_mat:
                        nmode = (state.ndim - 1) // 2
                    else:
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
                if den_mat:
                    state_ts = state_ts.reshape([self.cutoff ** self.nmode, 1])
                    state_ts = (state_ts @ state_ts.mH).reshape([-1] + [self.cutoff] * 2 * self.nmode)
            if den_mat:
                assert state_ts.ndim == 2 * self.nmode + 1
            else:
                assert state_ts.ndim == self.nmode + 1
            assert all(i == self.cutoff for i in state_ts.shape[1:])
        self.register_buffer('state', state_ts)

    def to(self, arg: Any) -> 'FockState':
        """Set dtype or device of the ``FockState``."""
        if arg == torch.float:
            if not self.basis:
                self.state = self.state.to(torch.cfloat)
        elif arg == torch.double:
            if not self.basis:
                self.state = self.state.to(torch.cdouble)
        else:
            self.state = self.state.to(arg)
        return self

    def __repr__(self) -> str:
        """Return a string representation of the ``FockState``."""
        if self.basis:
            # Represent Fock basis state as a string
            if self.state.ndim == 1:
                state_str = ''.join(map(str, self.state.tolist()))
                return f'|{state_str}>'
            else:
                temp = ''
                for i in range(self.state.shape[0]):
                    state_str = ''.join(map(str, self.state[i].tolist()))
                    temp += f'state_{i}: |{state_str}>\n'
                return temp
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

    def __lt__(self, other: 'FockState') -> bool:
        tuple_self = tuple(self.state.tolist())
        tuple_other = tuple(other.state.tolist())
        return tuple_self < tuple_other


class GaussianState(nn.Module):
    r"""A Gaussian state of n modes, representing by covariance matrix and displacement vector.

    Args:
        state (str or List): The Gaussian state. It can be a vacuum state with ``'vac'``, or arbitrary Gaussian state
            with ``[cov, mean]``. ``cov`` and ``mean`` are the covariance matrix and the displacement vector of
            the Gaussian state, respectively. Use ``xxpp`` convention and :math:`\hbar=2` by default.
            Default: ``'vac'``
        nmode (int or None, optional): The number of modes in the state. Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
    """
    def __init__(
        self,
        state: Union[str, List] = 'vac',
        nmode: Optional[int] = None,
        cutoff: Optional[int] = None
    ) -> None:
        super().__init__()
        if state == 'vac':
            if nmode is None:
                nmode = 1
            cov = torch.eye(2 * nmode) * dqp.hbar / (4 * dqp.kappa ** 2)
            mean = torch.zeros(2 * nmode, 1)
        elif isinstance(state, list):
            cov = state[0]
            mean = state[1]
            if not isinstance(cov, torch.Tensor):
                cov = torch.tensor(cov, dtype=torch.float)
            if not isinstance(mean, torch.Tensor):
                mean = torch.tensor(mean, dtype=torch.float)
            if nmode is None:
                nmode = cov.shape[-1] // 2
        cov = cov.reshape(-1, 2 * nmode, 2 * nmode)
        mean = mean.reshape(-1, 2 * nmode, 1)
        assert cov.ndim == mean.ndim == 3
        assert cov.shape[-2] == cov.shape[-1] == 2 * nmode, (
            'The shape of the covariance matrix should be (2*nmode, 2*nmode)')
        assert mean.shape[-2] == 2 * nmode, 'The length of the mean vector should be 2*nmode'
        self.register_buffer('cov', cov)
        self.register_buffer('mean', mean)
        self.nmode = nmode
        if cutoff is None:
            cutoff = 5
        self.cutoff = cutoff
        self.is_pure = self.check_purity()

    def check_purity(self, rtol = 1e-5, atol = 1e-8):
        """Check if the Gaussian state is pure state

        See https://arxiv.org/pdf/quant-ph/0503237.pdf Eq.(2.5)
        """
        purity = 1 / torch.sqrt(torch.det(4 * dqp.kappa ** 2 / dqp.hbar * self.cov))
        unity = torch.tensor(1.0, dtype=purity.dtype, device=purity.device)
        return torch.allclose(purity, unity, rtol=rtol, atol=atol)


class BosonicState(nn.Module):
    r"""A Bosoncic state of n modes, representing by a linear combination of Gaussian states.

    Args:
        state (str or List): The Bosoncic state. It can be a vacuum state with ``'vac'``, or arbitrary
            linear combinations of Gaussian states with ``[cov, mean, weight]``. ``cov``,``mean`` and ``weight`` are
            the covariance matrices, the displacement vectors and the weights of the Gaussian states, respectively.
            Use ``xxpp`` convention and :math:`\hbar=2` by default.
            Default: ``'vac'``
        nmode (int or None, optional): The number of modes in the state. Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
    """
    def __init__(
        self,
        state: Union[str, List] = 'vac',
        nmode: Optional[int] = None,
        cutoff: Optional[int] = None
    ) -> None:
        super().__init__()
        if state == 'vac':
            if nmode is None:
                nmode = 1
            cov = torch.eye(2 * nmode) * dqp.hbar / (4 * dqp.kappa ** 2)
            mean = torch.zeros(2 * nmode, 1) + 0j
            weight = torch.tensor([1]) + 0j
        elif isinstance(state, list):
            cov = state[0]
            mean = state[1]
            weight = state[2]
            if not isinstance(cov, torch.Tensor):
                cov = torch.tensor(cov, dtype=torch.float)
            if not isinstance(mean, torch.Tensor):
                mean = torch.tensor(mean, dtype=torch.cfloat)
            if not isinstance(weight, torch.Tensor):
                weight = torch.tensor(weight, dtype=torch.cfloat)
            if nmode is None:
                nmode = cov.shape[-1] // 2
        ncomb = weight.shape[-1]
        if cov.ndim == 2:
            cov = cov.reshape(1, 1, 2 * nmode, 2 * nmode)
        elif cov.ndim == 3:
            cov = cov.reshape(-1, ncomb, 2 * nmode, 2 * nmode)
        if mean.ndim == 2:
            if mean.shape[-1] == 1:
                mean = mean.reshape(1, 1, 2 * nmode, 1)
            elif mean.shape[0] == ncomb:
                mean = mean.reshape(1, ncomb, 2 * nmode, 1)
        elif mean.ndim == 3:
            mean = mean.reshape(-1, ncomb, 2 * nmode, 1)
        weight = weight.reshape(-1, ncomb)
        assert cov.ndim == mean.ndim == 4
        assert cov.shape[0] == mean.shape[0]
        assert cov.shape[-2] == cov.shape[-1] == 2 * nmode, (
            'The shape of the covariance matrix should be (2*nmode, 2*nmode)')
        assert mean.shape[-2] == 2 * nmode, 'The length of the mean vector should be 2*nmode'
        self.register_buffer('cov', cov)
        self.register_buffer('mean', mean)
        self.register_buffer('weight', weight)
        self.nmode = nmode
        if cutoff is None:
            cutoff = 5
        self.cutoff = cutoff

    def to(self, arg: Any) -> 'BosonicState':
        """Set dtype or device of the ``BosonicState``."""
        if arg == torch.float:
            self.cov = self.cov.to(arg)
            self.mean = self.mean.to(torch.cfloat)
            self.weight = self.weight.to(torch.cfloat)
        elif arg == torch.double:
            self.cov = self.cov.to(arg)
            self.mean = self.mean.to(torch.cdouble)
            self.weight = self.weight.to(torch.cdouble)
        else:
            self.cov = self.cov.to(arg)
            self.mean = self.mean.to(arg)
            self.weight = self.weight.to(arg)
        return self

    def tensor_product(self, state: 'BosonicState') -> 'BosonicState':
        """Get the tensor product of two Bosonic states."""
        return combine_bosonic_states([self, state])

    def wigner(self, wire: int, qvec: torch.Tensor, pvec: torch.Tensor, plot: bool = False, k: int = 0):
        r"""Get the discretized Wigner function of the specified mode.

        Args:
            wire (int): The wigner function for given wire.
            qvec (torch.Tensor): The discrete values for quadrature q.
            pvec (torch.Tensor): The discrete values for quadrature p.
            plot (bool, optional): Whether to plot the wigner function. Default: ``False``
            k (int, optional): The wigner function of kth batch to plot. Default: 0
        """
        grid_x, grid_y = torch.meshgrid(qvec, pvec, indexing='ij')
        coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)]).mT
        coords2 = coords.unsqueeze(1).unsqueeze(2) # (npoints, 1, 1, 2)
        coords3 = coords.unsqueeze(-1).unsqueeze(-3)
        if not isinstance(wire, torch.Tensor):
            wire = torch.tensor(wire).reshape(1)
        idx = torch.cat([wire, wire + self.nmode]) # xxpp order
        cov  = self.cov[..., idx[:, None], idx]
        mean = self.mean[..., idx, :]
        gauss_b = MultivariateNormal(mean.squeeze(-1).real, cov) # mean shape: (batch, ncomb, 2)
        prob_g = gauss_b.log_prob(coords2).exp() # (npoints, batch, ncomb)
        exp_real = torch.exp(mean.imag.mT @ torch.linalg.solve(cov, mean.imag) / 2).squeeze(-2, -1) # (batch, ncomb)
        # (batch, npoints, ncomb)
        exp_imag = torch.exp((coords3 - mean.real.unsqueeze(1)).mT @
                             torch.linalg.solve(cov, mean.imag).unsqueeze(1) * 1j).squeeze()
        wigner_vals = exp_real.unsqueeze(-2) * prob_g.permute(1, 0, 2) * exp_imag * self.weight.unsqueeze(-2)
        wigner_vals = wigner_vals.sum(dim=2).reshape(-1, len(qvec), len(pvec))
        if plot:
            plt.subplots(1, 1, figsize=(12, 10))
            plt.xlabel('Quadrature q')
            plt.ylabel('Quadrature p')
            plt.contourf(grid_x.cpu(), grid_y.cpu(), wigner_vals[k].cpu(), 60, cmap=cm.RdBu)
            plt.colorbar()
            plt.show()
        return wigner_vals

    def marginal(self, wire: int, qvec: torch.Tensor, phi: float = 0., plot: bool = False, k: int = 0):
        r"""Get the discretized marginal distribution of the specified mode along :math:`x\cos\phi + p\sin\phi`.

        Args:
            wire (int): The marginal function for given wire.
            qvec (torch.Tensor): The discrete values for quadrature q.
            phi (float): The angle used to compute the linear combination of quadratures.
            plot (bool, optional): Whether to plot the marginal function. Default: ``False``
            k (int, optional): The marginal function of kth batch to plot. Default: 0
        """
        if not isinstance(wire, torch.Tensor):
            wire = torch.tensor(wire).reshape(1)
        idx = torch.cat([wire, wire + self.nmode]) # xxpp order
        cov  = self.cov[..., idx[:, None], idx]
        mean = self.mean[..., idx, :]
        r = PhaseShift(inputs=-phi, nmode=1, wires=wire.tolist(), cutoff=self.cutoff)
        r.to(cov.dtype).to(cov.device)
        cov, mean = r([cov, mean]) # (batch, ncomb, 2, 2)
        cov = cov[..., 0, 0].unsqueeze(1)
        mean = mean[..., 0, 0].unsqueeze(1)
        prefactor = 1 / (torch.sqrt(2 * torch.pi * cov)) # (batch, 1, ncomb)
        # (batch, npoints, ncomb)
        marginal_vals = self.weight.unsqueeze(1) * prefactor * torch.exp(-0.5 * (qvec.reshape(-1, 1) - mean)**2 / cov)
        marginal_vals = marginal_vals.sum(2) # (batch, npoints)
        if plot:
            plt.subplots(1, 1, figsize=(12, 10))
            plt.xlabel('Quadrature q')
            plt.ylabel('Wave_function')
            plt.plot(qvec.cpu(), marginal_vals[k].cpu())
            plt.show()
        return marginal_vals


class CatState(BosonicState):
    r"""Single-mode cat state.

    The cat state is a superposition of coherent states.

    See https://arxiv.org/abs/2103.05530 Section IV B.

    Args:
        r (Any, optional): Displacement magnitude :math:`|r|`. Default: ``None``
        theta (Any, optional): Displacement angle :math:`\theta`. Default: ``None``
        p (int, optional): Parity, where :math:`\theta=p\pi`. ``p=0`` corresponds to an even
            cat state, and ``p=1`` an odd cat state. Default: 1
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
    """
    def __init__(self, r: Any = None, theta: Any = None, p: int = 1, cutoff: Optional[int] = None) -> None:
        nmode = 1
        covs = torch.eye(2) * dqp.hbar / (4 * dqp.kappa**2)
        if r is None:
            r = torch.rand(1)[0]
        if theta is None:
            theta = torch.rand(1)[0] * 2 * torch.pi
        if not isinstance(r, torch.Tensor):
            r = torch.tensor(r, dtype=torch.float)
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, dtype=torch.float)
        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, dtype=torch.long)
        real_part = r * torch.cos(theta)
        imag_part = r * torch.sin(theta)
        means = torch.stack([torch.stack([real_part, imag_part]),
                            -torch.stack([real_part, imag_part]),
                             torch.stack([imag_part, -real_part]) * 1j,
                            -torch.stack([imag_part, -real_part]) * 1j]) * dqp.hbar**0.5 / dqp.kappa
        temp = torch.exp(-2 * r**2)
        w0 = 0.5 / (1 + temp * torch.cos(p * torch.pi))
        w1 = w0
        w2 = torch.exp(-1j * torch.pi * p) * temp * w0
        w3 = torch.exp(1j * torch.pi * p) * temp * w0
        weights = torch.stack([w0, w1, w2, w3])
        state = [covs, means, weights]
        super().__init__(state, nmode, cutoff)


class GKPState(BosonicState):
    r"""Finite-energy single-mode GKP state.

    Using GKP states to encode qubits, with the qubit state defined by:
    :math:`\ket{\psi}_{gkp} = \cos\frac{\theta}{2}\ket{0}_{gkp} + e^{-i\phi}\sin\frac{\theta}{2}\ket{1}_{gkp}`

    See https://arxiv.org/abs/2103.05530 Section IV A.

    Args:
        theta (Any, optional): angle :math:`\theta` in Bloch sphere. Default: ``None``
        phi (Any, optional): angle :math:`\phi` in Bloch sphere. Default: ``None``
        amp_cutoff (float, optional): The amplitude threshold for keeping the terms. Default: 0.1
        epsilon (float, optional): The finite energy damping parameter. Default: 0.05
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
    """
    def __init__(
        self,
        theta: Any = None,
        phi: Any = None,
        amp_cutoff: float = 0.1,
        epsilon: float = 0.05,
        cutoff: Optional[int] = None
    ) -> None:
        nmode = 1
        if theta is None:
            theta = torch.rand(1)[0] * 2 * torch.pi
        if phi is None:
            phi = torch.rand(1)[0] * 2 * torch.pi
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, dtype=torch.float)
        if not isinstance(phi, torch.Tensor):
            phi = torch.tensor(phi, dtype=torch.float)
        if not isinstance(epsilon, torch.Tensor):
            epsilon = torch.tensor(epsilon, dtype=torch.float)
        if not isinstance(amp_cutoff, torch.Tensor):
            amp_cutoff = torch.tensor(amp_cutoff, dtype=torch.float)
        self.epsilon = epsilon
        self.amp_cutoff = amp_cutoff
        exp_eps = torch.exp(-2 * epsilon)
        # gaussian envelope
        z_max = torch.ceil(torch.sqrt(-4 / torch.pi * torch.log(amp_cutoff) * (1 + exp_eps) / (1 - exp_eps)))
        coords = torch.arange(-z_max, z_max + 1)
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing='ij')
        means = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

        k = means[:, 0]
        l = means[:, 1]
        thetas = torch.tensor([theta] * len(k))
        phis = torch.tensor([phi] * len(k))
        weights = self._update_weight(k, l, thetas, phis)

        filt = abs(weights) > amp_cutoff
        weights = weights[filt] + 0j
        weights /= torch.sum(weights)
        means = means[filt]
        means = means * torch.exp(-epsilon) / (1 + exp_eps) * (torch.pi * dqp.hbar / 2)**0.5 / dqp.kappa + 0j
        covs = torch.eye(2) * dqp.hbar / (4 * dqp.kappa**2) * (1 - exp_eps) / (1 + exp_eps)
        state = [covs, means, weights]
        super().__init__(state, nmode, cutoff)

    def _update_weight(self, k, l, theta, phi):
        """Compute the updated coefficients c_{k, l}(theta, phi) for given k, l, theta, and phi.

        See https://arxiv.org/abs/2103.05530 Eq.(43) and (B1)
        """
        # Ensure that k and l are integers
        k = k.long()
        l = l.long()

        k_mod_2 = k % 2
        l_mod_2 = l % 2
        k_mod_4 = k % 4
        l_mod_4 = l % 4

        result = torch.zeros_like(theta)

        # Case 1: k mod 2 == 0 and l mod 2 == 0
        mask1 = (k_mod_2 == 0) & (l_mod_2 == 0)
        result[mask1] = 1

        # Case 2: k mod 4 == 0 and l mod 2 == 1
        mask2 = (k_mod_4 == 0) & (l_mod_2 == 1)
        result[mask2] = torch.cos(theta[mask2])

        # Case 3: k mod 4 == 2 and l mod 2 == 1
        mask3 = (k_mod_4 == 2) & (l_mod_2 == 1)
        result[mask3] = -torch.cos(theta[mask3])

        # Case 4: k mod 4 == 3 and l mod 4 == 0
        mask4_1 = (k_mod_4 == 3) & (l_mod_4 == 0)
        result[mask4_1] = torch.sin(theta[mask4_1]) * torch.cos(phi[mask4_1])

        # Case 5: k mod 4 == 1 and l mod 4 == 0
        mask4_2 = (k_mod_4 == 1) & (l_mod_4 == 0)
        result[mask4_2] = torch.sin(theta[mask4_2]) * torch.cos(phi[mask4_2])

        # Case 6: k mod 4 == 3 and l mod 4 == 2
        mask4_3 = (k_mod_4 == 3) & (l_mod_4 == 2)
        result[mask4_3] = -torch.sin(theta[mask4_3]) * torch.cos(phi[mask4_3])

        # Case 7: k mod 4 == 1 and l mod 4 == 2
        mask4_4 = (k_mod_4 == 1) & (l_mod_4 == 2)
        result[mask4_4] = -torch.sin(theta[mask4_4]) * torch.cos(phi[mask4_4])

        # Case 8: k mod 4 == 3 and l mod 4 == 3
        mask5_1 = (k_mod_4 == 3) & (l_mod_4 == 3)
        result[mask5_1] = -torch.sin(theta[mask5_1]) * torch.sin(phi[mask5_1])

        # Case 9: k mod 4 == 1 and l mod 4 == 1
        mask5_2 = (k_mod_4 == 1) & (l_mod_4 == 1)
        result[mask5_2] = -torch.sin(theta[mask5_2]) * torch.sin(phi[mask5_2])

        # Case 10: k mod 4 == 3 and l mod 4 == 1
        mask5_3 = (k_mod_4 == 3) & (l_mod_4 == 1)
        result[mask5_3] = torch.sin(theta[mask5_3]) * torch.sin(phi[mask5_3])

        # Case 11: k mod 4 == 1 and l mod 4 == 3
        mask5_4 = (k_mod_4 == 1) & (l_mod_4 == 3)
        result[mask5_4] = torch.sin(theta[mask5_4]) * torch.sin(phi[mask5_4])

        exp_eps = torch.exp(-2 * self.epsilon)
        prefactor = torch.exp(-0.25 * torch.pi * (l**2 + k**2) * (1 - exp_eps)/(1 + exp_eps))

        weight = result * prefactor # update coefficient
        return weight


class FockStateBosonic(BosonicState):
    """Single-mode Fock state, representing by a linear combination of Gaussian states.

    See https://arxiv.org/abs/2103.05530 Section IV C.

    Args:
        n (int): Particle number.
        r (Any, optional): The quality parameter for the approximation. Default: 0.05
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
    """
    def __init__(self, n: int, r: Any = 0.05, cutoff: Optional[int] = None) -> None:
        if not isinstance(r, torch.Tensor):
            r = torch.tensor(r, dtype=torch.float)
        assert r ** 2 < 1 / n, 'NOT a physical state'
        nmode = 1
        m = np.arange(n + 1)
        combs = torch.tensor(comb(n, m))
        cov = torch.eye(2) * dqp.hbar / (4 * dqp.kappa**2) * (1 + (n - m) * r**2) / (1 - (n - m) * r**2)
        mean = torch.zeros([n + 1, 2]) + 0j
        weight = (-1)**(n - m) * combs * (1 - n * r**2) / (1 - (n - m) * r**2)
        weight = weight / weight.sum(-1, keepdims=True) + 0j
        state = [cov, mean, weight]
        if cutoff is None:
            cutoff = n + 1
        super().__init__(state, nmode, cutoff)


def combine_tensors(tensors: List[torch.Tensor], ndim_ds: int = 2) -> torch.Tensor:
    """Combine a list of 3D tensors for Bosonic states according to the dimension of direct sum.

    Args:
        tensors (List[torch.Tensor]): The list of 3D tensors to combine.
        ndim_ds (int, optional): The dimension of direct sum. Use 1 for direct sum along rows,
            or use 2 for direct sum along both rows and columns. Default: 2
    """
    assert ndim_ds in (1, 2)
    # Get number of tensors and their shapes
    n = len(tensors)
    shape_lst = [tensor.shape for tensor in tensors]
    len_lst, hs, ws = map(list, zip(*shape_lst))
    size_h = sum(hs)
    if ndim_ds == 1:
        size_w = ws[0]
    elif ndim_ds == 2:
        size_w = sum(ws)
    # Expand tensors to combination dimensions
    expanded_tensors = []
    for i in range(n):
        # Insert new dimensions and expand for combination
        view_shape = [1] * n + list(tensors[i].shape[1:]) # tensors[i]: (len_i, h_i, w_i)
        view_shape[i] = tensors[i].shape[0]
        expand_shape = len_lst + list(tensors[i].shape[1:])
        expanded = tensors[i].view(*view_shape).expand(*expand_shape)
        expanded_tensors.append(expanded)
    # Create zero-initialized result template (len_1, len_2, ..., len_n, size_h, size_w)
    result = tensors[0].new_zeros(*len_lst, size_h, size_w)
    # Calculate block offsets
    row_offsets = torch.cumsum(torch.tensor([0] + hs[:-1]), 0).tolist()
    if ndim_ds == 2:
        col_offsets = torch.cumsum(torch.tensor([0] + ws[:-1]), 0).tolist()
    # Place each block in corresponding position
    for i in range(n):
        h, w = hs[i], ws[i]
        row_start = row_offsets[i]
        if ndim_ds == 1:
            result[..., row_start:row_start+h, :w] = expanded_tensors[i]
        elif ndim_ds == 2:
            col_start = col_offsets[i]
            result[..., row_start:row_start+h, col_start:col_start+w] = expanded_tensors[i]
    # Flatten result to (len_1*len_2*...*len_n, size_h, size_w)
    return result.view(-1, size_h, size_w)


def combine_bosonic_states(states: List[BosonicState], cutoff: Optional[int] = None) -> BosonicState:
    """Combine multiple Bosonic states into a single state.

    Args:
        states (List[BosonicState]): List of Bosonic states to combine.
        cutoff (int or None, optional): The Fock space truncation. If ``None``, the cutoff of the first state is used.
            Default: ``None``
    """
    covs = []
    means = []
    weights = []
    nmode = 0
    if cutoff is None:
        cutoff = states[0].cutoff
    for state in states:
        covs.append(xxpp_to_xpxp(state.cov))
        means.append(xxpp_to_xpxp(state.mean))
        weights.append(state.weight)
        nmode += state.nmode
    cov = xpxp_to_xxpp(vmap(combine_tensors)(covs))
    mean = xpxp_to_xxpp(vmap(combine_tensors)(means, ndim_ds=1))
    weight = vmap(multi_kron)(weights)
    return BosonicState([cov, mean, weight], nmode, cutoff)
