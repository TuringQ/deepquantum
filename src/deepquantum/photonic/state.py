"""
Quantum states
"""

from typing import Any, List, Optional, Union
import matplotlib.pyplot as plt
from matplotlib import cm

import torch
from torch import nn

import deepquantum.photonic as dqp
from .gate import PhaseShift
from .qmath import dirac_ket, xpxp_to_xxpp

import itertools


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
        state (str or List): The Gaussian state. It can be a vacuum state with ``'vac'``, or arbitrary Gaussian states
            with [``cov``, ``mean``]. ``cov`` and ``mean`` are the covariance matrix and the displacement vector
            of the Gaussian state, respectively. Use ``xxpp`` convention and :math:`\hbar=2` by default.
            Default: ``'vac'``
        nmode (int or None, optional): The number of modes in the state. Default: ``None``
        cutoff (int, optional): The Fock space truncation. Default: 5
    """
    def __init__(
        self,
        state: Union[str, List] = 'vac',
        nmode: Optional[int] = None,
        cutoff: int = 5
    ) -> None:
        super().__init__()
        if state == 'vac':
            cov = torch.eye(2 * nmode) * dqp.hbar / (4 * dqp.kappa ** 2)
            mean = torch.zeros(2 * nmode, 1)
        elif isinstance(state, list):
            cov = state[0]
            mean = state[1]
            if not isinstance(cov, torch.Tensor):
                cov = torch.tensor(cov)
            if not isinstance(mean, torch.Tensor):
                mean = torch.tensor(mean)
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
    r"""A linear combination of Gaussian state of n modes, representing by covariance matrix, displacement vector and weight.

    Args:
        state (str or List): A linear combination of Gaussian state. It can be a vacuum state with ``'vac'``,
        or arbitrary linear combination of Gaussian states with [``cov``, ``mean``, ``weight``]. ``cov``,``mean``
        and ``weight`` are the covariance matrix, the displacement vector and combination weight
        of the Gaussian state, respectively. Use ``xxpp`` convention and :math:`\hbar=2` by default.
        Default: ``'vac'``
        nmode (int or None, optional): The number of modes in the state. Default: ``None``
        cutoff (int, optional): The Fock space truncation. Default: 5
    """
    def __init__(
        self,
        state: Union[str, List] = 'vac',
        nmode: Optional[int] = None,
        cutoff: int = 5
    ) -> None:
        super().__init__()
        if state == 'vac':
            nmode = 1
            cov = torch.eye(2 * nmode) * dqp.hbar / (4 * dqp.kappa ** 2)
            mean = torch.zeros(2 * nmode, 1)
            weight = torch.tensor([1])
        elif isinstance(state, list):
            cov = state[0]
            mean = state[1]
            weight = state[2]
        if not isinstance(cov, torch.Tensor):
            cov = torch.tensor(cov)
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight)
        if nmode is None:
            nmode = cov.shape[-1] // 2
        if weight.dim() == 1:
            n_combi = len(weight)
        if weight.dim() == 2:
            n_combi = len(weight[0])
        cov = cov.reshape(-1, n_combi, 2 * nmode, 2 * nmode)
        mean = mean.reshape(-1, n_combi, 2 * nmode, 1)
        weight = weight.reshape(-1, n_combi)
        assert cov.shape[0] == mean.shape[0] == weight.shape[0]
        assert cov.ndim == mean.ndim == 4
        assert cov.shape[-2] == cov.shape[-1] == 2 * nmode, (
            'The shape of the covariance matrix should be (2*nmode, 2*nmode)')
        assert mean.shape[-2] == 2 * nmode, 'The length of the mean vector should be 2*nmode'
        self.register_buffer('cov', cov)
        self.register_buffer('mean', mean)
        self.register_buffer('weight', weight)
        self.nmode = nmode
        self.cutoff = cutoff

    @staticmethod
    def combine_states(nmode:int, states:list):
        """
        Combine local covs, means and weights to obtain the global covs, means and weights,
        use ``xxpp`` convention and :math:`\hbar=2` by default.
        """
        def block_diag_func(mat):
            return xpxp_to_xxpp(torch.block_diag(*mat))
        def flat_func(mat):
            return mat.flatten()
        assert len(states) == nmode
        covs = [ ]
        means = [ ]
        weights = [ ]
        temp_= [ ]
        batch = states[0].weight.shape[0]
        k = 0
        for s in states:
            assert isinstance(s, BosonicState)
            covs.append(s.cov)
            means.append(s.mean)
            weights.append(s.weight)
            temp_.append(range(k, k + s.weight.shape[1]))
            k = k + s.weight.shape[1]
            temp_batch = s.weight.shape[0]
            assert batch == temp_batch, 'the inputs batch should be the same'
        covs = torch.cat(covs, dim=1)
        means = torch.cat(means, dim=1)
        weights = torch.cat(weights, dim=1)
        covs_ = [ ]
        means_ = [ ]
        weights_ = [ ]
        for i in itertools.product(*temp_):
            cov_temp = covs[:, i]
            mean_temp = means[:, i]
            cov_all = torch.vmap(block_diag_func)(cov_temp)
            mean_all = torch.vmap(flat_func)(mean_temp)
            covs_.append(cov_all)
            means_.append(mean_all)
            weights_.append(weights[:, i].prod(dim=1))
        covs_ = torch.stack(covs_).permute(1, 0, 2, 3).reshape(batch, -1, 2 * nmode, 2 * nmode)
        means_xxpp = torch.zeros_like(torch.stack(means_))
        means_xxpp[:, :, torch.arange(0, nmode)] = \
        torch.stack(means_)[:, :, torch.arange(0, 2 * nmode, 2)]
        means_xxpp[:, :, torch.arange(nmode, 2 * nmode)] = \
        torch.stack(means_)[:,:, torch.arange(1, 2 * nmode, 2)]
        means_ = means_xxpp.permute(1, 0, 2).reshape(batch, -1, 2 * nmode, 1)
        weights_ = torch.stack(weights_).mT
        return [covs_, means_, weights_]

    def wigner(self, wire, qvec, pvec, plot=False, k=0):
        r"""Calculates the discretized Wigner function of the specified mode."""

        def gaussian_func(cov, mean, x_vals):
            """Calculate gaussian function values for batched x_vals.
            """
            mean = mean.flatten()
            prefactor = 1 / torch.sqrt(torch.linalg.det(2 * torch.pi * cov))
            diff = x_vals - mean.unsqueeze(0)
            cov = cov.to(mean.dtype)
            solved = torch.linalg.solve(cov, diff.T).T
            quad = torch.sum(diff * solved, dim=1)
            f_vals = torch.exp(-0.5 * quad)
            return prefactor * f_vals

        if not isinstance(wire, torch.Tensor):
            wire = torch.tensor(wire).reshape(1)
        idx = torch.cat([wire, wire + self.nmode]) # xxpp order
        batch = self.cov.shape[0]
        wigner_vals = []
        for i in range(batch):
            cov_sub  = self.cov[i][:, idx[:, None], idx]
            mean_sub = self.mean[i][:, idx]
            weight_sub = self.weight[i]
            grid_x, grid_y = torch.meshgrid(pvec, qvec, indexing='ij')
            coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)]).mT
            vals = torch.vmap(gaussian_func, in_dims=(0, 0, None))(cov_sub, mean_sub, coords)
            weighted_vals = weight_sub.reshape(-1, 1) * vals
            wigner_vals.append(weighted_vals.sum(0).reshape(len(pvec), len(qvec)).mT)
        wigner_vals = torch.stack(wigner_vals)
        if plot:
            fig, axs = plt.subplots(1, 1, figsize=(12, 10))
            plt.xlabel('Quadrature q')
            plt.ylabel('Quadrature p')
            plt.contourf(qvec, pvec, wigner_vals[k], 60, cmap=cm.RdBu)
            plt.colorbar()
            plt.show()
        return wigner_vals

    def marginal(self, wire, qvec, phi=0., plot=False, k=0):
        r"""Calculates the discretized marginal distribution of the specified mode along
        the :math:`x\cos\phi + p\sin\phi` quadrature."""

        if not isinstance(wire, torch.Tensor):
            wire = torch.tensor(wire).reshape(1)
        idx = torch.cat([wire, wire + self.nmode]) # xxpp order
        batch = self.cov.shape[0]
        marginal_vals = []
        for i in range(batch):
            weight = self.weight[i]
            cov_sub  = self.cov[i][:, idx[:, None], idx]
            mean_sub = self.mean[i][:, idx]
            cov_sub = cov_sub.to(mean_sub.dtype)

            r = PhaseShift(inputs=-phi, nmode=1, wires=wire.tolist(), cutoff=self.cutoff)
            r.to(mean_sub.dtype).to(mean_sub.device)
            cov_out, mean_out = r([cov_sub, mean_sub])
            temp = 0
            for i, weight_i in enumerate(weight):
                prefactor = 1 / (torch.sqrt(2 * torch.pi * cov_out[i][0,0]))
                temp += weight_i * prefactor * torch.exp(-0.5 * (qvec - mean_out[i][0])**2 / cov_out[i][0,0])
            marginal_vals.append(temp)
        marginal_vals = torch.stack(marginal_vals)
        if plot:
            fig, axs = plt.subplots(1, 1, figsize=(12, 10))
            plt.xlabel('Quadrature q')
            plt.ylabel('Wave_function')
            plt.plot(qvec, marginal_vals[k])
            plt.show()
        return marginal_vals

class CatState(BosonicState):
    r"""
    Cat state for single mode, The cat state is a non-Gaussian superposition of coherent states

    see https://arxiv.org/abs/2103.05530

    Args:
        r (float): Displacement magnitude :math:`|r|`
        theta (float): Displacement angle :math:`\theta`
        p (int): Parity, where :math:`\theta=p\pi`. ``p=0`` corresponds to an even
            cat state, and ``p=1`` an odd cat state.
        cutoff (int, optional): The Fock space truncation. Default: 5
    """
    def __init__(
        self,
        r: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        p: int = 1,
        cutoff: int = 5
    ) -> None:
        nmode = 1
        covs  = torch.stack([torch.eye(2)] * 4)
        if r is None:
            r = torch.rand(1)[0]
        if theta is None:
            theta = torch.rand(1)[0] * 2 * torch.pi
        if not isinstance(r, torch.Tensor):
            r = torch.tensor(r)
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta)
        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p)
        real_part = r * torch.cos(theta)
        imag_part = r * torch.sin(theta)
        means = torch.sqrt(torch.tensor(2 * dqp.hbar)) * \
        torch.stack([torch.stack([real_part, imag_part]),
                    -torch.stack([real_part, imag_part]),
                    1j*torch.stack([imag_part, -real_part]),
                    -1j*torch.stack([imag_part, -real_part])])
        temp = torch.exp(-2 * r**2)
        w0 = 0.5 / (1 + temp * torch.cos(p * torch.pi))
        w1 = w0
        w2 = torch.exp(-1j * torch.pi * p) * temp * w0
        w3 = torch.exp(1j * torch.pi * p) * temp * w0
        weights = torch.stack([w0, w1, w2, w3])
        state = [covs, means, weights]
        super().__init__(state, nmode, cutoff)


class GKPState(BosonicState):
    r"""
    Finite energy of GKP state for single mode. Using GKP states to encode qubits, with the qubit state defined by:
    :math:`\ket{\psi}_{gkp} = \cos\frac{\theta}{2}\ket{0}_{gkp} + e^{-i\phi}\sin\frac{\theta}{2}\ket{1}_{gkp}`

    see https://arxiv.org/abs/2103.05530

    Args:
        theta (float): angle :math:`\theta` in Bloch sphere
        phi (float): angle :math:`\phi` in Bloch sphere
        amp_cutoff (float): amplitude threshold for keeping the terms. Default: 0.5
        epsilon (float): finite energy damping parameter. Default: 0.1
        cutoff (int, optional): the Fock space truncation. Default: 5
    """
    def __init__(
        self,
        theta: Optional[torch.Tensor] = None,
        phi: Optional[torch.Tensor] = None,
        amp_cutoff: float = 0.1,
        epsilon: float = 0.05,
        cutoff: int = 5
    ) -> None:
        nmode = 1
        if not isinstance(epsilon, torch.Tensor):
            epsilon = torch.tensor(epsilon)
        if not isinstance(amp_cutoff, torch.Tensor):
            amp_cutoff = torch.tensor(amp_cutoff)
        self.epsilon = epsilon
        self.amp_cutoff = amp_cutoff
        exp_eps = torch.exp(-2 * epsilon)
        # gaussian envelope
        z_max = torch.ceil(torch.sqrt(-4 / torch.pi * torch.log(amp_cutoff) * (1 + exp_eps) / (1-exp_eps)))
        coords = torch.arange(-z_max, z_max + 1)
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing='ij')
        means = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

        k = means[:, 0]
        l = means[:, 1]
        thetas = torch.tensor([theta] * len(k))
        phis = torch.tensor([phi] * len(k))
        weights = self._update_weight(k, l, thetas, phis)

        filt = abs(weights) > amp_cutoff
        weights = weights[filt]
        weights /= torch.sum(weights)
        means = means[filt]
        means = means * 2 * torch.exp(-epsilon) / (1 + exp_eps)
        means = means * 0.5 * torch.sqrt(torch.tensor(torch.pi * dqp.hbar))   # lattice spacing
        covs = torch.stack([torch.eye(2)] * len(means))
        covs = covs * 0.5 * dqp.hbar * (1 - exp_eps) / (1 + exp_eps)
        state = [covs, means, weights]
        super().__init__(state, nmode, cutoff)

    def _update_weight(self, k, l, theta, phi):
        """
        Compute the updated coefficients c_{k, l}(theta, phi) for given k, l, theta, and phi.

        see https://arxiv.org/abs/2103.05530 eq.43
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