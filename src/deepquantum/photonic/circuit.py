"""
Photonic quantum circuit
"""

import itertools
import random
from collections import defaultdict, Counter
from copy import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from thewalrus import hafnian, tor
from torch import nn, vmap
from torch.distributions.multivariate_normal import MultivariateNormal

from .decompose import UnitaryDecomposer
from .draw import DrawCircuit
from .gate import PhaseShift, BeamSplitter, MZI, BeamSplitterTheta, BeamSplitterPhi, BeamSplitterSingle, UAnyGate
from .gate import Squeezing, Displacement
from .operation import Operation, Gate
from .qmath import fock_combinations, permanent, product_factorial, sort_dict_fock_basis, sub_matrix
from .qmath import photon_number_mean_var, quadrature_to_ladder, sample_sc_mcmc
from .state import FockState, GaussianState
from ..state import MatrixProductState


class QumodeCircuit(Operation):
    r"""Photonic quantum circuit.

    Args:
        nmode (int): The number of modes in the circuit.
        init_state (Any): The initial state of the circuit. It can be a vacuum state with ``'vac'``.
            For Fock backend, it can be a Fock basis state, e.g., ``[1,0,0]``, or a Fock state tensor,
            e.g., ``[(1/2**0.5, [1,0]), (1/2**0.5, [0,1])]``. Alternatively, it can be a tensor representation.
            For Gaussian backend, it can be arbitrary Gaussian states with ``[cov, mean]``.
            Use ``xxpp`` convention and :math:`\hbar=2` by default.
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        backend (str, optional): Use ``'fock'`` for Fock backend or ``'gaussian'`` for Gaussian backend.
            Default: ``'fock'``
        basis (bool, optional): Whether to use the representation of Fock basis state for the initial state.
            Default: ``True``
        detector (str, optional): For Gaussian backend, use ``'pnrd'`` for the photon-number-resolving detector
            or ``'threshold'`` for the threshold detector. Default: ``'pnrd'``
        name (str or None, optional): The name of the circuit. Default: ``None``
        mps (bool, optional): Whether to use matrix product state representation. Default: ``False``
        chi (int or None, optional): The bond dimension for matrix product state representation.
            Default: ``None``
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """
    def __init__(
        self,
        nmode: int,
        init_state: Any,
        cutoff: Optional[int] = None,
        backend: str = 'fock',
        basis: bool = True,
        detector: str = 'pnrd',
        name: Optional[str] = None,
        mps: bool = False,
        chi: Optional[int] = None,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1
    ) -> None:
        super().__init__(name=name, nmode=nmode, wires=list(range(nmode)))
        if isinstance(init_state, (FockState, GaussianState, MatrixProductState)):
            if isinstance(init_state, MatrixProductState):
                assert nmode == init_state.nsite
                assert backend == 'fock' and not basis, 'Only support MPS for Fock backend with Fock state tensor.'
                mps = True
                chi = init_state.chi
                cutoff = init_state.qudit
            else:
                assert nmode == init_state.nmode
                mps = False
                cutoff = init_state.cutoff
                if isinstance(init_state, FockState):
                    backend = 'fock'
                    basis = init_state.basis
                elif isinstance(init_state, GaussianState):
                    backend = 'gaussian'
            self.init_state = init_state
        else:
            if mps:
                assert backend == 'fock' and not basis, 'Only support MPS for Fock backend with Fock state tensor.'
                self.init_state = MatrixProductState(nsite=nmode, state=init_state, chi=chi, qudit=cutoff,
                                                     normalize=False)
            else:
                if backend == 'fock':
                    self.init_state = FockState(state=init_state, nmode=nmode, cutoff=cutoff, basis=basis)
                elif backend == 'gaussian':
                    self.init_state = GaussianState(state=init_state, nmode=nmode, cutoff=cutoff)
                cutoff = self.init_state.cutoff

        self.operators = nn.Sequential()
        self.encoders = []
        self.cutoff = cutoff
        self.backend = backend
        self.basis = basis
        self.detector = detector.lower()
        self.mps = mps
        self.chi = chi
        self.noise = noise
        self.mu = mu
        self.sigma = sigma
        self.state = None
        self.npara = 0
        self.ndata = 0
        self.depth = np.array([0] * nmode)

    def __add__(self, rhs: 'QumodeCircuit') -> 'QumodeCircuit':
        """Addition of the ``QumodeCircuit``.

        The initial state is the same as the first ``QumodeCircuit``.
        """
        assert self.nmode == rhs.nmode
        cir = QumodeCircuit(nmode=self.nmode, init_state=self.init_state, cutoff=self.cutoff, basis=self.basis,
                            name=self.name, mps=self.mps, chi=self.chi, noise=self.noise, mu=self.mu, sigma=self.sigma)
        cir.operators = self.operators + rhs.operators
        cir.encoders = self.encoders + rhs.encoders
        cir.npara = self.npara + rhs.npara
        cir.ndata = self.ndata + rhs.ndata
        cir.depth = self.depth + rhs.depth
        return cir

    def to(self, arg: Any) -> 'QumodeCircuit':
        """Set dtype or device of the ``QumodeCircuit``."""
        if arg == torch.float:
            if self.backend == 'fock' and not self.basis:
                self.init_state.to(torch.cfloat)
            elif self.backend == 'gaussian':
                self.init_state.to(torch.float)
            for op in self.operators:
                if op.npara == 0:
                    op.to(torch.cfloat)
                elif op.npara > 0:
                    op.to(torch.float)
        elif arg == torch.double:
            if self.backend == 'fock' and not self.basis:
                self.init_state.to(torch.cdouble)
            elif self.backend == 'gaussian':
                self.init_state.to(torch.double)
            for op in self.operators:
                if op.npara == 0:
                    op.to(torch.cdouble)
                elif op.npara > 0:
                    op.to(torch.double)
        else:
            self.init_state.to(arg)
            self.operators.to(arg)
        return self

    # pylint: disable=arguments-renamed
    def forward(
        self,
        data: Optional[torch.Tensor] = None,
        state: Any = None,
        is_prob: bool = False,
        detector: Optional[str] = None,
        stepwise: bool = False
    ) -> Union[torch.Tensor, Dict, List[torch.Tensor]]:
        """Perform a forward pass of the photonic quantum circuit and return the final state.

        Args:
            data (torch.Tensor or None, optional): The input data for the ``encoders``. Default: ``None``
            state (Any, optional): The initial state for the photonic quantum circuit. Default: ``None``
            is_prob (bool, optional): Whether to return probabilities for Fock basis states or Gaussian backend.
                Default: ``False``
            detector (str or None, optional): For Gaussian backend, use ``'pnrd'`` for the photon-number-resolving
                detector or ``'threshold'`` for the threshold detector. Default: ``None``
            stepwise (bool, optional): Whether to use the forward function of each operator for Gaussian backend.
                Default: ``False``

        Returns:
            Union[torch.Tensor, Dict, List[torch.Tensor]]: The final state of the photonic quantum circuit after
            applying the ``operators``.
        """
        if self.backend == 'fock':
            return self._forward_fock(data, state, is_prob)
        elif self.backend == 'gaussian':
            return self._forward_gaussian(data, state, is_prob, detector, stepwise)

    def _forward_fock(
        self,
        data: Optional[torch.Tensor] = None,
        state: Any = None,
        is_prob: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor], Dict]:
        """Perform a forward pass based on the Fock backend.

        Args:
            data (torch.Tensor or None, optional): The input data for the ``encoders``. Default: ``None``
            state (Any, optional): The initial state for the photonic quantum circuit. Default: ``None``
            is_prob (bool, optional): Whether to return probabilities for Fock basis states. Default: ``False``

        Returns:
            Union[torch.Tensor, Dict]: The final state of the photonic quantum circuit after
            applying the ``operators``.
        """
        if state is None:
            state = self.init_state
        if isinstance(state, MatrixProductState):
            assert not self.basis
            state = state.tensors
        elif isinstance(state, FockState):
            state = state.state
        elif not isinstance(state, torch.Tensor):
            state = FockState(state=state, nmode=self.nmode, cutoff=self.cutoff, basis=self.basis).state
        if data is None:
            if self.basis:
                state_dict = self._forward_helper_basis(state=state, is_prob=is_prob)
                self.state = sort_dict_fock_basis(state_dict)
            else:
                self.state = self._forward_helper_tensor(state=state)
                if not self.mps and self.state.ndim == self.nmode:
                    self.state = self.state.unsqueeze(0)
        else:
            if data.ndim == 1:
                data = data.unsqueeze(0)
            assert data.ndim == 2
            if self.basis:
                state_dict = vmap(self._forward_helper_basis, in_dims=(0, None, None))(data, state, is_prob)
                self.state = sort_dict_fock_basis(state_dict)
            else:
                if self.mps:
                    assert state[0].ndim in (3, 4)
                    if state[0].ndim == 3:
                        self.state = vmap(self._forward_helper_tensor, in_dims=(0, None))(data, state)
                    elif state[0].ndim == 4:
                        self.state = vmap(self._forward_helper_tensor)(data, state)
                else:
                    if state.shape[0] == 1:
                        self.state = vmap(self._forward_helper_tensor, in_dims=(0, None))(data, state)
                    else:
                        self.state = vmap(self._forward_helper_tensor)(data, state)
            # for plotting the last data
            self.encode(data[-1])
        return self.state

    def _forward_helper_basis(
        self,
        data: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        is_prob: bool = False
    ) -> Dict:
        """Perform a forward pass for one sample if the input is a Fock basis state."""
        self.encode(data)
        if state is None:
            state = self.init_state.state
        out_dict = {}
        final_states = self._get_all_fock_basis(state)
        sub_mats = self._get_sub_matrices(state, final_states)
        per_norms = self._get_permanent_norms(state, final_states)
        if is_prob:
            rst = vmap(self._get_prob_fock_vmap)(sub_mats, per_norms)
        else:
            rst = vmap(self._get_amplitude_fock_vmap)(sub_mats, per_norms)
        for i in range(len(final_states)):
            final_state = FockState(state=final_states[i], nmode=self.nmode, cutoff=self.cutoff, basis=self.basis)
            out_dict[final_state] = rst[i]
        return out_dict

    def _forward_helper_tensor(
        self,
        data: Optional[torch.Tensor] = None,
        state: Union[torch.Tensor, List[torch.Tensor], MatrixProductState, None] = None
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Perform a forward pass for one sample if the input is a Fock state tensor."""
        self.encode(data)
        if state is None:
            state = self.init_state
        if self.mps:
            if not isinstance(state, MatrixProductState):
                state = MatrixProductState(nsite=self.nmode, state=state, chi=self.chi, qudit=self.cutoff,
                                           normalize=self.init_state.normalize)
            return self.operators(state).tensors
        if isinstance(state, FockState):
            state = state.state
        x = self.operators(self.tensor_rep(state)).squeeze(0)
        return x

    def _forward_gaussian(
        self,
        data: Optional[torch.Tensor] = None,
        state: Any = None,
        is_prob: bool = False,
        detector: Optional[str] = None,
        stepwise: bool = False
    ) -> List[torch.Tensor]:
        """Perform a forward pass based on the Gaussian backend.

        Args:
            data (torch.Tensor or None, optional): The input data for the ``encoders``. Default: ``None``
            state (Any, optional): The initial state for the photonic quantum circuit. Default: ``None``
            is_prob (bool, optional): Whether to return probabilities. Default: ``False``
            detector (str or None, optional): Use ``'pnrd'`` for the photon-number-resolving detector or
                ``'threshold'`` for the threshold detector. Only valid when ``is_prob`` is ``True``.
                Default: ``None``
            stepwise (bool, optional): Whether to use the forward function of each operator. Default: ``False``

        Returns:
            List[torch.Tensor]: The covariance matrix and displacement vector of the final state
            of the photonic quantum circuit after applying the ``operators``.
        """
        if state is None:
            state = self.init_state
        elif not isinstance(state, GaussianState):
            state = GaussianState(state=state, nmode=self.nmode, cutoff=self.cutoff)
        state = [state.cov, state.mean]
        if data is None:
            self.state = self._forward_helper_gaussian(state=state, stepwise=stepwise)
            if self.state[0].ndim == 2:
                self.state[0] = self.state[0].unsqueeze(0)
            if self.state[1].ndim == 2:
                self.state[1] = self.state[1].unsqueeze(0)
        else:
            if data.ndim == 1:
                data = data.unsqueeze(0)
            assert data.ndim == 2
            if state[0].shape[0] == 1:
                self.state = vmap(self._forward_helper_gaussian, in_dims=(0, None, None))(data, state, stepwise)
            else:
                self.state = vmap(self._forward_helper_gaussian, in_dims=(0, 0, None))(data, state, stepwise)
            self.encode(data[-1])
        if is_prob:
            return self._forward_gaussian_prob(detector)
        else:
            return self.state

    def _forward_helper_gaussian(
        self,
        data: Optional[torch.Tensor] = None,
        state: Optional[List[torch.Tensor]] = None,
        stepwise: bool = False
    ) -> List[torch.Tensor]:
        """Perform a forward pass for one sample if the input is a Gaussian state."""
        self.encode(data)
        if state is None:
            cov = self.init_state.cov
            mean = self.init_state.mean
        else:
            cov, mean = state
        if stepwise:
            self.state = self.operators([cov, mean])
        else:
            sp_mat = self.get_symplectic()
            cov = sp_mat @ cov @ sp_mat.mT
            mean = self.get_displacement(mean)
            self.state = [cov.squeeze(0), mean.squeeze(0)]
        return self.state

    def _forward_gaussian_prob(self, detector: Optional[str] = None) -> Dict:
        """Get the probabilities of all possible final states for Gaussian backend by different detectors.

        Args:
            detector (str or None, optional): Use ``'pnrd'`` for the photon-number-resolving detector or
                ``'threshold'`` for the threshold detector. Default: ``None``
        """
        cov, mean = self.state
        batch = cov.shape[0]
        if detector is None:
            detector = self.detector
        else:
            detector = detector.lower()
            self.detector = detector
        if detector == 'pnrd':
            odd_basis, even_basis = self._get_odd_even_fock_basis()
            final_states = even_basis
            final_states_d = torch.cat([even_basis, odd_basis])
            keys = list(map(FockState, final_states_d.tolist()))
        elif detector == 'threshold':
            final_states = torch.tensor(list(itertools.product(range(2), repeat=self.nmode)))
            keys = list(map(FockState, final_states.tolist()))
        probs = []
        for i in range(batch):
            if detector == 'pnrd':
                if not torch.allclose(mean[i], torch.zeros_like(mean[i])):
                    probs_i = self._get_probs_gaussian_helper(final_states_d, cov[i], mean[i], detector)
                else:
                    probs_i = self._get_probs_gaussian_helper(final_states, cov[i], mean[i], detector)
                    probs_i = torch.cat([probs_i.squeeze(), torch.zeros(len(odd_basis))])
            elif detector == 'threshold':
                probs_i = self._get_probs_gaussian_helper(final_states, cov[i], mean[i], detector)
            probs.append(probs_i.squeeze())
        return dict(zip(keys, torch.stack(probs).mT))

    def  _get_odd_even_fock_basis(self):
        """Split the fock basis into the odd and even photon number parts."""
        max_photon = self.nmode * (self.cutoff - 1)
        odd_lst = []
        even_lst = []
        for i in range(0, max_photon + 1):
            state_tmp = torch.tensor([i] + [0] * (self.nmode - 1), dtype=torch.int)
            temp_basis = self._get_all_fock_basis(state_tmp)
            if i % 2 == 0:
                even_lst.append(temp_basis)
            else:
                odd_lst.append(temp_basis)
        return torch.cat(odd_lst), torch.cat(even_lst)

    def encode(self, data: Optional[torch.Tensor]) -> None:
        """Encode the input data into the photonic quantum circuit parameters.

        This method iterates over the ``encoders`` of the circuit and initializes their parameters
        with the input data.

        Args:
            data (torch.Tensor or None): The input data for the ``encoders``, must be a 1D tensor.
        """
        if data is None:
            return
        assert len(data) >= self.ndata
        count = 0
        for op in self.encoders:
            count_up = count + op.npara
            op.init_para(data[count:count_up])
            count = count_up

    def get_unitary(self) -> torch.Tensor:
        """Get the unitary matrix of the photonic quantum circuit."""
        u = None
        for op in self.operators:
            if u is None:
                u = op.get_unitary()
            else:
                u = op.get_unitary() @ u
        if u is None:
            return torch.eye(self.nmode, dtype=torch.cfloat)
        else:
            return u

    def get_symplectic(self) -> torch.Tensor:
        """Get the symplectic matrix of the photonic quantum circuit."""
        s = None
        for op in self.operators:
            if s is None:
                s = op.get_symplectic()
            else:
                s = op.get_symplectic() @ s
        return s

    def get_displacement(self, init_mean: Any) -> torch.Tensor:
        """Get the final mean value of the Gaussian state in ``xxpp`` order."""
        if not isinstance(init_mean, torch.Tensor):
            init_mean = torch.tensor(init_mean)
        mean = init_mean.reshape(-1, 2 * self.nmode, 1)
        for op in self.operators:
            mean = op.get_symplectic() @ mean + op.get_displacement()
        return mean

    def _get_all_fock_basis(self, init_state: torch.Tensor) -> torch.Tensor:
        """Get all possible fock basis states according to the initial state."""
        nmode = len(init_state)
        nphoton = int(sum(init_state))
        states = torch.tensor(fock_combinations(nmode, nphoton), dtype=torch.int, device=init_state.device)
        max_values, _ = torch.max(states, dim=1)
        mask = max_values < self.cutoff
        return torch.masked_select(states, mask.unsqueeze(1)).view(-1, states.shape[-1])

    def _get_sub_matrices(self, init_state: torch.Tensor, final_states: torch.Tensor) -> torch.Tensor:
        """Get the sub-matrices for permanent."""
        sub_mats = []
        u = self.get_unitary()
        for state in final_states:
            sub_mats.append(sub_matrix(u, init_state, state))
        return torch.stack(sub_mats)

    def _get_permanent_norms(self, init_state: torch.Tensor, final_states: torch.Tensor) -> torch.Tensor:
        """Get the normalization factors for permanent."""
        return torch.sqrt(product_factorial(init_state) * product_factorial(final_states))

    def get_amplitude(self, final_state: Any, init_state: Optional[FockState] = None) -> torch.Tensor:
        """Get the transfer amplitude between the final state and the initial state.

        Args:
            final_state (Any): The final Fock basis state.
            init_state (FockState or None, optional): The initial Fock basis state. Default: ``None``
        """
        assert self.backend == 'fock'
        if not isinstance(final_state, torch.Tensor):
            final_state = torch.tensor(final_state, dtype=torch.int)
        if init_state is None:
            init_state = self.init_state
        assert init_state.basis, 'The initial state must be a Fock basis state'
        assert max(final_state) < self.cutoff, 'The number of photons in the final state must be less than cutoff'
        assert sum(final_state) == sum(init_state.state), 'The number of photons should be conserved'
        u = self.get_unitary()
        sub_mat = sub_matrix(u, init_state.state, final_state)
        nphoton = sum(init_state.state)
        if nphoton == 0:
            amp = torch.tensor(1.)
        else:
            per = permanent(sub_mat)
            amp = per / self._get_permanent_norms(init_state.state, final_state).to(per.dtype).to(per.device)
        return amp

    def _get_amplitude_fock_vmap(self, sub_mat: torch.Tensor, per_norm: torch.Tensor) -> torch.Tensor:
        """Get the transfer amplitude."""
        per = permanent(sub_mat)
        amp = per / per_norm.to(per.dtype).to(per.device)
        return amp.reshape(-1)

    def get_prob(
        self,
        final_state: Any,
        refer_state: Union[FockState, GaussianState, None] = None
    ) -> torch.Tensor:
        """Get the probability of the final state related to the reference state.

        Args:
            final_state (Any): The final Fock basis state.
            refer_state (FockState or GaussianState or None, optional): The initial Fock basis state or
                the final Gaussian state. Default: ``None``
        """
        if self.backend == 'fock':
            return self._get_prob_fock(final_state, refer_state)
        elif self.backend == 'gaussian':
            return self._get_prob_gaussian(final_state, refer_state)

    def _get_prob_fock(self, final_state: Any, init_state: Optional[FockState] = None) -> torch.Tensor:
        """Get the transfer probability between the final state and the initial state for the Fock backend.

        Args:
            final_state (Any): The final Fock basis state.
            init_state (FockState or None, optional): The initial Fock basis state. Default: ``None``
        """
        amplitude = self.get_amplitude(final_state, init_state)
        prob = torch.abs(amplitude) ** 2
        return prob

    def _get_prob_fock_vmap(self, sub_mat: torch.Tensor, per_norm: torch.Tensor) -> torch.Tensor:
        """Get the transfer probability."""
        amplitude = self._get_amplitude_fock_vmap(sub_mat, per_norm)
        prob = torch.abs(amplitude) ** 2
        return prob

    def _get_prob_gaussian(self, final_state: Any, state: Optional[GaussianState] = None) -> torch.Tensor:
        """Get the batched probabilities of the final state for Gaussian backend."""
        if not isinstance(final_state, torch.Tensor):
            final_state = torch.tensor(final_state, dtype=torch.int)
        if state is None:
            cov, mean = self.state
        else:
            cov = state.cov
            mean = state.mean
        if cov.ndim == 2:
            cov = cov.unsqueeze(0)
        assert cov.ndim == 3
        batch = cov.shape[0]
        probs = []
        for i in range(batch):
            prob = self._get_probs_gaussian_helper(final_state, cov=cov[i], mean=mean[i], detector=self.detector)[0]
            probs.append(prob)
        return torch.stack(probs).squeeze()

    def _get_probs_gaussian_helper(
        self,
        final_states: torch.Tensor,
        cov: torch.Tensor,
        mean: torch.tensor,
        detector: str = 'pnrd'
    ) -> torch.Tensor:
        """Get the probabilities of the final states for Gaussian backend."""
        if final_states.ndim == 1:
            final_states = final_states.unsqueeze(0)
        assert final_states.ndim == 2
        nmode = final_states.shape[-1]
        identity = torch.eye(nmode, dtype=cov.dtype)
        cov_ladder = quadrature_to_ladder(cov)
        mean_ladder = quadrature_to_ladder(mean)
        q = cov_ladder + torch.eye(2 * nmode) / 2
        det_q = torch.det(q)
        x_mat = torch.block_diag(identity.fliplr(), identity.fliplr()).fliplr() + 0j
        o_mat = torch.eye(2 * nmode) - torch.inverse(q)
        a_mat = x_mat @ o_mat
        gamma = mean_ladder.conj().mT @ torch.inverse(q)
        if detector == 'pnrd':
            matrix = a_mat
        elif detector == 'threshold':
            matrix = o_mat
        probs = []
        purity = GaussianState(self.state).check_purity()
        for i in range(len(final_states)):
            prob = self._get_prob_gaussian_base(final_states[i], matrix, det_q, mean_ladder, gamma, detector, purity)
            probs.append(prob)
        return torch.stack(probs)

    def _get_prob_gaussian_base(
        self,
        final_state: torch.Tensor,
        matrix: torch.Tensor,
        det_q: torch.Tensor,
        mean_ladder: torch.Tensor,
        gamma: torch.Tensor,
        detector: str = 'pnrd',
        purity: bool = True
    ) -> torch.Tensor:
        """Get the probability of the final state for Gaussian backend."""
        if_loop = False
        gamma = gamma.squeeze()
        nmode = len(final_state)
        if not torch.allclose(gamma, torch.zeros_like(gamma)):
            if_loop = True
            gamma_n1 = torch.cat([torch.cat([gamma[i].repeat(final_state[i])]) for i in range(nmode)])
            gamma_n2 = torch.cat([torch.cat([gamma[i + nmode].repeat(final_state[i])]) for i in range(nmode)])
            sub_gamma = torch.cat([gamma_n1, gamma_n2])
        else:
            sub_gamma = torch.tensor([0] * (2 * final_state.sum()))
        if purity and detector == 'pnrd':
            sub_mat = sub_matrix(matrix[:nmode, :nmode], final_state, final_state)
            sub_gamma = sub_gamma[:final_state.sum()]
        else:
            final_state_double = torch.cat([final_state, final_state])
            sub_mat = sub_matrix(matrix, final_state_double, final_state_double)
        sub_mat = sub_mat.cpu().numpy()
        np.fill_diagonal(sub_mat, sub_gamma) # replace all diagonal terms
        if detector == 'pnrd':
            p_vac = torch.exp(-gamma @ mean_ladder.squeeze() / 2) / torch.sqrt(det_q)
            if purity:
                haf = abs(hafnian(sub_mat, loop=if_loop, atol=1e-5)) ** 2
            else:
                haf = hafnian(sub_mat, loop=if_loop, atol=1e-5)
            prob = p_vac * haf / product_factorial(final_state)
        elif detector == 'threshold':
            assert max(final_state) < 2, 'Threshold detector with maximum 1 photon'
            if if_loop:
                raise NotImplementedError('Threshold measurement for the displaced states is NOT supported')
            else:
                prob = tor(sub_mat) / torch.sqrt(det_q)
        return abs(prob.real)

    def measure(
        self,
        shots: int = 1024,
        with_prob: bool = False,
        wires: Union[int, List[int], None] = None,
        detector: Optional[str] = None
    ) -> Union[Dict, List[Dict], None]:
        """Measure the final state.

        Args:
            shots (int, optional): The number of times to sample from the quantum state. Default: 1024
            with_prob (bool, optional): A flag that indicates whether to return the probabilities along with
                the number of occurrences. Default: ``False``
            wires (int, List[int] or None, optional): The wires to measure. It can be an integer or a list of
                integers specifying the indices of the wires. Only valid for Fock backend.
                Default: ``None`` (which means all wires are measured)
            detector (str or None, optional): For Gaussian backend, use ``'pnrd'`` for the photon-number-resolving
                detector or ``'threshold'`` for the threshold detector. Default: ``None``
        """
        assert not self.mps, 'Currently NOT supported.'
        if self.state is None:
            return
        if self.backend == 'fock':
            return self._measure_fock(shots, with_prob, wires)
        elif self.backend == 'gaussian':
            return self._measure_gaussian(shots, with_prob, detector)

    def _measure_fock(
        self,
        shots: int = 1024,
        with_prob: bool = False,
        wires: Union[int, List[int], None] = None
    ) -> Union[Dict, List[Dict]]:
        """Measure the final state for Fock backend."""
        if wires is None:
            wires = self.wires
        wires = sorted(self._convert_indices(wires))
        amp_dis = self.state
        all_results = []
        if self.basis:
            batch = len(amp_dis[list(amp_dis.keys())[0]])
            for i in range(batch):
                prob_dict = defaultdict(list)
                for key in amp_dis.keys():
                    state_b = key.state[wires]
                    state_b = FockState(state=state_b)
                    prob_dict[state_b].append(abs(amp_dis[key][i]) ** 2)
                for key in prob_dict.keys():
                    prob_dict[key] = sum(prob_dict[key])
                samples = random.choices(list(prob_dict.keys()), list(prob_dict.values()), k=shots)
                results = dict(Counter(samples))
                if with_prob:
                    for k in results:
                        results[k] = results[k], prob_dict[k]
                all_results.append(results)
        else:
            state_tensor = self.tensor_rep(amp_dis)
            batch = state_tensor.shape[0]
            combi = list(itertools.product(range(self.cutoff), repeat=len(wires)))
            for i in range(batch):
                prob_dict = {}
                state = state_tensor[i]
                probs = abs(state) ** 2
                if wires == self.wires:
                    ptrace_probs = probs
                else:
                    sum_idx = list(range(self.nmode))
                    for idx in wires:
                        sum_idx.remove(idx)
                    ptrace_probs = probs.sum(dim=sum_idx)
                for p_state in combi:
                    p_state_b = FockState(list(p_state))
                    prob_dict[p_state_b] = ptrace_probs[p_state]
                samples = random.choices(list(prob_dict.keys()), list(prob_dict.values()), k=shots)
                results = dict(Counter(samples))
                if with_prob:
                    for k in results:
                        results[k] = results[k], prob_dict[k]
                all_results.append(results)
        if batch == 1:
            return all_results[0]
        else:
            return all_results

    def _measure_gaussian(self, shots: int = 1024, with_prob: bool = False, detector: Optional[str] = None) -> Dict:
        """Measure the final state for Gaussian backend.

        See https://arxiv.org/pdf/2108.01622
        """
        if detector is None:
            detector = self.detector
        else:
            detector = detector.lower()
        cov, mean = self.state
        batch = cov.shape[0]
        all_results = []
        for i in range(batch):
            samples_i = self._sample_mcmc_gaussian(shots=shots, cov=cov[i], mean=mean[i],
                                                   detector=detector, num_chain=5)
            keys = list(map(FockState, samples_i.keys()))
            results = dict(zip(keys, samples_i.values()))
            if with_prob:
                for k in results:
                    prob = self._get_probs_gaussian_helper(k.state, cov=cov[i], mean=mean[i], detector=detector)[0]
                    results[k] = results[k], prob
            all_results.append(results)
        if batch == 1:
            return all_results[0]
        else:
            return all_results

    def _sample_mcmc_gaussian(self, shots: int, cov: torch.Tensor, mean: torch.Tensor, detector: str, num_chain: int):
        """Sample the output states for Gaussian backend via SC-MCMC method."""
        self.cov = cov
        self.mean = mean
        self.detector = detector
        if detector == 'threshold' and not torch.allclose(mean, torch.zeros_like(mean)):
            # For the displaced state, aggregate PNRD detector samples to derive threshold detector results
            self.detector = 'pnrd'
            merged_samples_pnrd = sample_sc_mcmc(prob_func=self._prob_func_gaussian,
                                                 proposal_sampler=self._proposal_sampler,
                                                 shots=shots,
                                                 num_chain=num_chain)
            merged_samples = defaultdict(int)
            for key in list(merged_samples_pnrd.keys()):
                key_threshold = (torch.tensor(key) != 0).int()
                key_threshold = tuple(key_threshold.tolist())
                merged_samples[key_threshold] += merged_samples_pnrd[key]
            self.detector = 'threshold'
        else:
            merged_samples = sample_sc_mcmc(prob_func=self._prob_func_gaussian,
                                            proposal_sampler=self._proposal_sampler,
                                            shots=shots,
                                            num_chain=num_chain)
        return merged_samples

    def _prob_func_gaussian(self, state: Any) -> torch.Tensor:
        """Get the probability of the state for Gaussian backend."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.int)
        prob = self._get_probs_gaussian_helper(state, cov=self.cov, mean=self.mean, detector=self.detector)[0]
        return prob

    def _proposal_sampler(self):
        """The proposal sampler for MCMC sampling."""
        sample = self._generate_rand_sample(self.detector)
        return sample

    def _generate_rand_sample(self, detector: str = 'pnrd'):
        """Generate random sample according to uniform proposal distribution."""
        if detector == 'threshold':
            sample = torch.randint(0, 2, [self.nmode])
        elif detector == 'pnrd':
            if torch.allclose(self.mean, torch.zeros_like(self.mean)):
                while True:
                    sample = torch.randint(0, self.cutoff, [self.nmode])
                    if sample.sum() % 2 == 0:
                        break
            else:
                sample = torch.randint(0, self.cutoff, [self.nmode])
        return sample

    def photon_number_mean_var(
        self,
        wires: Union[int, List[int], None] = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], None]:
        """Get the expectation value and variance of the photon number operator.

        Args:
            wires (int, List[int] or None, optional): The wires to measure. It can be an integer or a list of
                integers specifying the indices of the wires. Default: ``None`` (which means all wires are
                measured)
        """
        assert self.backend == 'gaussian'
        if self.state is None:
            return
        cov, mean = self.state
        if wires is None:
            wires = self.wires
        wires = sorted(self._convert_indices(wires))
        batch = cov.shape[0]
        cov_lst = [] # batch * nwire
        mean_lst = []
        for i in range(batch):
            cov_lst_i, mean_lst_i = self._get_local_covs_means(cov[i], mean[i], wires)
            cov_lst += cov_lst_i
            mean_lst += mean_lst_i
        covs = torch.stack(cov_lst)
        means = torch.stack(mean_lst)
        exp, var = photon_number_mean_var(covs, means)
        exp = exp.reshape(batch, len(wires)).squeeze()
        var = var.reshape(batch, len(wires)).squeeze()
        return exp, var

    def _get_local_covs_means(
        self,
        cov: torch.Tensor,
        mean: torch.Tensor,
        wires: List[int]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Get the local covariance matrices and mean vectors of a Gaussian state according to the wires to measure."""
        cov_lst = []
        mean_lst = []
        for wire in wires:
            indices = [wire] + [wire + self.nmode]
            cov_lst.append(cov[indices][:, indices])
            mean_lst.append(mean[indices])
        return cov_lst, mean_lst

    def measure_homodyne(
        self,
        shots: int = 1024,
        wires: Union[int, List[int], None] = None
    ) -> Union[torch.Tensor, None]:
        """Get the homodyne measurement results for quadratures x and p.

        Args:
            shots (int, optional): The number of times to sample from the quantum state. Default: 1024
            wires (int, List[int] or None, optional): The wires to measure. It can be an integer or a list of
                integers specifying the indices of the wires. Default: ``None`` (which means all wires are
                measured)
        """
        assert self.backend == 'gaussian'
        if self.state is None:
            return
        cov, mean = self.state
        if wires is None:
            wires = self.wires
        wires = sorted(self._convert_indices(wires))
        indices = wires + [wire + self.nmode for wire in wires]
        cov_sub = cov[:, indices][:, :, indices]
        mean_sub = mean[:, indices].squeeze(-1)
        samples = MultivariateNormal(mean_sub, cov_sub).sample([shots]) # (shots, batch, 2 * nwire)
        return samples.permute(1, 0, 2).squeeze()

    def draw(self, filename: Optional[str] = None):
        """Visualize the photonic quantum circuit.

        Args:
            filename (str or None, optional): The path for saving the figure.
        """
        self.draw_circuit = DrawCircuit(self.name, self.nmode, self.operators)
        self.draw_circuit.draw()
        if filename is not None:
            self.draw_circuit.save(filename)
        else:
            if self.nmode > 50:
                print('Too many modes in the circuit, please set filename to save the figure.')
        return self.draw_circuit.draw_

    def add(
        self,
        op: Operation,
        encode: bool = False,
        wires: Union[int, List[int], None] = None
    ) -> None:
        """A method that adds an operation to the photonic quantum circuit.

        The operation can be a gate or another photonic quantum circuit. The method also updates the
        attributes of the photonic quantum circuit. If ``wires`` is specified, the parameters of gates
        are shared.

        Args:
            op (Operation): The operation to add. It is an instance of ``Operation`` class or its subclasses,
                such as ``Gate``, or ``QumodeCircuit``.
            encode (bool): Whether the gate is to encode data. Default: ``False``
            wires (Union[int, List[int], None]): The wires to apply the gate on. It can be an integer
                or a list of integers specifying the indices of the wires. Default: ``None`` (which means
                the gate has its own wires)

        Raises:
            AssertionError: If the input arguments are invalid or incompatible with the quantum circuit.
        """
        assert isinstance(op, Operation)
        if wires is not None:
            assert isinstance(op, Gate)
            wires = self._convert_indices(wires)
            assert len(wires) == len(op.wires), 'Invalid input'
            op = copy(op)
            op.wires = wires
        if isinstance(op, QumodeCircuit):
            assert self.nmode == op.nmode
            self.operators += op.operators
            self.encoders  += op.encoders
            self.npara += op.npara
            self.ndata += op.ndata
            self.depth += op.depth
        else:
            self.operators.append(op)
            if isinstance(op, Gate):
                for i in op.wires:
                    self.depth[i] += 1
            if encode:
                assert not op.requires_grad, 'Please set requires_grad of the operation to be False'
                self.encoders.append(op)
                self.ndata += op.npara
            else:
                self.npara += op.npara

    def ps(
        self,
        wires: int,
        inputs: Any = None,
        encode: bool = False,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a phase shifter."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        ps = PhaseShift(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff,
                        requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
        self.add(ps, encode=encode)

    def bs(
        self,
        wires: List[int],
        inputs: Any = None,
        encode: bool = False,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a beam splitter."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        bs = BeamSplitter(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff,
                          requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
        self.add(bs, encode=encode)

    def mzi(
        self,
        wires: List[int],
        inputs: Any = None,
        phi_first: bool = True,
        encode: bool = False,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a Mach-Zehnder interferometer."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        mzi = MZI(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff, phi_first=phi_first,
                  requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
        self.add(mzi, encode=encode)

    def bs_theta(
        self,
        wires: List[int],
        inputs: Any = None,
        encode: bool = False,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        r"""Add a beam splitter with fixed :math:`\phi` at :math:`\pi/2`."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        bs = BeamSplitterTheta(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff,
                               requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
        self.add(bs, encode=encode)

    def bs_phi(
        self,
        wires: List[int],
        inputs: Any = None,
        encode: bool = False,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        r"""Add a beam splitter with fixed :math:`\theta` at :math:`\pi/4`."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        bs = BeamSplitterPhi(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff,
                             requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
        self.add(bs, encode=encode)

    def bs_rx(
        self,
        wires: List[int],
        inputs: Any = None,
        encode: bool = False,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add an Rx-type beam splitter."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        bs = BeamSplitterSingle(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff, convention='rx',
                                requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
        self.add(bs, encode=encode)

    def bs_ry(
        self,
        wires: List[int],
        inputs: Any = None,
        encode: bool = False,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add an Ry-type beam splitter."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        bs = BeamSplitterSingle(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff, convention='ry',
                                requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
        self.add(bs, encode=encode)

    def bs_h(
        self,
        wires: List[int],
        inputs: Any = None,
        encode: bool = False,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add an H-type beam splitter."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        bs = BeamSplitterSingle(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff, convention='h',
                                requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
        self.add(bs, encode=encode)

    def dc(
        self,
        wires: List[int],
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a directional coupler."""
        theta = torch.pi / 2
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        bs = BeamSplitterSingle(inputs=theta, nmode=self.nmode, wires=wires, cutoff=self.cutoff, convention='rx',
                                requires_grad=False, noise=self.noise, mu=mu, sigma=sigma)
        self.add(bs, encode=False)

    def h(
        self,
        wires: List[int],
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a photonic Hadamard gate."""
        theta = torch.pi / 2
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        bs = BeamSplitterSingle(inputs=theta, nmode=self.nmode, wires=wires, cutoff=self.cutoff, convention='h',
                                requires_grad=False, noise=self.noise, mu=mu, sigma=sigma)
        self.add(bs, encode=False)

    def any(
        self,
        unitary: Any,
        wires: Union[int, List[int], None] = None,
        minmax: Optional[List[int]] = None,
        name: str = 'uany'
    ) -> None:
        """Add an arbitrary unitary gate."""
        uany = UAnyGate(unitary=unitary, nmode=self.nmode, wires=wires, minmax=minmax, cutoff=self.cutoff,
                        name=name)
        self.add(uany)

    def clements(
        self,
        unitary: Any,
        wires: Union[int, List[int], None] = None,
        minmax: Optional[List[int]] = None,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add the Clements architecture of the unitary matrix.

        This is equivalent to ``any``, using `'cssr'`-type Clements decomposition.
        When ``basis`` is ``False``, this implementation is much faster.
        """
        if wires is None:
            if minmax is None:
                minmax = [0, self.nmode - 1]
            self._check_minmax(minmax)
            wires = list(range(minmax[0], minmax[1] + 1))
        else:
            wires = self._convert_indices(wires)
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        # clements decomposition
        ud = UnitaryDecomposer(unitary, 'cssr')
        mzi_info = ud.decomp()
        dic_mzi = mzi_info[1]
        phase_angle = mzi_info[0]['phase_angle']
        assert len(phase_angle) == len(wires), 'Please check wires'
        wires1 = wires[1::2]
        wires2 = wires[2::2]
        shift = wires[0] # clements decomposition starts from 0
        for i in range(len(wires)):
            if i % 2 == 0:
                idx = i // 2
                for j in range(len(wires1)):
                    phi, theta = dic_mzi[(wires1[j] - 1 - shift, wires1[j] - shift)][idx]
                    self.mzi(wires=[wires1[j] - 1, wires1[j]], inputs=[theta, phi], mu=mu, sigma=sigma)
            else:
                idx = (i - 1) // 2
                for j in range(len(wires2)):
                    phi, theta = dic_mzi[(wires2[j] - 1 - shift, wires2[j] - shift)][idx]
                    self.mzi(wires=[wires2[j] - 1, wires2[j]], inputs=[theta, phi], mu=mu, sigma=sigma)
        for wire in wires:
            self.ps(wires=wire, inputs=phase_angle[wire-shift], mu=mu, sigma=sigma)

    def s(
        self,
        wires: int,
        inputs: Any = None,
        encode: bool = False,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a squeezing gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        s = Squeezing(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff,
                      requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
        self.add(s, encode=encode)

    def d(
        self,
        wires: int,
        inputs: Any = None,
        encode: bool = False,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a displacement gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        d = Displacement(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff,
                         requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
        self.add(d, encode=encode)

    def r(
        self,
        wires: int,
        inputs: Any = None,
        encode: bool = False,
        inv_mode: bool = False,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a rotation gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        r = PhaseShift(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff,
                       requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma, inv_mode=inv_mode)
        self.add(r, encode=encode)
