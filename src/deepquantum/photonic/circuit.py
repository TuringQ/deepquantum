"""
Photonic quantum circuit
"""

import itertools
import random
import warnings
from collections import defaultdict, Counter
from copy import copy, deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn, vmap
from torch.distributions.multivariate_normal import MultivariateNormal

import deepquantum.photonic as dqp
from ..qmath import get_prob_mps, inner_product_mps, is_positive_definite, sample_sc_mcmc
from ..state import MatrixProductState
from .channel import PhotonLoss
from .decompose import UnitaryDecomposer
from .distributed import measure_dist
from .draw import DrawCircuit
from .gate import PhaseShift, BeamSplitter, MZI, BeamSplitterTheta, BeamSplitterPhi, BeamSplitterSingle, UAnyGate
from .gate import Squeezing, Squeezing2, Displacement, DisplacementPosition, DisplacementMomentum
from .gate import QuadraticPhase, ControlledX, ControlledZ, CubicPhase, Kerr, CrossKerr, DelayBS, DelayMZI, Barrier
from .hafnian_ import hafnian
from .measurement import Homodyne, Generaldyne
from .operation import Operation, Gate, Channel, Delay
from .qmath import fock_combinations, permanent, product_factorial, sort_dict_fock_basis, sub_matrix
from .qmath import photon_number_mean_var, measure_fock_tensor, sample_homodyne_fock, sample_reject_bosonic
from .qmath import quadrature_to_ladder, shift_func, align_shape, williamson
from .state import FockState, GaussianState, BosonicState, CatState, GKPState, DistributedFockState
from .state import combine_bosonic_states
from .torontonian_ import torontonian


class QumodeCircuit(Operation):
    r"""Photonic quantum circuit.

    Args:
        nmode (int): The number of modes in the circuit.
        init_state (Any): The initial state of the circuit. It can be a vacuum state with ``'vac'`` or ``'zeros'``.
            For Fock backend, it can be a Fock basis state, e.g., ``[1,0,0]``, or a Fock state tensor,
            e.g., ``[(1/2**0.5, [1,0]), (1/2**0.5, [0,1])]``. Alternatively, it can be a tensor representation.
            For Gaussian backend, it can be arbitrary Gaussian states with ``[cov, mean]``.
            For Bosonic backend, it can be arbitrary linear combinations of Gaussian states
            with ``[cov, mean, weight]``, or a list of local Bosonic states.
            Use ``xxpp`` convention and :math:`\hbar=2` by default.
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        backend (str, optional): Use ``'fock'`` for Fock backend, ``'gaussian'`` for Gaussian backend or
            ``'bosonic'`` for Bosonic backend. Default: ``'fock'``
        basis (bool, optional): Whether to use the representation of Fock basis state for the initial state.
            Default: ``True``
        den_mat (bool, optional): Whether to use density matrix representation. Only valid for Fock state tensor.
            Default: ``False``
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
        den_mat: bool = False,
        detector: str = 'pnrd',
        name: Optional[str] = None,
        mps: bool = False,
        chi: Optional[int] = None,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1
    ) -> None:
        super().__init__(name=name, nmode=nmode, wires=list(range(nmode)), cutoff=cutoff, den_mat=den_mat,
                         noise=noise, mu=mu, sigma=sigma)
        self.backend = backend
        self.basis = basis
        self.detector = detector.lower()
        self.mps = mps
        self.chi = chi
        self.set_init_state(init_state)
        self.operators = nn.Sequential()
        self.encoders = []
        self.measurements = nn.ModuleList()
        self.state = None
        self.state_measured = None
        self.ndata = 0
        self.depth = np.array([0] * nmode)

        self._bosonic_states = None # list of initial Bosonic states
        self._lossy = False
        self._nloss = 0
        self._is_batch_expand = False # whether batch states are expanded out of photons conservation
        self._expand_state = None # expanded state (init_state + lossy + batch expand)
        self._all_fock_basis = None
        # TDM
        self._if_delayloop = False
        self._nmode_tdm = self.nmode
        self._ntau_dict = defaultdict(list) # {wire: [tau1, tau2, ...]}
        self._unroll_dict = None # {wire_space: [wires_delay_n, ..., wires_delay_1, wire_space_concurrent]}
        self._operators_tdm = None
        self._measurements_tdm = None
        self.wires_homodyne = []

    def set_init_state(self, init_state: Any) -> None:
        """Set the initial state of the circuit."""
        if isinstance(init_state, (FockState, GaussianState, BosonicState, MatrixProductState)):
            if isinstance(init_state, MatrixProductState):
                assert self.nmode == init_state.nsite
                assert self.backend == 'fock' and not self.basis, \
                    'Only support MPS for Fock backend with Fock state tensor.'
                self.mps = True
                self.chi = init_state.chi
                self.cutoff = init_state.qudit # if self.cutoff is changed, the operators should be reset
            else:
                assert self.nmode == init_state.nmode
                self.mps = False
                self.cutoff = init_state.cutoff # if self.cutoff is changed, the operators should be reset
                if isinstance(init_state, FockState):
                    self.backend = 'fock'
                    self.basis = init_state.basis
                elif isinstance(init_state, GaussianState):
                    self.backend = 'gaussian'
                elif isinstance(init_state, BosonicState):
                    self.backend = 'bosonic'
            self.init_state = init_state
        else:
            if self.mps:
                assert self.backend == 'fock' and not self.basis, \
                    'Only support MPS for Fock backend with Fock state tensor.'
                assert self.cutoff is not None, 'Please set the cutoff.'
                self.init_state = MatrixProductState(nsite=self.nmode, state=init_state, chi=self.chi,
                                                     qudit=self.cutoff, normalize=False)
            else:
                if self.backend == 'fock':
                    self.init_state = FockState(state=init_state, nmode=self.nmode, cutoff=self.cutoff,
                                                basis=self.basis, den_mat=self.den_mat)
                elif self.backend == 'gaussian':
                    self.init_state = GaussianState(state=init_state, nmode=self.nmode, cutoff=self.cutoff)
                elif self.backend == 'bosonic':
                    if isinstance(init_state, list) and all(isinstance(s, BosonicState) for s in init_state):
                        self.init_state = combine_bosonic_states(states=init_state, cutoff=self.cutoff)
                        if self.init_state.nmode < self.nmode:
                            nmode = self.nmode - self.init_state.nmode
                            vac = BosonicState(state='vac', nmode=nmode, cutoff=self.cutoff)
                            self.init_state.tensor_product(vac)
                        assert self.init_state.nmode == self.nmode
                    else:
                        self.init_state = BosonicState(state=init_state, nmode=self.nmode, cutoff=self.cutoff)
                self.cutoff = self.init_state.cutoff

    def __add__(self, rhs: 'QumodeCircuit') -> 'QumodeCircuit':
        """Addition of the ``QumodeCircuit``.

        The initial state is the same as the first ``QumodeCircuit``.
        """
        assert self.nmode == rhs.nmode
        cir = QumodeCircuit(nmode=self.nmode, init_state=self.init_state, cutoff=self.cutoff, backend=self.backend,
                            basis=self.basis, den_mat=self.den_mat, detector=self.detector, name=self.name,
                            mps=self.mps, chi=self.chi, noise=self.noise, mu=self.mu, sigma=self.sigma)
        cir.operators = self.operators + rhs.operators
        cir.encoders = self.encoders + rhs.encoders
        cir.measurements = rhs.measurements
        cir.npara = self.npara + rhs.npara
        cir.ndata = self.ndata + rhs.ndata
        cir.depth = self.depth + rhs.depth

        cir._bosonic_states = self._bosonic_states
        cir._lossy = self._lossy or rhs._lossy
        cir._nloss = self._nloss + rhs._nloss

        cir._if_delayloop = self._if_delayloop or rhs._if_delayloop
        cir._nmode_tdm = self._nmode_tdm + rhs._nmode_tdm - self.nmode
        cir._ntau_dict = defaultdict(list)
        for key, value in self._ntau_dict.items():
            cir._ntau_dict[key].extend(value)
        for key, value in rhs._ntau_dict.items():
            cir._ntau_dict[key].extend(value)
        cir.wires_homodyne = rhs.wires_homodyne
        return cir

    def to(self, arg: Any) -> 'QumodeCircuit':
        """Set dtype or device of the ``QumodeCircuit``."""
        self.init_state.to(arg)
        if arg in (torch.float, torch.double):
            for op in self.operators:
                op.to(arg)
            for op_m in self.measurements:
                op_m.to(arg)
        else:
            self.operators.to(arg)
            self.measurements.to(arg)
        if self.backend == 'bosonic' and isinstance(self._bosonic_states, list):
            for bs in self._bosonic_states:
                bs.to(arg)
        return self

    # pylint: disable=arguments-renamed
    def forward(
        self,
        data: Optional[torch.Tensor] = None,
        state: Any = None,
        is_prob: Optional[bool] = None,
        detector: Optional[str] = None,
        stepwise: bool = False
    ) -> Union[torch.Tensor, Dict, List[torch.Tensor]]:
        """Perform a forward pass of the photonic quantum circuit and return the final-state-related result.

        Args:
            data (torch.Tensor or None, optional): The input data for the ``encoders``. Default: ``None``
            state (Any, optional): The initial state for the photonic quantum circuit. Default: ``None``
            is_prob (bool or None, optional): For Fock backend, whether to return probabilities or amplitudes.
                For Gaussian (Bosonic) backend, whether to return probabilities or the final Gaussian (Bosonic) state.
                For Fock backend with ``basis=True``, set ``None`` to return the unitary matrix. Default: ``None``
            detector (str or None, optional): For Gaussian backend, use ``'pnrd'`` for the photon-number-resolving
                detector or ``'threshold'`` for the threshold detector. Default: ``None``
            stepwise (bool, optional): Whether to use the forward function of each operator for Gaussian backend.
                Default: ``False``

        Returns:
            Union[torch.Tensor, Dict, List[torch.Tensor]]: The result of the photonic quantum circuit after
            applying the ``operators``.
        """
        if self.backend == 'fock':
            return self._forward_fock(data, state, is_prob)
        elif self.backend in ('gaussian', 'bosonic'):
            return self._forward_cv(data, state, is_prob, detector, stepwise)

    def _forward_fock(
        self,
        data: Optional[torch.Tensor] = None,
        state: Any = None,
        is_prob: Optional[bool] = None
    ) -> Union[torch.Tensor, Dict, List[torch.Tensor]]:
        """Perform a forward pass based on the Fock backend.

        Args:
            data (torch.Tensor or None, optional): The input data for the ``encoders``. Default: ``None``
            state (Any, optional): The initial state for the photonic quantum circuit. Default: ``None``
            is_prob (bool or None, optional): Whether to return probabilities or amplitudes.
                When ``basis=True``, set ``None`` to return the unitary matrix. Default: ``None``

        Returns:
            Union[torch.Tensor, Dict, List[torch.Tensor]]: Unitary matrix, Fock state tensor,
            a dictionary of probabilities or amplitudes, or a list of tensors for MPS.
        """
        if self.mps:
            assert not is_prob
        if state is None:
            state = self.init_state
        else:
            if self.basis:
                self.init_state = FockState(state, nmode=self.nmode, cutoff=self.cutoff, basis=self.basis)
        if isinstance(state, MatrixProductState):
            assert not self.basis
            state = state.tensors
        elif isinstance(state, FockState):
            state = state.state
        elif not isinstance(state, torch.Tensor):
            state = FockState(state=state, nmode=self.nmode, cutoff=self.cutoff, basis=self.basis).state
        # preprocessing of batched initial states
        if self.basis:
            self._is_batch_expand = False # reset
            self._expand_state = None # reset
            state = self._prepare_expand_state(state, cal_all_fock_basis=True)
        if data is None or data.ndim == 1:
            if self.basis:
                assert state.ndim in (1, 2)
                if state.ndim == 1:
                    self.state = self._forward_helper_basis(data, state, is_prob)
                elif state.ndim == 2:
                    self.state = vmap(self._forward_helper_basis, in_dims=(None, 0, None))(data, state, is_prob)
            else:
                self.state = self._forward_helper_tensor(data, state, is_prob)
                if not self.mps and self.state.ndim == self.nmode:
                    self.state = self.state.unsqueeze(0)
        else:
            assert data.ndim == 2
            if self.basis:
                assert state.ndim in (1, 2)
                if state.ndim == 1:
                    self.state = vmap(self._forward_helper_basis, in_dims=(0, None, None))(data, state, is_prob)
                elif state.ndim == 2:
                    if data.shape[0] == 1:
                        self.state = vmap(self._forward_helper_basis, in_dims=(None, 0, None))(data[0], state, is_prob)
                    else:
                        self.state = vmap(self._forward_helper_basis, in_dims=(0, 0, None))(data, state, is_prob)
            else:
                if self.mps:
                    assert state[0].ndim in (3, 4)
                    if state[0].ndim == 3:
                        self.state = vmap(self._forward_helper_tensor, in_dims=(0, None, None))(data, state, is_prob)
                    elif state[0].ndim == 4:
                        self.state = vmap(self._forward_helper_tensor, in_dims=(0, 0, None))(data, state, is_prob)
                else:
                    if state.shape[0] == 1:
                        self.state = vmap(self._forward_helper_tensor, in_dims=(0, None, None))(data, state, is_prob)
                    else:
                        self.state = vmap(self._forward_helper_tensor, in_dims=(0, 0, None))(data, state, is_prob)
            # for plotting the last data
            self.encode(data[-1])
        if self.basis and is_prob is not None:
            self.state = sort_dict_fock_basis(self.state)
        return self.state

    def _forward_helper_basis(
        self,
        data: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        is_prob: Optional[bool] = None
    ) -> Union[torch.Tensor, Dict]:
        """Perform a forward pass for one sample if the input is a Fock basis state."""
        self.encode(data)
        unitary = self.get_unitary()
        if is_prob is None:
            return unitary
        else:
            if state is None:
                state = self.init_state.state
            out_dict = defaultdict(float)
            final_states = self._all_fock_basis
            if self._is_batch_expand:
                unitary = torch.block_diag(unitary, torch.eye(1, dtype=unitary.dtype, device=unitary.device))
            sub_mats = vmap(sub_matrix, in_dims=(None, None, 0))(unitary, state, final_states)
            per_norms = self._get_permanent_norms(state, final_states).to(unitary.dtype)
            if is_prob:
                rst = vmap(self._get_prob_fock_vmap)(sub_mats, per_norms)
            else:
                rst = vmap(self._get_amplitude_fock_vmap)(sub_mats, per_norms)
            for i in range(len(final_states)):
                final_state = FockState(state=final_states[i], nmode=self.nmode, cutoff=self.cutoff, basis=self.basis)
                if not is_prob:
                    assert final_state not in out_dict, \
                        'Amplitudes of reduced states can not be added, please set "is_prob" to be True.'
                out_dict[final_state] += rst[i]
            return out_dict

    def _forward_helper_tensor(
        self,
        data: Optional[torch.Tensor] = None,
        state: Union[torch.Tensor, List[torch.Tensor], None] = None,
        is_prob: Optional[bool] = None
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
        else:
            if isinstance(state, FockState):
                state = state.state
            x = self.operators(self.tensor_rep(state)).squeeze(0)
            if is_prob:
                if self.den_mat:
                    x = x.reshape(-1, self.cutoff ** self.nmode, self.cutoff ** self.nmode).diagonal(dim1=-2, dim2=-1)
                    x = abs(x).reshape([-1] + [self.cutoff] * self.nmode).squeeze(0)
                else:
                    x = abs(x) ** 2
            return x

    def _forward_cv(
        self,
        data: Optional[torch.Tensor] = None,
        state: Any = None,
        is_prob: Optional[bool] = None,
        detector: Optional[str] = None,
        stepwise: bool = False
    ) -> Union[List[torch.Tensor], Dict]:
        """Perform a forward pass based on the Gaussian (Bosonic) backend.

        Args:
            data (torch.Tensor or None, optional): The input data for the ``encoders``. Default: ``None``
            state (Any, optional): The initial state for the photonic quantum circuit. Default: ``None``
            is_prob (bool or None, optional): Whether to return probabilities or the final Gaussian (Bosonic) state.
                Default: ``None``
            detector (str or None, optional): Use ``'pnrd'`` for the photon-number-resolving detector or
                ``'threshold'`` for the threshold detector. Only valid when ``is_prob`` is ``True``.
                Default: ``None``
            stepwise (bool, optional): Whether to use the forward function of each operator. Default: ``False``

        Returns:
            Union[List[torch.Tensor], Dict]: The final Gaussian (Bosonic) state or a dictionary of probabilities.
        """
        if state is None:
            if self.backend == 'bosonic' and self._bosonic_states is not None:
                state = combine_bosonic_states(states=self._bosonic_states, cutoff=self.cutoff)
            else:
                state = self.init_state
        elif not isinstance(state, (GaussianState, BosonicState)):
            nmode = self.nmode
            if self._nmode_tdm is not None and isinstance(state, list):
                if isinstance(state[0], torch.Tensor) and state[0].shape[-1] // 2 == self._nmode_tdm:
                    nmode = self._nmode_tdm
            if self.backend == 'gaussian':
                state = GaussianState(state=state, nmode=nmode, cutoff=self.cutoff)
            elif self.backend == 'bosonic':
                state = BosonicState(state=state, nmode=nmode, cutoff=self.cutoff)
        cov, mean = state.cov, state.mean
        if self.backend == 'bosonic':
            weight = state.weight
        else:
            weight = None
        if self._if_delayloop:
            self._prepare_unroll_dict()
            cov, mean = self._unroll_init_state([cov, mean])
            self._unroll_circuit()
        if data is None or data.ndim == 1:
            cov, mean = self._forward_helper_gaussian(data, [cov, mean], stepwise)
            if cov.ndim < state.cov.ndim:
                cov = cov.unsqueeze(0)
            if mean.ndim < state.mean.ndim:
                mean = mean.unsqueeze(0)
        else:
            assert data.ndim == 2
            if cov.shape[0] == 1:
                cov, mean = vmap(self._forward_helper_gaussian, in_dims=(0, None, None))(data, [cov, mean], stepwise)
            else:
                cov, mean = vmap(self._forward_helper_gaussian, in_dims=(0, 0, None))(data, [cov, mean], stepwise)
            self.encode(data[-1])
        if is_prob:
            self.state = [cov, mean] # for checking purity
            self.state = self._forward_cv_prob(cov, mean, weight, detector)
        else:
            if self._if_delayloop:
                cov, mean = self._shift_state([cov, mean])
            if self.backend == 'gaussian':
                self.state = [cov, mean]
            elif self.backend == 'bosonic':
                self.state = [cov, mean, weight]
        return self.state

    def _forward_helper_gaussian(
        self,
        data: Optional[torch.Tensor] = None,
        state: Optional[List[torch.Tensor]] = None,
        stepwise: bool = False
    ) -> List[torch.Tensor]:
        """Perform a forward pass for one sample if the input is a Gaussian state."""
        if self._lossy:
            stepwise = True
        self.encode(data)
        if self._if_delayloop:
            operators = self._operators_tdm
        else:
            operators = self.operators
        if state is None:
            cov = self.init_state.cov
            mean = self.init_state.mean
        else:
            cov, mean = state
        if self.backend == 'bosonic':
            if cov.ndim == 3:
                cov = cov.unsqueeze(0)
        if stepwise:
            cov, mean = operators([cov, mean])
        else:
            sp_mat = self.get_symplectic()
            cov = sp_mat @ cov @ sp_mat.mT
            mean = self.get_displacement(mean)
        return [cov.squeeze(0), mean.squeeze(0)]

    def _forward_cv_prob(
        self,
        cov: torch.Tensor,
        mean: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        detector: Optional[str] = None
    ) -> Dict:
        """Get the probabilities of all possible final states for Gaussian (Bosonic) backend by different detectors.

        Args:
            cov (torch.Tensor): The covariance matrices of the Gaussian states.
            mean (torch.Tensor): The displacement vectors of the Gaussian states.
            weight (torch.Tensor or None, optional): The weights of the Gaussian states. Default: ``None``
            detector (str or None, optional): Use ``'pnrd'`` for the photon-number-resolving detector or
                ``'threshold'`` for the threshold detector. Default: ``None``
        """
        assert weight is None, 'Currently Fock probability is not supported in Bosonic backend'
        shape_cov = cov.shape
        shape_mean = mean.shape
        if shape_cov[1] == 1:
            cov = cov.expand(-1, shape_mean[1], -1, -1)
        if shape_mean[1] == 1:
            mean = mean.expand(-1, shape_cov[1], -1, -1)
        cov = cov.reshape(-1, *shape_cov[-2:])
        mean = mean.reshape(-1, *shape_mean[-2:])
        purity = GaussianState([cov, mean]).is_pure
        batch_forward = vmap(self._forward_gaussian_prob_helper, in_dims=(0, 0, None, None, None, None))
        if detector is None:
            detector = self.detector
        else:
            detector = detector.lower()
            self.detector = detector
        basis = self._get_odd_even_fock_basis(detector=detector)
        if detector == 'pnrd':
            idx_loop = torch.all(mean==0, dim=1)
            idx_loop = idx_loop.squeeze(1)
            cov_0 = cov[idx_loop]
            mean_0 = mean[idx_loop]
            cov_1 = cov[~idx_loop]
            mean_1 = mean[~idx_loop]
            final_states = torch.cat([torch.cat(basis[1]), torch.cat(basis[0])])
            probs = []
            if len(cov_0) > 0:
                loop = False
                probs_0 = batch_forward(cov_0, mean_0, basis, detector, purity, loop)
                probs.append(probs_0)
            if len(cov_1) > 0:
                loop = True
                probs_1 = batch_forward(cov_1, mean_1, basis, detector, purity, loop)
                probs.append(probs_1)
            probs = torch.cat(probs) # reorder the result here
            if len(cov_0) * len(cov_1) > 0:
                idx0 = torch.where(~idx_loop==0)[0]
                idx1 = torch.where(~idx_loop==1)[0]
                probs = probs[torch.argsort(torch.cat([idx0, idx1]))]
        elif detector == 'threshold':
            final_states = torch.cat(basis)
            loop = True
            probs = batch_forward(cov, mean, basis, detector, purity, loop)
        keys = list(map(FockState, final_states.tolist()))
        # TODO: Fock probabilities for Bosonic state with weights
        # if weight is not None:
        #     probs = probs.reshape(weight.shape[0], weight.shape[1], -1) # (batch, ncomb, nfock)
        #     probs = (probs * weight.unsqueeze(-1)).sum(1).real
        return dict(zip(keys, probs.mT))

    def _forward_gaussian_prob_helper(self, cov, mean, basis, detector, purity, loop):
        prob_lst = []
        if detector == 'pnrd':
            odd_basis = basis[0]
            even_basis = basis[1]
            for state in even_basis:
                prob_even = self._get_probs_gaussian_helper(state, cov, mean, detector, purity, loop)
                prob_lst.append(prob_even)
            if loop or not purity:
                for state in odd_basis:
                    prob_odd = self._get_probs_gaussian_helper(state, cov, mean, detector, purity, loop)
                    prob_lst.append(prob_odd)
                probs = torch.cat(prob_lst)
            else:
                probs = torch.cat(prob_lst)
                probs = torch.cat([probs, torch.zeros(len(torch.cat(odd_basis)), device=probs.device)])
        elif detector == 'threshold':
            for state in basis:
                prob = self._get_probs_gaussian_helper(state, cov, mean, detector, purity, loop)
                prob_lst.append(prob)
            probs = torch.cat(prob_lst)
        return probs

    def _prepare_expand_state(self, state: torch.Tensor, cal_all_fock_basis: bool = False) -> torch.Tensor:
        """Check and expand the Fock state if necessary."""
        if state.ndim == 1:
            if self._lossy:
                state = torch.cat([state, state.new_zeros(self._nloss)], dim=-1)
            if cal_all_fock_basis:
                self._all_fock_basis = self._get_all_fock_basis(state)
        elif state.ndim == 2:
            if self._lossy:
                state = torch.cat([state, state.new_zeros(state.shape[0], self._nloss)], dim=-1)
            nphotons = torch.sum(state, dim=-1, keepdim=True)
            max_photon = torch.max(nphotons).item()
            # expand the Fock state if the photon number is not conserved
            if any(nphoton < max_photon for nphoton in nphotons):
                state = torch.cat([state, max_photon - nphotons], dim=-1)
                self._is_batch_expand = True
            if cal_all_fock_basis:
                self._all_fock_basis = self._get_all_fock_basis(state[0])
        if self._lossy or self._is_batch_expand:
            self._expand_state = state
        return state

    def _prepare_unroll_dict(self) -> Dict[int, List]:
        """Create a dictionary that maps spatial modes to concurrent modes."""
        if self._unroll_dict is None:
            self._unroll_dict = defaultdict(list)
            wires = list(range(self._nmode_tdm))
            start = 0
            for i in range(self.nmode):
                for ntau in reversed(self._ntau_dict[i]):
                    self._unroll_dict[i].append(wires[start:start+ntau]) # modes in delay line
                    start += ntau
                self._unroll_dict[i].append(wires[start]) # spatial mode
                start += 1
        return self._unroll_dict

    def _unroll_init_state(self, state: List[torch.Tensor]) -> List[torch.Tensor]:
        """Unroll the initial state from spatial modes to concurrent modes."""
        idx = torch.tensor([value[-1] for value in self._unroll_dict.values()])
        idx = torch.cat([idx, idx + self._nmode_tdm])
        cov, mean = state
        size = cov.size()
        size_tdm = 2 * self._nmode_tdm
        if size[-1] == size_tdm:
            return state
        else:
            cov_tdm = cov.new_ones(size[:-2].numel() * size_tdm).reshape(*size[:-2], size_tdm).diag_embed()
            mean_tdm = mean.new_zeros(*size[:-2], size_tdm, 1)
            cov_tdm[..., idx[:, None], idx] = cov
            mean_tdm[..., idx, :] = mean
            return [cov_tdm, mean_tdm]

    def _unroll_circuit(self) -> None:
        """Unroll the circuit from spatial modes to concurrent modes."""
        nmode = self._nmode_tdm
        if self._operators_tdm is None:
            self._operators_tdm = nn.Sequential()
            ndelay = np.array([0] * self.nmode) # counter of delay loops for each mode
            for op in self.operators:
                if isinstance(op, Delay):
                    wire = op.wires[0]
                    ndelay[wire] += 1
                    idx_delay = -ndelay[wire] - 1
                    wires = [self._unroll_dict[wire][idx_delay][0], self._unroll_dict[wire][-1]]
                    op.gates[0].nmode = nmode
                    op.gates[0].wires = wires
                    self._operators_tdm.append(op.gates[0])
                    if len(op.gates) > 1:
                        for gate in op.gates[1:]:
                            gate.nmode = nmode
                            gate.wires = wires[0:1]
                            if isinstance(gate, PhotonLoss):
                                self._lossy = True
                                self._nloss += 1
                            self._operators_tdm.append(gate)
                else:
                    op_tdm = copy(op)
                    op_tdm.nmode = nmode
                    op_tdm.wires = [self._unroll_dict[wire][-1] for wire in op.wires]
                    self._operators_tdm.append(op_tdm)
        if self._measurements_tdm is None:
            self._measurements_tdm = nn.ModuleList()
            for op_m in self.measurements:
                op_m_tdm = copy(op_m)
                op_m_tdm.nmode = nmode
                op_m_tdm.wires = [self._unroll_dict[wire][-1] for wire in op_m.wires]
                self._measurements_tdm.append(op_m_tdm)

    def global_circuit(self, nstep: int, use_deepcopy: bool = False) -> 'QumodeCircuit':
        """Get the global circuit given the number of time steps.

        Note:
            The initial state of the global circuit is always the vacuum state.
        """
        self._prepare_unroll_dict()
        nmode = self._nmode_tdm + (nstep - 1) * self.nmode
        cir = QumodeCircuit(nmode, init_state='vac', cutoff=self.cutoff, backend=self.backend, basis=self.basis,
                            den_mat=self.den_mat, detector=self.detector, name=self.name, mps=self.mps, chi=self.chi,
                            noise=self.noise, mu=self.mu, sigma=self.sigma)
        for i in range(nstep):
            ndelay = np.array([0] * self.nmode) # counter of delay loops for each mode
            for op in self.operators:
                encode = op in self.encoders
                if isinstance(op, Delay):
                    wire = op.wires[0]
                    ndelay[wire] += 1
                    idx_delay = -ndelay[wire] - 1
                    wire1 = self._unroll_dict[wire][idx_delay][i % op.ntau]
                    if i == 0:
                        wire2 = self._unroll_dict[wire][-1]
                    else:
                        wire2 = self._nmode_tdm + self.nmode * (i - 1) + wire
                    if use_deepcopy or encode:
                        op_tdm = deepcopy(op.gates[0])
                    else:
                        op_tdm = copy(op.gates[0])
                    op_tdm.nmode = nmode
                    op_tdm.wires = [wire1, wire2]
                    cir.add(op_tdm, encode=encode)
                    if len(op.gates) > 1:
                        for gate in op.gates[1:]:
                            if use_deepcopy or encode:
                                op_gate = deepcopy(gate)
                            else:
                                op_gate = copy(gate)
                            op_gate.nmode = nmode
                            op_gate.wires = [wire1]
                            if isinstance(gate, PhotonLoss):
                                cir._lossy = True
                                cir._nloss += 1
                            cir.add(op_gate, encode=encode)
                else:
                    if use_deepcopy or encode:
                        op_tdm = deepcopy(op)
                    else:
                        op_tdm = copy(op)
                    op_tdm.nmode = nmode
                    if i == 0:
                        op_tdm.wires = [self._unroll_dict[wire][-1] for wire in op.wires]
                    else:
                        op_tdm.wires = [self._nmode_tdm + self.nmode * (i - 1) + wire for wire in op.wires]
                    if isinstance(op, PhotonLoss):
                        cir._lossy = True
                        cir._nloss += 1
                    cir.add(op_tdm, encode=encode)
            for op_m in self.measurements:
                op_m_tdm = copy(op_m)
                op_m_tdm.nmode = nmode
                if i == 0:
                    op_m_tdm.wires = [self._unroll_dict[wire][-1] for wire in op_m.wires]
                else:
                    op_m_tdm.wires = [self._nmode_tdm + self.nmode * (i - 1) + wire for wire in op_m.wires]
                cir.add(op_m_tdm)
            cir.barrier()
        return cir

    def _shift_state(self, state: List[torch.Tensor], nstep: int = 1, reverse: bool = False) -> List[torch.Tensor]:
        """Shift the state according to ``nstep``, which is equivalent to shifting the TDM circuit."""
        cov, mean = state
        idx_shift = []
        for wire in self._unroll_dict:
            for idx in self._unroll_dict[wire]:
                if isinstance(idx, int):
                    idx_shift.append(idx)
                elif isinstance(idx, list):
                    if reverse:
                        idx_shift.extend(shift_func(idx, -nstep))
                    else:
                        idx_shift.extend(shift_func(idx, nstep))
        idx_shift = torch.tensor(idx_shift)
        idx_shift = torch.cat([idx_shift, idx_shift + self._nmode_tdm])
        cov = cov[..., idx_shift[:, None], idx_shift]
        mean = mean[..., idx_shift, :]
        return [cov, mean]

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
        if self._if_delayloop:
            operators = self._operators_tdm
        else:
            operators = self.operators
        nloss = 0
        for op in operators:
            if isinstance(op, Barrier):
                continue
            if isinstance(op, PhotonLoss):
                nloss += 1
                op.gate.wires = [op.wires[0], op.nmode + nloss - 1]
                op.gate.nmode = op.nmode + nloss
                if u is None:
                    u = op.gate.get_unitary()
                else:
                    idx_r = torch.tensor(op.gate.wires, device=u.device)
                    idx_c = torch.arange(op.gate.nmode, device=u.device)
                    u = torch.block_diag(u, torch.eye(1, dtype=u.dtype, device=u.device))
                    u_local = op.gate.update_matrix()
                    u_update = u[idx_r[:, None], idx_c]
                    u[idx_r[:, None], idx_c] = u_local @ u_update
            else:
                if u is None:
                    u = op.get_unitary()
                else:
                    idx_r = torch.tensor(op.wires, device=u.device)
                    idx_c = torch.arange(op.nmode + nloss, device=u.device)
                    u_local = op.update_matrix()
                    u_update = u[idx_r[:, None], idx_c]
                    u[idx_r[:, None], idx_c] = u_local @ u_update
        if u is None:
            return torch.eye(self.nmode, dtype=torch.cfloat)
        else:
            return u

    def get_symplectic(self) -> torch.Tensor:
        """Get the symplectic matrix of the photonic quantum circuit."""
        s = None
        if self._if_delayloop:
            operators = self._operators_tdm
            nmode = self._nmode_tdm
        else:
            operators = self.operators
            nmode = self.nmode
        for op in operators:
            if isinstance(op, Barrier):
                continue
            if s is None:
                s = op.get_symplectic()
            else:
                s = op.get_symplectic() @ s
        if s is None:
            return torch.eye(2 * nmode, dtype=torch.float)
        return s

    def get_displacement(self, init_mean: Any) -> torch.Tensor:
        """Get the final mean value of the Gaussian state in ``xxpp`` order."""
        if not isinstance(init_mean, torch.Tensor):
            init_mean = torch.tensor(init_mean)
        if self._if_delayloop:
            operators = self._operators_tdm
            nmode = self._nmode_tdm
        else:
            operators = self.operators
            nmode = self.nmode
        mean = init_mean
        if self.backend == 'gaussian':
            mean = mean.reshape(-1, 2 * nmode, 1)
        elif self.backend == 'bosonic':
            if mean.ndim == 2:
                mean = mean.unsqueeze(0).unsqueeze(-1)
            elif mean.ndim == 3:
                if mean.shape[-1] == 1:
                    mean = mean.unsqueeze(0)
                elif mean.shape[-1] == 2 * nmode:
                    mean = mean.unsqueeze(-1)
            assert mean.ndim == 4
        for op in operators:
            if isinstance(op, Barrier):
                continue
            mean = op.get_symplectic().to(mean.dtype) @ mean + op.get_displacement()
        return mean

    def _get_all_fock_basis(self, init_state: torch.Tensor) -> torch.Tensor:
        """Get all possible fock basis states according to the initial state."""
        nphoton = torch.max(torch.sum(init_state, dim=-1))
        nmode = len(init_state)
        if self._if_delayloop:
            nancilla = nmode - self._nmode_tdm
        else:
            nancilla = nmode - self.nmode
        states = torch.tensor(fock_combinations(nmode, nphoton, self.cutoff, nancilla=nancilla),
                              dtype=torch.long, device=init_state.device)
        return states

    def _get_odd_even_fock_basis(self, detector: Optional[str] = None) -> Union[Tuple[List, List], List]:
        """Split the fock basis into the odd and even photon number parts."""
        if detector is None:
            detector = self.detector
        if self._if_delayloop:
            nmode = self._nmode_tdm
        else:
            nmode = self.nmode
        if detector == 'pnrd':
            max_photon = nmode * (self.cutoff - 1)
            odd_lst = []
            even_lst = []
            for i in range(0, max_photon + 1):
                state_tmp = torch.tensor([i] + [0] * (nmode - 1))
                temp_basis = self._get_all_fock_basis(state_tmp)
                if i % 2 == 0:
                    even_lst.append(temp_basis)
                else:
                    odd_lst.append(temp_basis)
            return odd_lst, even_lst
        elif detector == 'threshold':
            final_states = torch.tensor(list(itertools.product(range(2), repeat=nmode)))
            keys = torch.sum(final_states, dim=1)
            dic_temp = defaultdict(list)
            for state, s in zip(final_states, keys):
                dic_temp[s.item()].append(state)
            state_lst = [torch.stack(i) for i in list(dic_temp.values())]
            return state_lst

    def _get_permanent_norms(self, init_state: torch.Tensor, final_state: torch.Tensor) -> torch.Tensor:
        """Get the normalization factors for permanent."""
        return torch.sqrt(product_factorial(init_state) * product_factorial(final_state))

    def get_amplitude(
        self,
        final_state: Any,
        init_state: Any = None,
        unitary: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get the transfer amplitude between the final state and the initial state.

        Note:
            When states are expanded due to photon loss or batched initial states,
            the amplitudes of the reduced states can not be added, please try ``get_prob`` instead.

        Args:
            final_state (Any): The final Fock basis state.
            init_state (Any, optional): The initial Fock basis state. Default: ``None``
            unitary (torch.Tensor or None, optional): The unitary matrix. Default: ``None``
        """
        assert self.backend == 'fock'
        if not isinstance(final_state, torch.Tensor):
            final_state = torch.tensor(final_state, dtype=torch.long)
        if init_state is None:
            init_state = self.init_state
        elif not isinstance(init_state, FockState):
            init_state = FockState(state=init_state, nmode=self.nmode, cutoff=self.cutoff, basis=self.basis)
        assert init_state.basis, 'The initial state must be a Fock basis state'
        assert max(final_state) < self.cutoff, 'The number of photons in the final state must be less than cutoff'
        if unitary is None:
            unitary = self.get_unitary()
        else:
            assert unitary.ndim == 2, 'The unitary matrix must be 2D'
        state = init_state.state.to(unitary.device)
        final_state = final_state.to(unitary.device)
        if state.ndim == 1:
            sub_mat = sub_matrix(unitary, state, final_state)
            per = permanent(sub_mat)
            amp = per / self._get_permanent_norms(state, final_state).to(per.dtype)
        else:
            idx_nonzero = torch.where(torch.sum(state, dim=-1) == torch.sum(final_state))[0]
            amp = torch.zeros(state.shape[0], dtype=unitary.dtype, device=unitary.device)
            if idx_nonzero.numel() != 0:
                sub_mats = vmap(sub_matrix, in_dims=(None, 0, None))(unitary, state[idx_nonzero], final_state)
                per_norms = self._get_permanent_norms(state[idx_nonzero], final_state).to(unitary.dtype)
                rst = vmap(self._get_amplitude_fock_vmap)(sub_mats, per_norms).flatten()
                amp[idx_nonzero] = rst
        return amp

    def _get_amplitude_fock_vmap(self, sub_mat: torch.Tensor, per_norm: torch.Tensor) -> torch.Tensor:
        """Get the transfer amplitude."""
        per = permanent(sub_mat)
        amp = per / per_norm
        return amp.reshape(-1)

    def get_prob(
        self,
        final_state: Any,
        refer_state: Any = None,
        unitary: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get the probability of the final state related to the reference state.

        Args:
            final_state (Any): The final Fock basis state.
            refer_state (Any, optional): The initial Fock basis state or the final Gaussian state. Default: ``None``
            unitary (torch.Tensor or None, optional): The unitary matrix. Default: ``None``
        """
        if not isinstance(final_state, torch.Tensor):
            final_state = torch.tensor(final_state, dtype=torch.long)
        assert max(final_state) < self.cutoff, 'The number of photons in the final state must be less than cutoff'
        if self.backend == 'fock':
            if refer_state is None:
                if self._expand_state is not None:
                    refer_state = self._expand_state
                else:
                    refer_state = self._prepare_expand_state(self.init_state.state)
            if unitary is None:
                unitary = self.get_unitary()
            else:
                assert unitary.ndim == 2, 'The unitary matrix must be 2D'
            if self._is_batch_expand:
                identity = torch.eye(1, dtype=unitary.dtype, device=unitary.device)
                unitary = torch.block_diag(unitary, identity)
            nmode = final_state.shape[-1]
            if refer_state.shape[-1] == nmode:
                return self._get_prob_fock(final_state, refer_state, unitary)
            else:
                wires = list(range(nmode))
                nphoton_final = torch.sum(final_state, dim=-1)
                max_photon = torch.sum(refer_state, dim=-1).max().item()
                nmode_expand = refer_state.shape[-1] - nmode
                expand_state = torch.tensor(fock_combinations(nmode_expand, max_photon - nphoton_final),
                                            dtype=torch.long, device=final_state.device)
                final_state = final_state.reshape(-1, nmode).expand(expand_state.shape[0], -1)
                final_states = torch.cat([final_state, expand_state], dim=-1)
                if refer_state.ndim == 1:
                    rst = self._measure_fock_unitary_helper(refer_state, unitary, wires, final_states)
                else:
                    rst = vmap(self._measure_fock_unitary_helper,
                               in_dims=(0, None, None, None))(refer_state, unitary, wires, final_states)
                rst = list(rst.values())[0]
            return rst
        elif self.backend == 'gaussian':
            if self._if_delayloop:
                nmode = self._nmode_tdm
            else:
                nmode = self.nmode
            if refer_state is None:
                refer_state = GaussianState(self.state, nmode=nmode, cutoff=self.cutoff)
            return self._get_prob_gaussian(final_state, refer_state)

    def _get_prob_fock(
        self,
        final_state: Any,
        init_state: Any = None,
        unitary: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get the transfer probability between the final state and the initial state for the Fock backend.

        Args:
            final_state (Any): The final Fock basis state.
            init_state (Any, optional): The initial Fock basis state. Default: ``None``
            unitary (torch.Tensor or None, optional): The unitary matrix. Default: ``None``
        """
        if init_state is None: # when mcmc
            nmode = self.nmode + self._nloss + self._is_batch_expand
            init_state = FockState(state=self._init_state, nmode=nmode, cutoff=self.cutoff, basis=self.basis)
        if unitary is None: # when mcmc
            unitary = self._unitary
        amplitude = self.get_amplitude(final_state, init_state, unitary)
        prob = torch.abs(amplitude) ** 2
        return prob

    def _get_prob_fock_vmap(self, sub_mat: torch.Tensor, per_norm: torch.Tensor) -> torch.Tensor:
        """Get the transfer probability."""
        amplitude = self._get_amplitude_fock_vmap(sub_mat, per_norm)
        prob = torch.abs(amplitude) ** 2
        return prob

    def _get_prob_gaussian(
        self,
        final_state: Any,
        state: Any = None
    ) -> torch.Tensor:
        """Get the batched probabilities of the final state for Gaussian backend."""
        if not isinstance(final_state, torch.Tensor):
            final_state = torch.tensor(final_state, dtype=torch.long)
        if state is None:
            cov = self._cov
            mean = self._mean
        else:
            if not isinstance(state, GaussianState):
                state = GaussianState(state=state, cutoff=self.cutoff)
            cov = state.cov
            mean = state.mean
        if cov.ndim == 2:
            cov = cov.unsqueeze(0)
        if mean.ndim == 2:
            mean = mean.unsqueeze(0)
        assert cov.ndim == mean.ndim == 3
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
        mean: torch.Tensor,
        detector: str = 'pnrd',
        purity: Optional[bool] = None,
        loop: Optional[bool] = None
    ) -> torch.Tensor:
        """Get the probabilities of the final states for Gaussian backend."""
        if loop is None:
            loop = ~torch.all(mean==0)
        if final_states.ndim == 1:
            final_states = final_states.unsqueeze(0)
        assert final_states.ndim == 2
        nmode = final_states.shape[-1]
        final_states = final_states.to(cov.device)
        identity = cov.new_ones(2 * nmode).diag_embed()
        cov_ladder = quadrature_to_ladder(cov)
        mean_ladder = quadrature_to_ladder(mean)
        q = cov_ladder + identity / 2
        det_q = q.det()
        x_mat = identity.reshape(2, nmode, 2 * nmode).flip(0).reshape(2 * nmode, 2 * nmode) + 0j
        o_mat = identity - q.inverse()
        a_mat = x_mat @ o_mat
        gamma = mean_ladder.mH @ q.inverse()
        if detector == 'pnrd':
            matrix = a_mat
        elif detector == 'threshold':
            matrix = o_mat
        if purity is None:
            purity = GaussianState([cov, mean]).is_pure
        p_vac = torch.exp(-0.5 * mean_ladder.mH @ q.inverse() @ mean_ladder) / det_q.sqrt()
        batch_get_prob = vmap(self._get_prob_gaussian_base, in_dims=(0, None, None, None, None, None, None))
        probs = batch_get_prob(final_states, matrix, gamma, p_vac, detector, purity, loop)
        return probs

    def _get_prob_gaussian_base(
        self,
        final_state: torch.Tensor,
        matrix: torch.Tensor,
        gamma: torch.Tensor,
        p_vac: torch.Tensor,
        detector: str = 'pnrd',
        purity: bool = True,
        loop: bool = False
    ) -> torch.Tensor:
        """Get the probability of the final state for Gaussian backend."""
        gamma = gamma.squeeze()
        nmode = len(final_state)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore') # local warning
            gamma_n1 = torch.repeat_interleave(gamma[:nmode], final_state)
            gamma_n2 = torch.repeat_interleave(gamma[nmode:], final_state)
        sub_gamma = torch.cat([gamma_n1, gamma_n2])
        if detector == 'pnrd':
            if purity:
                sub_mat = sub_matrix(matrix[:nmode, :nmode], final_state, final_state)
                half_len = len(sub_gamma) // 2
                sub_gamma = sub_gamma[:half_len]
            else:
                final_state_double = torch.cat([final_state, final_state])
                sub_mat = sub_matrix(matrix, final_state_double, final_state_double)
            if len(sub_gamma) == 1:
                sub_mat = sub_gamma
            else:
                sub_mat[torch.arange(len(sub_gamma)), torch.arange(len(sub_gamma))] = sub_gamma
            if purity:
                haf = abs(hafnian(sub_mat, loop=loop)) ** 2
            else:
                haf = hafnian(sub_mat, loop=loop)
            prob = p_vac * haf / product_factorial(final_state).to(device=haf.device, dtype=haf.dtype)
        elif detector == 'threshold':
            final_state_double = torch.cat([final_state, final_state])
            sub_mat = sub_matrix(matrix, final_state_double, final_state_double)
            prob = p_vac * torontonian(sub_mat, sub_gamma)
        return abs(prob.real).squeeze()

    def _get_prob_mps(self, final_state: Any, wires: Union[int, List[int], None] = None) -> torch.Tensor:
        """Get the probability of the given bit string for MPS.

        Args:
            final_state (Any): The final Fock basis state.
            wires (int, List[int] or None, optional): The wires to measure. It can be an integer or a list of
                integers specifying the indices of the wires.
        """
        if isinstance(final_state, FockState):
            final_state = final_state.state.tolist()
        if wires is None:
            wires = list(range(self.nmode))
        else:
            wires = self._convert_indices(wires)
        assert len(final_state) == len(wires)
        state = copy(self.state)
        if self.state[0].ndim == 3:
            state = [site.unsqueeze(0) for site in state]
        for i, wire in enumerate(wires):
            state[wire] = state[wire][..., [final_state[i]], :]
        return inner_product_mps(state, state).real

    def measure(
        self,
        shots: int = 1024,
        with_prob: bool = False,
        wires: Union[int, List[int], None] = None,
        detector: Optional[str] = None,
        mcmc: bool = False
    ) -> Union[Dict, List[Dict], None]:
        """Measure the final state.

        Args:
            shots (int, optional): The number of times to sample from the quantum state. Default: 1024
            with_prob (bool, optional): A flag that indicates whether to return the probabilities along with
                the number of occurrences. Default: ``False``
            wires (int, List[int] or None, optional): The wires to measure. It can be an integer or a list of
                integers specifying the indices of the wires. Default: ``None`` (which means all wires are measured)
            detector (str or None, optional): For Gaussian backend, use ``'pnrd'`` for the photon-number-resolving
                detector or ``'threshold'`` for the threshold detector. Default: ``None``
            mcmc (bool, optional): Whether to use MCMC sampling method. Default: ``False``

        See https://arxiv.org/pdf/2108.01622 for MCMC.
        """
        assert self.backend in ('fock', 'gaussian'), 'Currently Fock measurement is not supported in Bosonic backend'
        if self.state is None:
            return
        if wires is None:
            wires = self.wires
        wires = sorted(self._convert_indices(wires))
        if self.backend == 'fock':
            results = self._measure_fock(shots, with_prob, wires, mcmc)
        elif self.backend == 'gaussian':
            if detector is None:
                detector = self.detector
            else:
                detector = detector.lower()
            results = self._measure_gaussian(shots, with_prob, wires, detector, mcmc)
        if len(results) == 1:
            results = results[0]
        return results

    def _prob_dict_to_measure_result(self, prob_dict: Dict, shots: int, with_prob: bool) -> Dict:
        """Get the measurement result from the dictionary of probabilities."""
        samples = random.choices(list(prob_dict.keys()), list(prob_dict.values()), k=shots)
        results = dict(Counter(samples))
        if with_prob:
            results = {key: (value, prob_dict[key]) for key, value in results.items()}
        return results

    def _measure_fock(self, shots: int, with_prob: bool, wires: List[int], mcmc: bool) -> List[Dict]:
        """Measure the final state for Fock backend."""
        if isinstance(self.state, torch.Tensor):
            if self.basis:
                return self._measure_fock_unitary(shots, with_prob, wires, mcmc)
            else:
                assert not mcmc, "Final states have been calculated, we don't need mcmc!"
                return self._measure_fock_tensor(shots, with_prob, wires)
        elif isinstance(self.state, Dict):
            assert not mcmc, "Final states have been calculated, we don't need mcmc!"
            return self._measure_dict(shots, with_prob, wires)
        elif isinstance(self.state, List):
            assert not mcmc, "Final states have been calculated, we don't need mcmc!"
            return self._measure_mps(shots, with_prob, wires)
        else:
            assert False, 'Check your forward function or input!'

    def _measure_fock_unitary(self, shots: int, with_prob: bool, wires: List[int], mcmc: bool) -> List[Dict]:
        """Measure the final state according to the unitary matrix for Fock backend."""
        if self.state.ndim == 2:
            self.state = self.state.unsqueeze(0)
        batch = self.state.shape[0]
        init_state = self.init_state.state if self._expand_state is None else self._expand_state
        if init_state.ndim == 1:
            init_state = init_state.unsqueeze(0)
        batch_init = init_state.shape[0]
        unitary = self.state
        if self._is_batch_expand:
            identity = torch.eye(1, dtype=self.state.dtype, device=self.state.device)
            unitary = vmap(torch.block_diag, in_dims=(0, None))(self.state, identity)
        all_results = []
        if mcmc:
            for i in range(batch):
                if batch_init == 1:
                    samples_i = self._sample_mcmc_fock(shots=shots, init_state=init_state[0], unitary=unitary[i],
                                                       num_chain=5)
                else:
                    samples_i = self._sample_mcmc_fock(shots=shots, init_state=init_state[i], unitary=unitary[i],
                                                       num_chain=5)
                results = defaultdict(list)
                if with_prob:
                    for k in samples_i:
                        prob = self._get_prob_fock(k)
                        samples_i[k] = samples_i[k], prob
                for key in samples_i.keys():
                    state_b = [key[wire] for wire in wires]
                    state_b = FockState(state=state_b)
                    results[state_b].append(samples_i[key])
                if with_prob:
                    results = {
                        key: (
                            sum(count for count, _ in value),
                            sum(prob for _, prob in value)
                        )
                        for key, value in results.items()
                    }
                else:
                    results = {key: sum(value) for key, value in results.items()}
                all_results.append(results)
        else:
            if batch_init == 1:
                prob_dict_batch = vmap(self._measure_fock_unitary_helper,
                                       in_dims=(None, 0, None))(init_state[0], unitary, wires)
            else:
                prob_dict_batch = vmap(self._measure_fock_unitary_helper,
                                       in_dims=(0, 0, None))(init_state, unitary, wires)
            for i in range(batch):
                prob_dict = {key: value[i] for key, value in prob_dict_batch.items()}
                results = self._prob_dict_to_measure_result(prob_dict, shots, with_prob)
                all_results.append(results)
        return all_results

    def _measure_fock_unitary_helper(
        self,
        init_state: torch.Tensor,
        unitary: torch.Tensor,
        wires: Union[int, List[int], None] = None,
        final_states: Optional[torch.Tensor] = None
    ) -> Dict:
        """VMAP helper for measuring the final state according to the unitary matrix for Fock backend.

        Returns:
            Dict: A dictionary of probabilities for final states.
        """
        if final_states is None:
            final_states = self._all_fock_basis
        sub_mats = vmap(sub_matrix, in_dims=(None, None, 0))(unitary, init_state, final_states)
        per_norms = self._get_permanent_norms(init_state, final_states).to(unitary.dtype)
        rst = vmap(self._get_prob_fock_vmap)(sub_mats, per_norms)
        state_dict = {}
        prob_dict = defaultdict(list)
        for i in range(len(final_states)):
            final_state = FockState(state=final_states[i])
            state_dict[final_state] = rst[i]
        for key in state_dict.keys():
            state_b = key.state[wires]
            state_b = FockState(state=state_b)
            prob_dict[state_b].append(state_dict[key])
        prob_dict = {key: sum(value) for key, value in prob_dict.items()}
        return prob_dict

    def _measure_dict(self, shots: int, with_prob: bool, wires: List[int]) -> List[Dict]:
        """Measure the final state according to the dictionary of amplitudes or probabilities."""
        if self._if_delayloop:
            wires = [self._unroll_dict[wire][-1] for wire in wires]
        all_results = []
        batch = len(self.state[list(self.state.keys())[0]])
        if self.backend == 'fock' and any(value.dtype.is_complex for value in self.state.values()):
            is_prob = False
        else:
            is_prob = True
        for i in range(batch):
            prob_dict = defaultdict(list)
            for key in self.state.keys():
                if wires == self.wires:
                    state_b = key
                else:
                    state_b = key.state[wires]
                    state_b = FockState(state=state_b)
                if is_prob:
                    prob_dict[state_b].append(self.state[key][i])
                else:
                    prob_dict[state_b].append(abs(self.state[key][i]) ** 2)
            prob_dict = {key: sum(value) for key, value in prob_dict.items()}
            results = self._prob_dict_to_measure_result(prob_dict, shots, with_prob)
            all_results.append(results)
        return all_results

    def _measure_fock_tensor(self, shots: int, with_prob: bool, wires: List[int]) -> List[Dict]:
        """Measure the final state according to Fock state tensor for Fock backend."""
        all_results = []
        if self.state.is_complex():
            if self.den_mat:
                state_tensor = self.state.reshape(-1, self.cutoff ** self.nmode, self.cutoff ** self.nmode)
                state_tensor = self.tensor_rep(abs(state_tensor.diagonal(dim1=-2, dim2=-1)))
            else:
                state_tensor = self.tensor_rep(abs(self.state) ** 2)
        else:
            state_tensor = self.tensor_rep(self.state)
        batch = state_tensor.shape[0]
        combi = list(itertools.product(range(self.cutoff), repeat=len(wires)))
        for i in range(batch):
            prob_dict = {}
            probs = state_tensor[i]
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
            results = self._prob_dict_to_measure_result(prob_dict, shots, with_prob)
            all_results.append(results)
        return all_results

    def _measure_mps(self, shots: int, with_prob: bool, wires: List[int]) -> List[Dict]:
        """Measure the final state according to MPS."""
        all_results = []
        samples = []
        for _ in range(shots):
            samples.append(self._generate_chain_sample(wires))
        for j in range(samples[0].shape[0]):
            samples_j = [tuple(sample[j].tolist()) for sample in samples]
            samples_j = dict(Counter(samples_j))
            keys = list(map(FockState, samples_j.keys()))
            results = dict(zip(keys, samples_j.values()))
            if with_prob:
                for k in results:
                    prob = self._get_prob_mps(k, wires)[j]
                    results[k] = results[k], prob
            all_results.append(results)
        return all_results

    def _sample_mcmc_fock(self, shots: int, init_state: torch.Tensor, unitary: torch.Tensor, num_chain: int):
        """Sample the output states for Fock backend via SC-MCMC method."""
        self._init_state = init_state
        self._unitary = unitary
        self._all_fock_basis = self._get_all_fock_basis(init_state)
        merged_samples = sample_sc_mcmc(prob_func=self._get_prob_fock,
                                        proposal_sampler=self._proposal_sampler,
                                        shots=shots,
                                        num_chain=num_chain)
        return merged_samples

    def _measure_gaussian(
        self,
        shots: int,
        with_prob: bool,
        wires: List[int],
        detector: str,
        mcmc: bool
    ) -> List[Dict]:
        """Measure the final state for Gaussian backend."""
        if isinstance(self.state, List):
            return self._measure_gaussian_state(shots, with_prob, wires, detector, mcmc)
        elif isinstance(self.state, Dict):
            assert not mcmc, "Final states have been calculated, we don't need mcmc!"
            print('Automatically using the default detector!')
            return self._measure_dict(shots, with_prob, wires)
        else:
            assert False, 'Check your forward function or input!'

    def _measure_gaussian_state(
        self,
        shots: int,
        with_prob: bool,
        wires: List[int],
        detector: str,
        mcmc: bool
    ) -> List[Dict]:
        """Measure the final state according to Gaussian state for Gaussian backend.

        See https://arxiv.org/pdf/2108.01622
        """
        cov, mean = self.state
        batch = cov.shape[0]
        all_results = []
        all_samples = []
        if mcmc:
            print('Using MCMC method to sample the final states!')
            for i in range(batch):
                samples_i = self._sample_mcmc_gaussian(shots=shots, cov=cov[i], mean=mean[i],
                                                       detector=detector, num_chain=5)
                all_samples.append(samples_i)
        else: # chain-rule method with small number of shots
            print('Using chain-rule method to sample the final states!')
            samples = []
            for _ in range(shots):
                sample = self._generate_chain_sample(wires)
                samples.append(sample)
            samples = torch.stack(samples).permute(1, 0, 2) # (batch, shots, wires)
            for i in range(batch):
                sample_lst = samples[i].tolist()
                sample_tup = [tuple(s) for s in sample_lst]
                samples_i = defaultdict(int, Counter(sample_tup))
                all_samples.append(samples_i)
        for i, samples_i in enumerate(all_samples): # post-process samples
            results = defaultdict(list)
            if with_prob:
                for k in samples_i:
                    if mcmc:
                        prob = self._get_prob_gaussian(k, [cov[i], mean[i]])
                    else:
                        wires_ = sorted(self._convert_indices(wires))
                        wires_ = torch.tensor(wires_, device=cov.device)
                        idx = torch.cat([wires_, wires_ + self.nmode])
                        prob = self._get_prob_gaussian(k, [cov[i][idx[:, None], idx], mean[i][idx, :]])
                    samples_i[k] = samples_i[k], prob
            for key in samples_i.keys():
                if mcmc:
                    state_b = [key[wire] for wire in wires]
                else:
                    state_b = list(key)
                state_b = FockState(state=state_b)
                results[state_b].append(samples_i[key])
            if with_prob:
                results = {
                    key: (
                        sum(count for count, _ in value),
                        sum(prob for _, prob in value)
                    )
                    for key, value in results.items()
                }
            else:
                results = {key: sum(value) for key, value in results.items()}
            all_results.append(results)
        return all_results

    def _sample_mcmc_gaussian(self, shots: int, cov: torch.Tensor, mean: torch.Tensor, detector: str, num_chain: int):
        """Sample the output states for Gaussian backend via SC-MCMC method."""
        self._cov = cov
        self._mean = mean
        self.detector = detector
        if detector == 'threshold' and not torch.allclose(mean, torch.zeros_like(mean)):
            # For the displaced state, aggregate PNRD detector samples to derive threshold detector results
            self.detector = 'pnrd'
            merged_samples_pnrd = sample_sc_mcmc(prob_func=self._get_prob_gaussian,
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
            merged_samples = sample_sc_mcmc(prob_func=self._get_prob_gaussian,
                                            proposal_sampler=self._proposal_sampler,
                                            shots=shots,
                                            num_chain=num_chain)
        return merged_samples

    def _proposal_sampler(self):
        """The proposal sampler for MCMC sampling."""
        if self.backend == 'fock':
            assert self.basis, 'Currently NOT supported.'
            sample = self._all_fock_basis[torch.randint(0, len(self._all_fock_basis), (1,))[0]]
        elif self.backend == 'gaussian':
            sample = self._generate_rand_sample(self.detector)
        return tuple(sample.tolist())

    def _generate_rand_sample(self, detector: str = 'pnrd'):
        """Generate a random sample according to uniform proposal distribution."""
        if self._if_delayloop:
            nmode = self._nmode_tdm
        else:
            nmode = self.nmode
        if detector == 'threshold':
            sample = torch.randint(0, 2, [nmode])
        elif detector == 'pnrd':
            sample = torch.randint(0, self.cutoff, [nmode])
        return sample

    def _generate_chain_sample(self, wires: Union[int, List[int], None] = None) -> torch.Tensor:
        """Generate batched random samples via chain rule.

        Args:
            wires (int, List[int] or None, optional): The wires to measure. It can be an integer or a list of
                integers specifying the indices of the wires. Default: ``None`` (which means all wires are
                measured)

        Returns:
            torch.Tensor: Tensor of shape (batch, nwire).
        """
        if wires is None:
            wires = self.wires
        wires = sorted(self._convert_indices(wires))
        sample = []
        if self.backend == 'fock':
            assert self.mps
            mps = copy(self.state)
            if mps[0].ndim == 3:
                mps = [site.unsqueeze(0) for site in mps]
            for i in wires:
                p = vmap(get_prob_mps)(mps, wire=i)
                sample_single_wire = torch.multinomial(p, num_samples=1)
                sample.append(sample_single_wire)
                index = sample_single_wire.reshape(-1, 1, 1, 1).expand(-1, mps[i].shape[-3], -1, mps[i].shape[-1])
                mps[i] = torch.gather(mps[i], dim=2, index=index)
            sample = torch.stack(sample, dim=-1).squeeze(1)
        elif self.backend == 'gaussian': # chain rule for GBS
            sample = self._generate_chain_sample_gaussian(wires)
        return sample

    def _generate_chain_sample_gaussian(self, wires: List[int]) -> torch.Tensor:
        """Generate batched random samples via chain rule for Gaussian backend.

        See https://research-information.bris.ac.uk/en/studentTheses/classical-simulations-of-gaussian-boson-sampling
        Chapter 5
        """
        def _sample_wire(sample, cov_sub, mean_sub, cutoff, detector):
            """Sample for a wire"""
            states = [torch.tensor(sample + [i], device=cov_sub.device) for i in range(cutoff)]
            probs = [self._get_probs_gaussian_helper(s, cov_sub, mean_sub, detector) for s in states]
            sample_wire = torch.multinomial(torch.cat(probs), num_samples=1)
            return sample_wire

        def _sample_pure(cov, mean, wires, nmode, cutoff, detector):
            """Sample for a pure state"""
            wires = torch.tensor(wires, device=cov.device)
            sample = []
            for i in range(1, len(wires) + 1):
                idx = torch.cat([wires[:i], wires[:i] + nmode])
                cov_sub = cov[idx[:, None], idx]
                mean_sub = mean[idx, :]
                sample_wire = _sample_wire(sample, cov_sub, mean_sub, cutoff, detector)
                sample.append(sample_wire)
            return torch.cat(sample)

        def _sample_mixed(cov, mean, wires, nmode, cutoff, detector, eps = 5e-5):
            """Sample for a mixed state"""
            wires = torch.tensor(wires, device=cov.device)
            _, s = williamson(cov)
            cov_t = s @ s.mT * dqp.hbar / (4 * dqp.kappa**2)
            cov_w = cov - cov_t # cov_mix = cov_t + cov_w
            cov_w += cov.new_ones(cov_w.shape[-1]).diag_embed() * eps
            mean0 = MultivariateNormal(mean.squeeze(-1), cov_w).sample([1])[0] # may be numerically unstable
            sample = []
            mean_m = None
            for i in range(1, len(wires) + 1):
                wires_i = wires[i:].tolist()
                cov_m = cov.new_ones(2 * len(wires_i)).diag_embed() * dqp.hbar / (4 * dqp.kappa**2) # See Eq.(5.18)
                heterodyne = Generaldyne(cov_m=cov_m, nmode=nmode, wires=wires_i)
                # collapse the state
                state = [cov_t.unsqueeze(0), mean0.reshape(1, -1, 1)]
                if i < len(wires):
                    cov_out, mean_out = heterodyne(state, mean_m)
                    mean_m = heterodyne.samples[0] # with batch
                    mask = torch.ones_like(mean_m, dtype=bool)
                    idx_discard = torch.tensor([0, len(mean_m) // 2], device=mask.device)
                    mask[idx_discard] = False
                    mean_m = mean_m[mask] # discard the first mode
                else:
                    cov_out, mean_out = state
                idx = torch.cat([wires[:i], wires[:i] + nmode])
                cov_sub = cov_out[0, idx[:, None], idx]
                mean_sub = mean_out[0, idx, :]
                sample_wire = _sample_wire(sample, cov_sub, mean_sub, cutoff, detector)
                sample.append(sample_wire)
            return torch.cat(sample)

        sample = []
        purity = GaussianState(self.state).is_pure
        cov, mean = self.state
        batch = cov.shape[0]
        if purity:
            for i in range(batch):
                sample.append(_sample_pure(cov[i], mean[i], wires, self.nmode, self.cutoff, self.detector))
        else:
            for i in range(batch):
                sample.append(_sample_mixed(cov[i], mean[i], wires, self.nmode, self.cutoff, self.detector))
        sample = torch.stack(sample)
        return sample

    def photon_number_mean_var(
        self,
        wires: Union[int, List[int], None] = None
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get the expectation value and variance of the photon number operator.

        Args:
            wires (int, List[int] or None, optional): The wires to measure. It can be an integer or a list of
                integers specifying the indices of the wires. Default: ``None`` (which means all wires are
                measured)
        """
        assert self.backend in ('gaussian', 'bosonic')
        if self.state is None:
            return
        assert isinstance(self.state, list), 'NOT valid when "is_prob" is True'
        if self.backend == 'gaussian':
            cov, mean = self.state
        elif self.backend == 'bosonic':
            cov, mean, weight = self.state
        if wires is None:
            wires = self.wires
        wires = sorted(self._convert_indices(wires))
        if self._if_delayloop:
            wires = [self._unroll_dict[wire][-1] for wire in wires]
        shape_cov = cov.shape
        shape_mean = mean.shape
        batch = shape_cov[0]
        nwire = len(wires)
        cov = cov.reshape(-1, *shape_cov[-2:])
        mean = mean.reshape(-1, *shape_mean[-2:])
        covs, means = self._get_local_covs_means(cov, mean, wires)
        if self.backend == 'gaussian':
            weights = None
        elif self.backend == 'bosonic':
            covs = covs.reshape(*shape_cov[:2], nwire, 2, 2).transpose(1, 2)
            covs = covs.reshape(-1, shape_cov[-3], 2, 2) # (batch*nwire, ncomb, 2, 2)
            means = means.reshape(*shape_mean[:2], nwire, 2, 1).transpose(1, 2)
            means = means.reshape(-1, shape_mean[-3], 2, 1)
            if weight.shape[0] == 1:
                weights = weight
            else:
                weights = torch.stack([weight] * nwire, dim=-2).reshape(batch * nwire, weight.shape[-1])
            ncomb = weights.shape[-1]
            if covs.shape[1] == 1:
                covs = covs.expand(-1, ncomb, -1, -1)
            if means.shape[1] == 1:
                means = means.expand(-1, ncomb, -1, -1)
        exp, var = photon_number_mean_var(covs, means, weights)
        exp = exp.reshape(batch, nwire).squeeze()
        var = var.reshape(batch, nwire).squeeze()
        return exp, var

    def _get_local_covs_means(
        self,
        cov: torch.Tensor,
        mean: torch.Tensor,
        wires: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the local covariance matrices and mean vectors of a Gaussian state according to the wires to measure."""
        def extract_blocks(mat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            """Extract specified blocks from the input tensor.

            Args:
                mat (torch.Tensor): Input tensor.
                idx (torch.Tensor): Index tensor of shape (nblock, block_size), where each row contains
                    the row/column indices for a block.

            Returns:
                torch.Tensor: Output tensor of shape (batch, nblock, block_size, -1) containing all extracted blocks.
            """
            nblock, block_size = idx.shape
            if mat.shape[-2] == mat.shape[-1]: # cov
                rows = idx[:, :, None].expand(-1, -1, block_size) # (nblock, block_size, block_size)
                cols = idx[:, None, :].expand(-1, block_size, -1) # (nblock, block_size, block_size)
                all_rows = rows.reshape(-1)
                all_cols = cols.reshape(-1)
                out = mat[:, all_rows, all_cols]
            elif mat.shape[-1] == 1: # mean
                out = mat[:, idx, :]
            return out.reshape(mat.shape[0], nblock, block_size, -1)

        indices = []
        if self._if_delayloop:
            nmode = self._nmode_tdm
        else:
            nmode = self.nmode
        for wire in wires:
            indices.append([wire] + [wire + nmode])
        indices = torch.tensor(indices, device=cov.device)
        covs = extract_blocks(cov, indices).reshape(-1, 2, 2) # batch * nwire
        means = extract_blocks(mean, indices).reshape(-1, 2, 1)
        return covs, means

    def measure_homodyne(
        self,
        shots: int = 10,
        wires: Union[int, List[int], None] = None
    ) -> Optional[torch.Tensor]:
        """Get the homodyne measurement results.

        If ``self.measurements`` is specified via ``self.homodyne``, return the results of
        the conditional homodyne measurement. Otherwise, return the results of the ideal homodyne measurement.
        The Gaussian states after measurements are stored in ``self.state_measured``.

        Note:
            ``batch`` * ``shots`` can not be too large for Fock backend.

        Args:
            shots (int, optional): The number of times to sample from the quantum state. Default: 10
            wires (int, List[int] or None, optional): The wires to measure for the ideal homodyne. It can be
                an integer or a list of integers specifying the indices of the wires. Default: ``None`` (which means
                all wires are measured)
        """
        if self.state is None:
            return
        assert isinstance(self.state, (list, torch.Tensor)), 'NOT valid when "is_prob" is True'
        if len(self.measurements) > 0:
            if self._if_delayloop:
                measurements = self._measurements_tdm
            else:
                measurements = self.measurements
            samples = []
            if self.backend == 'fock':
                assert not self.basis
                assert not self.mps, 'Currently NOT supported.'
                shape = self.state.shape
                batch = shape[0]
                self.state_measured = torch.stack([self.state] * shots).reshape(-1, *shape[1:])
            else:
                batch = self.state[0].shape[0]
                self.state_measured = []
                if self.backend == 'bosonic':
                    state = align_shape(*self.state)
                else:
                    state = self.state
                for s in state: # [cov, mean, weight]
                    shape = s.shape
                    self.state_measured.append(torch.stack([s] * shots).reshape(-1, *shape[1:]))
            for op_m in measurements:
                self.state_measured = op_m(self.state_measured)
                nwire = len(op_m.wires)
                samples.append(op_m.samples[:, :nwire].reshape(shots, batch, nwire).permute(1, 0, 2))
            return torch.cat(samples, dim=-1).squeeze() # (batch, shots, nwire)
        else:
            if wires is None:
                wires = self.wires
            wires = torch.tensor(sorted(self._convert_indices(wires)))
            if self.backend == 'fock':
                assert not self.basis
                assert len(wires) == 1
                # (batch, shots, 1)
                samples = sample_homodyne_fock(self.state, wires[0], self.nmode, self.cutoff, shots, self.den_mat)
            else:
                cov, mean = self.state[:2]
                if not is_positive_definite(cov):
                    size = cov.size()
                    if cov.dtype == torch.double:
                        epsilon = 1e-16
                    elif cov.dtype == torch.float:
                        epsilon = 1e-8
                    else:
                        raise ValueError('Unsupported dtype.')
                    cov += epsilon * cov.new_ones(size[:-1].numel()).reshape(size[:-1]).diag_embed()
                idx = torch.cat([wires, wires + self.nmode])
                cov_sub = cov[..., idx[:, None], idx]
                mean_sub = mean[..., idx, :]
                if len(self.state) == 2:
                    # (shots, batch, 2 * nwire)
                    samples = MultivariateNormal(mean_sub.squeeze(-1), cov_sub).sample([shots])
                    samples = samples.permute(1, 0, 2)
                elif len(self.state) == 3:
                    cov_sub, mean_sub, weight = align_shape(cov_sub, mean_sub, self.state[2])
                    samples = sample_reject_bosonic(cov_sub, mean_sub, weight, cov_sub.new_zeros(1), shots)
            return samples.squeeze()

    @property
    def max_depth(self) -> int:
        """Get the max number of gates on the wires."""
        return max(self.depth)

    def draw(self, filename: Optional[str] = None, unroll: bool = False):
        """Visualize the photonic quantum circuit.

        Args:
            filename (str or None, optional): The path for saving the figure.
            unroll (bool, optional): Whether to draw the unrolled circuit.
        """
        if self._if_delayloop and unroll:
            self._prepare_unroll_dict()
            self._unroll_circuit()
            nmode = self._nmode_tdm
            operators = self._operators_tdm
            measurements = self._measurements_tdm
        else:
            nmode = self.nmode
            operators = self.operators
            measurements = self.measurements
        self.draw_circuit = DrawCircuit(self.name, nmode, operators, measurements)
        self.draw_circuit.draw()
        if filename is not None:
            self.draw_circuit.save(filename)
        else:
            if self.nmode > 50:
                print('Too many modes in the circuit, please set filename to save the figure.')
        return self.draw_circuit.draw_

    def cat(
        self,
        wires: int,
        r: Any = None,
        theta: Any = None,
        p: int = 1
    ) -> None:
        """Prepare a cat state.

        ``r`` and ``theta`` are the displacement magnitude and angle respectively.
        ``p`` is the parity, corresponding to an even or odd cat state when ``p=0`` or ``p=1`` respectively.
        """
        if self._bosonic_states is None:
            self._bosonic_states = [BosonicState(state='vac', nmode=1, cutoff=self.cutoff)] * self.nmode
        cat = CatState(r=r, theta=theta, p=p, cutoff=self.cutoff)
        self._bosonic_states[wires] = cat

    def gkp(
        self,
        wires: int,
        theta: Any = None,
        phi: Any = None,
        amp_cutoff: float = 0.1,
        epsilon: float = 0.05
    ) -> None:
        """Prepare a GKP state.

        ``theta`` and ``phi`` are angles in Bloch sphere.
        ``amp_cutoff`` is the amplitude threshold for keeping the terms.
        ``epsilon`` is the finite energy damping parameter.
        """
        if self._bosonic_states is None:
            self._bosonic_states = [BosonicState(state='vac', nmode=1, cutoff=self.cutoff)] * self.nmode
        gkp = GKPState(theta=theta, phi=phi, amp_cutoff=amp_cutoff, epsilon=epsilon, cutoff=self.cutoff)
        self._bosonic_states[wires] = gkp

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
            encode (bool, optional): Whether the gate is to encode data. Default: ``False``
            wires (int, List[int] or None, optional): The wires to apply the gate on. It can be an integer
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
            self.measurements = op.measurements
            self.npara += op.npara
            self.ndata += op.ndata
            self.depth += op.depth
            self._lossy = self._lossy or op._lossy
            self._nloss += op._nloss
            self._if_delayloop = self._if_delayloop or op._if_delayloop
            self._nmode_tdm += op._nmode_tdm - self.nmode
            for key, value in op._ntau_dict.items():
                self._ntau_dict[key].extend(value)
            self._unroll_dict = None
            self._operators_tdm = None
            self._measurements_tdm = None
            self.wires_homodyne = op.wires_homodyne
        elif isinstance(op, (Gate, Channel, Delay)):
            self.operators.append(op)
            for i in op.wires:
                self.depth[i] += 1
            if encode:
                assert not op.requires_grad, 'Please set requires_grad of the operation to be False'
                self.encoders.append(op)
                self.ndata += op.npara
            else:
                self.npara += op.npara
            if isinstance(op, Delay):
                self._if_delayloop = True
                self._nmode_tdm += op.ntau
                self._ntau_dict[op.wires[0]].append(op.ntau)
        elif isinstance(op, Homodyne):
            self.measurements.append(op)
            self.wires_homodyne.append(op.wires[0])

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
        ps = PhaseShift(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
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
        bs = BeamSplitter(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
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
        mzi = MZI(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
                  phi_first=phi_first, requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
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
        bs = BeamSplitterTheta(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
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
        bs = BeamSplitterPhi(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
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
        bs = BeamSplitterSingle(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
                                convention='rx', requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
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
        bs = BeamSplitterSingle(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
                                convention='ry', requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
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
        bs = BeamSplitterSingle(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
                                convention='h', requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
        self.add(bs, encode=encode)

    def dc(self, wires: List[int], mu: Optional[float] = None, sigma: Optional[float] = None) -> None:
        """Add a directional coupler."""
        theta = torch.pi / 2
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        bs = BeamSplitterSingle(inputs=theta, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
                                convention='rx', requires_grad=False, noise=self.noise, mu=mu, sigma=sigma)
        self.add(bs)

    def h(self, wires: List[int], mu: Optional[float] = None, sigma: Optional[float] = None) -> None:
        """Add a photonic Hadamard gate."""
        theta = torch.pi / 2
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        bs = BeamSplitterSingle(inputs=theta, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
                                convention='h', requires_grad=False, noise=self.noise, mu=mu, sigma=sigma)
        self.add(bs)

    def any(
        self,
        unitary: Any,
        wires: Union[int, List[int], None] = None,
        minmax: Optional[List[int]] = None,
        name: str = 'uany'
    ) -> None:
        """Add an arbitrary unitary gate."""
        uany = UAnyGate(unitary=unitary, nmode=self.nmode, wires=wires, minmax=minmax, cutoff=self.cutoff,
                        den_mat=self.den_mat, name=name)
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
        r: Any = None,
        theta: Any = None,
        encode: bool = False,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a squeezing gate."""
        requires_grad = not encode
        if r is None and theta is None:
            inputs = None
        else:
            requires_grad = False
            if r is None:
                inputs = [torch.rand(1)[0], theta]
            elif theta is None:
                inputs = [r, 0]
            else:
                inputs = [r, theta]
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        s = Squeezing(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
                      requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
        self.add(s, encode=encode)

    def s2(
        self,
        wires: List[int],
        r: Any = None,
        theta: Any = None,
        encode: bool = False,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a two-mode squeezing gate."""
        requires_grad = not encode
        if r is None and theta is None:
            inputs = None
        else:
            requires_grad = False
            if r is None:
                inputs = [torch.rand(1)[0], theta]
            elif theta is None:
                inputs = [r, 0]
            else:
                inputs = [r, theta]
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        s2 = Squeezing2(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
                        requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
        self.add(s2, encode=encode)

    def d(
        self,
        wires: int,
        r: Any = None,
        theta: Any = None,
        encode: bool = False,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a displacement gate."""
        requires_grad = not encode
        if r is None and theta is None:
            inputs = None
        else:
            requires_grad = False
            if r is None:
                inputs = [torch.rand(1)[0], theta]
            elif theta is None:
                inputs = [r, 0]
            else:
                inputs = [r, theta]
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        d = Displacement(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
                         requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
        self.add(d, encode=encode)

    def x(
        self,
        wires: int,
        inputs: Any = None,
        encode: bool = False,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a position displacement gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        dx = DisplacementPosition(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff,
                                  den_mat=self.den_mat, requires_grad=requires_grad, noise=self.noise,
                                  mu=mu, sigma=sigma)
        self.add(dx, encode=encode)

    def z(
        self,
        wires: int,
        inputs: Any = None,
        encode: bool = False,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a momentum displacement gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        dp = DisplacementMomentum(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff,
                                  den_mat=self.den_mat, requires_grad=requires_grad, noise=self.noise,
                                  mu=mu, sigma=sigma)
        self.add(dp, encode=encode)

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
        r = PhaseShift(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
                       requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma, inv_mode=inv_mode)
        self.add(r, encode=encode)

    def f(self, wires: int, mu: Optional[float] = None, sigma: Optional[float] = None) -> None:
        """Add a Fourier gate."""
        theta = torch.pi / 2
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        f = PhaseShift(inputs=theta, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
                       requires_grad=False, noise=self.noise, mu=mu, sigma=sigma)
        self.add(f)

    def qp(
        self,
        wires: int,
        inputs: Any = None,
        encode: bool = False,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a quadratic phase gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        qp = QuadraticPhase(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
                            requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
        self.add(qp, encode=encode)

    def cx(
        self,
        wires: List[int],
        inputs: Any = None,
        encode: bool = False,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a controlled-X gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        cx = ControlledX(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
                         requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
        self.add(cx, encode=encode)

    def cz(
        self,
        wires: List[int],
        inputs: Any = None,
        encode: bool = False,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a controlled-Z gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        cz = ControlledZ(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
                         requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
        self.add(cz, encode=encode)

    def cp(
        self,
        wires: int,
        inputs: Any = None,
        encode: bool = False,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a cubic phase gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        cp = CubicPhase(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
                        requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
        self.add(cp, encode=encode)

    def k(
        self,
        wires: int,
        inputs: Any = None,
        encode: bool = False,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a Kerr gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        k = Kerr(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
                 requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
        self.add(k, encode=encode)

    def ck(
        self,
        wires: List[int],
        inputs: Any = None,
        encode: bool = False,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a cross-Kerr gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        ck = CrossKerr(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
                       requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
        self.add(ck, encode=encode)

    def delay(
        self,
        wires: int,
        ntau: int = 1,
        inputs: Any = None,
        convention: str = 'bs',
        encode: bool = False,
        loop_gates: Optional[List] = None,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a delay loop."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        if convention == 'bs':
            delay = DelayBS(inputs=inputs, ntau=ntau, nmode=self.nmode, wires=wires, cutoff=self.cutoff,
                            den_mat=self.den_mat, requires_grad=requires_grad, loop_gates=loop_gates,
                            noise=self.noise, mu=mu, sigma=sigma)
        elif convention == 'mzi':
            delay = DelayMZI(inputs=inputs, ntau=ntau, nmode=self.nmode, wires=wires, cutoff=self.cutoff,
                             den_mat=self.den_mat, requires_grad=requires_grad, loop_gates=loop_gates,
                             noise=self.noise, mu=mu, sigma=sigma)
        self.add(delay, encode=encode)

    def homodyne(
        self,
        wires: int,
        phi: Any = None,
        eps: float = 2e-4,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a homodyne measurement."""
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        homodyne = Homodyne(phi=phi, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
                            eps=eps, requires_grad=False, noise=self.noise, mu=mu, sigma=sigma)
        self.add(homodyne)

    def homodyne_x(
        self,
        wires: int,
        eps: float = 2e-4,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a homodyne measurement for quadrature x."""
        phi = 0.
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        homodyne = Homodyne(phi=phi, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
                            eps=eps, requires_grad=False, noise=self.noise, mu=mu, sigma=sigma)
        self.add(homodyne)

    def homodyne_p(
        self,
        wires: int,
        eps: float = 2e-4,
        mu: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> None:
        """Add a homodyne measurement for quadrature p."""
        phi = np.pi / 2
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        homodyne = Homodyne(phi=phi, nmode=self.nmode, wires=wires, cutoff=self.cutoff, den_mat=self.den_mat,
                            eps=eps, requires_grad=False, noise=self.noise, mu=mu, sigma=sigma)
        self.add(homodyne)

    def loss(
        self,
        wires: int,
        inputs: Any = None,
        encode: bool = False
    ) -> None:
        """Add a photon loss channel.

        The `inputs` corresponds to `theta` of the loss channel.
        """
        if self.backend == 'fock' and not self.basis:
            assert self.den_mat, 'Please use the density matrix representation'
        self._lossy = True
        self._nloss += 1
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        loss = PhotonLoss(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff,
                          requires_grad=requires_grad)
        self.add(loss, encode=encode)

    def loss_t(
        self,
        wires: int,
        inputs: Any = None,
        encode: bool = False
    ) -> None:
        """Add a photon loss channel.

        The `inputs` corresponds to the transmittance of the loss channel.
        """
        if self.backend == 'fock' and not self.basis:
            assert self.den_mat, 'Please use the density matrix representation'
        self._lossy = True
        self._nloss += 1
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs, dtype=torch.float)
            theta = torch.arccos(inputs ** 0.5) * 2
        loss = PhotonLoss(inputs=theta, nmode=self.nmode, wires=wires, cutoff=self.cutoff,
                          requires_grad=requires_grad)
        self.add(loss, encode=encode)

    def loss_db(
        self,
        wires: int,
        inputs: Any = None,
        encode: bool = False
    ) -> None:
        """Add a photon loss channel.

        The `inputs` corresponds to the probability of loss with the unit of dB and is positive.
        """
        if self.backend == 'fock' and not self.basis:
            assert self.den_mat, 'Please use the density matrix representation'
        self._lossy = True
        self._nloss += 1
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs, dtype=torch.float)
            t = 10 ** (-inputs / 10)
            theta = torch.arccos(t ** 0.5) * 2
        loss = PhotonLoss(inputs=theta, nmode=self.nmode, wires=wires, cutoff=self.cutoff,
                          requires_grad=requires_grad)
        self.add(loss, encode=encode)

    def barrier(self, wires: Union[int, List[int], None] = None) -> None:
        """Add a barrier."""
        br = Barrier(nmode=self.nmode, wires=wires, cutoff=self.cutoff)
        self.add(br)

class DistributedQumodeCircuit(QumodeCircuit):
    """Photonic quantum circuit for a distributed Fock state.

    Args:
        nmode (int): The number of modes in the circuit.
        init_state (Any): The initial state of the circuit. It can be a vacuum state with ``'vac'`` or ``'zeros'``.
            It can be a Fock basis state, e.g., ``[1,0,0]``, or a Fock state tensor,
            e.g., ``[(1/2**0.5, [1,0]), (1/2**0.5, [0,1])]``.
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        name (str or None, optional): The name of the circuit. Default: ``None``
    """
    def __init__(self, nmode: int, init_state: Any, cutoff: Optional[int] = None, name: Optional[str] = None) -> None:
        super().__init__(nmode, init_state, cutoff, backend='fock', basis=False, den_mat=False,
                         detector='pnrd', name=name, mps=False, chi=None, noise=False, mu=0, sigma=0.1)

    def set_init_state(self, init_state: Any = None) -> None:
        """Set the initial state of the circuit."""
        if isinstance(init_state, DistributedFockState):
            self.init_state = init_state
        else:
            self.init_state = DistributedFockState(init_state, self.nmode, self.cutoff)
            self.cutoff = self.init_state.cutoff

    # pylint: disable=arguments-renamed
    @torch.no_grad()
    def forward(
        self,
        data: Optional[torch.Tensor] = None,
        state: Optional[DistributedFockState] = None
    ) -> DistributedFockState:
        """Perform a forward pass of the photonic quantum circuit and return the final state.

        This method applies the ``operators`` of the photonic quantum circuit to the initial state or the given state
        and returns the resulting state. If ``data`` is given, it is used as the input for the ``encoders``.
        The ``data`` must be a 1D tensor.

        Args:
            data (torch.Tensor or None, optional): The input data for the ``encoders``. Default: ``None``
            state (DistributedFockState or None, optional): The initial state for the photonic quantum circuit.
                Default: ``None``
        """
        if state is None:
            self.init_state.reset()
        else:
            self.init_state = state
        self.encode(data)
        self.state = self.operators(self.init_state)
        return self.state

    def measure(
        self,
        shots: int = 1024,
        with_prob: bool = False,
        wires: Union[int, List[int], None] = None,
        block_size: int = 2 ** 24
    ) -> Union[Dict, None]:
        """Measure the final state.

        Args:
            shots (int, optional): The number of times to sample from the quantum state. Default: 1024
            with_prob (bool, optional): A flag that indicates whether to return the probabilities along with
                the number of occurrences. Default: ``False``
            wires (int, List[int] or None, optional): The wires to measure. It can be an integer or a list of
                integers specifying the indices of the wires. Default: ``None`` (which means all wires are measured)
            block_size (int, optional): The block size for sampling. Default: 2 ** 24
        """
        if wires is None:
            wires = list(range(self.nmode))
        wires = sorted(self._convert_indices(wires))
        if self.state is None:
            return
        else:
            if self.state.world_size == 1:
                return measure_fock_tensor(self.state.amps.unsqueeze(0), shots, with_prob, wires, block_size)
            else:
                return measure_dist(self.state, shots, with_prob, wires, block_size)
