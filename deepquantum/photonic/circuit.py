"""
Photonic quantum circuit
"""

import itertools
import random
from collections import defaultdict, Counter
from copy import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn, vmap

from .decompose import UnitaryDecomposer
from .draw import DrawCircuit
from .gate import PhaseShift, BeamSplitter, MZI, BeamSplitterTheta, BeamSplitterPhi, BeamSplitterSingle, UAnyGate
from .gate import Squeezing, Displacement
from .operation import Operation, Gate
from .qmath import fock_combinations, permanent, product_factorial, sort_dict_fock_basis, sub_matrix
from .state import FockState, GaussianState


class QumodeCircuit(Operation):
    """Photonic quantum circuit.

    Args:
        nmode (int): The number of modes in the circuit.
        init_state (Any): The initial state of the circuit. For Fock backend, it can be a Fock basis state,
            e.g., ``[1,0,0]``, or a Fock state tensor, e.g., ``[(1/2**0.5, [1,0]), (1/2**0.5, [0,1])]``.
            Alternatively, it can be a tensor representation.
            For Gaussian backend, it can be a vacuum state with 'vac', or arbitrary Gaussian states with [cov, mean]
            Use ``xxpp`` convention and :math:`\hbar=2` by default.
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        backend (str, optional): Use Fock backend or Gaussian backend. Default: ``fock``
        basis (bool, optional): Whether to use the representation of Fock basis state for the initial state.
            Default: ``True``
        name (str or None, optional): The name of the circuit. Default: ``None``
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """
    def __init__(
        self,
        nmode: int,
        init_state: Any,
        cutoff: Optional[int] = None,
        backend: Optional[str] = 'fock',
        basis: bool = True,
        name: Optional[str] = None,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1
    ) -> None:
        super().__init__(name=name, nmode=nmode, wires=list(range(nmode)))
        if isinstance(init_state, (FockState, GaussianState)):
            assert nmode == init_state.nmode
            cutoff = init_state.cutoff
            self.init_state = init_state
            if isinstance(init_state, FockState):
                backend = 'fock'
                basis = init_state.basis
            elif isinstance(init_state, GaussianState):
                backend = 'gaussian'
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
                            name=self.name, noise=self.noise, mu=self.mu, sigma=self.sigma)
        cir.operators = self.operators + rhs.operators
        cir.encoders = self.encoders + rhs.encoders
        cir.npara = self.npara + rhs.npara
        cir.ndata = self.ndata + rhs.ndata
        cir.depth = self.depth + rhs.depth
        return cir

    def to(self, arg: Any) -> 'QumodeCircuit':
        """Set dtype or device of the ``QumodeCircuit``."""
        if arg == torch.float:
            self.init_state.to(torch.cfloat)
            for op in self.operators:
                if op.npara == 0:
                    op.to(torch.cfloat)
                elif op.npara > 0:
                    op.to(torch.float)
        elif arg == torch.double:
            self.init_state.to(torch.cdouble)
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
        stepwise: bool = False
    ) -> Union[torch.Tensor, Dict, List[torch.Tensor]]:
        """Perform a forward pass of the photonic quantum circuit and return the final state.

        Args:
            data (torch.Tensor or None, optional): The input data for the ``encoders``. Default: ``None``
            state (Any, optional): The initial state for the photonic quantum circuit. Default: ``None``
            is_prob (bool, optional): Whether to return probabilities for Fock basis states. Default: ``False``
            stepwise (bool, optional): Whether to use the forward function of each operator. Default: ``False``

        Returns:
            Union[torch.Tensor, Dict, List[torch.Tensor]]: The final state of the photonic quantum circuit after
            applying the ``operators``.
        """
        if self.backend == 'fock':
            return self.forward_fock(data, state, is_prob)
        elif self.backend == 'gaussian':
            return self.forward_gaussian(data, state, stepwise)

    def forward_fock(
        self,
        data: Optional[torch.Tensor] = None,
        state: Any = None,
        is_prob: bool = False
    ) -> Union[torch.Tensor, Dict]:
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
            state = self.init_state.state
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
                if self.state.ndim == self.nmode:
                    self.state = self.state.unsqueeze(0)
        else:
            if data.ndim == 1:
                data = data.unsqueeze(0)
            assert data.ndim == 2
            if self.basis:
                state_dict = vmap(self._forward_helper_basis, in_dims=(0, None, None))(data, state, is_prob)
                self.state = sort_dict_fock_basis(state_dict)
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
            rst = vmap(self._get_prob_vmap)(sub_mats, per_norms)
        else:
            rst = vmap(self._get_amplitude_vmap)(sub_mats, per_norms)
        for i in range(len(final_states)):
            final_state = FockState(state=final_states[i], nmode=self.nmode, cutoff=self.cutoff, basis=self.basis)
            out_dict[final_state] = rst[i]
        return out_dict

    def _forward_helper_tensor(
        self,
        data: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Perform a forward pass for one sample if the input is a Fock state tensor."""
        self.encode(data)
        if state is None:
            state = self.init_state.state
        x = self.operators(self.tensor_rep(state)).squeeze(0)
        return x

    def forward_gaussian(
        self,
        data: Optional[torch.Tensor] = None,
        state: Any = None,
        stepwise: bool = False
    ) -> List[torch.Tensor]:
        """Perform a forward pass based on the Gaussian backend.

        Args:
            data (torch.Tensor or None, optional): The input data for the ``encoders``. Default: ``None``
            state (Any, optional): The initial state for the photonic quantum circuit. Default: ``None``
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
            # self.state = self.state.unsqueeze(0)
        else:
            if data.ndim == 1:
                data = data.unsqueeze(0)
            assert data.ndim == 2
            if state[0].shape[0] == 1:
                self.state = vmap(self._forward_helper_gaussian, in_dims=(0, None, None))(data, state, stepwise)
            else:
                self.state = vmap(self._forward_helper_gaussian, in_dims=(0, 0, None))(data, state, stepwise)
            self.encode(data[-1])
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
            cov = state[0]
            mean = state[1]
        if stepwise:
            self.state = self.operators([cov, mean])
        else:
            sp_mat = self.get_symplectic()
            cov = sp_mat @ cov @ sp_mat.mT
            mean = self.get_displacement(mean)
            self.state = [cov, mean]
        return self.state

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
        """Get the final mean value of the Gaussian state in xxpp order."""
        if not isinstance(init_mean, torch.Tensor):
            init_mean = torch.tensor(init_mean)
        mean = init_mean.reshape(-1, 2 * self.nmode, 1)
        for op in self.operators:
            mean = op.get_symplectic() @ mean + op.get_displacement()
        return mean

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

    def get_unitary_op(self) -> torch.Tensor:
        """Get the unitary matrix of the photonic quantum circuit."""
        u = None
        for op in self.operators:
            print(op)
            print(op.get_unitary_op())
            if u is None:
                u = op.get_unitary_op()
            else:
                u = op.get_unitary_op() @ u
        if u is None:
            return torch.eye(self.nmode, dtype=torch.cfloat)
        else:
            return u

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
        u = self.get_unitary_op()
        for state in final_states:
            sub_mats.append(sub_matrix(u, init_state, state))
        return torch.stack(sub_mats)

    def _get_permanent_norms(self, init_state: torch.Tensor, final_states: torch.Tensor) -> torch.Tensor:
        """Get the normalization factors for permanent."""
        return torch.sqrt(product_factorial(init_state) * product_factorial(final_states))

    def get_amplitude(self, final_state: Any, init_state: Optional['FockState'] = None) -> torch.Tensor:
        """Get the transfer amplitude between the final state and the initial state.

        Args:
            final_state (Any): The final Fock basis state.
            init_state (FockState or None, optional): The initial Fock basis state. Default: ``None``
        """
        if not isinstance(final_state, torch.Tensor):
            final_state = torch.tensor(final_state, dtype=torch.int)
        if init_state is None:
            init_state = self.init_state
        assert init_state.basis, 'The initial state must be a Fock basis state'
        assert max(final_state) < self.cutoff, 'The number of photons in the final state must be less than cutoff'
        assert sum(final_state) == sum(init_state.state), 'The number of photons should be conserved'
        u = self.get_unitary_op()
        sub_mat = sub_matrix(u, init_state.state, final_state)
        nphoton = sum(init_state.state)
        if nphoton == 0:
            amp = torch.tensor(1.)
        else:
            per = permanent(sub_mat)
            amp = per / self._get_permanent_norms(init_state.state, final_state).to(per.dtype).to(per.device)
        return amp

    def _get_amplitude_vmap(self, sub_mat: torch.Tensor, per_norm: torch.Tensor) -> torch.Tensor:
        """Get the transfer amplitude."""
        per = permanent(sub_mat)
        amp = per / per_norm.to(per.dtype).to(per.device)
        return amp.reshape(-1)

    def get_prob(self, final_state: Any, init_state: Optional['FockState'] = None) -> torch.Tensor:
        """Get the transfer probability between the final state and the initial state.

        Args:
            final_state (Any): The final Fock basis state.
            init_state (FockState or None, optional): The initial Fock basis state. Default: ``None``
        """
        amplitude = self.get_amplitude(final_state, init_state)
        prob = torch.abs(amplitude) ** 2
        return prob

    def _get_prob_vmap(self, sub_mat: torch.Tensor, per_norm: torch.Tensor) -> torch.Tensor:
        """Get the transfer probability."""
        amplitude = self._get_amplitude_vmap(sub_mat, per_norm)
        prob = torch.abs(amplitude) ** 2
        return prob

    def measure(
        self,
        shots: int = 1024,
        with_prob: bool = False,
        wires: Union[int, List[int], None] = None
    ) -> Union[Dict, List[Dict], None]:
        """Measure the final state.

        Args:
            shots (int, optional): The number of times to sample from the quantum state. Default: 1024
            with_prob (bool, optional): A flag that indicates whether to return the probabilities along with
                the number of occurrences. Default: ``False``
            wires (int, List[int] or None, optional): The wires to measure. It can be an integer or a list of
                integers specifying the indices of the wires. Default: ``None`` (which means all wires are
                measured)
        """
        if self.state is None:
            return
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
                    state_str = ''.join(map(str, p_state))
                    p_str = f'|{state_str}>'
                    prob_dict[p_str] = ptrace_probs[tuple(p_state)]
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

    def draw(self, filename: Optional[str] = None):
        """Visualize the photonic quantum circuit.

        Args:
            filename (str or None, optional): The path for saving the figure.
        """
        self.draw_circuit = DrawCircuit(self.name, self.nmode, self.operators)
        if filename is not None:
            self.draw_circuit.save(filename)
        else:
            if self.nmode > 50:
                print('Too many modes in the circuit, please save the figure.')
        self.draw_circuit.draw()
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
                       requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma, inv_mode=True)
        self.add(r, encode=encode)
