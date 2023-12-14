import itertools
import random
from collections import defaultdict, Counter
from copy import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn, vmap

from .draw import DrawCircuit
from .gate import PhaseShift, BeamSplitter, BeamSplitterTheta, BeamSplitterPhi, UAnyGate
from .operation import Operation, Gate
from .qmath import fock_combinations, permanent, product_factorial, sort_dict_fock_basis, sub_matrix
from .state import FockState


class QumodeCircuit(Operation):
    """
    Constructing quantum optical circuit
    Args:
        nmode: int, the total wires of the circuit
        cutoff: int, the maximum number of photons in each mode
        init_state: The initial state of the circuit. It can be a Fock basis state, e.g., [1,0,0],
            or a Fock state tensor, e.g., [(sqrt(2)/2, [1,0]), (sqrt(2)/2, [0,1])].
        name: str, the name of the circuit
        basis: Whether to use the representation of Fock basis state for the initial state.
            Default: ``False``
    """
    def __init__(
        self,
        nmode: int,
        init_state: Any,
        cutoff: int = None,
        basis: bool = False,
        name: Optional[str] = None,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1
    ) -> None:
        super().__init__(name=name, nmode=nmode, wires=list(range(nmode)))
        if isinstance(init_state, FockState):
            assert nmode == init_state.nmode
            cutoff = init_state.cutoff
            basis = init_state.basis
            self.init_state = init_state
        else:
            self.init_state = FockState(state=init_state, nmode=nmode, cutoff=cutoff, basis=basis)
            cutoff = self.init_state.cutoff
        self.operators = nn.Sequential()
        self.encoders = []
        self.cutoff = cutoff
        self.basis = basis
        self.noise = noise
        self.mu = mu
        self.sigma = sigma
        self.state = None
        self.u = None
        self.npara = 0
        self.ndata = 0
        self.depth = np.array([0] * nmode)

    # pylint: disable=arguments-renamed
    def forward(self, data = None, state = None, is_prob = False) -> Union[torch.Tensor, FockState]:
        """Perform a forward pass of the quantum circuit and return the final state.
        Args:
            state: the state to be evolved.  Default: ``None``
            data: the circuit parameters(angles).  Default: ``None``
        """
        if state is None:
            state = self.init_state
        else:
            state = FockState(state=state, nmode=self.nmode, cutoff=self.cutoff, basis=self.basis)
        if data is None:
            if self.basis:
                state = self._forward_helper_basis(state=state, is_prob=is_prob)
                self.state = sort_dict_fock_basis(state)
            else:
                self.state = self._forward_helper_tensor(state=state.state)
        else:
            if self.basis:
                 state = vmap(self._forward_helper_basis, in_dims=(0, None, None))(data, state, is_prob)
                 self.state = sort_dict_fock_basis(state)
                 # for plotting the last data in the circuit
                 self.encode(data[-1])
                 self.u = self.get_unitary_op()
            else:
                if state.state.shape[0] == 1:
                    self.state = vmap(self._forward_helper_tensor, in_dims=(0, None))(data, state.state)
                else:
                    self.state = vmap(self._forward_helper_tensor)(data, state.state)
                self.encode(data[-1])
        return self.state

    def _forward_helper_basis(self, data = None, state = None, is_prob = False):
        """Perform a forward pass for one sample if the input is a Fock basis state."""
        self.encode(data)
        out_dict = {}
        self.u = self.get_unitary_op()
        final_states = self.get_all_fock_basis(state)
        sub_mats = self.get_sub_matrices(final_states, state)
        per_norms = self._get_permanent_norms(final_states, state)
        if is_prob:
            rst = vmap(self._get_prob_vmap, in_dims=(0, 0, None))(sub_mats, per_norms, state)
        else:
            rst = vmap(self._get_amplitude_vmap, in_dims=(0, 0, None))(sub_mats, per_norms, state)
        for i in range(len(final_states)):
            final_state = FockState(state=final_states[i], nmode=self.nmode, cutoff=self.cutoff, basis=self.basis)
            out_dict[final_state] = rst[i]
        return out_dict

    def _forward_helper_tensor(self, data = None, state = None):
        """Perform a forward pass for one sample if the input is a Fock state tensor."""
        self.encode(data)
        x = self.operators(state).squeeze(0)
        return x

    def encode(self, data: torch.Tensor) -> None:
        """Encode the input data into thecircuit parameters.

        This method iterates over the ``encoders`` of the circuit and initializes their parameters
        with the input data. Here we assume phaseshifter and beamsplitter with single parameters

        Args:
            data (torch.Tensor): The input data for the ``encoders``, must be a 1D tensor.
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
        """
        Get the unitary matrix of the nmode optical quantum circuit.
        """
        u = None
        for op in self.operators:
            if u is None:
                u = op.get_unitary_op()
            else:
                u = op.get_unitary_op() @ u
        self.u = u
        return u

    def get_all_fock_basis(self, init_state = None):
        """Get all possible fock basis states according to the initial state."""
        if init_state is None:
            init_state = self.init_state
        assert init_state.basis
        nmode = init_state.nmode
        nphoton = sum(init_state.state)
        states = torch.tensor(fock_combinations(nmode, nphoton), dtype=torch.int)
        max_values, _ = torch.max(states, dim=1)
        mask = max_values < self.cutoff
        return torch.masked_select(states, mask.unsqueeze(1)).view(-1, states.shape[-1])

    def get_sub_matrices(self, final_states, init_state = None):
        if init_state is None:
            init_state = self.init_state
        assert init_state.basis
        if self.u is None:
            u = self.get_unitary_op()
        else:
            u = self.u
        sub_mats = []
        for state in final_states:
            sub_mats.append(sub_matrix(u, init_state.state, state))
        return torch.stack(sub_mats)

    def _get_permanent_norms(self, final_states, init_state = None):
        if not isinstance(final_states, torch.Tensor):
            final_states = torch.tensor(final_states, dtype=torch.int)
        if init_state is None:
            init_state = self.init_state
        assert init_state.basis
        return torch.sqrt((product_factorial(init_state.state) * product_factorial(final_states)))

    def get_amplitude(self, final_state, init_state = None):
        if not isinstance(final_state, torch.Tensor):
            final_state = torch.tensor(final_state, dtype=torch.int)
        if init_state is None:
            init_state = self.init_state
        assert init_state.basis
        assert(max(final_state) < self.cutoff), 'the number of photons in the final state must be less than cutoff'
        assert(sum(final_state) == sum(init_state.state)), 'the number of photons should be conserved'
        if self.u is None:
            u = self.get_unitary_op()
        else:
            u = self.u
        sub_mat = sub_matrix(u, init_state.state, final_state)
        nphoton = sum(init_state.state)
        if nphoton == 0:
            amp = torch.tensor(1.)
        else:
            per = permanent(sub_mat)
            amp = per / np.sqrt((product_factorial(init_state.state) * product_factorial(final_state)))
        return amp

    def _get_amplitude_vmap(self, sub_mat, per_norm, init_state = None):
        """Calculating the transfer amplitude of the given final state

        final_state: fock state, list or torch.tensor
        """
        if init_state is None:
            init_state = self.init_state
        nphoton = sum(init_state.state)
        if nphoton == 0:
            amp = torch.tensor(1.)
        else:
            per = permanent(sub_mat)
            amp = per / per_norm
        return amp.reshape(-1)

    def get_prob(self, final_state, init_state = None):
        """Calculating the transfer probability of the given final state

        final_state: fock state, list or torch.tensor
        """
        amplitude = self.get_amplitude(final_state, init_state)
        prob = torch.abs(amplitude) ** 2
        return prob

    def _get_prob_vmap(self, sub_mat, per_norm, init_state = None):
        """Calculating the transfer probability of the given final state

        final_state: fock state, list or torch.tensor
        """
        amplitude = self._get_amplitude_vmap(sub_mat, per_norm, init_state)
        prob = torch.abs(amplitude) ** 2
        return prob

    def measure(self, wires = None, shots = 1024):
        """
        measure several wires outputs, default shots = 1024
        Args:
             wires: list, the wires to be measured
             shots: total measurement times, default 1024
        """
        if wires is None:
            wires = self.wires
        if self.state is None:
            return
        else:
            prob_dis = self.state
        all_results = []
        if self.basis:
            batch = len(prob_dis[list(prob_dis.keys())[0]])
            for k in range(batch):
                prob_measure_dict = defaultdict(list)
                for key in prob_dis.keys():
                    s_ = key.state[(wires)]
                    s_ = FockState(state=s_, basis=self.basis)
                    temp = abs(prob_dis[key][k]) ** 2
                    prob_measure_dict[s_].append(temp)
                for key in prob_measure_dict.keys():
                    prob_measure_dict[key] = sum(prob_measure_dict[key])
                samples = random.choices(list(prob_measure_dict.keys()), list(prob_measure_dict.values()), k=shots)
                results = dict(Counter(samples))
                all_results.append(results)
        else:  # tensor state with batch
            state_tensor = self.tensor_rep(prob_dis)
            batch = state_tensor.shape[0]
            combi = list(itertools.product(range(self.cutoff), repeat=len(wires)))
            for i in range(batch):
                dict_ = {}
                state_i = state_tensor[i]
                probs_i = abs(state_i) ** 2
                if wires == self.wires:
                    ptrace_probs_i = probs_i   # no need for ptrace if measure all
                else:
                    sum_idx = list(range(self.nmode))
                    for idx in wires:
                        sum_idx.remove(idx)
                    ptrace_probs_i = probs_i.sum(dim=sum_idx)  # here partial trace for the measurement wires,此处可能需要归一化
                for p_state in combi:
                    lst1 = list(map(lambda x:str(x), p_state))
                    state_str = ''.join(lst1)
                    p_str = ('|' + state_str + '>')
                    dict_[p_str] = ptrace_probs_i[tuple(p_state)]
                samples = random.choices(list(dict_.keys()), list(dict_.values()), k=shots)
                results = dict(Counter(samples))
                all_results.append(results)
        if batch == 1:
            return all_results[0]
        else:
            return all_results

    def draw(self):
        """
        circuit plotting
        """
        if self.nmode > 50:
            print('too many wires in the circuit, run circuit.save for the complete circuit')
        self.draw_circuit = DrawCircuit(self.name, self.nmode, self.operators, self.depth)
        self.draw_circuit.draw()
        return self.draw_circuit.draw_

    def save(self, filename):
        """
        save the circuit in svg
        filename: 'example.svg'
        """
        self.draw_circuit.save(filename)

    def add(
        self,
        op: Operation,
        encode: bool = False,
        wires: Union[int, List[int], None] = None
    ) -> None:
        """A method that adds an operation to the quantum circuit.

        The operation can be a gate, a layer, or another quantum circuit. The method also updates the
        attributes of the quantum circuit. If ``wires`` is specified, the parameters of gates are shared.

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
        wires: Union[int, List[int], None] = None,
        inputs: Any = None,
        mu = 0,
        sigma = 0,
        encode: bool = False
    ) -> None:
        """Add a phaseshifter"""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        ps = PhaseShift(inputs=inputs, nmode=self.nmode, wires=wires, cutoff=self.cutoff,
                        requires_grad=requires_grad, noise=self.noise, mu=mu, sigma=sigma)
        self.add(ps, encode=encode)

    def bs(
        self,
        wires: Union[int, List[int], None] = None,
        inputs: Any = None,
        mu = None,
        sigma = None,
        encode: bool = False
    ) -> None:
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

    def bs_theta(
        self,
        wires: Union[int, List[int], None] = None,
        inputs: Any = None,
        mu = None,
        sigma = None,
        encode: bool = False
    ) -> None:
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
        wires: Union[int, List[int], None] = None,
        inputs: Any = None,
        mu = None,
        sigma = None,
        encode: bool = False
    ) -> None:
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
