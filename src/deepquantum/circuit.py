"""
Quantum circuit
"""

from copy import copy
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union
from typing import TYPE_CHECKING

import numpy as np
import torch
from qiskit import QuantumCircuit
from torch import nn, vmap

from .channel import BitFlip, PhaseFlip, Depolarizing, Pauli, AmplitudeDamping, PhaseDamping
from .channel import GeneralizedAmplitudeDamping
from .qmath import sample_sc_mcmc
from .gate import ParametricSingleGate
from .gate import U3Gate, PhaseShift, PauliX, PauliY, PauliZ, Hadamard, SGate, SDaggerGate, TGate, TDaggerGate
from .gate import Rx, Ry, Rz, ProjectionJ, CNOT, Swap, Rxx, Ryy, Rzz, Rxy, ReconfigurableBeamSplitter, Toffoli, Fredkin
from .gate import UAnyGate, LatentGate, HamiltonianGate, Barrier
from .layer import Observable, U3Layer, XLayer, YLayer, ZLayer, HLayer, RxLayer, RyLayer, RzLayer, CnotLayer, CnotRing
from .operation import Operation, Gate, Layer, Channel
from .qmath import amplitude_encoding, expectation, get_prob_mps, measure, sample2expval, inner_product_mps
from .state import QubitState, MatrixProductState

if TYPE_CHECKING:
    from .mbqc import Pattern


class QubitCircuit(Operation):
    """Quantum circuit for qubits.

    This class inherits from the ``Operation`` class and implements methods for creating, manipulating,
    and measuring quantum states.

    Args:
        nqubit (int): The number of qubits in the circuit.
        init_state (Any, optional): The initial state of the circuit. Default: ``'zeros'``
        name (str or None, optional): The name of the circuit. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        reupload (bool, optional): Whether to use data re-uploading. Default: ``False``
        mps (bool, optional): Whether to use matrix product state representation. Default: ``False``
        chi (int or None, optional): The bond dimension for matrix product state representation.
            Default: ``None``
        shots (int, optional): The number of shots for the measurement. Default: ``1024``

    Raises:
        AssertionError: If the type or dimension of ``init_state`` does not match ``nqubit`` or ``den_mat``.
    """
    def __init__(
        self,
        nqubit: int,
        init_state: Any = 'zeros',
        name: Optional[str] = None,
        den_mat: bool = False,
        reupload: bool = False,
        mps: bool = False,
        chi: Optional[int] = None,
        shots: int = 1024
    ) -> None:
        super().__init__(name=name, nqubit=nqubit, wires=None, den_mat=den_mat)
        self.reupload = reupload
        self.mps = mps
        self.chi = chi
        self.shots = shots
        self.set_init_state(init_state)
        self.operators = nn.Sequential()
        self.encoders = []
        self.observables = nn.ModuleList()
        self.state = None
        self.ndata = 0
        self.depth = np.array([0] * nqubit)
        self.wires_measure = []
        self.wires_condition = []
        # MBQC
        self.wire2node_dict = defaultdict(lambda: None)

    def set_init_state(self, init_state: Any) -> None:
        """Set the initial state of the circuit."""
        if isinstance(init_state, (QubitState, MatrixProductState)):
            if isinstance(init_state, MatrixProductState):
                assert self.nqubit == init_state.nsite
                assert not self.den_mat, 'Currently, MPS for density matrix is NOT supported'
                self.mps = True
                self.chi = init_state.chi
            else:
                assert self.nqubit == init_state.nqubit
                self.mps = False
                self.den_mat = init_state.den_mat
            self.init_state = init_state
        else:
            if self.mps:
                self.init_state = MatrixProductState(nsite=self.nqubit, state=init_state, chi=self.chi)
                self.chi = self.init_state.chi
            else:
                self.init_state = QubitState(nqubit=self.nqubit, state=init_state, den_mat=self.den_mat)

    def __add__(self, rhs: 'QubitCircuit') -> 'QubitCircuit':
        """Addition of the ``QubitCircuit``.

        The initial state is the same as the first ``QubitCircuit``.
        The information of observables and measurements is the same as the second ``QubitCircuit``.
        """
        assert self.nqubit == rhs.nqubit
        cir = QubitCircuit(nqubit=self.nqubit, init_state=self.init_state, name=self.name, den_mat=self.den_mat,
                           reupload=self.reupload, mps=self.mps, chi=self.chi)
        cir.operators = self.operators + rhs.operators
        cir.encoders = self.encoders + rhs.encoders
        cir.observables = rhs.observables
        cir.npara = self.npara + rhs.npara
        cir.ndata = self.ndata + rhs.ndata
        cir.depth = self.depth + rhs.depth
        cir.wires_measure = rhs.wires_measure
        cir.wires_condition += rhs.wires_condition
        cir.wires_condition = list(set(cir.wires_condition))
        return cir

    def to(self, arg: Any) -> 'QubitCircuit':
        """Set dtype or device of the ``QubitCircuit``."""
        if arg == torch.float:
            self.init_state.to(torch.cfloat)
            for op in self.operators:
                if op.npara == 0:
                    op.to(torch.cfloat)
                elif op.npara > 0:
                    op.to(torch.float)
            for ob in self.observables:
                if ob.npara == 0:
                    ob.to(torch.cfloat)
                elif ob.npara > 0:
                    ob.to(torch.float)
        elif arg == torch.double:
            self.init_state.to(torch.cdouble)
            for op in self.operators:
                if op.npara == 0:
                    op.to(torch.cdouble)
                elif op.npara > 0:
                    op.to(torch.double)
            for ob in self.observables:
                if ob.npara == 0:
                    ob.to(torch.cdouble)
                elif ob.npara > 0:
                    ob.to(torch.double)
        else:
            self.init_state.to(arg)
            self.operators.to(arg)
            self.observables.to(arg)
        return self

    # pylint: disable=arguments-renamed
    def forward(
        self,
        data: Optional[torch.Tensor] = None,
        state: Union[torch.Tensor, QubitState, List[torch.Tensor], MatrixProductState, None] = None
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Perform a forward pass of the quantum circuit and return the final state.

        This method applies the ``operators`` of the quantum circuit to the initial state or the given state
        and returns the resulting state. If ``data`` is given, it is used as the input for the ``encoders``.
        The ``state`` can be either a ``MatrixProductState`` or a ``QubitState`` object, or a tensor
        representation of them. The ``data`` must be a 1D or 2D tensor.

        Args:
            data (torch.Tensor or None, optional): The input data for the ``encoders``. Default: ``None``
            state (torch.Tensor, QubitState, List[torch.Tensor], MatrixProductState or None, optional):
                The initial state for the quantum circuit. Default: ``None``

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: The final state of the quantum circuit after
            applying the ``operators``.
        """
        if state is None:
            state = self.init_state
        if isinstance(state, MatrixProductState):
            state = state.tensors
        elif isinstance(state, QubitState):
            state = state.state
        if data is None or data.ndim == 1:
            self.state = self._forward_helper(data, state)
            if not self.mps:
                if self.state.ndim == 2:
                    self.state = self.state.unsqueeze(0)
                if state.ndim == 2:
                    self.state = self.state.squeeze(0)
        else:
            assert data.ndim == 2
            if self.mps:
                assert state[0].ndim in (3, 4)
                if state[0].ndim == 3:
                    self.state = vmap(self._forward_helper, in_dims=(0, None))(data, state)
                elif state[0].ndim == 4:
                    self.state = vmap(self._forward_helper)(data, state)
            else:
                assert state.ndim in (2, 3)
                if state.ndim == 2:
                    self.state = vmap(self._forward_helper, in_dims=(0, None))(data, state)
                elif state.ndim == 3:
                    self.state = vmap(self._forward_helper)(data, state)
            self.encode(data[-1])
        return self.state

    def _forward_helper(
        self,
        data: Optional[torch.Tensor] = None,
        state: Union[torch.Tensor, QubitState, List[torch.Tensor], MatrixProductState, None] = None
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Perform a forward pass for one sample."""
        self.encode(data)
        if state is None:
            state = self.init_state
        if self.mps:
            if not isinstance(state, MatrixProductState):
                state = MatrixProductState(nsite=self.nqubit, state=state, chi=self.chi,
                                           normalize=self.init_state.normalize)
            return self.operators(state).tensors
        if isinstance(state, QubitState):
            state = state.state
        x = self.operators(self.tensor_rep(state))
        if self.den_mat:
            x = self.matrix_rep(x)
        else:
            x = self.vector_rep(x)
        return x.squeeze(0)

    def encode(self, data: Optional[torch.Tensor]) -> None:
        """Encode the input data into the quantum circuit parameters.

        This method iterates over the ``encoders`` of the quantum circuit and initializes their parameters
        with the input data. If ``reupload`` is ``False``, the input data must be at least as long as the number
        of parameters in the ``encoders``. If ``reupload`` is ``True``, the input data can be repeated to fill up
        the parameters.

        Args:
            data (torch.Tensor or None): The input data for the ``encoders``, must be a 1D tensor.

        Raises:
            AssertionError: If ``reupload`` is ``False`` and the input data is shorter than the number of
                parameters in the ``encoders``.
        """
        if data is None:
            return
        if not self.reupload:
            assert len(data) >= self.ndata, 'The circuit needs more data, or consider data re-uploading'
        count = 0
        for op in self.encoders:
            count_up = count + op.npara
            if self.reupload and count_up > len(data):
                n = int(np.ceil(count_up / len(data)))
                data_tmp = torch.cat([data] * n)[count:count_up]
                op.init_para(data_tmp)
            else:
                op.init_para(data[count:count_up])
            count = count_up % len(data)

    def init_para(self) -> None:
        """Initialize the parameters of the ``operators``."""
        for op in self.operators:
            op.init_para()

    def init_encoder(self) -> None: # deal with the problem of state_dict() with vmap
        """Initialize the parameters of the ``encoders``."""
        for op in self.encoders:
            op.init_para()

    def reset(self, init_state: Any = 'zeros') -> None:
        """Reset the ``QubitCircuit`` according to ``init_state``."""
        self.set_init_state(init_state)
        self.operators = nn.Sequential()
        self.encoders = []
        self.observables = nn.ModuleList()
        self.state = None
        self.npara = 0
        self.ndata = 0
        self.depth = np.array([0] * self.nqubit)
        self.wires_measure = []
        self.wires_condition = []

    def amplitude_encoding(self, data: Any) -> torch.Tensor:
        """Encode data into quantum states using amplitude encoding."""
        return amplitude_encoding(data, self.nqubit)

    def observable(self, wires: Union[int, List[int], None] = None, basis: str = 'z') -> None:
        """Add an ``Observable``.

        Args:
            wires (int, List[int] or None, optional): The wires to measure. Default: ``None`` (which means
                all wires are measured)
            basis (str, optional): The measurement basis for each wire. It can be ``'x'``, ``'y'``, or ``'z'``.
                If only one character is given, it is repeated for all wires. Default: ``'z'``
        """
        observable = Observable(nqubit=self.nqubit, wires=wires, basis=basis,
                                den_mat=self.den_mat, tsr_mode=False)
        self.observables.append(observable)

    def reset_observable(self) -> None:
        """Reset the ``observables``."""
        self.observables = nn.ModuleList()

    def measure(
        self,
        shots: Optional[int] = None,
        with_prob: bool = False,
        wires: Union[int, List[int], None] = None,
        block_size: int = 2 ** 24
    ) -> Union[Dict, List[Dict], None]:
        """Measure the final state.

        Args:
            shots (int or None, optional): The number of shots for the measurement. Default: ``None`` (which means
                ``self.shots``)
            with_prob (bool, optional): Whether to show the true probability of the measurement. Default: ``False``
            wires (int, List[int] or None, optional): The wires to measure. Default: ``None`` (which means all wires)
            block_size (int, optional): The block size for sampling. Default: 2 ** 24
        """
        if shots is None:
            shots = self.shots
        else:
            self.shots = shots
        if wires is None:
            wires = list(range(self.nqubit))
        self.wires_measure = self._convert_indices(wires)
        if self.mps:
            samples = sample_sc_mcmc(prob_func=self._get_prob,
                                  proposal_sampler=self._proposal_sampler,
                                  shots=shots,
                                  num_chain=5)
            return dict(samples)
        if self.state is None:
            return
        else:
            return measure(self.state, shots=shots, with_prob=with_prob, wires=self.wires_measure,
                           den_mat=self.den_mat, block_size=block_size)

    def expectation(self, shots: Optional[int] = None) -> torch.Tensor:
        """Get the expectation value according to the final state and ``observables``.

        Args:
            shots (int or None, optional): The number of shots for the expectation value.
                Default: ``None`` (which means the exact and differentiable expectation value).
        """
        assert len(self.observables) > 0, 'There is no observable'
        if isinstance(self.state, list):
            assert all(isinstance(i, torch.Tensor) for i in self.state), 'Invalid final state'
            assert len(self.state) == self.nqubit, 'Invalid final state'
        else:
            assert isinstance(self.state, torch.Tensor), 'There is no final state'
        assert self.wires_condition == [], 'Expectation with conditional measurement is NOT supported'
        out = []
        if shots is None:
            for observable in self.observables:
                expval = expectation(self.state, observable=observable, den_mat=self.den_mat, chi=self.chi)
                out.append(expval)
        else:
            self.shots = shots
            dtype = self.state[0].real.dtype # in order to be compatible with MPS
            device = self.state[0].device
            for observable in self.observables:
                cir_basis = QubitCircuit(nqubit=self.nqubit, den_mat=self.den_mat, mps=self.mps, chi=self.chi)
                for wire, basis in zip(observable.wires, observable.basis):
                    if basis == 'x':
                        cir_basis.h(wire)
                    elif basis == 'y':
                        cir_basis.sdg(wire)
                        cir_basis.h(wire)
                cir_basis.to(dtype).to(device)
                state = cir_basis(state=self.state)
                wires = sum(observable.wires, [])
                samples = measure(state=state, shots=shots, wires=wires, den_mat=self.den_mat)
                if isinstance(samples, list):
                    expval = []
                    for sample in samples:
                        expval_i = sample2expval(sample=sample).to(dtype).to(device)
                        expval.append(expval_i)
                    expval = torch.cat(expval)
                elif isinstance(samples, dict):
                    expval = sample2expval(sample=samples).to(dtype).to(device)
                    if self.state.ndim == 2:
                        expval = expval.squeeze(0)
                out.append(expval)
        out = torch.stack(out, dim=-1)
        return out

    def defer_measure(self, with_prob: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List, List]]:
        """Get the state vectors and the measurement results after deferred measurement."""
        assert not self.den_mat
        assert not self.mps
        rst = self.measure(shots=1, with_prob=with_prob, wires=self.wires_condition)
        if self.state.ndim == 2:
            key = [*rst][0]
            state = self._slice_state_vector(state=self.state, wires=self.wires_condition, bits=key)
            if with_prob:
                prob = rst[key][1]
                print(f'The probability of deferred measurement to get "{key}" is {prob}.')
                return state, key, prob
            else:
                return state
        elif self.state.ndim == 3:
            state = []
            keys = []
            probs = []
            for i, d in enumerate(rst):
                key = [*d][0]
                state.append(self._slice_state_vector(state=self.state[i], wires=self.wires_condition, bits=key))
                if with_prob:
                    prob = d[key][1]
                    print(f'The probability of deferred measurement to get "{key}" for sample {i} is {prob}.')
                    keys.append(key)
                    probs.append(prob)
            if with_prob:
                return torch.stack(state), keys, probs
            else:
                return torch.stack(state)

    def post_select(self, bits: str) -> torch.Tensor:
        """Get the state vectors after post selection."""
        assert not self.den_mat
        assert not self.mps
        return self._slice_state_vector(state=self.state, wires=self.wires_condition, bits=bits)

    def get_unitary(self) -> torch.Tensor:
        """Get the unitary matrix of the quantum circuit."""
        u = None
        for op in self.operators:
            if isinstance(op, Barrier):
                continue
            if u is None:
                u = op.get_unitary()
            else:
                u = op.get_unitary() @ u
        if u is None:
            return torch.eye(2 ** self.nqubit, dtype=torch.cfloat)
        else:
            return u

    def get_amplitude(self, bits: str) -> torch.Tensor:
        """Get the amplitude for the given bit string.

        Args:
            bits (str): A bit string.
        """
        assert not self.den_mat
        assert len(bits) == self.nqubit
        if self.mps:
            state = [self.state[i][..., [int(bits[i])], :] for i in range(self.nqubit)]
            amp = MatrixProductState(nsite=self.nqubit, state=state, qudit=1).full_tensor()
        else:
            state = self.state.reshape([-1] + [2] * self.nqubit)
            for i in range(self.nqubit):
                state = state[:, int(bits[i])]
            amp = state.squeeze()
        return amp

    def get_prob_mps(self, bits: str, wires: Union[int, List[int], None] = None) -> torch.Tensor:
        """Get the probability for the given bit string.

        Args:
            bits (str): A bit string.
        """
        if self.mps and wires is not None:
            wires = self._convert_indices(wires)
            assert len(bits) == len(wires)
            idx = 0
            state = copy(self.state)
            for i in wires:
                state[i] = state[i][:, [int(bits[idx])], :]
                idx += 1
            prob = inner_product_mps(state, state).real
        else:
            amp = self.get_amplitude(bits)
            prob = torch.abs(amp) ** 2
        return prob

    def _get_prob(self, bits: str) -> torch.Tensor:
        """During MCMC measurement, Get the probability for the given bit string.

        Args:
            bits (str): A bit string.
        """
        if self.mps and len(self.wires_measure) != self.nqubit:
            idx = 0
            state = copy(self.state)
            for i in self.wires_measure:
                state[i] = state[i][:, [int(bits[idx])], :]
                idx += 1
            prob = inner_product_mps(state, state).real
        else:
            amp = self.get_amplitude(bits)
            prob = torch.abs(amp) ** 2
        return prob

    def inverse(self, encode: bool = False) -> 'QubitCircuit':
        """Get the inversed circuit.

        Note:
            The inversed circuit shares the parameters with the original circuit.
            You should ONLY encode data onto the original circuit.
            If you want to encode data onto the inversed circuit, set ``encode`` to be ``True``.
        """
        if isinstance(self.name, str):
            name = self.name + '_inverse'
        else:
            name = self.name
        cir = QubitCircuit(nqubit=self.nqubit, name=name, den_mat=self.den_mat, reupload=self.reupload,
                           mps=self.mps, chi=self.chi)
        for op in reversed(self.operators):
            if isinstance(op, Channel):
                op_inv = op
            else:
                op_inv = op.inverse()
            cir.operators.append(op_inv)
            if encode and op in self.encoders:
                cir.encoders.append(op_inv)
        cir.depth = self.depth
        cir.npara = self.npara
        cir.wires_condition = self.wires_condition
        if encode:
            cir.ndata = self.ndata
        else:
            cir.ndata = 0
        return cir

    @property
    def max_depth(self) -> int:
        """Get the max number of gates on the wires."""
        return max(self.depth)

    def _slice_state_vector(
        self,
        state: torch.Tensor,
        wires: Union[int, List[int]],
        bits: str
    ) -> torch.Tensor:
        """Get the sliced state vectors according to ``wires`` and ``bits``."""
        assert not self.den_mat
        assert not self.mps
        wires = self._convert_indices(wires)
        if len(bits) == 1:
            bits = bits * len(wires)
        assert len(wires) == len(bits)
        wires = [i + 1 for i in wires]
        state = state.reshape([-1] + [2] * self.nqubit)
        batch = state.shape[0]
        permute_shape = list(range(self.nqubit + 1))
        for i in wires:
            permute_shape.remove(i)
        permute_shape = wires + permute_shape
        state = state.permute(permute_shape)
        for b in bits:
            b = int(b)
            assert b in (0, 1)
            state = state[b]
        state = nn.functional.normalize(state.reshape(batch, -1), p=2, dim=-1)
        return state

    def qasm(self) -> str:
        """Get QASM of the quantum circuit."""
        allowed_ops = (U3Gate, PhaseShift, PauliX, PauliY, PauliZ, Hadamard, SGate, SDaggerGate, TGate,
                       TDaggerGate, Rx, Ry, Rz, CNOT, Swap, Rxx, Ryy, Rzz, Toffoli, Fredkin, Barrier, Layer)
        without_control_gates = (TGate, TDaggerGate, CNOT, Rxx, Ryy, Rzz, Toffoli, Fredkin, Barrier)
        single_control_gates = (U3Gate, PhaseShift, PauliY, PauliZ, Hadamard, SGate, SDaggerGate, Rx, Ry, Rz, Swap)
        qasm_lst = ['OPENQASM 2.0;\n' + 'include "qelib1.inc";\n']
        if self.wires_measure == self.wires_condition == []:
            qasm_lst.append(f'qreg q[{self.nqubit}];\n')
        else:
            qasm_lst.append(f'qreg q[{self.nqubit}];\n' + f'creg c[{self.nqubit}];\n')
        for op in self.operators:
            if not isinstance(op, allowed_ops):
                Gate._reset_qasm_new_gate()
                raise ValueError(f'{op.name} is NOT supported')
            if isinstance(op, Gate):
                if op.condition:
                    Gate._reset_qasm_new_gate()
                    raise ValueError(f'Conditional mode is NOT supported for {op.name}')
                if isinstance(op, PauliX):
                    if len(op.controls) > 4:
                        Gate._reset_qasm_new_gate()
                        raise ValueError(f'Too many control bits for {op.name}')
                elif isinstance(op, without_control_gates):
                    if len(op.controls) > 0:
                        Gate._reset_qasm_new_gate()
                        raise ValueError(f'Too many control bits for {op.name}')
                elif isinstance(op, single_control_gates):
                    if len(op.controls) > 1:
                        Gate._reset_qasm_new_gate()
                        raise ValueError(f'Too many control bits for {op.name}')
            qasm_lst.append(op._qasm())
        for wire in self.wires_measure:
            qasm_lst.append(f'measure q[{wire}] -> c[{wire}];\n')
        Gate._reset_qasm_new_gate()
        return ''.join(qasm_lst)

    def _qasm(self):
        """Get QASM of the quantum circuit for visualization."""
        # pylint: disable=protected-access
        qasm_lst = ['OPENQASM 2.0;\n' + 'include "qelib1.inc";\n']
        if self.wires_measure == self.wires_condition == []:
            qasm_lst.append(f'qreg q[{self.nqubit}];\n')
        else:
            qasm_lst.append(f'qreg q[{self.nqubit}];\n' + f'creg c[{self.nqubit}];\n')
        for op in self.operators:
            qasm_lst.append(op._qasm())
        for wire in self.wires_measure:
            qasm_lst.append(f'measure q[{wire}] -> c[{wire}];\n')
        Gate._reset_qasm_new_gate()
        Channel._reset_qasm_new_gate()
        return ''.join(qasm_lst)

    def _proposal_sampler(self):
        """The proposal sampler for MCMC sampling."""
        sample_chain = ''
        mps_state = copy(self.state)
        idx = 0
        for i in self.wires_measure:
            if idx != 0:
                mps_s = MatrixProductState(nsite=self.nqubit, state=mps_state)
                mps_s.center_orthogonalization(c=-1) # stay center-orthogonal form
                mps_state = list(mps_s.state_dict().values())
            sample_single_wire = torch.multinomial(get_prob_mps(i, mps_state), num_samples=1)
            sample_chain += str(sample_single_wire.item())
            mps_state[i] = mps_state[i][:, [int(sample_single_wire)], :]
            idx += 1
        return sample_chain

    def pattern(self) -> 'Pattern':
        """Get the MBQC pattern."""
        assert not self.den_mat and not self.mps, 'Currently NOT supported'
        from .mbqc import Pattern
        allowed_ops = (PauliX, PauliY, PauliZ, Hadamard, SGate, Rx, Ry, Rz, CNOT, Toffoli, Barrier,
                       XLayer, YLayer, ZLayer, HLayer, RxLayer, RyLayer, RzLayer, CnotLayer, CnotRing)
        for i in range(self.nqubit):
            self.wire2node_dict[i] = i
        state_zero = torch.zeros_like(self.init_state.state)
        if state_zero.ndim == 2:
            state_zero[0] = 1
        elif state_zero.ndim == 3:
            state_zero[:, 0] = 1
        if torch.all(self.init_state.state == state_zero):
            pattern = Pattern()
            for i in range(self.nqubit):
                pattern.add_graph(nodes_state=[i], state='zero')
        else:
            pattern = Pattern(nodes_state=self.nqubit, state=self.init_state.state)
        pattern.reupload = self.reupload
        node_next = self.nqubit
        for op in self.operators:
            assert isinstance(op, allowed_ops), f'{op.name} is NOT supported for MBQC pattern transpiler'
            encode = op in self.encoders
            if isinstance(op, Gate):
                pattern = self._update_pattern(pattern, op, node_next, encode)
                node_next += op.nancilla
            elif isinstance(op, Layer):
                for gate in op.gates:
                    pattern = self._update_pattern(pattern, gate, node_next, encode)
                    node_next += gate.nancilla
        pattern.set_nodes_out_seq([self.wire2node_dict[i] for i in range(self.nqubit)])
        return pattern

    def _update_pattern(self, pattern: 'Pattern', gate: Gate, node_next: int, encode: bool = False) -> 'Pattern':
        assert len(gate.controls) == 0, f'Control bits are NOT supported for MBQC pattern transpiler'
        assert not gate.condition, f'Conditional mode is NOT supported for MBQC pattern transpiler'
        nodes = [self.wire2node_dict[i] for i in gate.wires]
        ancilla = [node_next + i for i in range(gate.nancilla)]
        if isinstance(gate, ParametricSingleGate):
            cmds = gate.pattern(nodes, ancilla, gate.theta, gate.requires_grad)
        else:
            cmds = gate.pattern(nodes, ancilla)
        pattern.commands.extend(cmds)
        if encode:
            for i in gate.idx_enc:
                pattern.encoders.append(cmds[i])
            pattern.npara += gate.nancilla - len(gate.idx_enc)
            pattern.ndata += len(gate.idx_enc)
        else:
            pattern.npara += gate.nancilla
        for wire, node in zip(gate.wires, gate.nodes):
            self.wire2node_dict[wire] = node
        return pattern

    def draw(self, output: str = 'mpl', **kwargs):
        """Visualize the quantum circuit."""
        qc = QuantumCircuit.from_qasm_str(self._qasm())
        return qc.draw(output=output, **kwargs)

    def add(
        self,
        op: Operation,
        encode: bool = False,
        wires: Union[int, List[int], None] = None,
        controls: Union[int, List[int], None] = None
    ) -> None:
        """A method that adds an operation to the quantum circuit.

        The operation can be a gate, a layer, or another quantum circuit. The method also updates the
        attributes of the quantum circuit. If ``wires`` is specified, the parameters of gates are shared.

        Args:
            op (Operation): The operation to add. It is an instance of ``Operation`` class or its subclasses,
                such as ``Gate``, ``Layer``, ``Channel``, or ``QubitCircuit``.
            encode (bool, optional): Whether the gate or layer is to encode data. Default: ``False``
            wires (int, List[int] or None, optional): The wires to apply the gate on. It can be an integer
                or a list of integers specifying the indices of the wires. Default: ``None`` (which means
                the gate has its own wires)
            controls (int, List[int] or None, optional): The control wires for the gate. It can be an integer
                or a list of integers specifying the indices of the control wires. Only valid when ``wires``
                is not ``None``. Default: ``None`` (which means the gate has its own control wires)

        Raises:
            AssertionError: If the input arguments are invalid or incompatible with the quantum circuit.
        """
        assert isinstance(op, Operation)
        if wires is not None:
            assert isinstance(op, Gate)
            if controls is None:
                controls = []
            wires = self._convert_indices(wires)
            controls = self._convert_indices(controls)
            for wire in wires:
                assert wire not in controls, 'Use repeated wires'
            assert len(wires) == len(op.wires), 'Invalid input'
            op = copy(op)
            op.wires = wires
            op.controls = controls
        if isinstance(op, QubitCircuit):
            assert self.nqubit == op.nqubit
            self.operators += op.operators
            self.encoders  += op.encoders
            self.observables = op.observables
            self.npara += op.npara
            self.ndata += op.ndata
            self.depth += op.depth
            self.wires_measure = op.wires_measure
            self.wires_condition += op.wires_condition
            self.wires_condition = list(set(self.wires_condition))
        else:
            op.tsr_mode = True
            self.operators.append(op)
            if isinstance(op, Gate):
                for i in op.wires + op.controls:
                    self.depth[i] += 1
                if op.condition:
                    self.wires_condition += op.controls
                    self.wires_condition = list(set(self.wires_condition))
            elif isinstance(op, Layer):
                for wire in op.wires:
                    for i in wire:
                        self.depth[i] += 1
            # elif isinstance(op, Channel):
            #     for i in op.wires:
            #         self.depth[i] += 1
            if encode:
                assert not op.requires_grad, 'Please set requires_grad of the operation to be False'
                self.encoders.append(op)
                self.ndata += op.npara
            else:
                self.npara += op.npara

    def u3(
        self,
        wires: int,
        inputs: Any = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        encode: bool = False
    ) -> None:
        """Add a U3 gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        u3 = U3Gate(inputs=inputs, nqubit=self.nqubit, wires=wires, controls=controls,
                    condition=condition, den_mat=self.den_mat, requires_grad=requires_grad)
        self.add(u3, encode=encode)

    def cu(self, control: int, target: int, inputs: Any = None, encode: bool = False) -> None:
        """Add a controlled U3 gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        cu = U3Gate(inputs=inputs, nqubit=self.nqubit, wires=[target], controls=[control],
                    den_mat=self.den_mat, requires_grad=requires_grad)
        self.add(cu, encode=encode)

    def p(
        self,
        wires: int,
        inputs: Any = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        encode: bool = False
    ) -> None:
        """Add a phase shift gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        p = PhaseShift(inputs=inputs, nqubit=self.nqubit, wires=wires, controls=controls,
                       condition=condition, den_mat=self.den_mat, requires_grad=requires_grad)
        self.add(p, encode=encode)

    def cp(self, control: int, target: int, inputs: Any = None, encode: bool = False) -> None:
        """Add a controlled phase shift gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        cp = PhaseShift(inputs=inputs, nqubit=self.nqubit, wires=[target], controls=[control],
                        den_mat=self.den_mat, requires_grad=requires_grad)
        self.add(cp, encode=encode)

    def x(self, wires: int, controls: Union[int, List[int], None] = None, condition: bool = False) -> None:
        """Add a Pauli-X gate."""
        x = PauliX(nqubit=self.nqubit, wires=wires, controls=controls, condition=condition, den_mat=self.den_mat)
        self.add(x)

    def y(self, wires: int, controls: Union[int, List[int], None] = None, condition: bool = False) -> None:
        """Add a Pauli-Y gate."""
        y = PauliY(nqubit=self.nqubit, wires=wires, controls=controls, condition=condition, den_mat=self.den_mat)
        self.add(y)

    def z(self, wires: int, controls: Union[int, List[int], None] = None, condition: bool = False) -> None:
        """Add a Pauli-Z gate."""
        z = PauliZ(nqubit=self.nqubit, wires=wires, controls=controls, condition=condition, den_mat=self.den_mat)
        self.add(z)

    def h(self, wires: int, controls: Union[int, List[int], None] = None, condition: bool = False) -> None:
        """Add a Hadamard gate."""
        h = Hadamard(nqubit=self.nqubit, wires=wires, controls=controls, condition=condition, den_mat=self.den_mat)
        self.add(h)

    def s(self, wires: int, controls: Union[int, List[int], None] = None, condition: bool = False) -> None:
        """Add an S gate."""
        s = SGate(nqubit=self.nqubit, wires=wires, controls=controls, condition=condition, den_mat=self.den_mat)
        self.add(s)

    def sdg(self, wires: int, controls: Union[int, List[int], None] = None, condition: bool = False) -> None:
        """Add an S dagger gate."""
        sdg = SDaggerGate(nqubit=self.nqubit, wires=wires, controls=controls, condition=condition,
                          den_mat=self.den_mat)
        self.add(sdg)

    def t(self, wires: int, controls: Union[int, List[int], None] = None, condition: bool = False) -> None:
        """Add a T gate."""
        t = TGate(nqubit=self.nqubit, wires=wires, controls=controls, condition=condition, den_mat=self.den_mat)
        self.add(t)

    def tdg(self, wires: int, controls: Union[int, List[int], None] = None, condition: bool = False) -> None:
        """Add a T dagger gate."""
        tdg = TDaggerGate(nqubit=self.nqubit, wires=wires, controls=controls, condition=condition,
                          den_mat=self.den_mat)
        self.add(tdg)

    def ch(self, control: int, target: int) -> None:
        """Add a controlled Hadamard gate."""
        ch = Hadamard(nqubit=self.nqubit, wires=[target], controls=[control], den_mat=self.den_mat)
        self.add(ch)

    def cs(self, control: int, target: int) -> None:
        """Add a controlled S gate."""
        cs = SGate(nqubit=self.nqubit, wires=[target], controls=[control], den_mat=self.den_mat)
        self.add(cs)

    def csdg(self, control: int, target: int) -> None:
        """Add a controlled S dagger gate."""
        csdg = SDaggerGate(nqubit=self.nqubit, wires=[target], controls=[control], den_mat=self.den_mat)
        self.add(csdg)

    def ct(self, control: int, target: int) -> None:
        """Add a controlled T gate."""
        ct = TGate(nqubit=self.nqubit, wires=[target], controls=[control], den_mat=self.den_mat)
        self.add(ct)

    def ctdg(self, control: int, target: int) -> None:
        """Add a controlled T dagger gate."""
        ctdg = TDaggerGate(nqubit=self.nqubit, wires=[target], controls=[control], den_mat=self.den_mat)
        self.add(ctdg)

    def rx(
        self,
        wires: int,
        inputs: Any = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        encode: bool = False
    ) -> None:
        """Add an Rx gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        rx = Rx(inputs=inputs, nqubit=self.nqubit, wires=wires, controls=controls, condition=condition,
                den_mat=self.den_mat, requires_grad=requires_grad)
        self.add(rx, encode=encode)

    def ry(
        self,
        wires: int,
        inputs: Any = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        encode: bool = False
    ) -> None:
        """Add an Ry gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        ry = Ry(inputs=inputs, nqubit=self.nqubit, wires=wires, controls=controls, condition=condition,
                den_mat=self.den_mat, requires_grad=requires_grad)
        self.add(ry, encode=encode)

    def rz(
        self,
        wires: int,
        inputs: Any = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        encode: bool = False
    ) -> None:
        """Add an Rz gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        rz = Rz(inputs=inputs, nqubit=self.nqubit, wires=wires, controls=controls, condition=condition,
                den_mat=self.den_mat, requires_grad=requires_grad)
        self.add(rz, encode=encode)

    def crx(self, control: int, target: int, inputs: Any = None, encode: bool = False) -> None:
        """Add a controlled Rx gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        crx = Rx(inputs=inputs, nqubit=self.nqubit, wires=[target], controls=[control],
                 den_mat=self.den_mat, requires_grad=requires_grad)
        self.add(crx, encode=encode)

    def cry(self, control: int, target: int, inputs: Any = None, encode: bool = False) -> None:
        """Add a controlled Ry gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        cry = Ry(inputs=inputs, nqubit=self.nqubit, wires=[target], controls=[control],
                 den_mat=self.den_mat, requires_grad=requires_grad)
        self.add(cry, encode=encode)

    def crz(self, control: int, target: int, inputs: Any = None, encode: bool = False) -> None:
        """Add a controlled Rz gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        crz = Rz(inputs=inputs, nqubit=self.nqubit, wires=[target], controls=[control],
                 den_mat=self.den_mat, requires_grad=requires_grad)
        self.add(crz, encode=encode)

    def j(
        self,
        wires: int,
        inputs: Any = None,
        plane: str = 'xy',
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        encode: bool = False
    ) -> None:
        """Add a projection matrix J."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        j = ProjectionJ(inputs=inputs, nqubit=self.nqubit, wires=wires, plane=plane, controls=controls,
                        condition=condition, den_mat=self.den_mat, requires_grad=requires_grad)
        self.add(j, encode=encode)

    def cnot(self, control: int, target: int) -> None:
        """Add a CNOT gate."""
        cnot = CNOT(nqubit=self.nqubit, wires=[control, target], den_mat=self.den_mat)
        self.add(cnot)

    def cx(self, control: int, target: int) -> None:
        """Add a CNOT gate."""
        cx = PauliX(nqubit=self.nqubit, wires=[target], controls=[control], den_mat=self.den_mat)
        self.add(cx)

    def cy(self, control: int, target: int) -> None:
        """Add a controlled Y gate."""
        cy = PauliY(nqubit=self.nqubit, wires=[target], controls=[control], den_mat=self.den_mat)
        self.add(cy)

    def cz(self, control: int, target: int) -> None:
        """Add a controlled Z gate."""
        cz = PauliZ(nqubit=self.nqubit, wires=[target], controls=[control], den_mat=self.den_mat)
        self.add(cz)

    def swap(self, wires: List[int], controls: Union[int, List[int], None] = None, condition: bool = False) -> None:
        """Add a SWAP gate."""
        swap = Swap(nqubit=self.nqubit, wires=wires, controls=controls, condition=condition, den_mat=self.den_mat)
        self.add(swap)

    def rxx(
        self,
        wires: List[int],
        inputs: Any = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        encode: bool = False
    ) -> None:
        """Add an Rxx gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        rxx = Rxx(inputs=inputs, nqubit=self.nqubit, wires=wires, controls=controls, condition=condition,
                  den_mat=self.den_mat, requires_grad=requires_grad)
        self.add(rxx, encode=encode)

    def ryy(
        self,
        wires: List[int],
        inputs: Any = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        encode: bool = False
    ) -> None:
        """Add an Ryy gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        ryy = Ryy(inputs=inputs, nqubit=self.nqubit, wires=wires, controls=controls, condition=condition,
                  den_mat=self.den_mat, requires_grad=requires_grad)
        self.add(ryy, encode=encode)

    def rzz(
        self,
        wires: List[int],
        inputs: Any = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        encode: bool = False
    ) -> None:
        """Add an Rzz gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        rzz = Rzz(inputs=inputs, nqubit=self.nqubit, wires=wires, controls=controls, condition=condition,
                  den_mat=self.den_mat, requires_grad=requires_grad)
        self.add(rzz, encode=encode)

    def rxy(
        self,
        wires: List[int],
        inputs: Any = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        encode: bool = False
    ) -> None:
        """Add an Rxy gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        rxy = Rxy(inputs=inputs, nqubit=self.nqubit, wires=wires, controls=controls, condition=condition,
                  den_mat=self.den_mat, requires_grad=requires_grad)
        self.add(rxy, encode=encode)

    def rbs(
        self,
        wires: List[int],
        inputs: Any = None,
        controls: Union[int, List[int], None] = None,
        condition: bool = False,
        encode: bool = False
    ) -> None:
        """Add a Reconfigurable Beam Splitter gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        rbs = ReconfigurableBeamSplitter(inputs=inputs, nqubit=self.nqubit, wires=wires, controls=controls,
                                         condition=condition, den_mat=self.den_mat, requires_grad=requires_grad)
        self.add(rbs, encode=encode)

    def crxx(self, control: int, target1: int, target2: int, inputs: Any = None, encode: bool = False) -> None:
        """Add a controlled Rxx gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        crxx = Rxx(inputs=inputs, nqubit=self.nqubit, wires=[target1, target2], controls=[control],
                   den_mat=self.den_mat, requires_grad=requires_grad)
        self.add(crxx, encode=encode)

    def cryy(self, control: int, target1: int, target2: int, inputs: Any = None, encode: bool = False) -> None:
        """Add a controlled Ryy gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        cryy = Ryy(inputs=inputs, nqubit=self.nqubit, wires=[target1, target2], controls=[control],
                   den_mat=self.den_mat, requires_grad=requires_grad)
        self.add(cryy, encode=encode)

    def crzz(self, control: int, target1: int, target2: int, inputs: Any = None, encode: bool = False) -> None:
        """Add a controlled Rzz gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        crzz = Rzz(inputs=inputs, nqubit=self.nqubit, wires=[target1, target2], controls=[control],
                   den_mat=self.den_mat, requires_grad=requires_grad)
        self.add(crzz, encode=encode)

    def crxy(self, control: int, target1: int, target2: int, inputs: Any = None, encode: bool = False) -> None:
        """Add a controlled Rxy gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        crxy = Rxy(inputs=inputs, nqubit=self.nqubit, wires=[target1, target2], controls=[control],
                   den_mat=self.den_mat, requires_grad=requires_grad)
        self.add(crxy, encode=encode)

    def toffoli(self, control1: int, control2: int, target: int) -> None:
        """Add a Toffoli gate."""
        toffoli = Toffoli(nqubit=self.nqubit, wires=[control1, control2, target], den_mat=self.den_mat)
        self.add(toffoli)

    def ccx(self, control1: int, control2: int, target: int) -> None:
        """Add a Toffoli gate."""
        ccx = PauliX(nqubit=self.nqubit, wires=[target], controls=[control1, control2], den_mat=self.den_mat)
        self.add(ccx)

    def fredkin(self, control: int, target1: int, target2: int) -> None:
        """Add a Fredkin gate."""
        fredkin = Fredkin(nqubit=self.nqubit, wires=[control, target1, target2], den_mat=self.den_mat)
        self.add(fredkin)

    def cswap(self, control: int, target1: int, target2: int) -> None:
        """Add a Fredkin gate."""
        cswap = Swap(nqubit=self.nqubit, wires=[target1, target2], controls=[control], den_mat=self.den_mat)
        self.add(cswap)

    def any(
        self,
        unitary: Any,
        wires: Union[int, List[int], None] = None,
        minmax: Optional[List[int]] = None,
        controls: Union[int, List[int], None] = None,
        name: str = 'uany'
    ) -> None:
        """Add an arbitrary unitary gate."""
        uany = UAnyGate(unitary=unitary, nqubit=self.nqubit, wires=wires, minmax=minmax, controls=controls,
                        name=name, den_mat=self.den_mat)
        self.add(uany)

    def latent(
        self,
        wires: Union[int, List[int], None] = None,
        minmax: Optional[List[int]] = None,
        inputs: Any = None,
        controls: Union[int, List[int], None] = None,
        encode: bool = False,
        name: str = 'latent'
    ) -> None:
        """Add a latent gate."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        latent = LatentGate(inputs=inputs, nqubit=self.nqubit, wires=wires, minmax=minmax, controls=controls,
                            name=name, den_mat=self.den_mat, requires_grad=requires_grad)
        self.add(latent, encode=encode)

    def hamiltonian(
        self,
        hamiltonian: Any,
        t: Any = None,
        wires: Union[int, List[int], None] = None,
        minmax: Optional[List[int]] = None,
        controls: Union[int, List[int], None] = None,
        encode: bool = False,
        name: str = 'hamiltonian'
    ) -> None:
        """Add a Hamiltonian gate."""
        requires_grad = not encode
        if t is not None:
            requires_grad = False
        ham = HamiltonianGate(hamiltonian=hamiltonian, t=t, nqubit=self.nqubit, wires=wires, minmax=minmax,
                              controls=controls, name=name, den_mat=self.den_mat, requires_grad=requires_grad)
        self.add(ham, encode=encode)

    def xlayer(self, wires: Union[int, List[int], None] = None) -> None:
        """Add a layer of Pauli-X gates."""
        xl = XLayer(nqubit=self.nqubit, wires=wires, den_mat=self.den_mat)
        self.add(xl)

    def ylayer(self, wires: Union[int, List[int], None] = None) -> None:
        """Add a layer of Pauli-Y gates."""
        yl = YLayer(nqubit=self.nqubit, wires=wires, den_mat=self.den_mat)
        self.add(yl)

    def zlayer(self, wires: Union[int, List[int], None] = None) -> None:
        """Add a layer of Pauli-Z gates."""
        zl = ZLayer(nqubit=self.nqubit, wires=wires, den_mat=self.den_mat)
        self.add(zl)

    def hlayer(self, wires: Union[int, List[int], None] = None) -> None:
        """Add a layer of Hadamard gates."""
        hl = HLayer(nqubit=self.nqubit, wires=wires, den_mat=self.den_mat)
        self.add(hl)

    def rxlayer(self, wires: Union[int, List[int], None] = None, inputs: Any = None, encode: bool = False) -> None:
        """Add a layer of Rx gates."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        rxl = RxLayer(nqubit=self.nqubit, wires=wires, inputs=inputs, den_mat=self.den_mat,
                      requires_grad=requires_grad)
        self.add(rxl, encode=encode)

    def rylayer(self, wires: Union[int, List[int], None] = None, inputs: Any = None, encode: bool = False) -> None:
        """Add a layer of Ry gates."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        ryl = RyLayer(nqubit=self.nqubit, wires=wires, inputs=inputs, den_mat=self.den_mat,
                      requires_grad=requires_grad)
        self.add(ryl, encode=encode)

    def rzlayer(self, wires: Union[int, List[int], None] = None, inputs: Any = None, encode: bool = False) -> None:
        """Add a layer of Rz gates."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        rzl = RzLayer(nqubit=self.nqubit, wires=wires, inputs=inputs, den_mat=self.den_mat,
                      requires_grad=requires_grad)
        self.add(rzl, encode=encode)

    def u3layer(self, wires: Union[int, List[int], None] = None, inputs: Any = None, encode: bool = False) -> None:
        """Add a layer of U3 gates."""
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        u3l = U3Layer(nqubit=self.nqubit, wires=wires, inputs=inputs, den_mat=self.den_mat,
                      requires_grad=requires_grad)
        self.add(u3l, encode=encode)

    def cxlayer(self, wires: Optional[List[List[int]]] = None) -> None:
        """Add a layer of CNOT gates."""
        cxl = CnotLayer(nqubit=self.nqubit, wires=wires, den_mat=self.den_mat)
        self.add(cxl)

    def cnot_ring(self, minmax: Optional[List[int]] = None, step: int = 1, reverse: bool = False) -> None:
        """Add a layer of CNOT gates in a cyclic way."""
        cxr = CnotRing(nqubit=self.nqubit, minmax=minmax, step=step, reverse=reverse, den_mat=self.den_mat)
        self.add(cxr)

    def bit_flip(
        self,
        wires: int,
        inputs: Any = None,
        encode: bool = False
    ) -> None:
        """Add a bit-flip channel."""
        assert self.den_mat
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        bf = BitFlip(inputs=inputs, nqubit=self.nqubit, wires=wires, requires_grad=requires_grad)
        self.add(bf, encode=encode)

    def phase_flip(
        self,
        wires: int,
        inputs: Any = None,
        encode: bool = False
    ) -> None:
        """Add a phase-flip channel."""
        assert self.den_mat
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        pf = PhaseFlip(inputs=inputs, nqubit=self.nqubit, wires=wires, requires_grad=requires_grad)
        self.add(pf, encode=encode)

    def depolarizing(
        self,
        wires: int,
        inputs: Any = None,
        encode: bool = False
    ) -> None:
        """Add a depolarizing channel."""
        assert self.den_mat
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        dp = Depolarizing(inputs=inputs, nqubit=self.nqubit, wires=wires, requires_grad=requires_grad)
        self.add(dp, encode=encode)

    def pauli(
        self,
        wires: int,
        inputs: Any = None,
        encode: bool = False
    ) -> None:
        """Add a Pauli channel."""
        assert self.den_mat
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        p = Pauli(inputs=inputs, nqubit=self.nqubit, wires=wires, requires_grad=requires_grad)
        self.add(p, encode=encode)

    def amp_damp(
        self,
        wires: int,
        inputs: Any = None,
        encode: bool = False
    ) -> None:
        """Add an amplitude-damping channel."""
        assert self.den_mat
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        ad = AmplitudeDamping(inputs=inputs, nqubit=self.nqubit, wires=wires, requires_grad=requires_grad)
        self.add(ad, encode=encode)

    def phase_damp(
        self,
        wires: int,
        inputs: Any = None,
        encode: bool = False
    ) -> None:
        """Add a phase-damping channel."""
        assert self.den_mat
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        pd = PhaseDamping(inputs=inputs, nqubit=self.nqubit, wires=wires, requires_grad=requires_grad)
        self.add(pd, encode=encode)

    def gen_amp_damp(
        self,
        wires: int,
        inputs: Any = None,
        encode: bool = False
    ) -> None:
        """Add a generalized amplitude-damping channel."""
        assert self.den_mat
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        gad = GeneralizedAmplitudeDamping(inputs=inputs, nqubit=self.nqubit, wires=wires, requires_grad=requires_grad)
        self.add(gad, encode=encode)

    def barrier(self, wires: Union[int, List[int], None] = None) -> None:
        """Add a barrier."""
        br = Barrier(nqubit=self.nqubit, wires=wires)
        self.add(br)
