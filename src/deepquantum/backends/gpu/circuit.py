"""Compiled single-GPU statevector circuits and adjoint differentiation."""

from __future__ import annotations

from collections import Counter
from contextlib import suppress
from typing import Any

import torch
from torch import nn

from ...circuit import QubitCircuit
from ...operation import Gate
from .compiler import CompiledCircuitPlan, compile_circuit, compile_pauli_observables
from .loader import load_ixquantum_executor


class QubitCircuitGPU(QubitCircuit):
    """Compile a pure statevector circuit for one Iluvatar GPU.

    Gate construction follows :class:`QubitCircuit`. The native executor owns the
    returned state storage; call ``clone()`` to retain a state across executions.
    Qubit 0 is the most significant statevector bit, as in :class:`QubitCircuit`.
    Only expectation values support differentiation, using a first-order adjoint.

    Args:
        nqubit: Number of qubits, from 1 to 30, subject to device memory.
        init_state: Fixed initial state or ``'zeros'``. It cannot require gradients.
        name: Circuit name.
        shots: Default number of terminal measurement samples.
        device: Fixed CUDA-compatible execution device. Defaults to the current device.
        dtype: Statevector precision. Only ``torch.complex64`` is supported.
        diff_method: ``'none'`` or ``'adjoint'``.
    """

    def __init__(
        self,
        nqubit: int,
        init_state: Any = 'zeros',
        name: str | None = None,
        shots: int = 1024,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.complex64,
        diff_method: str = 'none',
    ) -> None:
        if not isinstance(nqubit, int) or isinstance(nqubit, bool) or (not 1 <= nqubit <= 30):
            raise ValueError('nqubit must be an integer in [1, 30]')
        if dtype != torch.complex64:
            raise ValueError('QubitCircuitGPU supports torch.complex64 only')
        if diff_method not in ('none', 'adjoint'):
            raise ValueError('diff_method must be "none" or "adjoint"')
        selected = torch.device('cuda' if device is None else device)
        if selected.type != 'cuda':
            raise ValueError('QubitCircuitGPU requires a CUDA-compatible Iluvatar device')
        if not getattr(torch, 'corex', False) or not torch.cuda.is_available():
            raise RuntimeError('QubitCircuitGPU requires CoreX-enabled PyTorch and an Iluvatar GPU')
        index = torch.cuda.current_device() if selected.index is None else selected.index
        if index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device {index} is not visible')
        super().__init__(nqubit=nqubit, init_state=init_state, name=name, shots=shots)
        self.gpu_device = torch.device('cuda', index)
        self.diff_method = diff_method
        self._compiled_plan: CompiledCircuitPlan | None = None
        self._executor: Any = None
        self._matrix_observable: tuple[torch.Tensor, tuple[int, ...]] | None = None
        self._plan_dirty = True

    def set_init_state(self, init_state: Any) -> None:
        """Set a fixed initial state, allocating the zero state only on execution."""
        self._compiled_zero_initial = isinstance(init_state, str) and init_state == 'zeros'
        if self._compiled_zero_initial:
            self.init_state = _ZeroState(self.nqubit)
        else:
            super().set_init_state(init_state)
            if self.init_state.state.requires_grad:
                raise ValueError('QubitCircuitGPU does not support differentiable initial states')
        self._plan_dirty = True
        self.state = None

    def add(self, *args: Any, **kwargs: Any) -> Any:
        result = super().add(*args, **kwargs)
        self.invalidate()
        return result

    def multirz(self, wires: list[int], inputs: Any = None, controls: int | list[int] | None = None) -> None:
        """Add a multi-qubit Z Pauli rotation."""
        wire_list = list(wires)
        if not wire_list:
            raise ValueError('MultiRZ requires at least one target wire')
        self.add(
            MultiRZ(inputs=inputs, nqubit=self.nqubit, wires=wire_list, controls=controls, requires_grad=inputs is None)
        )

    def reset_circuit(self, init_state: Any = 'zeros') -> None:
        """Reset gates, observables, and the compiled state."""
        super().reset_circuit(init_state)
        self._matrix_observable = None
        self.invalidate()

    def __add__(self, rhs: Any) -> QubitCircuitGPU:
        raise NotImplementedError('QubitCircuitGPU does not support circuit addition; add gates to one circuit')

    def inverse(self, encode: bool = False) -> QubitCircuitGPU:
        """Reject whole-circuit inversion; individual inverse gates are supported."""
        raise NotImplementedError('QubitCircuitGPU does not support circuit inverse(); add individual inverse gates')

    def compile(self) -> QubitCircuitGPU:
        """Compile the circuit structure; parameter values remain dynamic."""
        plan = compile_circuit(self)
        self.invalidate()
        self._ensure_executor_resources()
        self._executor.compile(
            plan.opcodes.tolist(),
            [list(x) for x in plan.targets],
            [list(x) for x in plan.controls],
            [None if x is None else x.to(self.gpu_device) for x in plan.matrices],
            [list(x) for x in plan.permutations],
            [list(x) for x in plan.diagonals],
            plan.angles.tolist(),
            [list(x) for x in plan.paulis],
            plan.parameter_indices.tolist(),
            plan.parameter_scales.tolist(),
            plan.gradient_kinds.tolist(),
            plan.generator_coeffs.tolist(),
            [list(x) for x in plan.observable_paulis],
            [list(x) for x in plan.observable_bits],
            None if plan.observable_matrix is None else plan.observable_matrix.to(self.gpu_device),
            list(plan.observable_matrix_bits),
            None
            if self._compiled_zero_initial
            else self.init_state.state.reshape(-1)
            .detach()
            .to(device=self.gpu_device, dtype=torch.complex64)
            .contiguous(),
        )
        self._compiled_plan = plan
        self._plan_dirty = False
        self.state = None
        return self

    def invalidate(self) -> None:
        """Mark compiled Python gate structure or fixed data changed; the next call recompiles."""
        self._plan_dirty = True
        self.state = None

    def close(self) -> None:
        """Release native resources and invalidate the state, including on cleanup errors."""
        if getattr(self, '_executor', None) is not None:
            try:
                self._executor.close()
            finally:
                self._executor = None
                self.invalidate()

    def memory_report(self) -> dict[str, Any]:
        """Report statevector requirements and current native allocations."""
        result = {'one_state_bytes': 8 * (1 << self.nqubit)}
        if self._executor is not None:
            result.update(dict(self._executor.memory_report()))
        return result

    def observable(self, *args: Any, **kwargs: Any) -> None:
        super().observable(*args, **kwargs)
        self.invalidate()

    def observable_matrix(self, matrix: torch.Tensor, wires: int | list[int]) -> None:
        """Register one fixed Hermitian matrix observable for exact expectation.

        Matrix expectations are non-differentiable and cannot be combined with
        Pauli ``observable()`` entries.
        """
        if self.observables:
            raise ValueError('observable_matrix() cannot be mixed with Pauli observable()')
        if self._matrix_observable is not None:
            raise ValueError('QubitCircuitGPU supports exactly one observable_matrix()')
        wire_list = [wires] if isinstance(wires, int) else list(wires)
        self._matrix_observable = (matrix, tuple(int(wire) for wire in wire_list))
        self.invalidate()

    def reset_observable(self) -> None:
        super().reset_observable()
        self._matrix_observable = None
        self.invalidate()

    def forward(self, data: torch.Tensor | None = None, state: Any = None) -> torch.Tensor:
        """Execute the circuit and return a borrowed ``(2**nqubit, 1)`` state tensor.

        Runtime data encoding and external forward states are unsupported. Use
        ``expectation()`` with ``diff_method='adjoint'`` for differentiable results.
        """
        if data is not None or state is not None:
            raise ValueError('QubitCircuitGPU accepts neither runtime data nor an external forward state')
        self._ensure_compiled()
        self.state = self._executor.execute(self._parameter_tensor()).reshape(-1, 1)
        return self.state

    def measure(
        self,
        shots: int | None = None,
        with_prob: bool = False,
        wires: int | list[int] | None = None,
        block_size: int = 2**24,
    ) -> dict[str, int] | None:
        """Sample the terminal state with ixquantum without state collapse."""
        del block_size
        if with_prob:
            raise ValueError('QubitCircuitGPU sampling supports with_prob=False only')
        if self.state is None:
            return None
        shots = self.shots if shots is None else int(shots)
        if shots <= 0:
            raise ValueError('shots must be positive')
        selected_wires = list(range(self.nqubit)) if wires is None else wires
        self.wires_measure = self._convert_indices(selected_wires)
        bit_ordering = [self.nqubit - wire - 1 for wire in reversed(sorted(self.wires_measure))]
        randnums = torch.rand(shots, dtype=torch.float64, device='cpu')
        sampled = self._executor.sample(shots, bit_ordering, randnums)
        self.shots = shots
        return {
            format(int(value), f'0{len(self.wires_measure)}b'): int(count)
            for value, count in Counter(sampled.tolist()).items()
        }

    def expectation(self, shots: int | None = None, *, parameters: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluate terminal observables, using an adjoint when requested.

        Adjoint mode runs a forward pass itself. Otherwise call the circuit first.
        Explicit parameters are a packed CUDA float32 vector in first-use order,
        with one entry per distinct trainable scalar; they require adjoint mode.
        """
        if shots is not None:
            raise ValueError('QubitCircuitGPU supports exact expectation only (shots=None)')
        self._ensure_compiled()
        if not self._compiled_plan.observable_paulis and self._compiled_plan.observable_matrix is None:
            raise ValueError('There is no compiled observable')
        if self.diff_method == 'adjoint':
            self.state = None
            return _GPUAdjointExpectation.apply(self._resolve_parameters(parameters), self)
        if parameters is not None:
            raise ValueError('explicit parameters require diff_method="adjoint"')
        return self._executor.expectation()

    def expectation_from_state(self, state: torch.Tensor, observables: list[Any]) -> torch.Tensor:
        """Evaluate Pauli observables on an existing CUDA state.

        This non-differentiable readout borrows the state for the duration of the
        call. Gate compilation and execution are skipped.
        """
        if not isinstance(state, torch.Tensor):
            raise TypeError('state must be a torch.Tensor')
        if state.device.type != 'cuda':
            raise ValueError('state must be a CUDA tensor')
        if state.device != self.gpu_device:
            raise ValueError(f'state must be on the circuit device {self.gpu_device}')
        if state.dtype != torch.complex64:
            raise TypeError('state must use torch.complex64')
        if state.requires_grad:
            raise ValueError('expectation_from_state is non-differentiable; state.requires_grad must be False')
        dimension = 1 << self.nqubit
        if tuple(state.shape) not in {(dimension,), (dimension, 1)}:
            raise ValueError(f'state must have shape ({dimension},) or ({dimension}, 1)')
        if not state.is_contiguous():
            raise ValueError('state must be contiguous')
        try:
            observable_list = list(observables)
        except TypeError as exc:
            raise TypeError('observables must be an iterable of DeepQuantum Pauli observables') from exc
        paulis, bits = compile_pauli_observables(self.nqubit, observable_list)
        self._ensure_executor_resources()
        with torch.no_grad():
            return self._executor.expectation_from_state(
                state.reshape(-1), [list(value) for value in paulis], [list(value) for value in bits]
            )

    def adjoint_vjp(self, parameters: torch.Tensor | None, coefficients: torch.Tensor) -> torch.Tensor:
        """Compute the adjoint gradient of a weighted Pauli expectation."""
        if self.diff_method != 'adjoint':
            raise ValueError('adjoint_vjp requires diff_method="adjoint"')
        self._ensure_compiled()
        values = self._resolve_parameters(parameters)
        weights = coefficients.to(self.gpu_device, torch.float32).reshape(-1).contiguous()
        self.state = None
        return self._executor.adjoint_vjp(values.detach(), weights)

    def value_and_grad(
        self, parameters: torch.Tensor | None, coefficients: torch.Tensor, constant: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute weighted expectation value and adjoint gradient with one forward."""
        if self.diff_method != 'adjoint':
            raise ValueError('value_and_grad requires diff_method="adjoint"')
        self._ensure_compiled()
        values = self._resolve_parameters(parameters)
        weights = coefficients.to(self.gpu_device, torch.float32).reshape(-1).contiguous()
        self.state = None
        return self._executor.value_and_grad(values.detach(), weights, float(constant))

    def _ensure_executor_resources(self) -> None:
        """Create the native executor on first use."""
        if self._executor is None:
            module = load_ixquantum_executor()
            self._executor = module.GPUCircuitExecutor(self.nqubit, self.gpu_device.index or 0)

    def _ensure_compiled(self) -> None:
        if self._plan_dirty or self._compiled_plan is None or self._executor is None:
            self.compile()

    def _parameter_tensor(self) -> torch.Tensor:
        """Pack dynamic values once per execution while preserving autograd edges."""
        if not self._compiled_plan or not self._compiled_plan.parameter_sources:
            return torch.empty(0, dtype=torch.float32, device=self.gpu_device)
        values = [
            value.reshape(()).to(dtype=torch.float32, device=self.gpu_device)
            for value in self._compiled_plan.parameter_sources
        ]
        return torch.stack(values).contiguous()

    def _resolve_parameters(self, parameters: torch.Tensor | None) -> torch.Tensor:
        """Return the packed logical parameter vector used by the native plan."""
        if parameters is None:
            return self._parameter_tensor()
        if not isinstance(parameters, torch.Tensor):
            raise TypeError('parameters must be a torch.Tensor')
        if parameters.device != self.gpu_device:
            raise ValueError(f'parameters must be on {self.gpu_device}')
        if parameters.dtype != torch.float32:
            raise TypeError('parameters must use torch.float32')
        if parameters.ndim != 1 or not parameters.is_contiguous():
            raise ValueError('parameters must be a contiguous one-dimensional tensor')
        expected = len(self._compiled_plan.parameter_sources) if self._compiled_plan else 0
        if parameters.numel() != expected:
            raise ValueError(f'expected {expected} logical parameters, got {parameters.numel()}')
        return parameters

    def _apply(self, fn: Any) -> QubitCircuitGPU:
        probe = fn(torch.empty(0, device=self.gpu_device))
        if probe.device != self.gpu_device:
            raise ValueError('QubitCircuitGPU execution device is fixed; construct a new circuit on the desired device')
        self.invalidate()
        return super()._apply(fn)

    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:
        # This hook also runs when a parent module loads the circuit's checkpoint.
        self.invalidate()
        super()._load_from_state_dict(*args, **kwargs)

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


class _GPUAdjointExpectation(torch.autograd.Function):
    """Bridge native adjoint gradients to PyTorch autograd."""

    @staticmethod
    def forward(ctx: Any, parameters: torch.Tensor, circuit: QubitCircuitGPU) -> torch.Tensor:
        ctx.circuit = circuit
        ctx.executor = circuit._executor
        ctx.plan_version = ctx.executor.plan_version()
        ctx.save_for_backward(parameters)
        result = ctx.executor.forward_expectation(parameters.detach())
        ctx.generation = ctx.executor.generation()
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        circuit = ctx.circuit
        if (
            circuit._executor is not ctx.executor
            or circuit._plan_dirty
            or ctx.executor.plan_version() != ctx.plan_version
        ):
            raise RuntimeError('single-GPU circuit changed or closed before backward')
        (parameters,) = ctx.saved_tensors
        weights = grad_output.to(device=parameters.device, dtype=torch.float32).reshape(-1).contiguous()
        try:
            gradient = ctx.executor.adjoint_vjp_prepared(parameters.detach(), weights, ctx.generation)
            return (gradient, None)
        finally:
            circuit.state = None


class _ZeroState(nn.Module):
    """Store zero-state metadata until GPU allocation."""

    _compiled_zero_state = True

    def __init__(self, nqubit: int) -> None:
        super().__init__()
        self.nqubit = nqubit
        self.den_mat = False


class MultiRZ(Gate):
    """Describe exp(-i theta Z...Z / 2) by its angle and wires."""

    def __init__(
        self, *, inputs: Any, nqubit: int, wires: list[int], controls: int | list[int] | None, requires_grad: bool
    ) -> None:
        super().__init__(name='MultiRZ', nqubit=nqubit, wires=wires, controls=controls)
        if inputs is None:
            inputs = torch.rand((), dtype=torch.float32) * (4 * torch.pi)
        elif not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        if inputs.numel() != 1:
            raise ValueError('MultiRZ inputs must be one scalar')
        self.npara = 1
        self.inv_mode = False
        if requires_grad:
            self.theta = nn.Parameter(inputs.reshape(()))
        else:
            self.register_buffer('theta', inputs.reshape(()))

    def inverse(self) -> MultiRZ:
        from copy import copy

        gate = copy(self)
        gate.inv_mode = not self.inv_mode
        return gate
