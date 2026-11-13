"""Compile DeepQuantum gates into ixquantum execution plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ...gate import Barrier

OP_MATRIX = 0
OP_PAULI_ROTATION = 1
OP_PHASE = 2
OP_PERMUTATION = 3
GRAD_NONE = 0
GRAD_PAULI_GENERATOR = 1
GRAD_PHASE = 2
# Pauli labels match custatevecPauli_t: X=1, Y=2, Z=3.
_PAULI = {
    'Rx': (1,),
    'Ry': (2,),
    'Rz': (3,),
    'Rxx': (1, 1),
    'Ryy': (2, 2),
    'Rzz': (3, 3),
    'MultiRZ': None,
}


@dataclass(frozen=True)
class CompiledCircuitPlan:
    """Host-side circuit plan consumed in one call by the native executor."""

    nqubit: int
    opcodes: np.ndarray
    targets: tuple[tuple[int, ...], ...]
    controls: tuple[tuple[int, ...], ...]
    matrices: tuple[torch.Tensor | None, ...]
    permutations: tuple[tuple[int, ...], ...]
    diagonals: tuple[tuple[complex, ...], ...]
    angles: np.ndarray
    paulis: tuple[tuple[int, ...], ...]
    parameter_indices: np.ndarray
    parameter_scales: np.ndarray
    gradient_kinds: np.ndarray
    generator_coeffs: np.ndarray
    parameter_sources: tuple[torch.Tensor, ...]
    observable_paulis: tuple[tuple[int, ...], ...]
    observable_bits: tuple[tuple[int, ...], ...]
    observable_matrix: torch.Tensor | None
    observable_matrix_bits: tuple[int, ...]


def _error(index: int, gate: Any, message: str) -> ValueError:
    return ValueError(f'QubitCircuitGPU compile rejected gate #{index} ({type(gate).__name__}): {message}')


def _static_matrix(gate: Any, index: int) -> torch.Tensor:
    if any(parameter.requires_grad for parameter in gate.parameters()):
        raise _error(index, gate, 'trainable parameters are unsupported for generic matrix gates')
    try:
        matrix = gate.update_matrix()
    except Exception as exc:  # pragma: no cover - third-party gate subclasses.
        raise _error(index, gate, f'cannot materialize a fixed matrix: {exc}') from exc
    if not isinstance(matrix, torch.Tensor):
        raise _error(index, gate, 'matrix is not a torch.Tensor')
    if matrix.requires_grad:
        raise _error(index, gate, 'matrix requires grad')
    return matrix.detach().to(dtype=torch.complex64, device='cpu').contiguous()


def _vendor_bits(nqubit: int, wires: tuple[int, ...]) -> tuple[int, ...]:
    # DQ wire 0 is tensor/MSB; custatevec bit 0 is LSB.  For a multi-target
    # matrix DQ's first wire is the matrix MSB axis, so reverse target axes too.
    return tuple(reversed([nqubit - wire - 1 for wire in wires]))


def _parameter_key(theta: torch.Tensor) -> tuple[Any, ...]:
    """Return a stable key for one logical scalar parameter."""
    storage = theta.untyped_storage()
    return (
        'tensor',
        theta.device.type,
        theta.device.index,
        str(theta.dtype),
        storage.data_ptr(),
        theta.storage_offset(),
    )


def compile_pauli_observables(
    nqubit: int, observables: list[Any] | tuple[Any, ...]
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Validate DQ Pauli observables and map wire-0-as-MSB to vendor bits."""
    mapping = {'x': 1, 'y': 2, 'z': 3}
    paulis: list[tuple[int, ...]] = []
    bits: list[tuple[int, ...]] = []
    for index, observable in enumerate(observables):
        try:
            wires = tuple(wire[0] if isinstance(wire, list) else wire for wire in observable.wires)
            basis = tuple(observable.basis)
            if len(wires) != len(basis) or not wires:
                raise ValueError('observable must contain one non-empty Pauli basis string')
            normalized_wires = tuple(int(wire) for wire in wires)
            if len(set(normalized_wires)) != len(normalized_wires) or any(
                wire < 0 or wire >= nqubit for wire in normalized_wires
            ):
                raise ValueError('observable wires must be unique circuit wires')
            paulis.append(tuple(mapping[item.lower()] for item in basis))
            bits.append(tuple(nqubit - wire - 1 for wire in normalized_wires))
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'QubitCircuitGPU rejected observable #{index}: Pauli strings only ({exc})') from exc
    if not paulis:
        raise ValueError('at least one Pauli observable is required')
    return tuple(paulis), tuple(bits)


def _compile_observables(
    circuit: Any,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], torch.Tensor | None, tuple[int, ...]]:
    matrix_observable = getattr(circuit, '_matrix_observable', None)
    if matrix_observable is not None:
        if circuit.observables:
            raise ValueError('QubitCircuitGPU cannot mix Pauli observables with observable_matrix()')
        matrix, wires = matrix_observable
        wires = tuple(int(wire) for wire in wires)
        if not wires or len(set(wires)) != len(wires) or any(wire < 0 or wire >= circuit.nqubit for wire in wires):
            raise ValueError('observable_matrix wires must be non-empty, unique circuit wires')
        if not isinstance(matrix, torch.Tensor) or matrix.requires_grad:
            raise ValueError('observable_matrix requires a fixed torch.Tensor')
        dim = 1 << len(wires)
        matrix = matrix.detach().to(dtype=torch.complex64, device='cpu').contiguous()
        if matrix.shape != (dim, dim):
            raise ValueError(f'observable_matrix expected a {dim}x{dim} matrix, got {tuple(matrix.shape)}')
        if not torch.allclose(matrix, matrix.mH, atol=1e-5, rtol=1e-5):
            raise ValueError('observable_matrix must be Hermitian')
        # ixquantum 1.8.0 interprets basis bits in ascending order. Reorder both
        # matrix axes at compile time to preserve the caller's wire order.
        ordered_wires = tuple(sorted(wires))
        if wires != ordered_wires:
            count = len(wires)
            axes = tuple(wires.index(wire) for wire in ordered_wires)
            matrix = matrix.reshape((2,) * (2 * count)).permute(axes + tuple(axis + count for axis in axes))
            matrix = matrix.reshape(dim, dim).contiguous()
        return (), (), matrix, _vendor_bits(circuit.nqubit, ordered_wires)

    if not circuit.observables:
        return (), (), None, ()
    paulis, bits = compile_pauli_observables(circuit.nqubit, tuple(circuit.observables))
    return paulis, bits, None, ()


def compile_circuit(circuit: Any) -> CompiledCircuitPlan:
    """Compile a single-statevector circuit; scalar Pauli angles remain dynamic plan inputs."""
    if circuit.den_mat or circuit.mps:
        raise ValueError('QubitCircuitGPU supports pure statevectors only (den_mat=False, mps=False)')
    if circuit.reupload or circuit.ndata or circuit.encoders:
        raise ValueError('QubitCircuitGPU does not yet support data encoding/reupload')
    zero_initial = getattr(circuit.init_state, '_compiled_zero_state', False)
    if not zero_initial and getattr(circuit.init_state, 'state', None) is None:
        raise ValueError('QubitCircuitGPU requires init_state="zeros"')
    if not zero_initial:
        initial = circuit.init_state.state.reshape(-1)
        if initial.numel() != 1 << circuit.nqubit or initial.dtype not in (torch.complex64, torch.complex128):
            raise ValueError('QubitCircuitGPU initial state must be one pure complex statevector')
        if initial.requires_grad or not torch.isfinite(initial).all() or initial.norm() == 0:
            raise ValueError('QubitCircuitGPU initial state must be finite, nonzero, and non-differentiable')

    opcodes: list[int] = []
    targets: list[tuple[int, ...]] = []
    controls: list[tuple[int, ...]] = []
    matrices: list[torch.Tensor | None] = []
    angles: list[float] = []
    paulis: list[tuple[int, ...]] = []
    parameter_indices: list[int] = []
    parameter_scales: list[float] = []
    parameter_sources: list[torch.Tensor] = []
    parameter_slots: dict[tuple[Any, ...], int] = {}
    gradient_kinds: list[int] = []
    generator_coeffs: list[float] = []

    for index, gate in enumerate(circuit.operators):
        if type(gate) is Barrier:
            continue
        if getattr(gate, 'den_mat', False) or getattr(gate, 'condition', False):
            raise _error(index, gate, 'density-matrix and conditional operations are unsupported')
        wires = tuple(int(wire) for wire in getattr(gate, 'wires', []))
        gate_controls = tuple(int(wire) for wire in getattr(gate, 'controls', []))
        if not wires or len(set(wires)) != len(wires) or any(wire < 0 or wire >= circuit.nqubit for wire in wires):
            raise _error(index, gate, 'targets must be non-empty, unique circuit wires')
        if len(set(gate_controls)) != len(gate_controls) or any(
            wire < 0 or wire >= circuit.nqubit for wire in gate_controls
        ):
            raise _error(index, gate, 'controls must be unique circuit wires')
        if set(wires) & set(gate_controls):
            raise _error(index, gate, 'targets and controls must be disjoint')
        vendor_targets = _vendor_bits(circuit.nqubit, wires)
        vendor_controls = tuple(circuit.nqubit - wire - 1 for wire in gate_controls)
        name = type(gate).__name__

        if name == 'PhaseShift':
            theta = gate.theta
            scale = -1.0 if gate.inv_mode else 1.0
            opcodes.append(OP_PHASE)
            targets.append(vendor_targets)
            controls.append(vendor_controls)
            matrices.append(None)
            paulis.append(())
            parameter_scales.append(scale)
            generator_coeffs.append(scale)
            if theta.requires_grad:
                key = _parameter_key(theta)
                if key not in parameter_slots:
                    parameter_slots[key] = len(parameter_sources)
                    parameter_sources.append(theta)
                parameter_indices.append(parameter_slots[key])
                angles.append(0.0)
                gradient_kinds.append(GRAD_PHASE)
            else:
                parameter_indices.append(-1)
                angles.append(float(theta.detach().cpu()) * scale)
                gradient_kinds.append(GRAD_NONE)
            continue

        if name in _PAULI or name == 'Rxy':
            theta = getattr(gate, 'theta', None)
            if not isinstance(theta, torch.Tensor) or theta.numel() != 1:
                raise _error(index, gate, 'rotation angle must be one scalar tensor')
            labels = (3,) * len(wires) if name == 'MultiRZ' else _PAULI.get(name, (1, 1))
            if len(wires) != len(labels):
                raise _error(index, gate, f'{name} has an invalid target count')
            scale = -1.0 if getattr(gate, 'inv_mode', False) else 1.0
            # DQ Rxy is an exchange rotation: Rxx(theta/2) Ryy(theta/2).
            # Both commuting rotations share the original scalar parameter.
            groups = ((1, 1), (2, 2)) if name == 'Rxy' else (labels,)
            if name == 'Rxy':
                scale *= 0.5
            for labels in groups:
                opcodes.append(OP_PAULI_ROTATION)
                targets.append(vendor_targets)
                controls.append(vendor_controls)
                matrices.append(None)
                if theta.requires_grad:
                    key = _parameter_key(theta)
                    if key not in parameter_slots:
                        parameter_slots[key] = len(parameter_sources)
                        parameter_sources.append(theta)
                    parameter_indices.append(parameter_slots[key])
                    angles.append(0.0)
                    gradient_kinds.append(GRAD_PAULI_GENERATOR)
                    generator_coeffs.append(-0.5 * scale)
                else:
                    parameter_indices.append(-1)
                    angles.append(float(-0.5 * scale * theta.detach().cpu()))
                    gradient_kinds.append(GRAD_NONE)
                    generator_coeffs.append(0.0)
                paulis.append(tuple(reversed(labels)))
                parameter_scales.append(scale)
            continue

        if name == 'Identity':
            continue
        if name in ('CNOT', 'Toffoli', 'Fredkin'):
            control_count = 2 if name == 'Toffoli' else 1
            vendor_controls = (*vendor_controls, *(circuit.nqubit - w - 1 for w in wires[:control_count]))
            wires = wires[control_count:]
            vendor_targets = _vendor_bits(circuit.nqubit, wires)
            if name == 'Fredkin':
                matrix = torch.eye(4, dtype=torch.complex64)[[0, 2, 1, 3]]
            else:
                matrix = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        else:
            matrix = _static_matrix(gate, index)
        dim = 1 << len(wires)
        if matrix.shape != (dim, dim):
            raise _error(index, gate, f'expected a {dim}x{dim} matrix, got {tuple(matrix.shape)}')
        opcodes.append(OP_MATRIX)
        targets.append(vendor_targets)
        controls.append(vendor_controls)
        matrices.append(matrix)
        angles.append(0.0)
        paulis.append(())
        parameter_indices.append(-1)
        parameter_scales.append(1.0)
        gradient_kinds.append(GRAD_NONE)
        generator_coeffs.append(0.0)

    permutations = []
    diagonals = []
    for i, matrix in enumerate(matrices):
        permutation = ()
        diagonal = ()
        if matrix is not None:
            nonzero = matrix != 0
            if (nonzero.sum(0) == 1).all() and (nonzero.sum(1) == 1).all():
                columns = nonzero.to(torch.int64).argmax(1)
                permutation = tuple(columns.tolist())
                diagonal = tuple(matrix[torch.arange(len(columns)), columns].tolist())
                opcodes[i] = OP_PERMUTATION
                matrices[i] = None
        permutations.append(permutation)
        diagonals.append(diagonal)

    observable_paulis, observable_bits, observable_matrix, observable_matrix_bits = _compile_observables(circuit)
    if getattr(circuit, 'diff_method', 'none') == 'adjoint' and (
        not observable_paulis or observable_matrix is not None
    ):
        raise ValueError('adjoint requires one or more terminal Pauli observables')
    return CompiledCircuitPlan(
        nqubit=circuit.nqubit,
        opcodes=np.asarray(opcodes, dtype=np.int32),
        targets=tuple(targets),
        controls=tuple(controls),
        matrices=tuple(matrices),
        permutations=tuple(permutations),
        diagonals=tuple(diagonals),
        angles=np.asarray(angles, dtype=np.float64),
        paulis=tuple(paulis),
        parameter_indices=np.asarray(parameter_indices, dtype=np.int32),
        parameter_scales=np.asarray(parameter_scales, dtype=np.float64),
        gradient_kinds=np.asarray(gradient_kinds, dtype=np.int32),
        generator_coeffs=np.asarray(generator_coeffs, dtype=np.float64),
        parameter_sources=tuple(parameter_sources),
        observable_paulis=observable_paulis,
        observable_bits=observable_bits,
        observable_matrix=observable_matrix,
        observable_matrix_bits=observable_matrix_bits,
    )
