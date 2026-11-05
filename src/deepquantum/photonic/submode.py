"""Submode expansion and unresolved readout of spatial photonic circuits."""

import itertools
from collections import Counter
from copy import copy
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn, vmap
from torch.utils._pytree import register_pytree_node

import deepquantum.photonic as dqp

from .channel import OPA, PhotonLoss
from .circuit import QumodeCircuit
from .gate import (
    Barrier,
    BeamSplitter,
    BeamSplitterPhi,
    BeamSplitterSingle,
    BeamSplitterTheta,
    Displacement,
    DisplacementMomentum,
    DisplacementPosition,
    MZI,
    PhaseShift,
    Squeezing,
    Squeezing2,
    UAnyGate,
)
from .hardware import HardwareModel, Loss, _PluginBinding
from .qmath import (
    _convolve_counts,
    _fock_conditional_blocks,
    _fock_isometry,
    _group_count_axes,
    _normalize_coefficients,
    fock_combinations,
    group_probabilities,
)
from .source import SinglePhotonSource, Submodes
from .state import BosonicState, FockState, GaussianState

_PASSIVE = (BeamSplitter, BeamSplitterPhi, BeamSplitterSingle, BeamSplitterTheta, MZI, PhaseShift, UAnyGate, PhotonLoss)
_ACTIVE = (Squeezing, Squeezing2, Displacement, DisplacementPosition, DisplacementMomentum, OPA)


@dataclass(frozen=True)
class SubmodeResult:
    """Circuit output with an explicit orthogonal-basis mode layout.

    ``nmode`` counts spatial modes. Joint data use spatial-then-orthogonal-basis order.
    Factorized data have leading ``(batch, nsubmode)`` axes. Each factor contains
    nmode spatial modes. They are simultaneous independent quantum systems, not
    alternative samples. ``to_joint()`` explicitly materializes their tensor product.
    Gaussian factors contain [cov, mean]. Fock factors contain pure or mixed state tensors.
    With partially coupled groups, ``data`` is a tuple of joint block states and
    ``blocks`` records their orthogonal-basis mode indices. ``partition`` describes either layout.
    Bosonic and Fock basis state outputs currently use the joint representation.
    Fock basis state data are a transfer matrix when ``is_prob`` is None.
    """

    data: Any
    nmode: int
    nsubmode: int
    backend: str
    den_mat: bool
    factorized: bool
    blocks: tuple[tuple[int, ...], ...] = ()

    @property
    def partition(self):
        """Orthogonal-basis mode indices evolved jointly in each independent block."""
        return self.blocks or (
            tuple((k,) for k in range(self.nsubmode)) if self.factorized else (tuple(range(self.nsubmode)),)
        )

    def _factors(self):
        if self.blocks:
            return list(zip(self.blocks, self.data, strict=True))
        if not self.factorized:
            return [(tuple(range(self.nsubmode)), self.data)]
        axis = -(1 + self.nmode * (2 if self.den_mat else 1)) if self.backend == 'fock' else -3
        return [
            (
                (k,),
                self.data.select(axis, k) if self.backend == 'fock' else [value.select(axis, k) for value in self.data],
            )
            for k in range(self.nsubmode)
        ]

    def to_joint(self) -> Any:
        """Return ordinary backend tensors, explicitly allocating a joint state if needed."""
        if not self.factorized:
            return self.data
        blocks, states = zip(*self._factors(), strict=True)
        return _join_states(states, blocks, self.nmode, self.nsubmode, self.backend, self.den_mat)


register_pytree_node(
    SubmodeResult,
    lambda state: (
        [state.data],
        (state.nmode, state.nsubmode, state.backend, state.den_mat, state.factorized, state.blocks),
    ),
    lambda values, context: SubmodeResult(values[0], *context),
)


def _post_select_spatial(result, cutoff, wires, counts, keep, retained_submodes):
    """Convolve independent PNR conditional operators, tracing unretained orthogonal basis modes first."""
    accumulated, labels = None, []
    factors = sorted(result._factors(), key=lambda item: bool(set(retained_submodes).intersection(item[0])))
    for block, state in factors:
        size = len(block)
        retained = [(w, k) for w in keep for k in block if k in retained_submodes]
        selected = [w * size + block.index(k) for w, k in retained]
        measured = [w * size + k for w in wires for k in range(size)]
        values = _fock_conditional_blocks(state, result.nmode * size, cutoff, result.den_mat, measured, selected)
        if size == 1:
            values = values[(slice(None),) + tuple(slice(0, n + 1) for n in counts)]
        else:
            groups = [list(range(j * size, (j + 1) * size)) for j in range(len(wires))]
            values = _group_count_axes(values, groups, counts, operators=True)
        accumulated = values if accumulated is None else _convolve_counts(accumulated, values, counts, operators=True)
        labels.extend(retained)
    if any(n >= size for n, size in zip(counts, accumulated.shape[1:-2], strict=True)):
        raise ValueError('Cannot normalize a zero-probability PNR event.')
    density = accumulated[(slice(None),) + tuple(counts)]
    probability = density.diagonal(dim1=-2, dim2=-1).sum(-1).real
    if (probability <= 0).any():
        raise ValueError('Cannot normalize a zero-probability PNR event.')
    density = density / probability[:, None, None]
    order = [labels.index((w, k)) for w in keep for k in retained_submodes]
    density = density.reshape([-1] + [cutoff] * (2 * len(labels)))
    return density.permute([0] + [i + 1 for i in order] + [i + len(labels) + 1 for i in order]), probability


def _join_states(states, blocks, nmode, nsubmode, backend, den_mat):
    """Tensor independent blocks, then restore spatial-major coordinates."""
    labels = [(w, k) for block in blocks for w in range(nmode) for k in block]
    order = [labels.index((w, k)) for w in range(nmode) for k in range(nsubmode)]
    total = nmode * nsubmode
    if backend == 'gaussian':
        covs, means = zip(*states, strict=True)
        sizes = [cov.shape[-1] for cov in covs]
        rows = []
        for i, value in enumerate(covs):
            rows.append(torch.nn.functional.pad(value, (sum(sizes[:i]), sum(sizes[i + 1 :]))))
        cov = torch.cat(rows, -2)
        mean = torch.cat(means, -2)
        offset, x, p = 0, [], []
        for size in sizes:
            x += list(range(offset, offset + size // 2))
            p += list(range(offset + size // 2, offset + size))
            offset += size
        indices = [x[i] for i in order] + [p[i] for i in order]
        return [cov[..., indices, :][..., indices], mean[..., indices, :]]
    cutoff = states[0].shape[-1]
    rank = 2 if den_mat else 1
    batch_shape = states[0].shape[: -rank * nmode * len(blocks[0])]
    factors = [
        s.reshape(*batch_shape, *([cutoff ** (nmode * len(b))] * rank)) for s, b in zip(states, blocks, strict=True)
    ]
    result = factors[0]
    for value in factors[1:]:
        if den_mat:
            size = result.shape[-1] * value.shape[-1]
            result = torch.einsum('...ij,...kl->...ikjl', result, value).reshape(*batch_shape, size, size)
        else:
            result = (result.unsqueeze(-1) * value.unsqueeze(-2)).flatten(-2)
    result = result.reshape(*batch_shape, *([cutoff] * (rank * total)))
    leading = len(batch_shape)
    axes = list(range(leading)) + [i + leading for i in order]
    if den_mat:
        axes += [i + leading + total for i in order]
    return result.permute(axes)


def _profile_rotations(coefficients):
    """Complete the specified orthogonal basis and decompose its unitary into two-mode rotations."""
    columns = coefficients.mT
    n, p = columns.shape
    remaining = columns
    if n > p:
        complement = torch.linalg.qr(columns.detach(), mode='complete').Q[:, p:]
        for vector in complement.unbind(-1):
            vector = vector - remaining @ (remaining.mH @ vector)
            remaining = torch.cat([remaining, (vector / vector.norm()).unsqueeze(-1)], -1)
    rotations = []
    for col in range(n):
        for row in range(n - 1, col, -1):
            a, b = remaining[col, col], remaining[row, col]
            norm = (a.abs().square() + b.abs().square()).sqrt()
            denominator = torch.where(norm > 0, norm, torch.ones_like(norm))
            g = torch.stack([torch.stack([a.conj(), b.conj()]), torch.stack([-b, a])]) / denominator
            g = torch.where(norm > 0, g, torch.eye(2, device=g.device, dtype=g.dtype))
            indices = torch.tensor([col, row], device=g.device)
            remaining = remaining.index_copy(0, indices, g @ remaining[indices])
            rotations.append(([col, row], g.mH.resolve_conj()))
    phases = [([i], remaining[i, i].reshape(1, 1)) for i in range(n)]
    return phases + rotations[::-1]


def _lo_coefficients(lo, nwire, nsubmode, reference):
    """Normalize LO coefficients in the common orthogonal basis for each measured spatial mode."""
    if lo is None:
        coefficients = reference.new_zeros(nwire, nsubmode)
        coefficients[:, 0] = 1
    else:
        coefficients = torch.as_tensor(lo, device=reference.device)
        if coefficients.ndim == 1:
            coefficients = coefficients.unsqueeze(0).expand(nwire, -1)
        if coefficients.ndim != 2 or coefficients.shape[0] != nwire or not 0 < coefficients.shape[1] <= nsubmode:
            raise ValueError(
                'lo must be a coefficient vector or one row per selected spatial wire, in the common orthogonal basis.'
            )
        coefficients = coefficients.to(
            torch.complex128 if reference.dtype in (torch.float64, torch.complex128) else torch.complex64
        )
        if not torch.isfinite(coefficients).all() or (coefficients.abs().amax(dim=-1) == 0).any():
            raise ValueError('LO coefficient rows must be finite and nonzero.')
        coefficients = torch.nn.functional.pad(
            _normalize_coefficients(coefficients), (0, nsubmode - coefficients.shape[1])
        )
    return coefficients.to(
        torch.complex128 if reference.dtype in (torch.float64, torch.complex128) else torch.complex64
    )


def _lo_view(result, cutoff, wires, lo=None):
    """Reduce saved states to the LO modes without constructing the evolved joint state."""
    factors = result._factors()
    reference = factors[0][1] if result.backend == 'fock' else factors[0][1][0]
    coefficients = _lo_coefficients(lo, len(wires), result.nsubmode, reference)
    if result.backend in ('gaussian', 'bosonic'):
        cov, mean = 0, 0
        for block, state in factors:
            n = result.nmode * len(block)
            amplitude = coefficients[:, list(block)]
            embedding = amplitude.new_zeros(len(wires), n)
            for i, wire in enumerate(wires):
                embedding[i, wire * len(block) : (wire + 1) * len(block)] = amplitude[i]
            # b_g = ell^dagger a: x_g = Re(ell) x + Im(ell) p.
            projection = torch.cat(
                [torch.cat([embedding.real, embedding.imag], -1), torch.cat([-embedding.imag, embedding.real], -1)], -2
            ).to(state[0])
            cov = cov + projection @ state[0] @ projection.mT
            mean = mean + projection.to(state[1].dtype) @ state[1]
        data = [cov, mean] + ([factors[0][1][2]] if result.backend == 'bosonic' else [])
    else:
        active = (coefficients.abs().sum(0) != 0).nonzero().flatten().tolist()
        reduced, blocks, mass = [], [], 1
        for block, state in factors:
            kept = [k for k in block if k in active]
            selected = [w * len(block) + block.index(k) for w in wires for k in kept]
            density = _fock_conditional_blocks(state, result.nmode * len(block), cutoff, result.den_mat, [], selected)
            if kept:
                reduced.append(density.reshape([-1] + [cutoff] * (2 * len(selected))))
                blocks.append(tuple(active.index(k) for k in kept))
            else:
                mass = mass * density[:, 0, 0].real
        n = len(wires) * len(active)
        data = _join_states(reduced, blocks, len(wires), len(active), 'fock', True)
        if isinstance(mass, torch.Tensor):
            data = data * mass.reshape([-1] + [1] * (2 * n))
        for i, profile in enumerate(coefficients[:, active]):
            for local, matrix in reversed(_profile_rotations(profile.unsqueeze(0))):
                gate = UAnyGate(
                    matrix.mH.resolve_conj(), n, [i * len(active) + k for k in local], cutoff=cutoff, den_mat=True
                )
                data = gate(data)
        data = _fock_conditional_blocks(data, n, cutoff, True, [], [i * len(active) for i in range(len(wires))])
        data = data.reshape([-1] + [cutoff] * (2 * len(wires)))
    view = QumodeCircuit(len(wires), 'vac', cutoff=cutoff, backend=result.backend, basis=False, den_mat=True)
    view.state = data
    return view


def _clone_operation(op, wires, nmode):
    cloned = copy(op)
    cloned._parameters = op._parameters.copy()
    cloned._buffers = op._buffers.copy()
    cloned.wires, cloned.nmode = list(wires), nmode
    return cloned


def _embed_fock(state, profiles, cutoff, den_mat):
    """Apply one isometry per spatial ket/bra axis, preserving number coherences."""
    nspatial, nsubmode = profiles.shape
    state = state.to(torch.promote_types(state.dtype, profiles.dtype))
    state = state.reshape([-1] + [cutoff] * nspatial * (2 if den_mat else 1))
    isometries = [_fock_isometry(profile, cutoff).to(state) for profile in profiles]
    for axis in range(state.ndim - 1, 0, -1):
        matrix = isometries[(axis - 1) % nspatial]
        if axis > nspatial:
            matrix = matrix.conj()
        state = torch.tensordot(state, matrix, dims=([axis], [-1]))
        end = state.ndim - nsubmode
        state = state.permute(list(range(axis)) + list(range(end, state.ndim)) + list(range(axis, end)))
    return state


class _SubmodeOperation(nn.Module):
    """Expose physical controls to HardwareModel while adapting their action on modes."""

    def __init__(self, operation, circuit, modes, run):
        super().__init__()
        object.__setattr__(self, 'operation', operation)
        object.__setattr__(self, 'circuit', circuit)
        self._parameters = operation._parameters.copy()
        self._buffers = operation._buffers.copy()
        self.wires = operation.wires if circuit.factorized else [k for w in operation.wires for k in circuit.groups[w]]
        self.modes, self.run = modes, run

    def forward(self, state):
        operation = _clone_operation(self.operation, self.operation.wires, self.operation.nmode)
        operation._parameters, operation._buffers = self._parameters.copy(), self._buffers.copy()
        return self.circuit._lift(operation, state, self.modes, self.run)


class _SubmodeCircuit(QumodeCircuit):
    """Ephemeral execution view that reuses the ordinary backend and measurement kernels."""

    def __init__(self, parent: QumodeCircuit, state: Any = None, *, nsubmode=None, joint=False):
        if parent.mps or parent._with_delay:
            raise NotImplementedError('Submode expansion does not support MPS; unfold delays with global_circuit().')
        if parent.measurements:
            raise NotImplementedError('Use measure_homodyne() without conditional homodyne operations for submodes.')
        model = parent.hardware_model if parent.hardware_model is not None else HardwareModel()
        d = (
            nsubmode
            if nsubmode is not None
            else max(
                [model.nsubmode]
                + [s.components.shape[0] for s in parent.sources.values() if isinstance(s, SinglePhotonSource)]
            )
        )
        factorized = (
            not joint
            and d > 1
            and (parent.backend == 'gaussian' or parent.backend == 'fock' and not parent.basis)
            and not any(isinstance(source, SinglePhotonSource) for source in parent.sources.values())
            and all(b.plugin.components is None for b in model.bindings.values() if isinstance(b.plugin, Submodes))
        )
        nmode = parent.nmode if factorized else parent.nmode * d
        source = parent.init_state if state is None else state
        if parent.backend == 'bosonic' and state is None and parent._bosonic_states is not None:
            from .state import combine_bosonic_states

            source = combine_bosonic_states(parent._bosonic_states, cutoff=parent.cutoff)
        if not isinstance(source, (FockState, GaussianState, BosonicState)):
            if parent.backend == 'fock':
                source = FockState(source, parent.nmode, parent.cutoff, parent.basis, parent.den_mat)
            else:
                source = (GaussianState if parent.backend == 'gaussian' else BosonicState)(
                    source, parent.nmode, parent.cutoff
                )
        if source.nmode != parent.nmode:
            raise ValueError('Provide a spatial input state before submode expansion.')
        input_modes = {}
        for binding, wires in model._io_bindings(parent.wires, 'input'):
            if isinstance(binding.plugin, Submodes):
                input_modes.update(dict.fromkeys(wires, binding.plugin))
        for wire, preparation in parent.sources.items():
            if isinstance(preparation, SinglePhotonSource) and preparation.is_pure and not parent.den_mat:
                profile = Submodes(torch.ones(preparation.components.shape[0]))
                profile.components = torch.view_as_real(preparation.coefficients.resolve_conj()).unsqueeze(0)
                input_modes[int(wire)] = profile
        reference = source.state if parent.backend == 'fock' else source.cov
        dtype = torch.complex128 if reference.dtype in (torch.float64, torch.complex128) else torch.complex64
        default = torch.zeros(d, dtype=dtype, device=reference.device)
        default[0] = 1
        profiles = torch.stack(
            [
                torch.nn.functional.pad(
                    input_modes[wire].coefficients[0].to(default), (0, d - input_modes[wire].nsubmode)
                )
                if wire in input_modes
                else default
                for wire in parent.wires
            ]
        )
        if factorized:
            initial = source.state if parent.backend == 'fock' else [source.cov, source.mean]
        elif parent.backend == 'fock':
            if parent.basis:
                initial = torch.nn.functional.pad(source.state.unsqueeze(-1), (0, d - 1))
                initial = initial.reshape(*source.state.shape[:-1], nmode)
            else:
                initial = _embed_fock(source.state, profiles, parent.cutoff, parent.den_mat)
        else:
            identity = torch.eye(parent.nmode, dtype=profiles.dtype, device=profiles.device)
            embedding = (profiles.unsqueeze(-1) * identity.unsqueeze(1)).reshape(nmode, parent.nmode)
            real = torch.cat(
                [torch.cat([embedding.real, -embedding.imag], -1), torch.cat([embedding.imag, embedding.real], -1)], -2
            ).to(source.cov)
            vacuum = dqp.hbar / (4 * dqp.kappa**2)
            initial = [
                real
                @ (source.cov - vacuum * torch.eye(2 * parent.nmode, device=real.device, dtype=real.dtype))
                @ real.mT
                + vacuum * torch.eye(2 * nmode, device=real.device, dtype=real.dtype),
                real.to(source.mean.dtype) @ source.mean,
            ]
            if parent.backend == 'bosonic':
                initial.append(source.weight)
        # Isometric vacuum embedding preserves Gaussian purity; do not re-test it
        # inside vmap or rebuild state metadata with data-dependent validation.
        initial_state = copy(source)
        initial_state._buffers = source._buffers.copy()
        initial_state.nmode = nmode
        if parent.backend == 'fock':
            initial_state.state = initial
        else:
            initial_state.cov, initial_state.mean = initial[:2]
        super().__init__(
            nmode,
            initial_state,
            cutoff=parent.cutoff,
            backend=parent.backend,
            basis=parent.basis,
            den_mat=parent.den_mat,
            detector=parent.detector,
        )
        object.__setattr__(self, 'parent', parent)
        self.factorized = factorized
        self.nsubmode = d
        self.groups = [list(range(w * d, (w + 1) * d)) for w in parent.wires]
        self.input_modes = input_modes
        self.hardware_model = model
        self.ndata = parent.ndata
        self._bindings = model._bind(parent.operators)
        self._rotations = {}
        if any(not isinstance(op, _PASSIVE + _ACTIVE + (Barrier,)) for op in parent.operators):
            raise NotImplementedError(
                'Submode expansion supports passive optics, squeezing, displacement, OPA and loss.'
            )

    def forward(self, data=None, state=None, is_prob=None, detector=None, sort=True):
        if self.backend == 'fock':
            result = self._forward_fock(data, state, is_prob, sort)
            if not self.basis and result.ndim == self.nmode * (2 if self.den_mat else 1) + self.factorized:
                result = result.unsqueeze(0)
        else:
            result = self._forward_cv(data, state, is_prob, detector, True)
            if self.factorized and result[0].ndim == 3:
                result = [value.unsqueeze(0) for value in result]
        self.state = result
        return result

    def result(self):
        return SubmodeResult(self.state, self.parent.nmode, self.nsubmode, self.backend, self.den_mat, self.factorized)

    def encode(self, data):
        self.parent.encode(data)

    def _prepare_init_state(self, state, reset_fock_basis=False, nloss=None):
        if nloss is None:
            nloss = self.nsubmode * (
                sum(isinstance(op, PhotonLoss) for op in self.parent.operators)
                + self.hardware_model._loss_count(self.parent.operators, self.parent.wires)
            )
        self._nloss = nloss
        self._lossy = self._nloss > 0
        return super()._prepare_init_state(state, reset_fock_basis, self._nloss)

    def _primitive(self, op, state, run):
        if not run.basis:
            return op(state)
        if isinstance(op, PhotonLoss):
            return PhotonLoss._apply_unitary(state, op.gate.update_matrix(), op.wires, self.nmode)
        matrix = op.update_matrix()
        if matrix.shape != (len(op.wires), len(op.wires)):
            raise NotImplementedError('Active operations require Fock state tensors (basis=False).')
        if state is None:
            state = torch.eye(self.nmode, dtype=matrix.dtype, device=matrix.device)
        dtype = torch.promote_types(state.dtype, matrix.dtype)
        state, matrix = state.to(dtype), matrix.to(device=state.device, dtype=dtype)
        rows = torch.tensor(op.wires, device=state.device)
        cols = torch.arange(state.shape[-1], device=state.device)
        return state.index_put((rows[:, None], cols), matrix @ state[rows])

    def _rotate(self, state, modes, wire, run, inverse=False):
        if modes.components is None or not len(modes.relative_r):
            return state
        if modes not in self._rotations:
            self._rotations[modes] = _profile_rotations(modes.coefficients)
        rotations = reversed(self._rotations[modes]) if inverse else self._rotations[modes]
        for local, matrix in rotations:
            wires = [wire * self.nsubmode + k for k in local]
            # Validate the constant identity at construction, not a mapped tensor
            # inside vmap. The compiled matrix is unitary by construction.
            gate = UAnyGate(
                torch.eye(len(wires), dtype=matrix.dtype, device=matrix.device),
                self.nmode,
                wires=wires,
                cutoff=self.cutoff,
                den_mat=self.den_mat,
            )
            gate.matrix = matrix.mH.resolve_conj() if inverse else matrix
            state = self._primitive(gate, state, run)
        return state

    def _lift(self, op, state, modes, run):
        if isinstance(op, Barrier):
            return state
        if self.factorized:
            if isinstance(op, _PASSIVE):
                return op(state)
            ratios = op.r.new_zeros(self.nsubmode)
            if modes is None:
                ratios[0] = 1
            else:
                ratios = torch.nn.functional.pad(modes.relative_r.to(op.r), (0, self.nsubmode - len(modes.relative_r)))
            values = state if self.backend == 'fock' else state[0]
            batch = values.shape[0] // self.nsubmode
            blocks = self._reshape_blocks(state, batch)

            def apply(block, ratio):
                gate = _clone_operation(op, op.wires, self.nmode)
                gate._parameters.pop('r', None)
                gate._buffers['r'] = op.r * ratio
                if isinstance(op, OPA):
                    gate._parameters.pop('loss_theta', None)
                    gate._buffers['loss_theta'] = op.loss_theta * ratio
                return gate(block)

            result = vmap(apply, randomness='same')(blocks, ratios)
            return self._flatten_blocks(result)
        if modes is not None:
            for wire in op.wires:
                state = self._rotate(state, modes, wire, run, inverse=True)
        count = self.nsubmode if isinstance(op, _PASSIVE) else 1 if modes is None else len(modes.relative_r)
        for k in range(count):
            gate = _clone_operation(op, [w * self.nsubmode + k for w in op.wires], self.nmode)
            if modes is not None and isinstance(op, _ACTIVE):
                gate._parameters.pop('r', None)
                gate._buffers['r'] = op.r * modes.relative_r[k].to(op.r)
                if isinstance(op, OPA):
                    gate._parameters.pop('loss_theta', None)
                    gate._buffers['loss_theta'] = op.loss_theta * modes.relative_r[k].to(op.loss_theta)
            state = self._primitive(gate, state, run)
        if modes is not None:
            for wire in op.wires:
                state = self._rotate(state, modes, wire, run)
        return state

    def _reshape_blocks(self, state, batch):
        if isinstance(state, list):
            return [value.reshape(self.nsubmode, batch, *value.shape[1:]) for value in state]
        return state.reshape(self.nsubmode, batch, *state.shape[1:])

    def _flatten_blocks(self, state):
        if isinstance(state, list):
            return [value.flatten(0, 1) for value in state]
        return state.flatten(0, 1)

    def _factor_input(self, state):
        if self.backend == 'fock':
            vacuum = torch.zeros_like(state)
            vacuum[(slice(None),) + (0,) * (state.ndim - 1)] = 1
            return torch.stack([state] + [vacuum] * (self.nsubmode - 1)).flatten(0, 1)
        cov, mean = state
        vacuum = torch.eye(2 * self.nmode, dtype=cov.dtype, device=cov.device).expand_as(cov)
        vacuum = vacuum * dqp.hbar / (4 * dqp.kappa**2)
        return [
            torch.stack([cov] + [vacuum] * (self.nsubmode - 1)).flatten(0, 1),
            torch.stack([mean] + [torch.zeros_like(mean)] * (self.nsubmode - 1)).flatten(0, 1),
        ]

    def _io(self, state, run, where):
        for binding, wires in self.hardware_model._io_bindings(self.parent.wires, where):
            plugin = binding.plugin
            if isinstance(plugin, Submodes):
                continue
            if isinstance(plugin, Loss):
                selected = tuple(wires) if self.factorized else tuple(k for wire in wires for k in self.groups[wire])
                state = plugin(state, replace(run, wires=selected))
            else:
                raise NotImplementedError('Submode input/output plugins support mode coefficients and loss.')
        return state

    def _execute(self, state, basis=False, run=None):
        if run is None:
            run = self.hardware_model._new_run(self.backend, range(self.nmode), basis)
        else:
            run = replace(run, wires=tuple(range(self.nmode)), nmode=self.nmode, basis=basis)
        if self.factorized:
            state = self._factor_input(state)
        if basis:
            for wire, modes in self.input_modes.items():
                state = self._rotate(state, modes, wire, run)
        if self.den_mat:
            for wire, preparation in self.parent.sources.items():
                if isinstance(preparation, SinglePhotonSource):
                    wire = int(wire)
                    state = preparation(state, replace(run, wires=tuple(self.groups[wire])))
        state = self._io(state, run, 'input')
        for op in self.parent.operators:
            bindings = self._bindings.get(op, ())
            modes = next((b.plugin for b, _ in bindings if isinstance(b.plugin, Submodes)), None)
            if any(
                binding.where in ('before', 'after') and not isinstance(binding.plugin, Loss) for binding, _ in bindings
            ):
                raise NotImplementedError('Submode gate boundaries support loss.')
            lifted = _SubmodeOperation(op, self, modes, run)
            state = self.hardware_model._apply_operation(state, lifted, run, bindings, source=op)
        state = self._io(state, run, 'output')
        if state is None:
            state = torch.eye(self.nmode, dtype=torch.cfloat)
        return state

    def _get_unitary(self, _operators, _model=None):
        # Parent hooks pass these arguments; this view executes the compiled parent circuit.
        return self._execute(None, basis=True)

    def _forward_helper_tensor(self, data=None, state=None, is_prob=None):
        self.encode(data)
        state = self.init_state.state if state is None else state
        result = self._execute(self.tensor_rep(state))
        if self.factorized:
            result = self._reshape_blocks(result, result.shape[0] // self.nsubmode).movedim(0, 1)
        if is_prob:
            result = (
                result.reshape(-1, self.cutoff**self.nmode, self.cutoff**self.nmode)
                .diagonal(dim1=-2, dim2=-1)
                .real.reshape([-1] + [self.cutoff] * self.nmode)
                if self.den_mat
                else result.abs().square()
            )
        return result.squeeze(0)

    def _forward_helper_gaussian(self, data=None, state=None, _stepwise=True):
        # The parent CV hook supplies a flag; submode operations always execute stepwise.
        self.encode(data)
        state = [self.init_state.cov, self.init_state.mean] if state is None else state
        result = self._execute(state)
        if self.factorized:
            result = [
                value.movedim(0, 1) for value in self._reshape_blocks(result, result[0].shape[0] // self.nsubmode)
            ]
        return [x.squeeze(0) for x in result]

    def readout_view(self):
        """Expose saved factors as a native backend batch without changing the parent."""
        view = copy(self)
        if self.factorized:
            view.state = (
                [value.flatten(0, 1) for value in self.state]
                if isinstance(self.state, list)
                else self.state.flatten(0, 1)
            )
        return view

    def _probabilities(self, wires=None, limits=None):
        """Resolved probabilities, with (batch, nsubmode) leading axes for product states."""
        selected = self.wires if wires is None else wires
        view = self.readout_view()
        if self.backend == 'fock':
            state = view.state
            if self.den_mat:
                state = state.reshape(-1, self.cutoff**self.nmode, self.cutoff**self.nmode)
                prob = state.diagonal(dim1=-2, dim2=-1).real.reshape([-1] + [self.cutoff] * self.nmode)
            else:
                prob = state.abs().square().reshape([-1] + [self.cutoff] * self.nmode)
            traced = tuple(w + 1 for w in self.wires if w not in selected)
            if traced:
                prob = prob.sum(traced)
            order = sorted(selected)
            prob = prob.permute([0] + [order.index(w) + 1 for w in selected])
            if limits is not None:
                prob = prob[(slice(None),) + tuple(slice(0, n + 1) for n in limits)]
        elif self.backend == 'gaussian':
            if selected != self.wires:
                indices = selected + [w + self.nmode for w in selected]
                cov, mean = view.state
                view.state = [cov[:, indices, :][:, :, indices], mean[:, indices]]
                view.nmode, view.wires = len(selected), list(range(len(selected)))
            if limits is None:
                probabilities = view._forward_cv_prob(*view.state, detector='pnrd')
                prob = torch.stack(
                    [
                        probabilities[FockState(pattern)]
                        for pattern in itertools.product(range(self.cutoff), repeat=view.nmode)
                    ],
                    -1,
                ).reshape([-1] + [self.cutoff] * view.nmode)
            else:
                shape = [min(self.cutoff, n + 1) for n in limits]
                terms = [
                    view.get_prob(pattern).reshape(-1) for pattern in itertools.product(*(range(n) for n in shape))
                ]
                prob = torch.stack(terms, -1).reshape([-1] + shape)
        else:
            raise NotImplementedError('Bosonic photon-count probabilities are not supported.')
        if self.factorized:
            prob = prob.reshape(-1, self.nsubmode, *prob.shape[1:])
        return prob

    def spatial_probabilities(self, detector=None):
        detector = self.detector if detector is None else detector
        if detector != 'pnrd':
            raise NotImplementedError('Unresolved submode readout currently supports PNR detection.')
        if isinstance(self.state, dict):
            return group_probabilities(self.state, self.groups)
        if self.backend == 'fock' and self.basis:
            unitary = self.state.unsqueeze(0) if self.state.ndim == 2 else self.state
            initial = self._init_state_forward.reshape(-1, self._init_state_forward.shape[-1])
            probabilities = vmap(self._evaluate_fock_unitary, in_dims=(0, 0, None))(
                unitary, initial.expand(len(unitary), -1), True
            )
            return group_probabilities(probabilities, self.groups)
        prob = self._probabilities()
        if self.factorized:
            result = prob[:, 0]
            for k in range(1, self.nsubmode):
                result = _convolve_counts(result, prob[:, k])
            return group_probabilities(result, [[w] for w in self.wires])
        return group_probabilities(prob, self.groups)

    def _factor_probability(self, counts, wires=None):
        prob = self._probabilities(wires, counts)
        result = prob[:, 0]
        for k in range(1, self.nsubmode):
            result = _convolve_counts(result, prob[:, k], counts)
        if any(n >= size for n, size in zip(counts, result.shape[1:], strict=True)):
            return result.new_zeros(result.shape[0])
        return result[(slice(None),) + tuple(counts)]

    def get_spatial_prob(self, final_state):
        counts = final_state.state if isinstance(final_state, FockState) else torch.as_tensor(final_state)
        if counts.ndim != 1 or len(counts) != len(self.groups) or (counts < 0).any() or (counts != counts.long()).any():
            raise ValueError('Specify one nonnegative integer count per spatial output.')
        counts = counts.tolist()
        if self.factorized:
            return self._factor_probability(counts)
        partitions = [fock_combinations(self.nsubmode, n, self.cutoff) for n in counts]
        patterns = [tuple(k for group in groups for k in group) for groups in itertools.product(*partitions)]
        if self.backend == 'fock' and not self.basis:
            prob = self._probabilities()
            if not patterns:
                return prob.new_zeros(prob.shape[0])
            indices = torch.tensor(patterns, device=prob.device).mT.unbind(0)
            return prob[(slice(None),) + indices].sum(-1)
        terms = []
        for pattern in patterns:
            if self.backend == 'fock':
                if isinstance(self.state, dict):
                    value = self.state.get(FockState(pattern), torch.zeros_like(next(iter(self.state.values()))))
                else:
                    unitary = self.state.unsqueeze(0) if self.state.ndim == 2 else self.state
                    initial = self._init_state_forward.reshape(-1, self._init_state_forward.shape[-1])
                    value = torch.stack(
                        [
                            super(_SubmodeCircuit, self).get_prob(pattern, initial[min(i, len(initial) - 1)], u)
                            for i, u in enumerate(unitary)
                        ]
                    ).flatten()
            elif self.backend == 'gaussian':
                value = super().get_prob(pattern).reshape(-1)
            else:
                raise NotImplementedError('Bosonic photon-count probabilities are not supported.')
            terms.append(value)
        if terms:
            return sum(terms)
        if isinstance(self.state, dict):
            return torch.zeros_like(next(iter(self.state.values())))
        reference = self.state[0] if isinstance(self.state, list) else self.state
        return reference.real.new_zeros(1 if self.basis and reference.ndim == 2 else reference.shape[0])

    def measure_spatial(self, shots, with_prob, wires, detector, mcmc):
        detector = self.detector if detector is None else detector
        if detector != 'pnrd':
            raise NotImplementedError('Submode readout currently supports PNR detection.')
        selected = self.parent.wires if wires is None else self.parent._convert_indices(wires)
        if self.factorized:
            if self.backend == 'fock':
                probabilities = self._probabilities(selected)
                batch = probabilities.shape[0]
                draws = torch.multinomial(probabilities.reshape(batch * self.nsubmode, -1), shots, replacement=True)
                patterns = torch.stack(
                    [(draws // self.cutoff**i) % self.cutoff for i in reversed(range(len(selected)))], -1
                )
                samples = patterns.reshape(batch, self.nsubmode, shots, len(selected)).sum(1)
            else:
                # Existing Gaussian samplers work on a batch of independent blocks.
                # Shuffle expanded count dictionaries so their ordering cannot correlate blocks.
                resolved = self.readout_view().measure(shots, False, selected, 'pnrd', mcmc)
                resolved = [resolved] if isinstance(resolved, dict) else resolved
                draws = []
                for values in resolved:
                    events = torch.stack([key.state for key, count in values.items() for _ in range(count)])
                    draws.append(events[torch.randperm(shots)])
                samples = torch.stack(draws).reshape(-1, self.nsubmode, shots, len(selected)).sum(1)
            results = []
            for i, samples_batch in enumerate(samples):
                counts = Counter(tuple(pattern) for pattern in samples_batch.tolist())
                if with_prob:
                    counts = {
                        FockState(pattern): (n, self._factor_probability(list(pattern), selected)[i])
                        for pattern, n in counts.items()
                    }
                else:
                    counts = {FockState(pattern): n for pattern, n in counts.items()}
                results.append(counts)
            return results[0] if len(results) == 1 else results
        if with_prob or self.backend == 'fock' and not self.basis:
            probabilities = self.spatial_probabilities()
            result = self.parent._measure_dict(shots, with_prob, sorted(selected), state=probabilities)
            return result[0] if len(result) == 1 else result
        resolved = super().measure(shots, False, [k for w in selected for k in self.groups[w]], 'pnrd', mcmc)
        batches = [resolved] if isinstance(resolved, dict) else resolved
        results = []
        for batch in batches:
            counts = {}
            for pattern, occurrences in batch.items():
                key = FockState(pattern.state.reshape(-1, self.nsubmode).sum(-1))
                counts[key] = counts.get(key, 0) + occurrences
            results.append(counts)
        return results[0] if len(results) == 1 else results

    def spatial_photon_number_mean_var(self, wires):
        selected = self.parent.wires if wires is None else self.parent._convert_indices(wires)
        if self.factorized:
            if self.backend == 'gaussian':
                mean, variance = self.readout_view().photon_number_mean_var(selected)
                batch = self.state[0].shape[0]
                shape = (batch, self.nsubmode, len(selected))
                return mean.reshape(shape).sum(1).squeeze(), variance.reshape(shape).sum(1).squeeze()
            prob = self._probabilities(selected)
            shape = [1] * prob.ndim
            mass = prob.flatten(2).sum(-1)
            safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
            total_mass = mass.prod(-1)
            means, variances = [], []
            for j in range(len(selected)):
                shape[j + 2] = self.cutoff
                numbers = torch.arange(self.cutoff, dtype=prob.dtype, device=prob.device).reshape(shape)
                first = (prob * numbers).flatten(2).sum(-1) / safe_mass
                second = (prob * numbers.square()).flatten(2).sum(-1) / safe_mass
                shape[j + 2] = 1
                mean = first.sum(-1)
                variance = (second - first.square()).sum(-1)
                means.append(total_mass * mean)
                variances.append(total_mass * (variance + mean.square()) - means[-1].square())
            return torch.stack(means, -1).squeeze(), torch.stack(variances, -1).squeeze()
        if self.backend == 'fock':
            probabilities = self.spatial_probabilities()
            exp, square = 0, 0
            for pattern, probability in probabilities.items():
                counts = pattern.state[selected].to(probability)
                exp = exp + probability.reshape(-1, 1) * counts
                square = square + probability.reshape(-1, 1) * counts.square()
            return exp.squeeze(), (square - exp.square()).squeeze()
        cov, mean = self.state[:2]
        vacuum = dqp.hbar / (4 * dqp.kappa**2)
        exps, variances = [], []
        for wire in selected:
            indices = self.groups[wire] + [w + self.nmode for w in self.groups[wire]]
            v, m = cov[..., indices, :][..., indices], mean[..., indices, :]
            exp = (v.diagonal(dim1=-2, dim2=-1).sum(-1) + (m.mT @ m).squeeze(-1).squeeze(-1)) / (4 * vacuum)
            exp = exp - self.nsubmode / 2
            var = (v @ v).diagonal(dim1=-2, dim2=-1).sum(-1) / (8 * vacuum**2)
            var = var + (m.mT @ v.to(m.dtype) @ m).squeeze(-1).squeeze(-1) / (4 * vacuum**2) - self.nsubmode / 4
            if self.backend == 'bosonic':
                weights = self.state[2]
                second = ((var + exp.square()) * weights).sum(-1)
                exp = (exp * weights).sum(-1)
                var = second - exp.square()
            exps.append(exp.real)
            variances.append(var.real)
        return torch.stack(exps, -1).squeeze(), torch.stack(variances, -1).squeeze()


def _slice_profiles(modes, block):
    """Restrict orthogonal-basis coefficients without renormalizing their squeezing reference."""
    rows = [i for i, support in enumerate(modes.supports) if support and set(support) <= set(block)]
    result = copy(modes)
    result._buffers = modes._buffers.copy()
    coefficients = modes.coefficients
    size = max(block, default=-1) + 1
    coefficients = torch.nn.functional.pad(coefficients, (0, max(0, size - coefficients.shape[-1])))
    result.components = torch.view_as_real(coefficients[rows][:, list(block)].resolve_conj())
    result.relative_r = modes.relative_r[rows]
    result.supports = tuple(tuple(block.index(k) for k in modes.supports[i]) for i in rows)
    return result


def _expand_submodes(parent, state=None):
    """Group orthogonal basis modes connected by nonzero input or operation coefficients."""
    model = parent.hardware_model if parent.hardware_model is not None else HardwareModel()
    photons = [s for s in parent.sources.values() if isinstance(s, SinglePhotonSource)]
    d = max([model.nsubmode] + [s.components.shape[0] for s in photons])
    # Preserve the batched native-kernel path for operations diagonal in the common orthogonal basis.
    profiles = [b.plugin for b in model.bindings.values() if isinstance(b.plugin, Submodes)]
    if (
        d == 1
        or parent.basis
        and parent.backend == 'fock'
        or parent.backend == 'bosonic'
        or (not photons and all(p.components is None for p in profiles))
    ):
        return _SubmodeCircuit(parent, state, nsubmode=d)
    components = [{k} for k in range(d)]

    def connect(indices):
        touched = [group for group in components if group.intersection(indices)]
        if touched:
            merged = set().union(*touched)
            components[:] = [group for group in components if group not in touched] + [merged]

    # An arbitrary spatial input may correlate the orthogonal basis modes it occupies.
    inputs = {w: (0,) for w in parent.wires}
    for binding, wires in model._io_bindings(parent.wires, 'input'):
        if isinstance(binding.plugin, Submodes):
            inputs.update(dict.fromkeys(wires, binding.plugin.supports[0]))
    for wire, preparation in parent.sources.items():
        if isinstance(preparation, SinglePhotonSource):
            inputs[int(wire)] = tuple(range(preparation.components.shape[0]))
    initial_support = set(k for support in inputs.values() for k in support)
    connect(initial_support)
    for bindings in model._bind(parent.operators).values():
        for binding, _ in bindings:
            if isinstance(binding.plugin, Submodes):
                for support in binding.plugin.supports:
                    connect(support)
    blocks = tuple(sorted((tuple(sorted(group)) for group in components), key=lambda group: group[0]))
    if len(blocks) == 1:
        return _SubmodeCircuit(parent, state, nsubmode=d, joint=True)
    return _SubmodeBlocks(parent, state, blocks, initial_support)


class _SubmodeBlocks:
    """Small execution plan for independent groups of coupled orthogonal basis modes."""

    def __init__(self, parent, state, blocks, initial_support):
        self.parent, self.blocks = parent, blocks
        self.backend, self.den_mat = parent.backend, parent.den_mat
        self.nsubmode = sum(map(len, blocks))
        self.factorized = True
        self.circuits = []
        model = parent.hardware_model if parent.hardware_model is not None else HardwareModel()
        self.model = model
        source = parent.init_state if state is None else state
        if not isinstance(source, (FockState, GaussianState)):
            source = (
                FockState(source, parent.nmode, parent.cutoff, parent.basis, parent.den_mat)
                if parent.backend == 'fock'
                else GaussianState(source, parent.nmode, parent.cutoff)
            )
        reference = source.state if parent.backend == 'fock' else source.cov
        resolved = model._bind(parent.operators)
        for block in blocks:
            view = copy(parent)
            view._modules = parent._modules.copy()
            local = copy(model)
            local._modules = model._modules.copy()
            local.bindings = nn.ModuleDict()
            for name, binding in model.bindings.items():
                if isinstance(binding.plugin, Submodes):
                    if binding.where == 'input' and not initial_support <= set(block):
                        continue
                    narrowed = copy(binding)
                    narrowed._modules = binding._modules.copy()
                    narrowed.plugin = _slice_profiles(binding.plugin, block)
                    if binding.where == 'input' and not len(narrowed.plugin.relative_r):
                        continue
                    local.bindings[name] = narrowed
                else:
                    # Identity matters: all blocks share one physical noise realization.
                    local.bindings[name] = binding
            if 0 not in block:
                empty = _slice_profiles(Submodes(relative_r=torch.ones(self.nsubmode)), ())
                empty.components = reference.real.new_zeros(0, len(block), 2)
                for i, op in enumerate(parent.operators):
                    if isinstance(op, _ACTIVE) and not any(
                        isinstance(b.plugin, Submodes) for b, _ in resolved.get(op, ())
                    ):
                        local.bindings[f'default_{i}'] = _PluginBinding(
                            empty, type(op), None, where='modes', bound_ops=frozenset([op])
                        )
            view.hardware_model = local
            view.sources = nn.ModuleDict()
            for wire, preparation in parent.sources.items():
                if isinstance(preparation, SinglePhotonSource) and set(range(preparation.components.shape[0])) <= set(
                    block
                ):
                    narrowed = copy(preparation)
                    narrowed._parameters = preparation._parameters.copy()
                    narrowed._buffers = preparation._buffers.copy()
                    narrowed._parameters.pop('components', None)
                    components = preparation.components
                    padding = self.nsubmode - components.shape[0]
                    narrowed._buffers['components'] = torch.nn.functional.pad(components, (0, 0, 0, 0, 0, padding))[
                        list(block)
                    ]
                    view.sources[wire] = narrowed
            if initial_support <= set(block):
                initial = source
            elif parent.backend == 'fock':
                initial = FockState('vac', parent.nmode, parent.cutoff, basis=False, den_mat=parent.den_mat).to(
                    reference.real
                )
                initial.state = initial.state.expand(reference.shape[0], *initial.state.shape[1:])
            else:
                initial = GaussianState('vac', parent.nmode, parent.cutoff).to(reference)
                initial.cov = initial.cov.expand(reference.shape[0], -1, -1)
                initial.mean = initial.mean.expand(reference.shape[0], -1, -1)
            self.circuits.append(_SubmodeCircuit(view, initial, nsubmode=len(block), joint=True))

    def forward(self, data=None):
        initial = [
            cir.tensor_rep(cir.init_state.state)
            if self.backend == 'fock'
            else [cir.init_state.cov, cir.init_state.mean]
            for cir in self.circuits
        ]

        def execute(values, inputs):
            self.parent.encode(values)
            run = self.model._new_run(self.backend, self.parent.wires)
            return [cir._execute(value, run=run) for cir, value in zip(self.circuits, inputs, strict=True)]

        if data is None or data.ndim == 1:
            output = execute(data, initial)
        else:
            batch = (initial[0] if self.backend == 'fock' else initial[0][0]).shape[0]
            randomness = 'same'
            if batch == 1:
                output = vmap(lambda value: execute(value, initial), randomness=randomness)(data)
            else:
                if data.shape[0] == 1:
                    data = data.expand(batch, -1)

                def one(value, inputs):
                    inputs = (
                        [x.unsqueeze(0) for x in inputs]
                        if self.backend == 'fock'
                        else [[x.unsqueeze(0) for x in block] for block in inputs]
                    )
                    return execute(value, inputs)

                output = vmap(one, randomness=randomness)(data, initial)
            output = (
                [x.squeeze(1) for x in output]
                if self.backend == 'fock'
                else [[x.squeeze(1) for x in block] for block in output]
            )
            self.parent.encode(data[-1])
        for cir, value in zip(self.circuits, output, strict=True):
            cir.state = value
        self.state = tuple(output)
        return self.state

    __call__ = forward

    def result(self):
        return SubmodeResult(
            self.state, self.parent.nmode, self.nsubmode, self.backend, self.den_mat, True, self.blocks
        )

    def _probability(self, wires, limits=None):
        accumulated = None
        for cir in self.circuits:
            selected = [k for w in wires for k in cir.groups[w]]
            local_limits = None if limits is None else [n for n in limits for _ in range(cir.nsubmode)]
            values = cir._probabilities(selected, local_limits)
            groups = [list(range(j * cir.nsubmode, (j + 1) * cir.nsubmode)) for j in range(len(wires))]
            values = _group_count_axes(values, groups, limits)
            accumulated = values if accumulated is None else _convolve_counts(accumulated, values, limits)
        return accumulated

    def spatial_probabilities(self, detector=None):
        if (self.parent.detector if detector is None else detector) != 'pnrd':
            raise NotImplementedError('Submode readout currently supports PNR detection.')
        return group_probabilities(self._probability(self.parent.wires), [[w] for w in self.parent.wires])

    def get_spatial_prob(self, final_state):
        counts = final_state.state if isinstance(final_state, FockState) else torch.as_tensor(final_state)
        if (
            counts.ndim != 1
            or len(counts) != self.parent.nmode
            or (counts < 0).any()
            or (counts != counts.long()).any()
        ):
            raise ValueError('Specify one nonnegative integer count per spatial output.')
        counts = counts.tolist()
        result = self._probability(self.parent.wires, counts)
        if any(n >= size for n, size in zip(counts, result.shape[1:], strict=True)):
            return result.new_zeros(result.shape[0])
        return result[(slice(None),) + tuple(counts)]

    def measure_spatial(self, shots, with_prob, wires, detector, mcmc):
        if (self.parent.detector if detector is None else detector) != 'pnrd':
            raise NotImplementedError('Submode readout currently supports PNR detection.')
        selected = self.parent.wires if wires is None else self.parent._convert_indices(wires)
        samples = []
        for cir in self.circuits:
            measured = cir.measure_spatial(shots, False, selected, 'pnrd', mcmc)
            batches = [measured] if isinstance(measured, dict) else measured
            samples.append(
                torch.stack(
                    [
                        torch.stack([key.state for key, count in batch.items() for _ in range(count)])[
                            torch.randperm(shots)
                        ]
                        for batch in batches
                    ]
                )
            )
        output = []
        probabilities = self._probability(selected) if with_prob else None
        for i, events in enumerate(torch.stack(samples).sum(0)):
            counts = Counter(map(tuple, events.tolist()))
            output.append(
                {
                    FockState(pattern): (count, probabilities[(i,) + pattern]) if with_prob else count
                    for pattern, count in counts.items()
                }
            )
        return output[0] if len(output) == 1 else output

    def spatial_photon_number_mean_var(self, wires):
        selected = self.parent.wires if wires is None else self.parent._convert_indices(wires)
        if self.backend == 'gaussian':
            moments = [cir.spatial_photon_number_mean_var(selected) for cir in self.circuits]
            return tuple(sum(value[j] for value in moments) for j in range(2))
        total_mass, mean, variance = 1, 0, 0
        for cir in self.circuits:
            batch = cir.state.shape[0]
            if cir.den_mat:
                mass = cir.state.reshape(batch, cir.cutoff**cir.nmode, -1).diagonal(dim1=-2, dim2=-1).real.sum(-1)
            else:
                mass = cir.state.abs().square().flatten(1).sum(-1)
            mass = mass[:, None]
            safe = torch.where(mass > 0, mass, 1)
            first, var = cir.spatial_photon_number_mean_var(selected)
            first, var = first.reshape(batch, -1), var.reshape(batch, -1)
            mean = mean + first / safe
            variance = variance + (var + first.square()) / safe - (first / safe).square()
            total_mass = total_mass * mass
        return (total_mass * mean).squeeze(), (
            total_mass * (variance + mean.square()) - (total_mass * mean).square()
        ).squeeze()
