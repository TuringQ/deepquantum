"""Hardware plugins for photonic circuits."""

from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import nn
from torch.func import functional_call

from .channel import OPA, PhotonLoss
from .gate import (
    BeamSplitter,
    BeamSplitterPhi,
    BeamSplitterSingle,
    BeamSplitterTheta,
    ControlledX,
    ControlledZ,
    CrossKerr,
    CubicPhase,
    Displacement,
    DisplacementMomentum,
    DisplacementPosition,
    Kerr,
    MZI,
    PhaseShift,
    QuadraticPhase,
    Squeezing,
    Squeezing2,
)
from .measurement import Homodyne
from .operation import Delay, Operation

_PARAMETERS = {
    PhaseShift: ('theta',),
    BeamSplitter: ('theta', 'phi'),
    MZI: ('theta', 'phi'),
    BeamSplitterTheta: ('theta',),
    BeamSplitterPhi: ('phi',),
    BeamSplitterSingle: ('theta',),
    Squeezing: ('r', 'theta'),
    OPA: ('r', 'theta', 'loss_theta'),
    Squeezing2: ('r', 'theta'),
    Displacement: ('r', 'theta'),
    DisplacementPosition: ('r',),
    DisplacementMomentum: ('r',),
    QuadraticPhase: ('s',),
    ControlledX: ('s',),
    ControlledZ: ('s',),
    CubicPhase: ('gamma',),
    Kerr: ('kappa',),
    CrossKerr: ('kappa',),
    Homodyne: ('phi',),
}


def _register_scalar(module: nn.Module, name: str, value: float | torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=torch.get_default_dtype())
    if isinstance(value, nn.Parameter):
        module.register_parameter(name, value)
    else:
        module.register_buffer(name, value)


def _check_scalar(value: torch.Tensor, name: str, low: float | None = None, high: float | None = None) -> None:
    if value.numel() != 1 or not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f'{name} must be a finite real scalar.')
    if low is not None and (value < low).any():
        raise ValueError(f'{name} must be >= {low}.')
    if high is not None and (value > high).any():
        raise ValueError(f'{name} must be <= {high}.')


@dataclass(frozen=True)
class _HardwareRun:
    """Execution context with a cache of random samples."""

    backend: str
    generator: torch.Generator | None
    wires: tuple[int, ...]
    nmode: int
    basis: bool = False
    cache: dict = field(default_factory=dict)
    cache_key: tuple | None = None


class Loss(nn.Module):
    """Photon loss initialized by transmittance t or loss in dB.

    Set requires_grad=True to train the internal angle theta, with t = cos(theta / 2)**2.
    """

    def __init__(
        self,
        t: float | torch.Tensor | None = None,
        *,
        db: float | torch.Tensor | None = None,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()
        if (t is None) == (db is None):
            raise ValueError('Specify exactly one of t and db.')
        value = t if db is None else db
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.get_default_dtype())
        convention = 't' if db is None else 'db'
        _check_scalar(value, convention, 0, 1 if db is None else None)
        channel = PhotonLoss(value, cutoff=2, convention=convention, requires_grad=requires_grad)
        _register_scalar(self, 'theta', channel.theta)

    @property
    def t(self) -> torch.Tensor:
        """Transmittance."""
        return torch.cos(self.theta / 2) ** 2

    @property
    def db(self) -> torch.Tensor:
        """Photon loss in dB."""
        return -10 * torch.log10(self.t)

    def forward(self, state: Any, run: _HardwareRun) -> Any:
        """Apply loss to the bound modes."""
        theta = self.theta
        if run.backend in ('gaussian', 'bosonic'):
            nmode, cutoff = state[0].shape[-1] // 2, 2
            theta = theta.to(state[0])
        elif run.backend == 'fock':
            nmode, cutoff = run.nmode, 2
            if not run.basis:
                if not isinstance(state, torch.Tensor) or state.ndim != 2 * nmode + 1:
                    raise NotImplementedError('Fock tensor loss requires density matrix representation (den_mat=True).')
                cutoff = state.shape[-1]
            if state is not None:
                theta = theta.to(state.real)
        else:
            raise NotImplementedError(f'Loss does not support backend {run.backend}.')
        channel = PhotonLoss(theta, nmode=nmode, cutoff=cutoff)
        if run.basis:
            return PhotonLoss._apply_unitary(state, channel.gate.update_matrix(), run.wires, nmode)
        for wire in run.wires:
            channel.wires = [wire]
            state = channel(state)
        return state


class GaussianNoise(nn.Module):
    """Add mu + sigma * epsilon with one sample per binding, operation and parameter per execution.

    Parameters are validated at construction. A numeric sigma=0 disables sampling.
    Tensor sigma remains dynamic and is sampled even at zero.
    """

    def __init__(self, mu: float | torch.Tensor = 0, sigma: float | torch.Tensor = 0.1) -> None:
        super().__init__()
        _register_scalar(self, 'mu', mu)
        _check_scalar(self.mu, 'mu')
        if not isinstance(sigma, torch.Tensor) and sigma == 0:
            self.register_buffer('sigma', None)
        else:
            _register_scalar(self, 'sigma', sigma)
            _check_scalar(self.sigma, 'sigma', 0)

    def forward(self, value: torch.Tensor, run: _HardwareRun) -> torch.Tensor:
        """Perturb a parameter without modifying its nominal value."""
        if value.is_complex():
            raise TypeError('Gaussian parameter noise requires a real control.')
        if not value.is_floating_point():
            value = value.to(torch.get_default_dtype())
        value = value + self.mu.to(value).reshape(())
        if self.sigma is None:
            return value
        key = (run.cache_key, self)
        if key not in run.cache:
            device = value.device if run.generator is None else run.generator.device
            run.cache[key] = torch.randn((), dtype=value.dtype, device=device, generator=run.generator).to(value.device)
        return value + self.sigma.to(value).reshape(()) * run.cache[key]


class _PluginBinding(nn.Module):
    """Associate a plugin with a selection rule or specific operation objects."""

    def __init__(
        self,
        plugin: nn.Module,
        op_type: type[Operation] | tuple[type[Operation], ...] | None,
        wires: int | list[int] | None,
        where: str | None = None,
        parameter: str | dict[type[Operation], str | tuple[str, ...]] | None = None,
        bound_ops: frozenset[Operation] | None = None,
        delay: tuple[int | None, int | None] | None = None,
        index: int | None = None,
        required: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(plugin, nn.Module):
            raise TypeError('A hardware plugin must be an nn.Module.')
        if index is not None and (type(index) is not int or index < 0):
            raise ValueError('index must be a nonnegative integer or None.')
        if isinstance(wires, int):
            wires = [wires]
        if wires is not None:
            if not wires or any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
                raise ValueError('wires must contain distinct nonnegative integers.')
            wires = tuple(wires) if op_type is None else tuple(sorted(wires))
        self.plugin = plugin
        self.op_types = () if op_type is None else op_type if isinstance(op_type, tuple) else (op_type,)
        self.wires = wires
        self.where, self.parameter = where, parameter
        self.bound_ops = bound_ops  # Identity references; do not register the circuit's operations again.
        self.delay = delay  # (spatial wire, zero-based delay occurrence)
        self.index = index  # Zero-based occurrence of the selected gate type within each delay.
        self.required = required

    def matches(self, op: Operation) -> bool:
        """Match operation identity when bound, otherwise exact type and unordered wires."""
        return (
            (self.bound_ops is None or op in self.bound_ops)
            and type(op) in self.op_types
            and (self.wires is None or set(self.wires) == set(op.wires))
        )

    def same_rule(self, other: '_PluginBinding') -> bool:
        """Compare selection, application point and plugin type."""
        return (
            type(self.plugin) is type(other.plugin)
            and set(self.op_types) == set(other.op_types)
            and self.wires == other.wires
            and self.where == other.where
            and self.parameter == other.parameter
            and self.bound_ops == other.bound_ops
            and self.delay == other.delay
            and self.index == other.index
            and self.required == other.required
        )

    def resolve(self, operators: Sequence[Operation]) -> list[tuple[int, ...]]:
        """Find operation locations (operator,) or (delay, inner gate) in the template."""
        if self.bound_ops is not None:
            locations = []
            for i, op in enumerate(operators):
                if isinstance(op, Delay):
                    locations.extend((i, j) for j, gate in enumerate(op.gates) if self.matches(gate))
                elif self.matches(op):
                    locations.append((i,))
            return locations
        if self.delay is None:
            return [(i,) for i, op in enumerate(operators) if self.matches(op)]
        wire, index = self.delay
        delays = [
            (i, op) for i, op in enumerate(operators) if isinstance(op, Delay) and (wire is None or wire in op.wires)
        ]
        if index is not None:
            delays = delays[index : index + 1]
        locations = []
        for i, op in delays:
            gates = [j for j, gate in enumerate(op.gates) if self.matches(gate)]
            if self.index is not None:
                gates = gates[self.index : self.index + 1]
            locations.extend((i, j) for j in gates)
        return locations

    def forward(
        self,
        value: Any,
        run: _HardwareRun,
        op: Operation,
        parameter: str | None = None,
        source: Operation | None = None,
    ) -> Any:
        cache_key = (self, op if source is None else source, parameter)
        return self.plugin(value, replace(run, wires=tuple(op.wires), cache_key=cache_key))


@dataclass
class _ParameterConfig:
    model: 'HardwareModel'
    op_type: type[Operation] | tuple[type[Operation], ...]
    wires: int | list[int] | None
    parameter: str | dict[type[Operation], str]
    delay: tuple[int | None, int | None] | None = None
    index: int | None = None
    bound_ops: frozenset[Operation] | None = None

    def use(self, plugin: nn.Module, *, name: str | None = None) -> 'HardwareModel':
        """Bind a plugin implementing forward(parameter, run)."""
        return self.model._register(
            _PluginBinding(
                plugin,
                self.op_type,
                self.wires,
                parameter=self.parameter,
                delay=self.delay,
                index=self.index,
                bound_ops=self.bound_ops,
            ),
            name,
        )

    def gaussian(
        self, mu: float | torch.Tensor = 0, sigma: float | torch.Tensor = 0.1, *, name: str | None = None
    ) -> 'HardwareModel':
        """Add Gaussian parameter noise."""
        return self.use(GaussianNoise(mu, sigma), name=name)


@dataclass
class _StateConfig:
    model: 'HardwareModel'
    where: str
    op_type: type[Operation] | None
    wires: int | list[int] | None
    delay: tuple[int | None, int | None] | None = None
    index: int | None = None
    bound_ops: frozenset[Operation] | None = None

    def use(self, plugin: nn.Module, *, name: str | None = None) -> 'HardwareModel':
        """Bind a plugin implementing forward(state, run)."""
        return self.model._register(
            _PluginBinding(
                plugin,
                self.op_type,
                self.wires,
                where=self.where,
                delay=self.delay,
                index=self.index,
                bound_ops=self.bound_ops,
            ),
            name,
        )

    def loss(
        self,
        t: float | torch.Tensor | None = None,
        *,
        db: float | torch.Tensor | None = None,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> 'HardwareModel':
        """Add photon loss initialized by t or db, optionally training its internal theta."""
        return self.use(Loss(t, db=db, requires_grad=requires_grad), name=name)


@dataclass
class _OperationConfig:
    model: 'HardwareModel'
    op_type: type[Operation]
    wires: int | list[int] | None
    delay: tuple[int | None, int | None] | None = None
    index: int | None = None
    bound_ops: frozenset[Operation] | None = None

    def gaussian(self, mu: float | torch.Tensor = 0, sigma: float | torch.Tensor = 0.1) -> 'HardwareModel':
        """Perturb all scalar controls of the selected gate type independently."""
        for parameter in _PARAMETERS[self.op_type]:
            _ParameterConfig(
                self.model, self.op_type, self.wires, parameter, self.delay, self.index, self.bound_ops
            ).gaussian(mu, sigma)
        return self.model

    @property
    def before(self) -> _StateConfig:
        """Before each selected gate."""
        return _StateConfig(self.model, 'before', self.op_type, self.wires, self.delay, self.index, self.bound_ops)

    @property
    def after(self) -> _StateConfig:
        """After each selected gate."""
        return _StateConfig(self.model, 'after', self.op_type, self.wires, self.delay, self.index, self.bound_ops)


class _PhaseConfig(_OperationConfig):
    @property
    def theta(self) -> _ParameterConfig:
        """Select theta."""
        return _ParameterConfig(self.model, self.op_type, self.wires, 'theta', self.delay, self.index, self.bound_ops)


class _BeamSplitterConfig(_PhaseConfig):
    @property
    def phi(self) -> _ParameterConfig:
        """Select phi."""
        return _ParameterConfig(self.model, self.op_type, self.wires, 'phi', self.delay, self.index, self.bound_ops)


class _AmplitudePhaseConfig(_PhaseConfig):
    @property
    def r(self) -> _ParameterConfig:
        """Select squeezing or displacement magnitude."""
        return _ParameterConfig(self.model, self.op_type, self.wires, 'r', self.delay, self.index, self.bound_ops)


class _OPAConfig(_AmplitudePhaseConfig):
    @property
    def loss_theta(self) -> _ParameterConfig:
        """Select the internal loss angle."""
        return _ParameterConfig(
            self.model, self.op_type, self.wires, 'loss_theta', self.delay, self.index, self.bound_ops
        )


class _HomodyneConfig(_OperationConfig):
    @property
    def phi(self) -> _ParameterConfig:
        """Select the local-oscillator phase."""
        return _ParameterConfig(self.model, self.op_type, self.wires, 'phi', bound_ops=self.bound_ops)


@dataclass
class _DelayConfig:
    """Configure internal gates by their zero-based occurrence within the selected gate type."""

    model: 'HardwareModel'
    wires: int | None
    index: int | None

    def __post_init__(self) -> None:
        for name, value in [('wires', self.wires), ('index', self.index)]:
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f'{name} must be a nonnegative integer or None.')

    @property
    def theta(self) -> _ParameterConfig:
        """Select the delay coupler's theta."""
        return _ParameterConfig(self.model, (BeamSplitterTheta, MZI), None, 'theta', (self.wires, self.index), 0)

    @property
    def phi(self) -> _ParameterConfig:
        """Select the internal phase: PS.theta for DelayBS, MZI.phi for DelayMZI."""
        return _ParameterConfig(
            self.model, (PhaseShift, MZI), None, {PhaseShift: 'theta', MZI: 'phi'}, (self.wires, self.index), 0
        )

    def ps(self, index: int | None = None) -> _PhaseConfig:
        """Select internal phase shifters, including the built-in one."""
        return _PhaseConfig(self.model, PhaseShift, None, (self.wires, self.index), index)

    def bs_theta(self, index: int | None = None) -> _PhaseConfig:
        """Select internal theta-only beam splitters."""
        return _PhaseConfig(self.model, BeamSplitterTheta, None, (self.wires, self.index), index)

    def mzi(self, index: int | None = None) -> _BeamSplitterConfig:
        """Select internal MZIs."""
        return _BeamSplitterConfig(self.model, MZI, None, (self.wires, self.index), index)

    def s(self, index: int | None = None) -> _AmplitudePhaseConfig:
        """Select internal single-mode squeezers."""
        return _AmplitudePhaseConfig(self.model, Squeezing, None, (self.wires, self.index), index)

    def opa(self, index: int | None = None) -> _OPAConfig:
        """Select internal OPA channels for control noise or before/after plugins."""
        return _OPAConfig(self.model, OPA, None, (self.wires, self.index), index)

    def d(self, index: int | None = None) -> _AmplitudePhaseConfig:
        """Select internal displacements."""
        return _AmplitudePhaseConfig(self.model, Displacement, None, (self.wires, self.index), index)


class HardwareModel(nn.Module):
    """Apply hardware plugins without changing ideal controls.

    Template selectors are re-evaluated against the current circuit. Unrolled bindings
    follow specific gate objects through insertion/reordering. Removing a bound gate is an error.
    Later plugins of the same type replace earlier ones on the same operation parameter or before/after hook.
    Input/output plugins of the same type override earlier ones per mode.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        super().__init__()
        self.bindings = nn.ModuleDict()
        self.manual_seed(seed)

    def manual_seed(self, seed: int | None) -> 'HardwareModel':
        """Reset the hardware RNG, using the global PyTorch RNG when seed is None."""
        self._generator = None if seed is None else torch.Generator().manual_seed(seed)
        return self

    def gaussian(self, mu: float | torch.Tensor = 0, sigma: float | torch.Tensor = 0.1) -> 'HardwareModel':
        """Perturb all supported controls, including delay gates and homodyne phases."""
        plugin = GaussianNoise(mu, sigma)
        for delay in (None, (None, None)):
            self._register(
                _PluginBinding(
                    plugin,
                    tuple(_PARAMETERS),
                    None,
                    parameter=_PARAMETERS,
                    delay=delay,
                    required=False,
                ),
                None,
            )
        return self

    def on(self, op: Operation) -> _OperationConfig:
        """Select one operation by identity, independent of its position in the circuit."""
        if isinstance(op, Delay):
            raise TypeError('Select an inner gate with on(), or select the delay with delay().')
        return _OperationConfig(self, type(op), None, bound_ops=frozenset([op]))

    def ps(self, wires: int | None = None) -> _PhaseConfig:
        """Select phase shifters."""
        return _PhaseConfig(self, PhaseShift, wires)

    def homodyne(self, wires: int | None = None) -> _HomodyneConfig:
        """Select explicit homodyne measurements with noise sampled per measurement call."""
        return _HomodyneConfig(self, Homodyne, wires)

    def bs(self, wires: list[int] | None = None) -> _BeamSplitterConfig:
        """Select beam splitters."""
        return _BeamSplitterConfig(self, BeamSplitter, wires)

    def mzi(self, wires: list[int] | None = None) -> _BeamSplitterConfig:
        """Select MZIs."""
        return _BeamSplitterConfig(self, MZI, wires)

    def s(self, wires: int | None = None) -> _AmplitudePhaseConfig:
        """Select single-mode squeezers."""
        return _AmplitudePhaseConfig(self, Squeezing, wires)

    def opa(self, wires: int | None = None) -> _OPAConfig:
        """Select explicit OPA channels for control noise or before/after plugins."""
        return _OPAConfig(self, OPA, wires)

    def s2(self, wires: list[int] | None = None) -> _AmplitudePhaseConfig:
        """Select two-mode squeezers."""
        return _AmplitudePhaseConfig(self, Squeezing2, wires)

    def d(self, wires: int | None = None) -> _AmplitudePhaseConfig:
        """Select displacements."""
        return _AmplitudePhaseConfig(self, Displacement, wires)

    def delay(self, wires: int | None = None, index: int | None = None) -> _DelayConfig:
        """Select delay loops by their zero-based occurrence on the selected wire."""
        return _DelayConfig(self, wires, index)

    def input(self, wires: int | list[int] | None = None) -> _StateConfig:
        """Select input modes, with None selecting all modes."""
        return _StateConfig(self, 'input', None, wires)

    def output(self, wires: int | list[int] | None = None) -> _StateConfig:
        """Select output modes, with None selecting all modes."""
        return _StateConfig(self, 'output', None, wires)

    def _register(self, binding: _PluginBinding, name: str | None) -> 'HardwareModel':
        previous_name = next((key for key, current in self.bindings.items() if current.same_rule(binding)), None)
        if name is None:
            name = previous_name
            if name is None:
                index = len(self.bindings)
                while f'binding_{index}' in self.bindings:
                    index += 1
                name = f'binding_{index}'
        if name in self.bindings and name != previous_name:
            raise ValueError(f'Duplicate binding name: {name}.')
        if previous_name is not None:
            del self.bindings[previous_name]
        self.bindings[name] = binding
        return self

    def _validate(self, operators: Sequence[Operation], nmode: int) -> None:
        operations = [gate for op in operators for gate in (op.gates if isinstance(op, Delay) else [op])]
        if len(set(operations)) != len(operations):
            raise ValueError('Use distinct operation objects for separate circuit positions; parameters may be shared.')
        operations = set(operations)
        for name, binding in self.bindings.items():
            if binding.bound_ops is not None and not binding.bound_ops <= operations:
                raise ValueError(f'{name}: a bound operation was removed or replaced; rebuild the hardware binding.')
            if binding.wires is not None and max(binding.wires) >= nmode:
                raise ValueError('Hardware binding wire index is outside the circuit.')
            if binding.delay is not None and binding.delay[0] is not None and binding.delay[0] >= nmode:
                raise ValueError('Hardware binding wire index is outside the circuit.')
            if binding.required and binding.op_types and not binding.resolve(operators):
                raise ValueError(f'{name}: binding matches no operations in this circuit.')

    def _unroll(
        self,
        operators: Sequence[Operation],
        operation_map: dict[tuple[int, ...], list[Operation]],
        wire_map: list[list[int]],
        use_deepcopy: bool = False,
    ) -> 'HardwareModel':
        """Remap bindings, optionally copying plugins once while sharing them across time steps."""
        model = HardwareModel()
        if self._generator is not None:
            model._generator = torch.Generator().set_state(self._generator.get_state())
        plugins = [binding.plugin for binding in self.bindings.values()]
        if use_deepcopy:
            plugins = deepcopy(plugins)  # Copy shared plugin parameters once, not the referenced circuit gates.
        for (name, binding), plugin in zip(self.bindings.items(), plugins, strict=True):
            if not binding.op_types:
                wires = range(len(wire_map[0])) if binding.wires is None else binding.wires
                wires = [step[wire] for step in wire_map for wire in wires]
                mapped = _PluginBinding(plugin, None, wires, where=binding.where)
            else:
                gates = frozenset(
                    target for op_location in binding.resolve(operators) for target in operation_map[op_location]
                )
                mapped = _PluginBinding(
                    plugin,
                    binding.op_types,
                    None,
                    binding.where,
                    binding.parameter,
                    bound_ops=gates,
                    required=binding.required,
                )
            model._register(mapped, name)
        return model

    def _bind(self, operators: Sequence[Operation]) -> dict[Operation, list[tuple[_PluginBinding, str | None]]]:
        """Resolve overrides in the registration order of the surviving plugins."""
        matches = defaultdict(dict)
        for binding in self.bindings.values():
            for op_location in binding.resolve(operators):
                gate = operators[op_location[0]]
                if len(op_location) == 2:
                    gate = gate.gates[op_location[1]]
                parameters = binding.parameter
                if isinstance(parameters, dict):
                    parameters = parameters[type(gate)]
                if parameters is None or isinstance(parameters, str):
                    parameters = (parameters,)
                for parameter in parameters:
                    key = (binding.where, parameter, type(binding.plugin))
                    # Keep the replacement at its own position relative to other plugin types.
                    matches[gate].pop(key, None)
                    matches[gate][key] = (binding, parameter)
        return {gate: list(bindings.values()) for gate, bindings in matches.items()}

    def _new_run(self, backend: str, wires: Sequence[int], basis: bool = False) -> _HardwareRun:
        return _HardwareRun(backend, self._generator, tuple(wires), len(wires), basis)

    def _io_bindings(self, wires: Sequence[int], where: str) -> list[tuple[_PluginBinding, tuple[int, ...]]]:
        """Resolve the last plugin of each type on each input or output mode."""
        covered = defaultdict(set)
        selected = []
        for binding in reversed(list(self.bindings.values())):
            if binding.where == where and not binding.op_types:
                selected_wires = tuple(wires) if binding.wires is None else tuple(wires[w] for w in binding.wires)
                covered_wires = covered[type(binding.plugin)]
                active_wires = tuple(wire for wire in selected_wires if wire not in covered_wires)
                covered_wires.update(selected_wires)
                if active_wires:
                    selected.append((binding, active_wires))
        return list(reversed(selected))

    def _loss_count(self, operators: Sequence[Operation], wires: Sequence[int]) -> int:
        """Count vacuum environment modes needed by Fock basis loss."""
        count = sum(
            len(op.wires)
            for op, bindings in self._bind(operators).items()
            for binding, _ in bindings
            if binding.where is not None and isinstance(binding.plugin, Loss)
        )
        return count + sum(
            len(selected)
            for where in ('input', 'output')
            for binding, selected in self._io_bindings(wires, where)
            if isinstance(binding.plugin, Loss)
        )

    def _apply_io_plugins(self, state: Any, run: _HardwareRun, where: str) -> Any:
        for binding, wires in self._io_bindings(run.wires, where):
            state = binding.plugin(state, replace(run, wires=wires, cache_key=(binding, None, None)))
        return state

    def _apply_state_plugins(self, state, op, run, bindings, where, source=None):
        """Apply state plugins at one gate boundary."""
        for binding, _ in bindings:
            if binding.where == where:
                state = binding(state, run, op, source=source)
        return state

    def _apply_operation(
        self,
        state: Any,
        op: Operation,
        run: _HardwareRun,
        bindings: Sequence[tuple[_PluginBinding, str | None]],
        source: Operation | None = None,
    ) -> Any:
        """Execute one operation with one coherent realization of its controls."""
        state = self._apply_state_plugins(state, op, run, bindings, 'before', source)
        overrides = {}
        for binding, parameter in bindings:
            if parameter is not None:
                value = overrides.get(parameter, getattr(op, parameter))
                overrides[parameter] = binding(value, run, op, parameter, source)
        state = functional_call(op, overrides, (state,)) if overrides else op(state)
        return self._apply_state_plugins(state, op, run, bindings, 'after', source)

    def forward(
        self,
        state: Any,
        operators: Operation | Sequence[Operation],
        backend: str,
        wires: Sequence[int] | None = None,
        sources: Sequence[Operation] | None = None,
    ) -> Any:
        """Execute a single operation or a sequence of operations."""
        if isinstance(operators, Operation):
            self._validate([operators], operators.nmode)
            wires = range(operators.nmode) if wires is None else wires
            operators = [operators]
        if wires is None:
            raise ValueError('Specify circuit wires when executing a sequence of operations.')
        sources = operators if sources is None else sources
        bindings = self._bind(sources)
        source_gates = [gate for op in sources for gate in (op.gates if isinstance(op, Delay) else [op])]
        run = self._new_run(backend, wires)
        state = self._apply_io_plugins(state, run, 'input')
        for op, source in zip(operators, source_gates, strict=True):
            state = self._apply_operation(state, op, run, bindings.get(source, ()), source)
        return self._apply_io_plugins(state, run, 'output')
