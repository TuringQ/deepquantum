"""Hardware plugins for photonic circuits."""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import nn
from torch.func import functional_call

import deepquantum.photonic as dqp

from .gate import BeamSplitter, BeamSplitterTheta, Displacement, MZI, PhaseShift, Squeezing, Squeezing2
from .operation import Delay, Gate, Operation


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
    """Per-forward state shared by hardware plugins, never stored on the model."""

    backend: str
    generator: torch.Generator | None
    wires: tuple[int, ...]
    cache: dict = field(default_factory=dict)
    site: tuple | None = None


class Loss(nn.Module):
    """Vacuum loss specified by power transmittance or loss in dB."""

    def __init__(self, t: float | torch.Tensor | None = None, *, db: float | torch.Tensor | None = None) -> None:
        super().__init__()
        if (t is None) == (db is None):
            raise ValueError('Specify exactly one of t and db.')
        _register_scalar(self, 't' if db is None else 'db', t if db is None else db)
        self._validate()

    def _validate(self) -> None:
        if hasattr(self, 't'):
            _check_scalar(self.t, 't', 0, 1)
        else:
            _check_scalar(self.db, 'db', 0)

    def forward(self, state: list[torch.Tensor], run: _HardwareRun) -> list[torch.Tensor]:
        """Apply loss to the bound modes."""
        self._validate()
        if run.backend != 'gaussian':
            raise NotImplementedError('Loss currently supports the Gaussian backend.')
        cov, mean = state
        t = self.t if hasattr(self, 't') else 10 ** (-self.db / 10)
        nmode = cov.shape[-1] // 2
        mask = cov.new_zeros(2 * nmode)
        mask[list(run.wires) + [w + nmode for w in run.wires]] = 1
        scale = 1 + mask * (t.to(cov).reshape(()).sqrt() - 1)
        vacuum = dqp.hbar / (4 * dqp.kappa**2)
        cov = (scale[:, None] * scale) * cov + torch.diag((1 - scale.square()) * vacuum)
        return [cov, scale[:, None] * mean]


class GaussianNoise(nn.Module):
    """Additive parameter noise, sampled once per binding and gate per forward."""

    def __init__(self, mu: float | torch.Tensor = 0, sigma: float | torch.Tensor = 0.1) -> None:
        super().__init__()
        _register_scalar(self, 'mu', mu)
        _register_scalar(self, 'sigma', sigma)
        self._validate()

    def _validate(self) -> None:
        _check_scalar(self.mu, 'mu')
        _check_scalar(self.sigma, 'sigma', 0)

    def forward(self, value: torch.Tensor, run: _HardwareRun) -> torch.Tensor:
        """Perturb a parameter without modifying its nominal value."""
        self._validate()
        key = (run.site, self)
        if key not in run.cache:
            sample = value.new_zeros(())
            if self.sigma.requires_grad or self.sigma.detach().ne(0).any():
                device = value.device if run.generator is None else run.generator.device
                sample = torch.randn((), dtype=value.dtype, device=device, generator=run.generator).to(value.device)
            run.cache[key] = sample
        return value + self.mu.to(value).reshape(()) + self.sigma.to(value).reshape(()) * run.cache[key]


class _Binding(nn.Module):
    """A registered plugin and the circuit location it acts on."""

    def __init__(
        self,
        plugin: nn.Module,
        op_type: type[Gate] | tuple[type[Gate], ...] | None,
        wires: int | list[int] | None,
        where: str | None = None,
        parameter: str | dict[type[Gate], str] | None = None,
        op_indices: frozenset[int] | None = None,
        delay: tuple[int | None, int | None] | None = None,
        index: int | None = None,
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
            wires = tuple(wires)
        self.plugin = plugin
        self.op_types = () if op_type is None else op_type if isinstance(op_type, tuple) else (op_type,)
        self.wires = wires
        self.where, self.parameter = where, parameter
        self.op_indices = op_indices
        self.delay = delay  # (spatial wire, delay occurrence)
        self.index = index  # gate occurrence within each selected delay

    def matches(self, op: Operation, index: int) -> bool:
        """Match exact gate types, unordered wires, and optional unrolled operator indices."""
        return (
            (self.op_indices is None or index in self.op_indices)
            and type(op) in self.op_types
            and (self.wires is None or set(self.wires) == set(op.wires))
        )

    def resolve(self, operators: nn.Sequential) -> list[tuple[int, ...]]:
        """Find template paths (operator,) or (delay, inner gate)."""
        if self.delay is None:
            return [(i,) for i, op in enumerate(operators) if self.matches(op, i)]
        wire, index = self.delay
        delays = [
            (i, op) for i, op in enumerate(operators) if isinstance(op, Delay) and (wire is None or wire in op.wires)
        ]
        if index is not None:
            delays = delays[index : index + 1]
        paths = []
        for i, op in delays:
            gates = [j for j, gate in enumerate(op.gates) if self.matches(gate, j)]
            if self.index is not None:
                gates = gates[self.index : self.index + 1]
            paths.extend((i, j) for j in gates)
        return paths

    def forward(self, value: Any, run: _HardwareRun, op: Operation | None = None) -> Any:
        wires = (self.wires or run.wires) if op is None else tuple(op.wires)
        return self.plugin(value, replace(run, wires=wires, site=(self, op)))


@dataclass
class _ParameterConfig:
    model: 'HardwareModel'
    op_type: type[Gate] | tuple[type[Gate], ...]
    wires: int | list[int] | None
    parameter: str | dict[type[Gate], str]
    delay: tuple[int | None, int | None] | None = None
    index: int | None = None

    def use(self, plugin: nn.Module, *, name: str | None = None) -> 'HardwareModel':
        """Bind a plugin implementing forward(parameter, run)."""
        return self.model._register(
            _Binding(plugin, self.op_type, self.wires, parameter=self.parameter, delay=self.delay, index=self.index),
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
    op_type: type[Gate] | None
    wires: int | list[int] | None
    delay: tuple[int | None, int | None] | None = None
    index: int | None = None

    def use(self, plugin: nn.Module, *, name: str | None = None) -> 'HardwareModel':
        """Bind a plugin implementing forward(state, run)."""
        return self.model._register(
            _Binding(plugin, self.op_type, self.wires, where=self.where, delay=self.delay, index=self.index), name
        )

    def loss(
        self, t: float | torch.Tensor | None = None, *, db: float | torch.Tensor | None = None, name: str | None = None
    ) -> 'HardwareModel':
        """Set power transmittance t or loss in dB."""
        return self.use(Loss(t, db=db), name=name)


@dataclass
class _GateConfig:
    model: 'HardwareModel'
    op_type: type[Gate]
    wires: int | list[int] | None
    delay: tuple[int | None, int | None] | None = None
    index: int | None = None

    @property
    def before(self) -> _StateConfig:
        """Before each selected gate."""
        return _StateConfig(self.model, 'before', self.op_type, self.wires, self.delay, self.index)

    @property
    def after(self) -> _StateConfig:
        """After each selected gate."""
        return _StateConfig(self.model, 'after', self.op_type, self.wires, self.delay, self.index)

    @property
    def theta(self) -> _ParameterConfig:
        """Select theta."""
        return _ParameterConfig(self.model, self.op_type, self.wires, 'theta', self.delay, self.index)


class _BeamSplitterConfig(_GateConfig):
    @property
    def phi(self) -> _ParameterConfig:
        """Select phi."""
        return _ParameterConfig(self.model, self.op_type, self.wires, 'phi', self.delay, self.index)


class _AmplitudePhaseConfig(_GateConfig):
    @property
    def r(self) -> _ParameterConfig:
        """Select squeezing or displacement magnitude."""
        return _ParameterConfig(self.model, self.op_type, self.wires, 'r', self.delay, self.index)


@dataclass
class _DelayConfig:
    """Configure internal gates; gate indices count occurrences of the selected type."""

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

    def ps(self, index: int | None = None) -> _GateConfig:
        """Select internal phase shifters, including the built-in one."""
        return _GateConfig(self.model, PhaseShift, None, (self.wires, self.index), index)

    def bs_theta(self, index: int | None = None) -> _GateConfig:
        """Select internal theta-only beam splitters."""
        return _GateConfig(self.model, BeamSplitterTheta, None, (self.wires, self.index), index)

    def mzi(self, index: int | None = None) -> _BeamSplitterConfig:
        """Select internal MZIs."""
        return _BeamSplitterConfig(self.model, MZI, None, (self.wires, self.index), index)

    def s(self, index: int | None = None) -> _AmplitudePhaseConfig:
        """Select internal single-mode squeezers."""
        return _AmplitudePhaseConfig(self.model, Squeezing, None, (self.wires, self.index), index)

    def d(self, index: int | None = None) -> _AmplitudePhaseConfig:
        """Select internal displacements."""
        return _AmplitudePhaseConfig(self.model, Displacement, None, (self.wires, self.index), index)


class HardwareModel(nn.Module):
    """Bind hardware plugins to circuit operations and boundaries."""

    def __init__(self, *, seed: int | None = None) -> None:
        super().__init__()
        self.bindings = nn.ModuleDict()
        self.manual_seed(seed)

    def manual_seed(self, seed: int | None) -> 'HardwareModel':
        """Reset the hardware RNG; None uses the global PyTorch RNG."""
        self._generator = None if seed is None else torch.Generator().manual_seed(seed)
        return self

    def ps(self, wires: int | None = None) -> _GateConfig:
        """Select phase shifters."""
        return _GateConfig(self, PhaseShift, wires)

    def bs(self, wires: list[int] | None = None) -> _BeamSplitterConfig:
        """Select beam splitters."""
        return _BeamSplitterConfig(self, BeamSplitter, wires)

    def mzi(self, wires: list[int] | None = None) -> _BeamSplitterConfig:
        """Select MZIs."""
        return _BeamSplitterConfig(self, MZI, wires)

    def s(self, wires: int | None = None) -> _AmplitudePhaseConfig:
        """Select single-mode squeezers."""
        return _AmplitudePhaseConfig(self, Squeezing, wires)

    def s2(self, wires: list[int] | None = None) -> _AmplitudePhaseConfig:
        """Select two-mode squeezers."""
        return _AmplitudePhaseConfig(self, Squeezing2, wires)

    def d(self, wires: int | None = None) -> _AmplitudePhaseConfig:
        """Select displacements."""
        return _AmplitudePhaseConfig(self, Displacement, wires)

    def delay(self, wires: int | None = None, index: int | None = None) -> _DelayConfig:
        """Select delay loops; index is zero-based among delays on the selected wire."""
        return _DelayConfig(self, wires, index)

    def input(self, wires: int | list[int] | None = None) -> _StateConfig:
        """Select input modes; None selects all."""
        return _StateConfig(self, 'input', None, wires)

    def output(self, wires: int | list[int] | None = None) -> _StateConfig:
        """Select output modes; None selects all."""
        return _StateConfig(self, 'output', None, wires)

    def _register(self, binding: _Binding, name: str | None) -> 'HardwareModel':
        name = f'binding_{len(self.bindings)}' if name is None else name
        if name in self.bindings:
            raise ValueError(f'Duplicate binding name: {name}.')
        self.bindings[name] = binding
        return self

    def _validate(self, operators: nn.Sequential, nmode: int) -> None:
        for name, binding in self.bindings.items():
            if binding.wires is not None and max(binding.wires) >= nmode:
                raise ValueError('Hardware binding wire index is outside the circuit.')
            if binding.delay is not None and binding.delay[0] is not None and binding.delay[0] >= nmode:
                raise ValueError('Hardware binding wire index is outside the circuit.')
            if binding.op_types and not binding.resolve(operators):
                raise ValueError(f'{name}: binding matches no operations in this circuit.')

    def _unroll(
        self,
        operators: nn.Sequential,
        gate_indices: dict[tuple[int, ...], list[int]],
        wire_map: list[list[int]],
        use_deepcopy: bool = False,
    ) -> 'HardwareModel':
        """Remap bindings, optionally copying plugins once while sharing them across time steps."""
        model = HardwareModel()
        if self._generator is not None:
            model._generator = torch.Generator().set_state(self._generator.get_state())
        bindings = deepcopy(self.bindings) if use_deepcopy else self.bindings
        for name, binding in bindings.items():
            if not binding.op_types:
                wires = range(len(wire_map[0])) if binding.wires is None else binding.wires
                wires = [step[wire] for step in wire_map for wire in wires]
                mapped = _Binding(binding.plugin, None, wires, where=binding.where)
            else:
                indices = frozenset(target for path in binding.resolve(operators) for target in gate_indices[path])
                mapped = _Binding(
                    binding.plugin, binding.op_types, None, binding.where, binding.parameter, op_indices=indices
                )
            model._register(mapped, name)
        return model

    def forward(self, state: Any, operators: nn.Sequential, backend: str, wires: Sequence[int]) -> Any:
        """Apply plugins in physical order without overwriting ideal parameters."""
        run = _HardwareRun(backend, self._generator, tuple(wires))
        bindings = self.bindings.values()
        for binding in bindings:
            if binding.where == 'input':
                state = binding(state, run)
        for i, op in enumerate(operators):
            matches = [binding for binding in bindings if binding.matches(op, i)]
            for binding in matches:
                if binding.where == 'before':
                    state = binding(state, run, op)
            parameters = {}
            for binding in matches:
                parameter = binding.parameter
                if isinstance(parameter, dict):
                    parameter = parameter[type(op)]
                if parameter is not None:
                    value = parameters.get(parameter, getattr(op, parameter))
                    parameters[parameter] = binding(value, run, op)
            state = functional_call(op, parameters, (state,)) if parameters else op(state)
            for binding in matches:
                if binding.where == 'after':
                    state = binding(state, run, op)
        for binding in bindings:
            if binding.where == 'output':
                state = binding(state, run)
        return state
