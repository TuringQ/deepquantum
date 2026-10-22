"""Hardware plugins for photonic circuits."""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import nn
from torch.func import functional_call

import deepquantum.photonic as dqp

from .gate import BeamSplitter, Displacement, MZI, PhaseShift, Squeezing, Squeezing2
from .operation import Gate, Operation


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
        op_type: type[Gate] | None,
        wires: int | list[int] | None,
        where: str | None = None,
        parameter: str | None = None,
        op_indices: frozenset[int] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(plugin, nn.Module):
            raise TypeError('A hardware plugin must be an nn.Module.')
        if isinstance(wires, int):
            wires = [wires]
        if wires is not None:
            if not wires or any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
                raise ValueError('wires must contain distinct nonnegative integers.')
            wires = tuple(wires)
        self.plugin = plugin
        self.op_type, self.wires = op_type, wires
        self.where, self.parameter = where, parameter
        self.op_indices = op_indices

    def matches(self, op: Operation, index: int) -> bool:
        """Match the exact gate type, unordered wires, and optional unrolled operator indices."""
        return (
            (self.op_indices is None or index in self.op_indices)
            and type(op) is self.op_type
            and (self.wires is None or set(self.wires) == set(op.wires))
        )

    def forward(self, value: Any, run: _HardwareRun, op: Operation | None = None) -> Any:
        wires = (self.wires or run.wires) if op is None else tuple(op.wires)
        return self.plugin(value, replace(run, wires=wires, site=(self, op)))


@dataclass
class _ParameterConfig:
    model: 'HardwareModel'
    op_type: type[Gate]
    wires: int | list[int] | None
    parameter: str

    def use(self, plugin: nn.Module, *, name: str | None = None) -> 'HardwareModel':
        """Bind a plugin implementing forward(parameter, run)."""
        return self.model._register(_Binding(plugin, self.op_type, self.wires, parameter=self.parameter), name)

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

    def use(self, plugin: nn.Module, *, name: str | None = None) -> 'HardwareModel':
        """Bind a plugin implementing forward(state, run)."""
        return self.model._register(_Binding(plugin, self.op_type, self.wires, where=self.where), name)

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

    @property
    def before(self) -> _StateConfig:
        """Before each selected gate."""
        return _StateConfig(self.model, 'before', self.op_type, self.wires)

    @property
    def after(self) -> _StateConfig:
        """After each selected gate."""
        return _StateConfig(self.model, 'after', self.op_type, self.wires)

    @property
    def theta(self) -> _ParameterConfig:
        """Select theta."""
        return _ParameterConfig(self.model, self.op_type, self.wires, 'theta')


class _BeamSplitterConfig(_GateConfig):
    @property
    def phi(self) -> _ParameterConfig:
        """Select phi."""
        return _ParameterConfig(self.model, self.op_type, self.wires, 'phi')


class _AmplitudePhaseConfig(_GateConfig):
    @property
    def r(self) -> _ParameterConfig:
        """Select squeezing or displacement magnitude."""
        return _ParameterConfig(self.model, self.op_type, self.wires, 'r')


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
        for binding in self.bindings.values():
            if binding.wires is not None and max(binding.wires) >= nmode:
                raise ValueError('Hardware binding wire index is outside the circuit.')
            if binding.op_type is not None and not any(binding.matches(op, i) for i, op in enumerate(operators)):
                raise ValueError(f'{binding.op_type.__name__}: binding matches no operations in this circuit.')

    def _unroll(
        self, operators: nn.Sequential, gate_indices: list[list[int]], wire_map: list[list[int]]
    ) -> 'HardwareModel':
        """Map bindings to an unrolled circuit while sharing plugin parameters."""
        model = HardwareModel()
        if self._generator is not None:
            model._generator = torch.Generator().set_state(self._generator.get_state())
        for name, binding in self.bindings.items():
            if binding.op_type is None:
                wires = range(len(wire_map[0])) if binding.wires is None else binding.wires
                wires = [step[wire] for step in wire_map for wire in wires]
                mapped = _Binding(binding.plugin, None, wires, where=binding.where)
            else:
                indices = frozenset(
                    target for i, op in enumerate(operators) if binding.matches(op, i) for target in gate_indices[i]
                )
                mapped = _Binding(
                    binding.plugin, binding.op_type, None, binding.where, binding.parameter, op_indices=indices
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
                if binding.parameter is not None:
                    value = parameters.get(binding.parameter, getattr(op, binding.parameter))
                    parameters[binding.parameter] = binding(value, run, op)
            state = functional_call(op, parameters, (state,)) if parameters else op(state)
            for binding in matches:
                if binding.where == 'after':
                    state = binding(state, run, op)
        for binding in bindings:
            if binding.where == 'output':
                state = binding(state, run)
        return state
