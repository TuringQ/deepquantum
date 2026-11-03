"""Photonic input preparations and calibrated submode profiles."""

from typing import Any, TYPE_CHECKING

import torch
from torch import nn

import deepquantum.photonic as dqp

from ..qmath import partial_trace

if TYPE_CHECKING:
    from .hardware import _HardwareRun


def _register_real_components(module: nn.Module, value: torch.Tensor, requires_grad: bool) -> None:
    """Store complex amplitudes as real components so .to(dtype) preserves their phase."""
    if not value.is_complex():
        value = torch.complex(value, torch.zeros_like(value))
    components = torch.view_as_real(value.resolve_conj()).clone()
    if requires_grad:
        module.register_parameter('components', nn.Parameter(components))
    else:
        module.register_buffer('components', components)


def _replace_fock_modes(state: torch.Tensor, density: torch.Tensor, wires: tuple[int, ...], nmode: int) -> torch.Tensor:
    """Trace out selected modes and insert an independent density matrix in their place."""
    if not isinstance(state, torch.Tensor) or state.ndim != 2 * nmode + 1:
        raise NotImplementedError('Source preparation requires Fock density matrices (basis=False, den_mat=True).')
    cutoff = state.shape[-1]
    rest = [wire for wire in range(nmode) if wire not in wires]
    reduced = partial_trace(state.reshape(-1, cutoff**nmode, cutoff**nmode), nmode, list(wires), cutoff)
    reduced = reduced.reshape(-1, cutoff ** len(rest), cutoff ** len(rest))
    density = density.to(device=state.device, dtype=state.dtype)
    combined = torch.einsum('bij,kl->bikjl', reduced, density).reshape([-1] + [cutoff] * (2 * nmode))
    order = rest + list(wires)
    row = [order.index(wire) + 1 for wire in range(nmode)]
    return combined.permute([0] + row + [wire + nmode for wire in row])


class Submodes(nn.Module):
    r"""Orthogonal modes within each spatial port, expressed in a common basis.

    Modes may describe temporal/spectral wave packets, polarization, transverse
    profiles, or their combinations. A wavepacket is one mode function, which may
    itself be a superposition of the common orthogonal basis functions.
    Ratios describe fixed couplings under a common pump-amplitude sweep; photon
    fractions generally change with gain. All calibration data are fixed buffers.

    Args:
        coefficients: One wavepacket vector, normalized internally, or a matrix whose rows are orthonormal
            wavepackets in the common basis. Omit to use the common basis functions themselves.
        relative_r: Nonnegative squeezing ratios with a positive first entry. Entries correspond to rows of
            ``coefficients``, or to common-basis modes when coefficients are omitted.
            Ratios are divided by the first entry, so ``[3, 2]`` means ``[1, 2/3]`` and the circuit's ``r``
            remains the squeezing of the first wavepacket. Required for multiple wavepackets.
    """

    def __init__(self, coefficients: Any = None, *, relative_r: Any = None) -> None:
        super().__init__()
        if coefficients is None and relative_r is None:
            raise ValueError('Specify coefficients or relative_r.')
        if relative_r is not None:
            relative_r = torch.as_tensor(relative_r)
            if not relative_r.is_floating_point():
                if relative_r.is_complex():
                    raise ValueError('relative_r must be real.')
                relative_r = relative_r.to(torch.get_default_dtype())
            if (
                relative_r.ndim != 1
                or not len(relative_r)
                or not torch.isfinite(relative_r).all()
                or (relative_r < 0).any()
                or relative_r[0] <= 0
            ):
                raise ValueError('relative_r must be finite, nonnegative and start with a positive reference.')
            relative_r = relative_r / relative_r[0]
            if not torch.isfinite(relative_r).all():
                raise ValueError('Squeezing ratios to the reference mode must be finite.')
        if coefficients is None:
            self.register_buffer('components', None)
        else:
            values = torch.as_tensor(coefficients)
            if not values.is_floating_point() and not values.is_complex():
                values = values.to(torch.get_default_dtype())
            if values.ndim == 1:
                if not values.norm() > 0:
                    raise ValueError('coefficients must be nonzero.')
                values = (values / values.norm()).unsqueeze(0)
            if values.ndim != 2 or min(values.shape) == 0 or not torch.isfinite(values).all():
                raise ValueError('coefficients must be a finite vector or matrix of orthonormal rows.')
            tol = 100 * torch.finfo(values.real.dtype).eps
            identity = torch.eye(len(values), dtype=values.dtype, device=values.device)
            if not torch.allclose(values @ values.mH, identity, atol=tol, rtol=tol):
                raise ValueError('Submode profiles must be orthonormal.')
            if relative_r is None:
                if len(values) != 1:
                    raise ValueError('Specify relative_r for multiple profiles.')
                relative_r = values.real.new_ones(1)
            if len(relative_r) != len(values):
                raise ValueError('relative_r must match the number of profiles.')
            _register_real_components(self, values.detach(), False)
        self.register_buffer('relative_r', relative_r.detach().clone())
        # Fixed calibration topology is Python metadata, also usable inside vmap.
        self.supports = (
            tuple((k,) for k in range(len(relative_r)))
            if coefficients is None
            else tuple(tuple(row.nonzero().flatten().tolist()) for row in values.abs() != 0)
        )

    @classmethod
    def from_nbar(cls, nbar: Any, coefficients: Any = None) -> 'Submodes':
        """Calibrate relative squeezing from per-mode vacuum-seeded mean photon numbers.

        This sets ratios only; set the circuit's reference squeezing separately.
        Input photon numbers must be corrected for propagation and detection loss.
        """
        nbar = torch.as_tensor(nbar)
        if nbar.is_complex() or nbar.ndim != 1 or not len(nbar) or not torch.isfinite(nbar).all():
            raise ValueError('nbar must be a finite real vector.')
        if (nbar < 0).any() or nbar[0] <= 0:
            raise ValueError('nbar must be nonnegative with a nonzero reference mode.')
        r = nbar.sqrt().asinh()
        return cls(coefficients, relative_r=r)

    @property
    def coefficients(self) -> torch.Tensor:
        """Profiles as rows of a complex matrix."""
        if self.components is None:
            eye = torch.eye(len(self.relative_r), device=self.relative_r.device, dtype=self.relative_r.dtype)
            return torch.complex(eye, torch.zeros_like(eye))
        return torch.view_as_complex(self.components.contiguous())

    @property
    def nsubmode(self) -> int:
        """Dimension of the submode basis."""
        return len(self.relative_r) if self.components is None else self.components.shape[1]

    def mean_photon_weights(self, r: Any) -> torch.Tensor:
        """Vacuum-generated photon fractions at reference squeezing r, including its zero-gain limit."""
        r = torch.as_tensor(r, device=self.relative_r.device, dtype=self.relative_r.dtype).unsqueeze(-1)
        denominator = torch.where(r == 0, torch.ones_like(r), r.sinh())
        amplitudes = torch.where(r == 0, self.relative_r, (r * self.relative_r).sinh() / denominator)
        photons = amplitudes.square()
        return photons / photons.sum(-1, keepdim=True)

    def effective_mode_number(self, r: Any) -> torch.Tensor:
        """Inverse squared photon-fraction sum at reference squeezing r."""
        return self.mean_photon_weights(r).square().sum(-1).reciprocal()


class ThermalSource(nn.Module):
    r"""Prepare an independent thermal state at each selected spatial input.

    Subsequent squeezing preserves the single-mode purity.
    Existing correlations with these modes are removed.
    Fock preparation retains the unnormalized projection below the circuit cutoff;
    its trace is ``1 - (nbar/(1+nbar))**cutoff`` per prepared mode.

    Args:
        nbar: Mean thermal photon number. Specify exactly one of ``nbar`` and ``purity``.
        purity: Single-mode purity, equal to ``1 / (2*nbar + 1)`` and in ``(0, 1]``.
        requires_grad: Whether to train the internal amplitude, with ``nbar = amplitude**2``. Default: ``False``.
    """

    def __init__(
        self,
        nbar: float | torch.Tensor | None = None,
        *,
        purity: float | torch.Tensor | None = None,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()
        if (nbar is None) == (purity is None):
            raise ValueError('Specify exactly one of nbar and purity.')
        value = torch.as_tensor(nbar if purity is None else purity)
        if not value.is_floating_point():
            if value.is_complex():
                raise ValueError('Thermal source parameters must be real.')
            value = value.to(torch.get_default_dtype())
        if value.numel() != 1 or not torch.isfinite(value).all():
            raise ValueError('Thermal source parameters must be finite scalars.')
        if purity is not None:
            if (value <= 0).any() or (value > 1).any():
                raise ValueError('purity must be in (0, 1].')
            value = (value.reciprocal() - 1) / 2
        elif (value < 0).any():
            raise ValueError('nbar must be nonnegative.')
        amplitude = value.reshape(()).sqrt()
        if requires_grad:
            self.register_parameter('amplitude', nn.Parameter(amplitude))
        else:
            self.register_buffer('amplitude', amplitude)

    @property
    def nbar(self) -> torch.Tensor:
        """Mean thermal photon number before Fock truncation."""
        return self.amplitude.square()

    @property
    def purity(self) -> torch.Tensor:
        """Single-mode purity before Fock truncation."""
        return (2 * self.nbar + 1).reciprocal()

    def forward(self, state: Any, run: '_HardwareRun') -> Any:
        """Reset selected modes to independent thermal states."""
        if run.backend in ('gaussian', 'bosonic'):
            cov, mean = state[:2]
            nmode = cov.shape[-1] // 2
            mask = cov.new_ones(2 * nmode)
            mask[list(run.wires) + [wire + nmode for wire in run.wires]] = 0
            vacuum = dqp.hbar / (4 * dqp.kappa**2)
            noise = torch.diag((1 - mask) * vacuum * (2 * self.nbar.to(cov) + 1))
            return [cov * mask[:, None] * mask[None, :] + noise, mean * mask[:, None]] + state[2:]
        if run.backend != 'fock' or run.basis:
            raise NotImplementedError('ThermalSource requires Gaussian, Bosonic or Fock density states.')
        cutoff = state.shape[-1]
        nbar = self.nbar.to(state.real)
        q = nbar / (1 + nbar)
        density = torch.diag(q ** torch.arange(cutoff, device=state.device) / (1 + nbar))
        for wire in run.wires:
            state = _replace_fock_modes(state, density, (wire,), run.nmode)
        return state


class SinglePhotonSource(nn.Module):
    r"""Single-photon input in a pure wavepacket or mixed submode state.

    Use ``QumodeCircuit.single_photon`` to prepare independent spatial inputs.
    Mixed inputs require Fock density representation; pure coefficients also
    support Fock kets and basis states through the circuit API.

    Args:
        density_matrix: Trace-one positive matrix on the single-photon submode basis, not the full Fock space.
        coefficients: Complex wavepacket vector in the common basis, normalized internally.
            Defaults to the first basis submode when all three state arguments are omitted.
        factor: Nonzero matrix with ``rho = factor @ factor.mH / trace(factor @ factor.mH)``.
            Supports differentiable inputs. Specify at most one of the three state arguments.
        requires_grad: Whether to train real and imaginary coefficient/factor components.
            A supplied density matrix is first factorized. Default: ``False``.
    """

    def __init__(
        self,
        density_matrix: Any = None,
        *,
        coefficients: Any = None,
        factor: Any = None,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()
        if sum(value is not None for value in (density_matrix, coefficients, factor)) > 1:
            raise ValueError('Specify only one of coefficients, density_matrix and factor.')
        self.is_pure = density_matrix is None and factor is None
        if self.is_pure:
            coefficients = torch.as_tensor([1.0] if coefficients is None else coefficients)
            if coefficients.ndim != 1:
                raise ValueError('coefficients must be a nonzero vector.')
            factor = coefficients[:, None]
        values = torch.as_tensor(density_matrix if factor is None else factor)
        if not values.is_floating_point() and not values.is_complex():
            values = values.to(torch.get_default_dtype())
        if values.ndim != 2 or min(values.shape) == 0 or not torch.isfinite(values).all():
            raise ValueError('The internal state must be a finite nonempty matrix.')
        if factor is None:
            if values.requires_grad:
                raise ValueError(
                    'Use factor for differentiable internal inputs, or requires_grad=True to train a source.'
                )
            tol = 100 * torch.finfo(values.real.dtype).eps
            if values.shape[0] != values.shape[1] or not torch.allclose(values, values.mH, atol=tol, rtol=tol):
                raise ValueError('density_matrix must be Hermitian.')
            eigenvalues, eigenvectors = torch.linalg.eigh(values)
            if (eigenvalues < -tol).any() or not torch.allclose(
                values.trace().real, values.real.new_tensor(1), atol=tol, rtol=tol
            ):
                raise ValueError('density_matrix must be positive semidefinite with trace one.')
            values = eigenvectors * eigenvalues.clamp_min(0).sqrt()
        if not values.norm() > 0:
            raise ValueError('factor must be nonzero.')
        _register_real_components(self, values, requires_grad)

    @property
    def coefficients(self) -> torch.Tensor:
        """Normalized wavepacket coefficients for a pure single photon."""
        if not self.is_pure:
            raise ValueError('A mixed single-photon state has no single wavepacket.')
        vector = torch.view_as_complex(self.components.contiguous())[:, 0]
        return vector / vector.norm()

    @property
    def density_matrix(self) -> torch.Tensor:
        """Normalized density matrix in the internal one-photon basis."""
        factor = torch.view_as_complex(self.components.contiguous())
        density = factor @ factor.mH
        return density / density.diagonal().sum().real

    @property
    def purity(self) -> torch.Tensor:
        """Internal-state purity, equal to HOM visibility for two independent identical sources."""
        density = self.density_matrix
        return (density @ density).diagonal().sum().real

    def forward(self, state: Any, run: '_HardwareRun') -> Any:
        """Replace the selected internal modes with one photon in the specified mixed state."""
        if run.backend != 'fock' or run.basis:
            raise NotImplementedError('SinglePhotonSource requires Fock density matrices (basis=False, den_mat=True).')
        density = self.density_matrix
        if len(run.wires) < density.shape[-1]:
            raise ValueError('The selected wire group must contain the single-photon submode basis.')
        cutoff = state.shape[-1]
        if cutoff < 2:
            raise ValueError('A single-photon source requires cutoff >= 2.')
        nsubmode = len(run.wires)
        # Occupations |1_k> have these flattened indices; all other submodes stay in vacuum.
        indices = torch.tensor([cutoff ** (nsubmode - i - 1) for i in range(len(density))], device=density.device)
        fock_density = density.new_zeros(cutoff**nsubmode, cutoff**nsubmode)
        fock_density = fock_density.index_put((indices[:, None], indices[None, :]), density)
        return _replace_fock_modes(state, fock_density, run.wires, run.nmode)
