"""Quantum states"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import comb
from torch import nn, vmap

import deepquantum.photonic as dqp

from ..communication import comm_get_rank, comm_get_world_size
from ..qmath import is_power, list_to_decimal, multi_kron
from ..utils import apply_complex_fix
from .bargmann import (
    BargmannPolyExp,
    _apply_bargmann,
    _project_pnr,
    fock_to_bargmann,
    gaussian_to_bargmann,
    vacuum_bargmann,
)
from .draw import GaussianGraphVisualizer
from .qmath import cv_to_wigner, dirac_rep, fock_to_wigner, xpxp_to_xxpp, xxpp_to_xpxp


class FockState(nn.Module):
    """A Fock state of n modes, including Fock basis states and Fock state tensors.

    Args:
        state: The Fock state. It can be a vacuum state with ``'vac'`` or ``'zeros'``.
            It can be a Fock basis state, e.g., ``[1,0,0]``, or a Fock state tensor,
            e.g., ``[(1/2**0.5, [1,0]), (1/2**0.5, [0,1])]``. Alternatively, it can be a tensor representation.
        nmode: The number of modes in the state. Default: ``None``
        cutoff: The Fock space truncation. Default: ``None``
        basis: Whether the state is a Fock basis state or not. Default: ``True``
        den_mat: Whether to use density matrix representation. Only valid for Fock state tensor. Default: ``False``
    """

    def __init__(
        self, state: Any, nmode: int | None = None, cutoff: int | None = None, basis: bool = True, den_mat: bool = False
    ) -> None:
        super().__init__()
        self.basis = basis
        self.den_mat = den_mat
        if self.basis:
            if state in ('vac', 'zeros'):
                state = [0] * nmode
            # Process Fock basis state
            state = torch.tensor(state, dtype=torch.long) if not isinstance(state, torch.Tensor) else state.long()
            if state.ndim == 1:
                state = state.unsqueeze(0)
            assert state.ndim == 2
            if nmode is None:
                nmode = state.shape[-1]
            if cutoff is None:
                cutoff = torch.max(torch.sum(state, dim=-1)).item() + 1
            self.nmode = nmode
            self.cutoff = cutoff
            batch, size = state.shape
            state_ts = torch.zeros([batch, nmode], dtype=torch.long, device=state.device)
            if nmode > size:
                state_ts[:, :size] = state[:, :]
            else:
                state_ts[:, :] = state[:, :nmode]
            state_ts = state_ts.squeeze(0)
            assert state_ts.max() < self.cutoff
        else:
            if state in ('vac', 'zeros'):
                state = [(1, [0] * nmode)]
            # Process Fock state tensor
            if isinstance(state, torch.Tensor):  # with the dimension of batch size
                if nmode is None:
                    nmode = (state.ndim - 1) // 2 if den_mat else state.ndim - 1
                if cutoff is None:
                    cutoff = state.shape[-1]
                self.nmode = nmode
                self.cutoff = cutoff
                state_ts = state
            else:
                # Process Fock state tensor from Fock basis states
                assert isinstance(state, list)
                if all(isinstance(i, int) for i in state):
                    state = [(1.0, state)]
                assert all(isinstance(i, tuple) for i in state)
                nphoton = 0
                # Determine the number of photons and modes
                for s in state:
                    nphoton = max(nphoton, sum(s[1]))
                    if nmode is None:
                        nmode = len(s[1])
                if cutoff is None:
                    cutoff = nphoton + 1
                self.nmode = nmode
                self.cutoff = cutoff
                state_ts = torch.zeros([self.cutoff] * self.nmode, dtype=torch.cfloat)
                # Populate Fock state tensor with the input Fock basis states
                for s in state:
                    amp = s[0]
                    fock_basis = tuple(s[1])
                    state_ts[fock_basis] = amp
                state_ts = state_ts.unsqueeze(0)  # add additional batch size
                if den_mat:
                    state_ts = state_ts.reshape([self.cutoff**self.nmode, 1])
                    state_ts = (state_ts @ state_ts.mH).reshape([-1] + [self.cutoff] * 2 * self.nmode)
            if den_mat:
                assert state_ts.ndim == 2 * self.nmode + 1
            else:
                assert state_ts.ndim == self.nmode + 1
            assert all(i == self.cutoff for i in state_ts.shape[1:])
        self.register_buffer('state', state_ts)

    def _apply(self, fn: Any) -> 'FockState':
        if self.basis:
            super()._apply(fn)
        else:
            tensors_dict = {}
            name = 'state'
            tensor = self._buffers.pop(name)
            if tensor is not None:
                tensors_dict[name] = tensor
            super()._apply(fn)
            corrected = apply_complex_fix(fn, tensors_dict)
            for key, value in corrected.items():
                self.register_buffer(key, value)
        return self

    def wigner(
        self,
        wire: int,
        xrange: int | list = 10,
        prange: int | list = 10,
        npoints: int | list = 100,
        plot: bool = True,
        k: int = 0,
    ):
        """Get the discretized Wigner function of the specified mode.

        Args:
            wire: The Wigner function for the given wire.
            xrange: The range of quadrature x. Default: 10
            prange: The range of quadrature p. Default: 10
            npoints: The number of discretization points for quadratures. Default: 100
            plot: Whether to plot the Wigner function. Default: ``True``
            k: The index of the Wigner function within the batch to plot. Default: 0
        """
        return fock_to_wigner(self.state, wire, self.nmode, self.cutoff, self.den_mat, xrange, prange, npoints, plot, k)

    def __repr__(self) -> str:
        """Return a string representation of the ``FockState``."""
        if self.basis:
            # Represent Fock basis state as a string
            if self.state.ndim == 1:
                state_str = ''.join(map(str, self.state.tolist()))
                return f'|{state_str}>'
            else:
                repr_str = ''
                for i in range(self.state.shape[0]):
                    state_str = ''.join(map(str, self.state[i].tolist()))
                    repr_str += f'state_{i}: |{state_str}>\n'
                return repr_str
        else:
            # Represent Fock state tensor using Dirac notation
            ket_dict = dirac_rep(self.state, self.den_mat)
            repr_str = ''
            for key, value in ket_dict.items():
                repr_str += f'{key}: {value}\n'
            return repr_str

    def __eq__(self, other: 'FockState') -> bool:
        """Check if two ``FockState`` instances are equal."""
        return all([self.nmode == other.nmode, self.state.equal(other.state)])

    def __hash__(self) -> int:
        """Compute the hash value for the ``FockState``."""
        if self.state.device.type == 'meta':
            return hash(id(self))
        return hash(self.__repr__())

    def __lt__(self, other: 'FockState') -> bool:
        tuple_self = tuple(self.state.tolist())
        tuple_other = tuple(other.state.tolist())
        return tuple_self < tuple_other


class GaussianState(nn.Module):
    r"""A Gaussian state of n modes, representing by covariance matrix and displacement vector.

    Args:
        state: The Gaussian state. It can be a vacuum state with ``'vac'``, or arbitrary Gaussian state
            with ``[cov, mean]``. ``cov`` and ``mean`` are the covariance matrix and the displacement vector of
            the Gaussian state, respectively. Use ``xxpp`` convention and :math:`\hbar=2` by default.
            Default: ``'vac'``
        nmode: The number of modes in the state. Default: ``None``
        cutoff: The Fock space truncation. Default: ``None``
    """

    def __init__(self, state: str | list = 'vac', nmode: int | None = None, cutoff: int | None = None) -> None:
        super().__init__()
        if state == 'vac':
            if nmode is None:
                nmode = 1
            cov = torch.eye(2 * nmode) * dqp.hbar / (4 * dqp.kappa**2)
            mean = torch.zeros(2 * nmode, 1)
        elif isinstance(state, list):
            cov = state[0]
            mean = state[1]
            if not isinstance(cov, torch.Tensor):
                cov = torch.tensor(cov, dtype=torch.float)
            if not isinstance(mean, torch.Tensor):
                mean = torch.tensor(mean, dtype=torch.float)
            if nmode is None:
                nmode = cov.shape[-1] // 2
        cov = cov.reshape(-1, 2 * nmode, 2 * nmode)
        mean = mean.reshape(-1, 2 * nmode, 1)
        assert cov.ndim == mean.ndim == 3
        assert cov.shape[-2] == cov.shape[-1] == 2 * nmode, (
            'The shape of the covariance matrix should be (2*nmode, 2*nmode)'
        )
        assert mean.shape[-2] == 2 * nmode, 'The length of the mean vector should be 2*nmode'
        self.register_buffer('cov', cov)
        self.register_buffer('mean', mean)
        self.nmode = nmode
        if cutoff is None:
            cutoff = 5
        self.cutoff = cutoff
        self.is_pure = self.check_purity()

    def check_purity(self, rtol: float = 3e-4, atol: float = 3e-4) -> bool:
        """Check if the Gaussian state is pure state

        See https://arxiv.org/pdf/quant-ph/0503237.pdf Eq.(2.5)
        """
        sign, log_det = torch.linalg.slogdet(4 * dqp.kappa**2 / dqp.hbar * self.cov)
        is_pure = (sign > 0).all() and torch.allclose(
            log_det, torch.tensor(0.0, device=log_det.device, dtype=log_det.dtype), rtol=rtol, atol=atol
        )
        return is_pure

    def wigner(
        self,
        wire: int,
        xrange: int | list = 10,
        prange: int | list = 10,
        npoints: int | list = 100,
        plot: bool = True,
        k: int = 0,
        normalize: bool = True,
    ):
        """Get the discretized Wigner function of the specified mode.

        Args:
            wire: The Wigner function for the given wire.
            xrange: The range of quadrature x. Default: 10
            prange: The range of quadrature p. Default: 10
            npoints: The number of discretization points for quadratures. Default: 100
            plot: Whether to plot the Wigner function. Default: ``True``
            k: The index of the Wigner function within the batch to plot. Default: 0
            normalize: Whether to normalize the Wigner function. Default: ``True``
        """
        return cv_to_wigner([self.cov, self.mean], wire, xrange, prange, npoints, plot, k, normalize)

    def graph(self, k: int = 0, threshold: float = 1e-3, layout: str = 'spring', mode: str = 'simplified'):
        """Draw Gaussian graph state.

        Args:
            k: The batch index for covariance matrix.
            threshold: Cutoff threshold for ignoring extremely small numerical edge weights
            in the visualization.
            layout: The layout algorithm (``'spring'``, ``'circular'`` or ``'kamada_kawai'``).
                Also accepts ``'kamada'`` for Kamada-Kawai layout. Default: ``'spring'``
            mode: The ``'simplified'`` mode for the core structure including node squeezing :math:`Im(Z_{jj})` and
            edge entanglement :math:`Re(Z_{jk})`. The ``'full'`` mode for all structure including
            local shear :math:`Re(Z_{jj})` and correlated noise :math:`Im(Z_{jk})`. Default: ``'simplified'``
        """
        assert self.is_pure, 'The visualization is valid for Gaussian pure state'
        cov = 2 * dqp.kappa**2 / dqp.hbar * self.cov[k]
        graph_visualizer = GaussianGraphVisualizer(cov, threshold, mode)
        graph_visualizer.draw(layout=layout)
        return graph_visualizer


class BargmannState(nn.Module):
    r"""A photonic state in the Bargmann (stellar) representation.

    The state is stored as a :class:`BargmannPolyExp` ``(A, b, c)``. A ket on ``M`` modes has ``M``
    free variables, one per mode in local mode order. A density matrix has ``2M`` free
    variables: all ket variables followed by all bra variables, so that

    .. math::

        K_\psi(z)=\sum_n\psi_n\frac{z^n}{\sqrt{n!}},\qquad
        K_\rho(z,w)=\sum_{n,m}\rho_{n,m}\frac{z^nw^m}{\sqrt{n!m!}}.

    Gaussian states have no auxiliary variables. Non-Gaussian states use auxiliary
    derivative variables. The state is not normalized automatically.

    Args:
        state: The state. It can be ``'vac'``/``'zeros'``; a Gaussian state ``[cov, mean]`` (xxpp,
            current ``hbar``/``kappa``) or a ``GaussianState``; a Fock basis state such as ``[1, 0, 2]``;
            a Fock superposition ``[(amp, [n0, n1, ...]), ...]``; a ``FockState``; a batched Fock tensor
            (ket ``(B, C, ..., C)`` or density matrix ``(B, ket axes..., bra axes...)``); or a
            :class:`BargmannPolyExp`. Default: ``'vac'``
        nmode: The number of modes. Default: ``None``
        cutoff: The default Fock dimension for readout only; it does not affect evolution. Default: ``None``
        den_mat: Whether to use the density-matrix representation. Default: ``False``

    Note:
        Gaussian moments are converted with their own precision. Pass float64 moments for
        double-precision results; a later ``.to(torch.float64)`` cannot recover lost digits.
        A ket built from moments uses the gauge in which the vacuum amplitude is real and positive.
    """

    def __init__(self, state: Any = 'vac', nmode: int | None = None, cutoff: int | None = None, den_mat: bool = False):
        super().__init__()
        self.den_mat = den_mat
        polyexp, nmode = self._to_polyexp(state, nmode, den_mat)
        self.nmode = nmode
        self.cutoff = 5 if cutoff is None else int(cutoff)
        for name in ('mat_a', 'vec_b', 'coeff_c'):
            self.register_buffer(name, getattr(polyexp, name))

    @staticmethod
    def _to_polyexp(state: Any, nmode: int | None, den_mat: bool) -> tuple[BargmannPolyExp, int]:
        if isinstance(state, BargmannState):
            if state.den_mat != den_mat:
                raise ValueError('den_mat does not match the representation of the given BargmannState')
            return state.polyexp, state.nmode
        if isinstance(state, BargmannPolyExp):
            if nmode is None:
                nmode = state.nfree // 2 if den_mat else state.nfree
            if state.nfree != (2 * nmode if den_mat else nmode):
                raise ValueError('The representation must have one free variable per mode (two for density matrices)')
            return state, nmode
        if isinstance(state, str):
            if state not in ('vac', 'zeros'):
                raise ValueError(f'Unknown state {state!r}')
            nmode = 1 if nmode is None else nmode
            return vacuum_bargmann(2 * nmode if den_mat else nmode), nmode
        if isinstance(state, GaussianState) or (
            isinstance(state, list) and len(state) == 2 and all(isinstance(s, torch.Tensor) for s in state)
        ):
            if not isinstance(state, GaussianState):
                state = GaussianState(state, nmode=nmode)
            return BargmannState._from_moments(state.cov, state.mean, den_mat), state.nmode
        if isinstance(state, list):
            if all(isinstance(s, int) for s in state):
                return BargmannState._from_fock_basis(state, den_mat), len(state)
            state = FockState(state, nmode=nmode, basis=False, den_mat=False)
        if isinstance(state, FockState):
            if state.basis:
                pattern = state.state.reshape(-1).tolist()
                if state.state.ndim != 1:
                    raise ValueError('Batched Fock basis states are not supported; use a Fock tensor')
                return BargmannState._from_fock_basis(pattern, den_mat), state.nmode
            tensor, is_dm, nmode = state.state, state.den_mat, state.nmode
        elif isinstance(state, torch.Tensor):
            if nmode is None:
                raise ValueError('nmode is required for a Fock tensor input')
            tensor = state
            if tensor.ndim == nmode:
                tensor = tensor.unsqueeze(0)
                is_dm = False
            elif tensor.ndim == 2 * nmode and den_mat:
                tensor = tensor.unsqueeze(0)
                is_dm = True
            else:
                is_dm = tensor.ndim == 2 * nmode + 1
        else:
            raise ValueError(f'Unsupported state type {type(state).__name__}')
        if not tensor.is_complex():
            tensor = tensor + 0j
        if is_dm and not den_mat:
            raise ValueError('A density-matrix input requires den_mat=True')
        if den_mat and not is_dm:
            batch = tensor.shape[0]
            ket = tensor.reshape(batch, -1)
            tensor = (ket[:, :, None] * ket.conj()[:, None, :]).reshape(batch, *tensor.shape[1:], *tensor.shape[1:])
        nvar = 2 * nmode if den_mat else nmode
        if tensor.ndim != nvar + 1:
            raise ValueError('The Fock tensor shape does not match nmode')
        return fock_to_bargmann(tensor, nvar), nmode

    @staticmethod
    def _from_fock_basis(pattern: list[int], den_mat: bool) -> BargmannPolyExp:
        if any(n < 0 for n in pattern):
            raise ValueError('Photon numbers must be nonnegative')
        pattern = pattern * (2 if den_mat else 1)
        tensor = torch.zeros(1, *[n + 1 for n in pattern], dtype=torch.cfloat)
        tensor[(0, *pattern)] = 1
        return fock_to_bargmann(tensor, tensor.ndim - 1)

    @staticmethod
    def _from_moments(cov: torch.Tensor, mean: torch.Tensor, den_mat: bool) -> BargmannPolyExp:
        mat_a, vec_b, coeff_c = gaussian_to_bargmann(cov, mean, den_mat=den_mat)
        batch = mat_a.shape[0]
        nvar = mat_a.shape[-1]
        return BargmannPolyExp._from_tensors(
            mat_a.reshape(batch, 1, nvar, nvar), vec_b.reshape(batch, 1, nvar), coeff_c.to(mat_a.dtype)[:, None]
        )

    def _apply(self, fn: Any) -> 'BargmannState':
        tensors = {name: self._buffers.pop(name) for name in ('mat_a', 'vec_b', 'coeff_c')}
        super()._apply(fn)
        for name, value in apply_complex_fix(fn, tensors).items():
            self.register_buffer(name, value)
        return self

    def _new(self, polyexp: BargmannPolyExp, den_mat: bool | None = None) -> 'BargmannState':
        den_mat = self.den_mat if den_mat is None else den_mat
        return BargmannState(polyexp, den_mat=den_mat, cutoff=self.cutoff)

    @property
    def polyexp(self) -> BargmannPolyExp:
        """The underlying :class:`BargmannPolyExp`."""
        return BargmannPolyExp._from_tensors(self.mat_a, self.vec_b, self.coeff_c)

    @property
    def nfree(self) -> int:
        """Number of Bargmann variables labeling the physical state."""
        return self.nmode * (2 if self.den_mat else 1)

    @property
    def representation(self) -> str:
        """``'dm'`` for density matrices, otherwise ``'ket'``."""
        return 'dm' if self.den_mat else 'ket'

    @property
    def batch_size(self) -> int:
        """Size of the batch axis."""
        return self.mat_a.shape[0]

    def _select_batch(self, index: int) -> 'BargmannState':
        """Return the batch element ``index`` as a state with batch size 1."""
        sl = slice(index, index + 1)
        return self._new(BargmannPolyExp._from_tensors(self.mat_a[sl], self.vec_b[sl], self.coeff_c[sl]))

    @staticmethod
    def _cat_batch(states: list['BargmannState']) -> 'BargmannState':
        """Concatenate states with identical structure along the batch axis."""
        first = states[0]
        mat_a = torch.cat([s.mat_a for s in states])
        vec_b = torch.cat([s.vec_b for s in states])
        coeff_c = torch.cat([s.coeff_c for s in states])
        polyexp = BargmannPolyExp._from_tensors(mat_a, vec_b, coeff_c)
        return first._new(polyexp)

    def apply_unitary(self, kernel: BargmannPolyExp, wires: list[int]) -> 'BargmannState':
        r"""Apply a unitary kernel: :math:`U|\psi\rangle` or :math:`U\rho U^\dagger`."""
        pos = list(wires)
        if len(set(pos)) != len(pos) or any(w < 0 or w >= self.nmode for w in pos):
            raise ValueError('wires must contain distinct local mode indices')
        polyexp = _apply_bargmann(self.polyexp, kernel, pos)
        if self.den_mat:
            polyexp = _apply_bargmann(polyexp, kernel.conj(), [p + self.nmode for p in pos])
        return self._new(polyexp)

    def apply_channel(self, kernel: BargmannPolyExp, wires: list[int]) -> 'BargmannState':
        """Apply a channel kernel with variables ``(z_out, w_out, u_in, v_in)``; requires a density matrix."""
        if not self.den_mat:
            raise ValueError('Channels require the density-matrix representation; use den_mat=True')
        pos = list(wires)
        if len(set(pos)) != len(pos) or any(w < 0 or w >= self.nmode for w in pos):
            raise ValueError('wires must contain distinct local mode indices')
        polyexp = _apply_bargmann(self.polyexp, kernel, pos + [p + self.nmode for p in pos])
        return self._new(polyexp)

    def to_dm(self) -> 'BargmannState':
        r"""Return :math:`|\psi\rangle\langle\psi|`; a density matrix is returned unchanged."""
        if self.den_mat:
            return self
        polyexp = self.polyexp
        return self._new(polyexp.tensor_product(polyexp.conj()), den_mat=True)

    def _project_pnr(self, positions: tuple[int, ...], photons: tuple[int, ...]) -> 'BargmannState':
        """Project validated modes onto Fock outcomes, returning locally numbered remaining modes."""
        polyexp = _project_pnr(self.polyexp, self.nmode, self.den_mat, positions, photons)
        return self._new(polyexp)

    def norm(self) -> torch.Tensor:
        r"""Squared norm :math:`\langle\psi|\psi\rangle` of a ket or trace of a density matrix, shape ``(B,)``.

        Computed by Gaussian integration, independently of any Fock cutoff.
        Nonfinite results or imaginary parts exceeding 100 dtype eps times
        ``1 + abs(real part)`` raise ``ValueError``; roundoff imaginary parts are discarded.
        """
        polyexp = self.polyexp
        if not self.den_mat:
            polyexp = polyexp.tensor_product(polyexp.conj())
        value = polyexp.contract([(j, j + self.nmode) for j in range(self.nmode)]).evaluate()
        if not bool(torch.isfinite(value).all()):
            raise ValueError('State norm or trace must be finite')
        tolerance = 100 * torch.finfo(value.real.dtype).eps * (1 + value.real.abs())
        if bool((value.imag.abs() > tolerance).any()):
            raise ValueError('State norm or trace has a significant imaginary component')
        return value.real

    def _normalize(self, probability: torch.Tensor) -> 'BargmannState':
        """Normalize using an already validated positive squared norm or trace."""
        denominator = probability if self.den_mat else probability.sqrt()
        coeff_c = self.coeff_c / denominator.reshape(-1, *([1] * (self.coeff_c.ndim - 1)))
        return self._new(BargmannPolyExp._from_tensors(self.mat_a, self.vec_b, coeff_c))

    def to_fock(self, cutoff: int | None = None) -> torch.Tensor:
        """Fock amplitudes ``(B, C, ..., C)`` or density matrix ``(B, ket axes..., bra axes...)``.

        Axes follow local mode order. ``cutoff=C`` keeps photon numbers ``0, ..., C - 1``.
        """
        cutoff = self.cutoff if cutoff is None else int(cutoff)
        return self.polyexp.to_fock((cutoff,) * self.nfree)

    def __repr__(self) -> str:
        return (
            f'BargmannState(nmode={self.nmode}, representation={self.representation!r}, '
            f'batch={self.batch_size}, aux_shape={tuple(self.coeff_c.shape[2:])}, dtype={self.mat_a.dtype})'
        )


class BosonicState(nn.Module):
    r"""A Bosoncic state of n modes, representing by a linear combination of Gaussian states.

    Args:
        state: The Bosoncic state. It can be a vacuum state with ``'vac'``, or arbitrary
            linear combinations of Gaussian states with ``[cov, mean, weight]``. ``cov``,``mean`` and ``weight`` are
            the covariance matrices, the displacement vectors and the weights of the Gaussian states, respectively.
            Use ``xxpp`` convention and :math:`\hbar=2` by default.
            Default: ``'vac'``
        nmode: The number of modes in the state. Default: ``None``
        cutoff: The Fock space truncation. Default: ``None``
    """

    def __init__(self, state: str | list = 'vac', nmode: int | None = None, cutoff: int | None = None) -> None:
        super().__init__()
        if state == 'vac':
            if nmode is None:
                nmode = 1
            cov = torch.eye(2 * nmode) * dqp.hbar / (4 * dqp.kappa**2)
            mean = torch.zeros(2 * nmode, 1) + 0j
            weight = torch.tensor([1]) + 0j
        elif isinstance(state, list):
            cov = state[0]
            mean = state[1]
            weight = state[2]
            if not isinstance(cov, torch.Tensor):
                cov = torch.tensor(cov, dtype=torch.float)
            if not isinstance(mean, torch.Tensor):
                mean = torch.tensor(mean, dtype=torch.cfloat)
            if not isinstance(weight, torch.Tensor):
                weight = torch.tensor(weight, dtype=torch.cfloat)
            if nmode is None:
                nmode = cov.shape[-1] // 2
        ncomb = weight.shape[-1]
        if cov.ndim == 2:
            cov = cov.reshape(1, 1, 2 * nmode, 2 * nmode)
        elif cov.ndim == 3:
            cov = cov.reshape(-1, ncomb, 2 * nmode, 2 * nmode)
        if mean.ndim == 2:
            if mean.shape[-1] == 1:
                mean = mean.reshape(1, 1, 2 * nmode, 1)
            elif mean.shape[0] == ncomb:
                mean = mean.reshape(1, ncomb, 2 * nmode, 1)
        elif mean.ndim == 3:
            mean = mean.reshape(-1, ncomb, 2 * nmode, 1)
        weight = weight.reshape(-1, ncomb)
        assert cov.ndim == mean.ndim == 4
        assert cov.shape[0] == mean.shape[0]
        assert cov.shape[-2] == cov.shape[-1] == 2 * nmode, (
            'The shape of the covariance matrix should be (2*nmode, 2*nmode)'
        )
        assert mean.shape[-2] == 2 * nmode, 'The length of the mean vector should be 2*nmode'
        self.register_buffer('cov', cov)
        self.register_buffer('mean', mean)
        self.register_buffer('weight', weight)
        self.nmode = nmode
        if cutoff is None:
            cutoff = 5
        self.cutoff = cutoff

    def _apply(self, fn: Any) -> 'BosonicState':
        tensors_dict = {}
        names = ['mean', 'weight']
        tensors_dict = {name: tensor for name in names if (tensor := self._buffers.pop(name)) is not None}
        super()._apply(fn)
        corrected = apply_complex_fix(fn, tensors_dict)
        for key, value in corrected.items():
            self.register_buffer(key, value)
        return self

    def tensor_product(self, state: 'BosonicState') -> 'BosonicState':
        """Get the tensor product of two Bosonic states."""
        return combine_bosonic_states([self, state])

    def wigner(
        self,
        wire: int,
        xrange: int | list = 10,
        prange: int | list = 10,
        npoints: int | list = 100,
        plot: bool = True,
        k: int = 0,
        normalize: bool = True,
    ):
        """Get the discretized Wigner function of the specified mode.

        Args:
            wire: The Wigner function for the given wire.
            xrange: The range of quadrature x. Default: 10
            prange: The range of quadrature p. Default: 10
            npoints: The number of discretization points for quadratures. Default: 100
            plot: Whether to plot the Wigner function. Default: ``True``
            k: The index of the Wigner function within the batch to plot. Default: 0
            normalize: Whether to normalize the Wigner function. Default: ``True``
        """
        return cv_to_wigner([self.cov, self.mean, self.weight], wire, xrange, prange, npoints, plot, k, normalize)

    def marginal(
        self, wire: int, phi: float = 0.0, xrange: int | list = 10, npoints: int = 100, plot: bool = True, k: int = 0
    ):
        r"""Get the discretized marginal distribution of the specified mode along :math:`x\cos\phi + p\sin\phi`.

        Args:
            wire: The marginal function for given wire.
            phi: The angle used to compute the linear combination of quadratures. Default: 0
            xrange: The range of quadrature. Default: 10
            npoints: The number of discretization points for quadrature. Default: 100
            plot: Whether to plot the marginal function. Default: ``True``
            k: The index of the marginal function within the batch to plot. Default: 0
        """
        from .gate import PhaseShift

        xlist = [-xrange, xrange] if isinstance(xrange, int) else xrange
        xlist.append(npoints)
        assert len(xlist) == 3
        xvec = torch.linspace(*xlist)
        if not isinstance(wire, torch.Tensor):
            wire = torch.tensor(wire).reshape(1)
        idx = torch.cat([wire, wire + self.nmode])  # xxpp order
        cov = self.cov[..., idx[:, None], idx]
        mean = self.mean[..., idx, :]
        r = PhaseShift(inputs=-phi, nmode=1, wires=[0], cutoff=self.cutoff)
        r.to(cov.device, cov.dtype)
        cov, mean = r([cov, mean])  # (batch, ncomb, 2, 2)
        cov = cov[..., 0, 0].unsqueeze(1)
        mean = mean[..., 0, 0].unsqueeze(1)
        prefactor = 1 / (torch.sqrt(2 * torch.pi * cov))  # (batch, 1, ncomb)
        # (batch, npoints, ncomb)
        log_marg_vals = torch.log(self.weight.unsqueeze(1) * prefactor) - 0.5 * (xvec.reshape(-1, 1) - mean) ** 2 / cov
        marginal_vals = torch.exp(log_marg_vals).sum(2).real
        if plot:
            plt.subplots(1, 1, figsize=(12, 10))
            plt.xlabel('Quadrature q')
            plt.ylabel('Wave function')
            plt.plot(xvec.cpu(), marginal_vals[k].cpu())
            plt.show()
        return marginal_vals


class CatState(BosonicState):
    r"""Single-mode cat state.

    The cat state is a superposition of coherent states.

    See https://arxiv.org/abs/2103.05530 Section IV B.

    Args:
        r: Displacement magnitude :math:`|r|`. Default: ``None``
        theta: Displacement angle :math:`\theta`. Default: ``None``
        p: Parity, where :math:`\theta=p\pi`. ``p=0`` corresponds to an even cat state, and ``p=1`` an odd cat state.
            Default: 1
        cutoff: The Fock space truncation. Default: ``None``
    """

    def __init__(self, r: Any = None, theta: Any = None, p: int = 1, cutoff: int | None = None) -> None:
        nmode = 1
        covs = torch.eye(2) * dqp.hbar / (4 * dqp.kappa**2)
        if r is None:
            r = torch.rand(1)[0]
        if theta is None:
            theta = torch.rand(1)[0] * 2 * torch.pi
        if not isinstance(r, torch.Tensor):
            r = torch.tensor(r, dtype=torch.float)
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, dtype=torch.float)
        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, dtype=torch.long)
        real_part = r * torch.cos(theta)
        imag_part = r * torch.sin(theta)
        means = (
            torch.stack(
                [
                    torch.stack([real_part, imag_part]) + 0j,
                    -torch.stack([real_part, imag_part]) + 0j,
                    torch.stack([imag_part, -real_part]) * 1j,
                    -torch.stack([imag_part, -real_part]) * 1j,
                ]
            )
            * dqp.hbar**0.5
            / dqp.kappa
        )
        temp = torch.exp(-2 * r**2)
        w0 = 0.5 / (1 + temp * torch.cos(p * torch.pi)) + 0j
        w1 = w0
        w2 = torch.exp(-1j * torch.pi * p) * temp * w0
        w3 = torch.exp(1j * torch.pi * p) * temp * w0
        weights = torch.stack([w0, w1, w2, w3])
        state = [covs, means, weights]
        super().__init__(state, nmode, cutoff)


# ruff: noqa: E741
class GKPState(BosonicState):
    r"""Finite-energy single-mode GKP state.

    Using GKP states to encode qubits, with the qubit state defined by:
    :math:`\ket{\psi}_{gkp} = \cos\frac{\theta}{2}\ket{0}_{gkp} + e^{-i\phi}\sin\frac{\theta}{2}\ket{1}_{gkp}`

    See https://arxiv.org/abs/2103.05530 Section IV A.

    Args:
        theta: angle :math:`\theta` in Bloch sphere. Default: ``None``
        phi: angle :math:`\phi` in Bloch sphere. Default: ``None``
        amp_cutoff: The amplitude threshold for keeping the terms. Default: 0.1
        epsilon: The finite energy damping parameter. Default: 0.05
        cutoff: The Fock space truncation. Default: ``None``
    """

    def __init__(
        self,
        theta: Any = None,
        phi: Any = None,
        amp_cutoff: float = 0.1,
        epsilon: float = 0.05,
        cutoff: int | None = None,
    ) -> None:
        nmode = 1
        if theta is None:
            theta = torch.rand(1)[0] * 2 * torch.pi
        if phi is None:
            phi = torch.rand(1)[0] * 2 * torch.pi
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, dtype=torch.float)
        if not isinstance(phi, torch.Tensor):
            phi = torch.tensor(phi, dtype=torch.float)
        if not isinstance(epsilon, torch.Tensor):
            epsilon = torch.tensor(epsilon, dtype=torch.float)
        if not isinstance(amp_cutoff, torch.Tensor):
            amp_cutoff = torch.tensor(amp_cutoff, dtype=torch.float)
        self.epsilon = epsilon
        self.amp_cutoff = amp_cutoff
        exp_eps = torch.exp(-2 * epsilon)
        # gaussian envelope
        z_max = torch.ceil(torch.sqrt(-4 / torch.pi * torch.log(amp_cutoff) * (1 + exp_eps) / (1 - exp_eps)))
        coords = torch.arange(-z_max, z_max + 1)
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing='ij')
        means = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

        k = means[:, 0]
        l = means[:, 1]
        thetas = torch.tensor([theta] * len(k))
        phis = torch.tensor([phi] * len(k))
        weights = self._update_weight(k, l, thetas, phis)

        filt = abs(weights) > amp_cutoff
        weights = weights[filt] + 0j
        weights /= torch.sum(weights)
        means = means[filt]
        means = means * torch.exp(-epsilon) / (1 + exp_eps) * (torch.pi * dqp.hbar / 2) ** 0.5 / dqp.kappa + 0j
        covs = torch.eye(2) * dqp.hbar / (4 * dqp.kappa**2) * (1 - exp_eps) / (1 + exp_eps)
        state = [covs, means, weights]
        super().__init__(state, nmode, cutoff)

    def _update_weight(self, k, l, theta, phi):
        """Compute the updated coefficients c_{k, l}(theta, phi) for given k, l, theta, and phi.

        See https://arxiv.org/abs/2103.05530 Eq.(43) and (B1)
        """
        # Ensure that k and l are integers
        k = k.long()
        l = l.long()

        k_mod_2 = k % 2
        l_mod_2 = l % 2
        k_mod_4 = k % 4
        l_mod_4 = l % 4

        result = torch.zeros_like(theta)

        # Case 1: k mod 2 == 0 and l mod 2 == 0
        mask1 = (k_mod_2 == 0) & (l_mod_2 == 0)
        result[mask1] = 1

        # Case 2: k mod 4 == 0 and l mod 2 == 1
        mask2 = (k_mod_4 == 0) & (l_mod_2 == 1)
        result[mask2] = torch.cos(theta[mask2])

        # Case 3: k mod 4 == 2 and l mod 2 == 1
        mask3 = (k_mod_4 == 2) & (l_mod_2 == 1)
        result[mask3] = -torch.cos(theta[mask3])

        # Case 4: k mod 4 == 3 and l mod 4 == 0
        mask4_1 = (k_mod_4 == 3) & (l_mod_4 == 0)
        result[mask4_1] = torch.sin(theta[mask4_1]) * torch.cos(phi[mask4_1])

        # Case 5: k mod 4 == 1 and l mod 4 == 0
        mask4_2 = (k_mod_4 == 1) & (l_mod_4 == 0)
        result[mask4_2] = torch.sin(theta[mask4_2]) * torch.cos(phi[mask4_2])

        # Case 6: k mod 4 == 3 and l mod 4 == 2
        mask4_3 = (k_mod_4 == 3) & (l_mod_4 == 2)
        result[mask4_3] = -torch.sin(theta[mask4_3]) * torch.cos(phi[mask4_3])

        # Case 7: k mod 4 == 1 and l mod 4 == 2
        mask4_4 = (k_mod_4 == 1) & (l_mod_4 == 2)
        result[mask4_4] = -torch.sin(theta[mask4_4]) * torch.cos(phi[mask4_4])

        # Case 8: k mod 4 == 3 and l mod 4 == 3
        mask5_1 = (k_mod_4 == 3) & (l_mod_4 == 3)
        result[mask5_1] = -torch.sin(theta[mask5_1]) * torch.sin(phi[mask5_1])

        # Case 9: k mod 4 == 1 and l mod 4 == 1
        mask5_2 = (k_mod_4 == 1) & (l_mod_4 == 1)
        result[mask5_2] = -torch.sin(theta[mask5_2]) * torch.sin(phi[mask5_2])

        # Case 10: k mod 4 == 3 and l mod 4 == 1
        mask5_3 = (k_mod_4 == 3) & (l_mod_4 == 1)
        result[mask5_3] = torch.sin(theta[mask5_3]) * torch.sin(phi[mask5_3])

        # Case 11: k mod 4 == 1 and l mod 4 == 3
        mask5_4 = (k_mod_4 == 1) & (l_mod_4 == 3)
        result[mask5_4] = torch.sin(theta[mask5_4]) * torch.sin(phi[mask5_4])

        exp_eps = torch.exp(-2 * self.epsilon)
        prefactor = torch.exp(-0.25 * torch.pi * (l**2 + k**2) * (1 - exp_eps) / (1 + exp_eps))

        weight = result * prefactor  # update coefficient
        return weight


class FockStateBosonic(BosonicState):
    """Single-mode Fock state, representing by a linear combination of Gaussian states.

    See https://arxiv.org/abs/2103.05530 Section IV C.

    Args:
        n: Particle number.
        r: The quality parameter for the approximation. Default: 0.05
        cutoff: The Fock space truncation. Default: ``None``
    """

    def __init__(self, n: int, r: Any = 0.05, cutoff: int | None = None) -> None:
        assert r**2 < 1 / n, 'NOT a physical state'
        nmode = 1
        m = np.arange(n + 1)
        combs = comb(n, m)
        weight = (1 - n * r**2) / (1 - (n - m) * r**2) * combs * (-1) ** (n - m)
        weight = torch.tensor(weight / weight.sum(), dtype=torch.float) + 0j
        mean = torch.zeros([n + 1, 2, 1]) + 0j
        m = torch.tensor(m).reshape(-1, 1, 1)
        cov = torch.eye(2) * dqp.hbar / (4 * dqp.kappa**2) * (1 + (n - m) * r**2) / (1 - (n - m) * r**2)
        state = [cov, mean, weight]
        if cutoff is None:
            cutoff = n + 1
        super().__init__(state, nmode, cutoff)


class DistributedFockState(nn.Module):
    """A Fock state of n modes distributed between w nodes.

    Args:
        state: The Fock state. It can be a vacuum state with ``'vac'`` or ``'zeros'``.
            It can be a Fock basis state, e.g., ``[1,0,0]``, or a Fock state tensor,
            e.g., ``[(1/2**0.5, [1,0]), (1/2**0.5, [0,1])]``.
        nmode: The number of modes in the state. Default: ``None``
        cutoff: The Fock space truncation. Default: ``None``
    """

    def __init__(self, state: Any, nmode: int | None = None, cutoff: int | None = None) -> None:
        super().__init__()
        self.world_size = comm_get_world_size()
        self.rank = comm_get_rank()
        assert 0 <= self.rank < self.world_size
        if state in ('vac', 'zeros'):
            state = [(1, [0] * nmode)]
        assert isinstance(state, list)
        if all(isinstance(i, int) for i in state):
            state = [(1.0, state)]
        assert all(isinstance(i, tuple) for i in state)
        nphoton = 0
        for s in state:
            nphoton = max(nphoton, sum(s[1]))
            if nmode is None:
                nmode = len(s[1])
        if cutoff is None:
            cutoff = nphoton + 1
        self.state = state  # list of tuples
        self.nmode = nmode
        self.cutoff = cutoff
        assert is_power(self.world_size, self.cutoff)
        assert cutoff**nmode >= self.world_size
        self.nmode_global = int(np.round(np.emath.logn(self.cutoff, self.world_size)))
        self.nmode_local = self.nmode - self.nmode_global
        self.num_amps_per_node = self.cutoff**self.nmode_local
        amps = torch.zeros([self.cutoff] * self.nmode_local, dtype=torch.cfloat)
        buffer = torch.zeros_like(amps)
        self.register_buffer('amps', amps)
        self.register_buffer('buffer', buffer)
        self.reset()

    def _apply(self, fn: Any) -> 'DistributedFockState':
        tensors_dict = {}
        names = ['amps', 'buffer']
        tensors_dict = {name: tensor for name in names if (tensor := self._buffers.pop(name)) is not None}
        super()._apply(fn)
        corrected = apply_complex_fix(fn, tensors_dict)
        for key, value in corrected.items():
            self.register_buffer(key, value)
        return self

    def reset(self):
        """Reset the state."""
        self.amps.zero_()
        self.buffer.zero_()
        for s in self.state:
            amp = s[0]
            rank = list_to_decimal(s[1][: self.nmode_global], self.cutoff)
            if self.rank == rank:
                fock_basis = tuple(s[1][self.nmode_global :])
                self.amps[fock_basis] = amp


def combine_tensors(tensors: list[torch.Tensor], ndim_ds: int = 2) -> torch.Tensor:
    """Combine a list of 3D tensors for Bosonic states according to the dimension of direct sum.

    Args:
        tensors: The list of 3D tensors to combine.
        ndim_ds: The dimension of direct sum. Use 1 for direct sum along rows,
            or use 2 for direct sum along both rows and columns. Default: 2
    """
    assert ndim_ds in (1, 2)
    # Get number of tensors and their shapes
    n = len(tensors)
    shape_lst = [tensor.shape for tensor in tensors]
    len_lst, hs, ws = map(list, zip(*shape_lst, strict=True))
    size_h = sum(hs)
    if ndim_ds == 1:
        size_w = ws[0]
    elif ndim_ds == 2:
        size_w = sum(ws)
    # Expand tensors to combination dimensions
    expanded_tensors = []
    for i in range(n):
        # Insert new dimensions and expand for combination
        view_shape = [1] * n + list(tensors[i].shape[1:])  # tensors[i]: (len_i, h_i, w_i)
        view_shape[i] = tensors[i].shape[0]
        expand_shape = len_lst + list(tensors[i].shape[1:])
        expanded = tensors[i].view(*view_shape).expand(*expand_shape)
        expanded_tensors.append(expanded)
    # Create zero-initialized result template (len_1, len_2, ..., len_n, size_h, size_w)
    result = tensors[0].new_zeros(*len_lst, size_h, size_w)
    # Calculate block offsets
    row_offsets = torch.cumsum(torch.tensor([0] + hs[:-1]), 0).tolist()
    if ndim_ds == 2:
        col_offsets = torch.cumsum(torch.tensor([0] + ws[:-1]), 0).tolist()
    # Place each block in corresponding position
    for i in range(n):
        h, w = hs[i], ws[i]
        row_start = row_offsets[i]
        if ndim_ds == 1:
            result[..., row_start : row_start + h, :w] = expanded_tensors[i]
        elif ndim_ds == 2:
            col_start = col_offsets[i]
            result[..., row_start : row_start + h, col_start : col_start + w] = expanded_tensors[i]
    # Flatten result to (len_1*len_2*...*len_n, size_h, size_w)
    return result.view(-1, size_h, size_w)


def combine_bosonic_states(states: list[BosonicState], cutoff: int | None = None) -> BosonicState:
    """Combine multiple Bosonic states into a single state.

    Args:
        states: List of Bosonic states to combine.
        cutoff: The Fock space truncation. If ``None``, the cutoff of the first state is used. Default: ``None``
    """
    covs = []
    means = []
    weights = []
    nmode = 0
    if cutoff is None:
        cutoff = states[0].cutoff
    for state in states:
        covs.append(xxpp_to_xpxp(state.cov))
        means.append(xxpp_to_xpxp(state.mean))
        weights.append(state.weight)
        nmode += state.nmode
    cov = xpxp_to_xxpp(vmap(combine_tensors)(covs))
    mean = xpxp_to_xxpp(vmap(combine_tensors)(means, ndim_ds=1))
    weight = vmap(multi_kron)(weights)
    return BosonicState([cov, mean, weight], nmode, cutoff)
