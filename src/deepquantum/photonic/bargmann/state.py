"""Bargmann-representation quantum states."""

from typing import Any

import torch
from torch import nn

from ...utils import apply_complex_fix
from .ansatz import PolyExpAnsatz, fock_to_ansatz, vacuum_ansatz
from .moments import gaussian_to_bargmann


class BargmannState(nn.Module):
    r"""A photonic state in the Bargmann (stellar) representation.

    The state is stored as a :class:`PolyExpAnsatz` ``(A, b, c)``. A ket on ``M`` modes has ``M``
    CV variables, one per mode in the order of :attr:`wires`. A density matrix has ``2M`` CV
    variables: all ket variables followed by all bra variables, so that

    .. math::

        K_\psi(z)=\sum_n\psi_n\frac{z^n}{\sqrt{n!}},\qquad
        K_\rho(z,w)=\sum_{n,m}\rho_{n,m}\frac{z^nw^m}{\sqrt{n!m!}}.

    Gaussian states have no derived variables; Fock states and (later) heralded states carry
    derived variables. The state is not normalized automatically.

    Args:
        state: The state. It can be ``'vac'``/``'zeros'``; a Gaussian state ``[cov, mean]`` (xxpp,
            current ``hbar``/``kappa``) or a ``GaussianState``; a Fock basis state such as ``[1, 0, 2]``;
            a Fock superposition ``[(amp, [n0, n1, ...]), ...]``; a ``FockState``; a batched Fock tensor
            (ket ``(B, C, ..., C)`` or density matrix ``(B, ket axes..., bra axes...)``); or a
            :class:`PolyExpAnsatz`. Default: ``'vac'``
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
        ansatz, nmode = self._to_ansatz(state, nmode, den_mat)
        self.nmode = nmode
        self.wires = tuple(range(nmode))
        self.cutoff = 5 if cutoff is None else int(cutoff)
        self._set_ansatz(ansatz)

    # ------------------------------------------------------------------ construction
    @staticmethod
    def _to_ansatz(state: Any, nmode: int | None, den_mat: bool) -> tuple[PolyExpAnsatz, int]:
        from ..state import FockState, GaussianState  # avoid a circular import

        if isinstance(state, BargmannState):
            if state.den_mat != den_mat:
                raise ValueError('den_mat does not match the representation of the given BargmannState')
            return state.ansatz, state.nmode
        if isinstance(state, PolyExpAnsatz):
            if nmode is None:
                nmode = state.n_cv // 2 if den_mat else state.n_cv
            if state.n_cv != (2 * nmode if den_mat else nmode):
                raise ValueError('The ansatz must have one CV variable per mode (two for density matrices)')
            return state, nmode
        if isinstance(state, str):
            if state not in ('vac', 'zeros'):
                raise ValueError(f'Unknown state {state!r}')
            nmode = 1 if nmode is None else nmode
            return vacuum_ansatz(2 * nmode if den_mat else nmode), nmode
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
            if tensor.ndim == 2 * nmode and den_mat:
                tensor = tensor.unsqueeze(0)
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
        return fock_to_ansatz(tensor, nvar), nmode

    @staticmethod
    def _from_fock_basis(pattern: list[int], den_mat: bool) -> PolyExpAnsatz:
        if any(n < 0 for n in pattern):
            raise ValueError('Photon numbers must be nonnegative')
        tensor = torch.zeros(1, *[n + 1 for n in pattern], dtype=torch.cfloat)
        tensor[(0, *pattern)] = 1
        if den_mat:
            tensor = torch.zeros(1, *[n + 1 for n in pattern * 2], dtype=torch.cfloat)
            tensor[(0, *(pattern * 2))] = 1
        return fock_to_ansatz(tensor, tensor.ndim - 1)

    @staticmethod
    def _from_moments(cov: torch.Tensor, mean: torch.Tensor, den_mat: bool) -> PolyExpAnsatz:
        a, b, c = gaussian_to_bargmann(cov, mean)
        nmode = cov.shape[-1] // 2
        batch = a.shape[0]
        if not den_mat:
            tol = 100 * torch.finfo(cov.dtype).eps
            if not bool((a[:, :nmode, nmode:].detach().abs() <= tol).all()):
                raise ValueError('A ket requires a pure Gaussian input; set den_mat=True for mixed inputs')
            # c_rho = |c_psi|^2; sqrt(c) fixes the unobservable global phase by c_psi > 0.
            a, b, c = a[:, :nmode, :nmode], b[:, :nmode], c.sqrt()
        nvar = a.shape[-1]
        return PolyExpAnsatz(a.reshape(batch, 1, nvar, nvar), b.reshape(batch, 1, nvar), c.to(a.dtype)[:, None], nvar)

    @classmethod
    def from_ansatz(
        cls, ansatz: PolyExpAnsatz, den_mat: bool, cutoff: int | None = None, wires: tuple[int, ...] | None = None
    ) -> 'BargmannState':
        """Wrap an ansatz whose CV variables follow the layout of this class."""
        state = cls(ansatz, den_mat=den_mat, cutoff=cutoff)
        if wires is not None:
            if len(wires) != state.nmode:
                raise ValueError('wires must label every mode of the state')
            state.wires = tuple(wires)
        return state

    def _set_ansatz(self, ansatz: PolyExpAnsatz) -> None:
        self.register_buffer('a', ansatz.a)
        self.register_buffer('b', ansatz.b)
        self.register_buffer('c', ansatz.c)
        self.n_cv = ansatz.n_cv

    def _apply(self, fn: Any) -> 'BargmannState':
        tensors = {name: self._buffers.pop(name) for name in ('a', 'b', 'c')}
        super()._apply(fn)
        for name, value in apply_complex_fix(fn, tensors).items():
            self.register_buffer(name, value)
        return self

    def _new(self, ansatz: PolyExpAnsatz, den_mat: bool | None = None) -> 'BargmannState':
        den_mat = self.den_mat if den_mat is None else den_mat
        return BargmannState.from_ansatz(ansatz, den_mat=den_mat, cutoff=self.cutoff, wires=self.wires)

    # ------------------------------------------------------------------ properties
    @property
    def ansatz(self) -> PolyExpAnsatz:
        """The underlying :class:`PolyExpAnsatz`."""
        return PolyExpAnsatz(self.a, self.b, self.c, self.n_cv)

    @property
    def representation(self) -> str:
        """``'dm'`` for density matrices, otherwise ``'ket'``."""
        return 'dm' if self.den_mat else 'ket'

    @property
    def batch_size(self) -> int:
        """Size of the batch axis."""
        return self.a.shape[0]

    def select(self, index: int) -> 'BargmannState':
        """Return the batch element ``index`` as a state with batch size 1."""
        sl = slice(index, index + 1)
        return self._new(PolyExpAnsatz(self.a[sl], self.b[sl], self.c[sl], self.n_cv))

    @staticmethod
    def cat(states: list['BargmannState']) -> 'BargmannState':
        """Concatenate states with identical structure along the batch axis."""
        first = states[0]
        a = torch.cat([s.a for s in states])
        b = torch.cat([s.b for s in states])
        c = torch.cat([s.c for s in states])
        ansatz = PolyExpAnsatz(a, b, c, first.n_cv)
        return first._new(ansatz)

    # ------------------------------------------------------------------ evolution
    def _positions(self, wires: list[int]) -> list[int]:
        if len(set(wires)) != len(wires):
            raise ValueError('Repeated wires')
        try:
            return [self.wires.index(w) for w in wires]
        except ValueError as exc:
            raise ValueError(f'Wires {list(wires)} are not all present in the state {self.wires}') from exc

    @staticmethod
    def _apply_kernel(state: PolyExpAnsatz, kernel: PolyExpAnsatz, positions: list[int]) -> PolyExpAnsatz:
        """Contract kernel inputs with the CV variables at ``positions`` and put the outputs there.

        ``kernel`` has ``len(positions)`` output variables followed by as many input variables.
        """
        k = len(positions)
        if kernel.n_cv != 2 * k:
            raise ValueError('The kernel must have one output and one input variable per position')
        dtype = torch.promote_types(state.dtype, kernel.dtype)
        state, kernel = state.to(dtype=dtype), kernel.to(dtype=dtype, device=state.device)
        joint = kernel.tensor_product(state)
        out = joint.contract([(k + j, 2 * k + p) for j, p in enumerate(positions)])
        current = list(positions) + [q for q in range(state.n_cv) if q not in positions]
        return out.reorder([current.index(q) for q in range(state.n_cv)])

    def apply_unitary(self, kernel: PolyExpAnsatz, wires: list[int]) -> 'BargmannState':
        r"""Apply a unitary kernel: :math:`U|\psi\rangle` or :math:`U\rho U^\dagger`."""
        pos = self._positions(list(wires))
        ansatz = self._apply_kernel(self.ansatz, kernel, pos)
        if self.den_mat:
            ansatz = self._apply_kernel(ansatz, kernel.conj(), [p + self.nmode for p in pos])
        return self._new(ansatz)

    def apply_channel(self, kernel: PolyExpAnsatz, wires: list[int]) -> 'BargmannState':
        """Apply a channel kernel with variables ``(z_out, w_out, u_in, v_in)``; requires a density matrix."""
        if not self.den_mat:
            raise ValueError('Channels require the density-matrix representation; use den_mat=True')
        pos = self._positions(list(wires))
        ansatz = self._apply_kernel(self.ansatz, kernel, pos + [p + self.nmode for p in pos])
        return self._new(ansatz)

    def to_dm(self) -> 'BargmannState':
        r"""Return :math:`|\psi\rangle\langle\psi|`; a density matrix is returned unchanged."""
        if self.den_mat:
            return self
        ansatz = self.ansatz
        return self._new(ansatz.tensor_product(ansatz.conj()), den_mat=True)

    # ------------------------------------------------------------------ readout
    def norm(self) -> torch.Tensor:
        r"""Squared norm :math:`\langle\psi|\psi\rangle` of a ket or trace of a density matrix, shape ``(B,)``.

        Computed by Gaussian integration, independently of any Fock cutoff.
        """
        ansatz = self.ansatz
        if not self.den_mat:
            ansatz = ansatz.tensor_product(ansatz.conj())
        value = ansatz.contract([(j, j + self.nmode) for j in range(self.nmode)]).evaluate()
        return value.real

    def to_fock(self, cutoff: int | None = None) -> torch.Tensor:
        """Fock amplitudes ``(B, C, ..., C)`` or density matrix ``(B, ket axes..., bra axes...)``.

        Axes follow :attr:`wires`. ``cutoff=C`` keeps photon numbers ``0, ..., C - 1``.
        """
        cutoff = self.cutoff if cutoff is None else int(cutoff)
        return self.ansatz.to_fock((cutoff,) * self.n_cv)

    def __repr__(self) -> str:
        return (
            f'BargmannState(nmode={self.nmode}, wires={self.wires}, representation={self.representation!r}, '
            f'batch={self.batch_size}, derived_shape={tuple(self.c.shape[2:])}, dtype={self.a.dtype})'
        )
