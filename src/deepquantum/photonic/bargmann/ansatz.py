r"""Poly-exponential Bargmann ansatz.

A :class:`PolyExpAnsatz` represents the function

.. math::

    F(z)=\sum_{l}\exp\!\left(\tfrac12 z^TA^{(l)}_{zz}z+b^{(l)T}_z z\right)
    \sum_k c^{(l)}_k\,G_k\!\left(A^{(l)}_{yy},\,b^{(l)}_y+A^{(l)}_{yz}z\right),

where :math:`z` are the ``n_cv`` continuous (CV) variables and :math:`y` are derived
variables that are differentiated at :math:`y=0`. :math:`G_k` is the renormalized Hermite
tensor :math:`\exp(\tfrac12y^TAy+\beta^Ty)=\sum_kG_k(A,\beta)\,y^k/\sqrt{k!}`, so that
``c`` pairs directly with Fock coefficients. A Gaussian kernel is the special case with
no derived variables.

Conventions (see ``Bargmann/claude_docs/01_bargmann_backend_requirements.md`` Section 6):

* ``A`` is complex symmetric; quadratic forms use the transpose, not the adjoint.
* Variables are ordered as all CV variables followed by all derived variables.
* Axis layout: ``A (B, L, N, N)``, ``b (B, L, N)``, ``c (B, L, *derived_shape)``, where ``B`` is
  the batch axis and ``L`` the axis of a coherent linear superposition (summed over).
* Integrals use the Bargmann measure :math:`d\mu(z)=e^{-|z|^2}d^2z/\pi`.

All methods return new objects; nothing is modified in place.
"""

import torch

from .integrals import gaussian_contract
from .recurrence import _fock_coefficients


class PolyExpAnsatz:
    r"""Batched poly-exponential Bargmann ansatz :math:`(A, b, c)`.

    Args:
        a: Complex symmetric tensor of shape ``(B, L, N, N)``.
        b: Complex tensor of shape ``(B, L, N)``.
        c: Complex tensor of shape ``(B, L, *derived_shape)`` with ``N - n_cv`` derived axes.
        n_cv: Number of CV variables, which come first.
    """

    def __init__(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, n_cv: int) -> None:
        if a.ndim != 4 or a.shape[-1] != a.shape[-2]:
            raise ValueError('a must have shape (B, L, N, N)')
        nvar = a.shape[-1]
        if b.shape != a.shape[:-1]:
            raise ValueError('b must have shape (B, L, N), matching a')
        if c.ndim < 2 or c.shape[:2] != a.shape[:2]:
            raise ValueError('c must have shape (B, L, *derived_shape), matching a')
        if not 0 <= n_cv <= nvar or c.ndim - 2 != nvar - n_cv:
            raise ValueError('c must have one axis per derived variable')
        if not (a.is_complex() and a.dtype == b.dtype == c.dtype):
            raise ValueError('a, b and c must share one complex dtype')
        if not a.device == b.device == c.device:
            raise ValueError('a, b and c must be on the same device')
        self.a = a
        self.b = b
        self.c = c
        self.n_cv = int(n_cv)

    @property
    def batch_size(self) -> int:
        """Size of the batch axis."""
        return self.a.shape[0]

    @property
    def nlin(self) -> int:
        """Number of terms in the linear superposition."""
        return self.a.shape[1]

    @property
    def nvar(self) -> int:
        """Total number of variables."""
        return self.a.shape[-1]

    @property
    def n_derived(self) -> int:
        """Number of derived variables."""
        return self.nvar - self.n_cv

    @property
    def derived_shape(self) -> tuple[int, ...]:
        """Shape of the derived axes of ``c``."""
        return tuple(self.c.shape[2:])

    @property
    def dtype(self) -> torch.dtype:
        """Complex dtype."""
        return self.a.dtype

    @property
    def device(self) -> torch.device:
        """Device."""
        return self.a.device

    def to(self, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> 'PolyExpAnsatz':
        """Return a copy with the given complex dtype and/or device."""
        return PolyExpAnsatz(
            self.a.to(device=device, dtype=dtype),
            self.b.to(device=device, dtype=dtype),
            self.c.to(device=device, dtype=dtype),
            self.n_cv,
        )

    def expand_batch(self, batch: int) -> 'PolyExpAnsatz':
        """Broadcast a batch axis of size 1 to ``batch``."""
        if self.batch_size == batch:
            return self
        if self.batch_size != 1:
            raise ValueError(f'Cannot broadcast batch size {self.batch_size} to {batch}')
        return PolyExpAnsatz(
            self.a.expand(batch, *self.a.shape[1:]),
            self.b.expand(batch, *self.b.shape[1:]),
            self.c.expand(batch, *self.c.shape[1:]),
            self.n_cv,
        )

    def conj(self) -> 'PolyExpAnsatz':
        r"""Complex-conjugate all coefficients, giving the kernel :math:`F^\#` of the bra side."""
        return PolyExpAnsatz(self.a.conj(), self.b.conj(), self.c.conj(), self.n_cv)

    def _permute_vars(self, perm: list[int]) -> 'PolyExpAnsatz':
        """Permute variables of ``A`` and ``b`` only; ``c`` must already match the new derived order."""
        idx = torch.tensor(perm, dtype=torch.long, device=self.device)
        a = self.a[..., idx, :][..., :, idx]
        b = self.b[..., idx]
        return PolyExpAnsatz(a, b, self.c, self.n_cv)

    def reorder(self, perm: list[int]) -> 'PolyExpAnsatz':
        """Reorder CV variables: new CV variable ``i`` is old CV variable ``perm[i]``."""
        if sorted(perm) != list(range(self.n_cv)):
            raise ValueError('perm must be a permutation of the CV variables')
        if list(perm) == list(range(self.n_cv)):
            return self
        return self._permute_vars(list(perm) + list(range(self.n_cv, self.nvar)))

    def tensor_product(self, other: 'PolyExpAnsatz') -> 'PolyExpAnsatz':
        """Tensor product ``F(x) G(y)``.

        CV variables of ``self`` come first, then those of ``other``; derived variables follow
        in the same order. Batches broadcast; superposition axes form an outer product.
        """
        if self.dtype != other.dtype or self.device != other.device:
            raise ValueError('Tensor product requires a common dtype and device')
        batch = max(self.batch_size, other.batch_size)
        f, g = self.expand_batch(batch), other.expand_batch(batch)
        l1, l2 = f.nlin, g.nlin
        n1, n2 = f.nvar, g.nvar
        a = f.a.new_zeros(batch, l1, l2, n1 + n2, n1 + n2)
        a[..., :n1, :n1] = f.a[:, :, None]
        a[..., n1:, n1:] = g.a[:, None]
        b = torch.cat([f.b[:, :, None].expand(-1, -1, l2, -1), g.b[:, None].expand(-1, l1, -1, -1)], dim=-1)
        d1, d2 = f.derived_shape, g.derived_shape
        c = f.c.reshape(batch, l1, 1, *d1, *([1] * len(d2))) * g.c.reshape(batch, 1, l2, *([1] * len(d1)), *d2)
        out = PolyExpAnsatz(
            a.reshape(batch, l1 * l2, n1 + n2, n1 + n2),
            b.reshape(batch, l1 * l2, n1 + n2),
            c.reshape(batch, l1 * l2, *d1, *d2),
            f.n_cv + g.n_cv,
        )
        perm = (
            list(range(f.n_cv))
            + [n1 + j for j in range(g.n_cv)]
            + list(range(f.n_cv, n1))
            + [n1 + j for j in range(g.n_cv, n2)]
        )
        return out._permute_vars(perm)

    def contract(self, pairs: list[tuple[int, int]]) -> 'PolyExpAnsatz':
        """Integrate pairs of CV variables with the Bargmann measure.

        The remaining CV variables keep their relative order.
        """
        if any(i >= self.n_cv or j >= self.n_cv for i, j in pairs):
            raise ValueError('Only CV variables can be integrated')
        batch, nlin, nvar = self.batch_size, self.nlin, self.nvar
        a = self.a.reshape(batch * nlin, nvar, nvar)
        b = self.b.reshape(batch * nlin, nvar)
        a, b, factor, _ = gaussian_contract(a, b, pairs)
        nkeep = a.shape[-1]
        c = self.c * factor.reshape(batch, nlin, *([1] * self.n_derived))
        return PolyExpAnsatz(
            a.reshape(batch, nlin, nkeep, nkeep), b.reshape(batch, nlin, nkeep), c, self.n_cv - 2 * len(pairs)
        )

    def to_fock(self, cutoffs: tuple[int, ...]) -> torch.Tensor:
        r"""Fock coefficients :math:`\psi_n=\sum_k c_k G_{(n,k)}(A,b)`, summed over the superposition axis.

        Args:
            cutoffs: Output dimension of each CV variable.

        Returns:
            Tensor of shape ``(B, *cutoffs)``.
        """
        cutoffs = tuple(int(x) for x in cutoffs)
        if len(cutoffs) != self.n_cv or any(x < 1 for x in cutoffs):
            raise ValueError('One positive cutoff is required for each CV variable')
        batch, nlin, nvar = self.batch_size, self.nlin, self.nvar
        derived = self.derived_shape
        bounds = cutoffs + derived
        ones = self.a.new_ones(batch * nlin)
        # Explicit sizes: with no variables left (nvar == 0) a reshape with -1 would be ambiguous.
        a = self.a.reshape(batch * nlin, nvar, nvar)
        b = self.b.reshape(batch * nlin, nvar)
        g = _fock_coefficients(a, b, ones, bounds)
        g = g.reshape(batch, nlin, *bounds)
        c = self.c.reshape(batch, nlin, *([1] * self.n_cv), *derived)
        g = g * c
        if derived:
            g = g.sum(dim=tuple(range(2 + self.n_cv, 2 + self.n_cv + len(derived))))
        return g.sum(1)

    def evaluate(self) -> torch.Tensor:
        """Value of an ansatz without CV variables, shape ``(B,)``."""
        if self.n_cv:
            raise ValueError('evaluate() requires all CV variables to be integrated')
        return self.to_fock(())

    def __repr__(self) -> str:
        return (
            f'PolyExpAnsatz(batch={self.batch_size}, nlin={self.nlin}, n_cv={self.n_cv}, '
            f'derived_shape={self.derived_shape}, dtype={self.dtype})'
        )


def fock_to_ansatz(tensor: torch.Tensor, nvar: int) -> PolyExpAnsatz:
    r"""Represent a batched Fock tensor with ``nvar`` axes exactly.

    Uses :math:`A=\left(\begin{smallmatrix}0&I\\I&0\end{smallmatrix}\right)`, :math:`b=0` and
    ``c`` equal to the tensor, so that :math:`F(z)=\sum_n c_n z^n/\sqrt{n!}`.

    Args:
        tensor: Complex tensor of shape ``(B, d_1, ..., d_nvar)``.
        nvar: Number of Fock axes (CV variables).
    """
    if tensor.ndim != nvar + 1:
        raise ValueError('The Fock tensor must have a batch axis plus one axis per variable')
    eye = torch.eye(nvar, dtype=tensor.dtype, device=tensor.device)
    zero = torch.zeros_like(eye)
    a = torch.cat([torch.cat([zero, eye], -1), torch.cat([eye, zero], -1)], -2)
    batch = tensor.shape[0]
    a = a.expand(batch, 1, 2 * nvar, 2 * nvar)
    b = tensor.new_zeros(batch, 1, 2 * nvar)
    return PolyExpAnsatz(a, b, tensor.unsqueeze(1), nvar)


def vacuum_ansatz(nvar: int, dtype: torch.dtype = torch.cfloat, device: torch.device | str | None = None):
    """Vacuum kernel ``F = 1`` on ``nvar`` CV variables (a ket on ``nvar`` modes or a DM on ``nvar / 2``)."""
    a = torch.zeros(1, 1, nvar, nvar, dtype=dtype, device=device)
    b = torch.zeros(1, 1, nvar, dtype=dtype, device=device)
    c = torch.ones(1, 1, dtype=dtype, device=device)
    return PolyExpAnsatz(a, b, c, nvar)

