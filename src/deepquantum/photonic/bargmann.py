r"""Bargmann mathematical representations and numerical helpers.

Quantum states live in :mod:`deepquantum.photonic.state`; this module supplies
poly-exponential algebra, Fock readout, Gaussian conversion and conditional projection.
Concrete gate and channel representations live with their operation classes.

Variables use free axes followed by auxiliary axes. Density kernels use ket variables
before bra variables; operator kernels use output variables before input variables.
"""

import math
from functools import lru_cache

import torch
from torch.nn import functional

import deepquantum.photonic as dqp

from .qmath import quadrature_to_ladder


def gaussian_contract(
    mat_a: torch.Tensor, vec_b: torch.Tensor, pairs: list[tuple[int, int]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    r"""Integrate pairs of variables of a batched Gaussian kernel ``exp(x.T A x / 2 + b.T x)``.

    Split the variables into the kept set :math:`o` and the integrated set
    :math:`i=(u_1,\dots,u_p,v_1,\dots,v_p)`. With
    :math:`\Omega=\left(\begin{smallmatrix}0&I_p\\I_p&0\end{smallmatrix}\right)` and
    :math:`M=\Omega-A_{ii}`:

    .. math::

        A'=A_{oo}+A_{oi}M^{-1}A_{io},\qquad
        b'=b_o+A_{oi}M^{-1}b_i,\qquad
        f=\frac{\exp(b_i^TM^{-1}b_i/2)}{\sqrt{\det(I-\Omega A_{ii})}}.

    The scalar ``f`` multiplies the coefficient tensor of the kernel. Integrate pairs
    one at a time so each square root follows the analytic branch; taking the principal
    square root of the combined complex determinant can introduce a sign error.
    Auxiliary variables are never integrated; they are simply kept, so delayed
    derivatives commute with the integral.

    Args:
        mat_a: Complex symmetric tensor of shape ``(K, N, N)``.
        vec_b: Complex tensor of shape ``(K, N)``.
        pairs: Pairs of distinct variable indices to integrate.

    Returns:
        ``(a', b', f, keep)`` where ``a'`` has shape ``(K, n, n)``, ``b'`` has shape
        ``(K, n)``, ``f`` has shape ``(K,)`` and ``keep`` lists the original indices of the
        kept variables, in their original order.
    """
    nvar = mat_a.shape[-1]
    first = [int(i) for i, _ in pairs]
    second = [int(j) for _, j in pairs]
    integrated = first + second
    if len(set(integrated)) != len(integrated) or any(i < 0 or i >= nvar for i in integrated):
        raise ValueError('Integrated variable pairs must use distinct valid indices')
    remaining = list(range(nvar))
    factor = mat_a.new_ones(mat_a.shape[0])
    omega = mat_a.new_tensor([[0, 1], [1, 0]])
    eye = torch.eye(2, dtype=mat_a.dtype, device=mat_a.device)
    for first, second in pairs:
        pair_idx = [remaining.index(first), remaining.index(second)]
        keep_positions = [k for k in range(len(remaining)) if k not in pair_idx]
        idx = torch.tensor(pair_idx, dtype=torch.long, device=mat_a.device)
        keep_idx = torch.tensor(keep_positions, dtype=torch.long, device=mat_a.device)
        mat_a_ii = mat_a[:, idx][:, :, idx]
        mat_a_oi = mat_a[:, keep_idx][:, :, idx]
        mat_a_oo = mat_a[:, keep_idx][:, :, keep_idx]
        vec_b_i = vec_b[:, idx]
        vec_b_o = vec_b[:, keep_idx]
        rhs = torch.cat([mat_a_oi.mT, vec_b_i.unsqueeze(-1)], dim=-1)
        try:
            sol = torch.linalg.solve(omega - mat_a_ii, rhs)
        except RuntimeError as exc:
            if 'singular' not in str(exc).lower():
                raise
            raise ValueError('Singular Gaussian integral; the kernel is not integrable over these variables') from exc
        mat_a = mat_a_oo + mat_a_oi @ sol[..., :-1]
        mat_a = (mat_a + mat_a.mT) / 2
        vec_b = vec_b_o + (mat_a_oi @ sol[..., -1:]).squeeze(-1)
        det = torch.linalg.det(eye - omega @ mat_a_ii)
        factor = factor * torch.exp((vec_b_i * sol[..., -1]).sum(-1) / 2) / torch.sqrt(det)
        remaining = [remaining[k] for k in keep_positions]
    return mat_a, vec_b, factor, remaining


# Bounds include recurrence indices and intermediates, not only the output.
_max_work_bytes = 256 * 1024**2


def _check_work(bounds: tuple[int, ...], batch: int, element_size: int, polynomial_size: int = 1) -> None:
    size = math.prod(bounds)
    # Conservative working-set estimate; autograd/framework overhead may add more.
    estimate = size * (80 * max(len(bounds), 1) + 4 * batch * element_size * polynomial_size)
    if estimate > _max_work_bytes:
        raise MemoryError('Bargmann recurrence exceeds the working-set limit; reduce cutoff or herald photon numbers')


@lru_cache(maxsize=4)
def _recurrence_plan(bounds: tuple[int, ...]) -> tuple:
    """Cache only discrete CPU indices and constant weights, never an autograd graph.

    See https://arxiv.org/pdf/2504.10455v3 Section A.3.1, Eq. (57).
    """
    ranges = [torch.arange(bound, device='cpu') for bound in bounds]
    states = torch.cartesian_prod(*ranges).reshape(-1, len(bounds))
    strides = torch.tensor([math.prod(bounds[i + 1 :]) for i in range(len(bounds))], device='cpu')
    totals = states.sum(-1)
    layers = []
    for degree in range(1, int(totals.max()) + 1):
        ids = torch.where(totals == degree)[0]
        selected = states[ids]
        mode = (selected > 0).long().argmax(-1)
        row = torch.arange(len(ids), device='cpu')
        # Eq. (57), with one pivot i per target; Eq. (58) averages over all pivots.
        # Target t = selected, i = mode, k = t - e_i; dependencies have lower total degree.
        previous = selected.clone()
        previous[row, mode] -= 1
        previous_ids = (previous * strides).sum(-1)
        valid = previous > 0
        lower = (previous_ids[:, None] - strides).clamp_min(0)
        number = selected[row, mode].double()
        # weights_j = sqrt((t_j - delta_ij) / t_i); negative-index terms have zero weight.
        weights = (previous / number[:, None]).sqrt() * valid
        layers.append((ids, mode, previous_ids, lower, weights, number.rsqrt()))
    return tuple(layers)


def _device_layers(bounds: tuple[int, ...], mat_a: torch.Tensor):
    """Place the cached recurrence indices and weights on the coefficient device."""
    for layer in _recurrence_plan(bounds):
        yield tuple(
            t.to(device=mat_a.device, dtype=mat_a.real.dtype if t.is_floating_point() else t.dtype) for t in layer
        )


def _fock_coefficients(
    mat_a: torch.Tensor, vec_b: torch.Tensor, coeff_c: torch.Tensor, bounds: tuple[int, ...]
) -> torch.Tensor:
    r"""Evaluate normalized Taylor coefficients on a rectangular index set.

    Uses Eqs. (54)-(55) for coefficient normalization and Eq. (57) for recursion.

    For multi-indices :math:`k`, with :math:`k!=\prod_j k_j!`, use

    .. math::

        G(\xi)=c e^{\xi^T A\xi/2+b^T\xi}
        =\sum_k g_k\frac{\xi^k}{\sqrt{k!}},\qquad
        \partial_iG=(b_i+\sum_j A_{ij}\xi_j)G.

    Matching powers gives Eq. (57) of the paper:

    .. math::

        g_{k+e_i}=\frac{b_i g_k+\sum_j A_{ij}\sqrt{k_j}\,g_{k-e_j}}
        {\sqrt{k_i+1}},\qquad g_0=c.

    Negative-index coefficients are zero. In code, the target is :math:`t=k+e_i`:
    ``inverse_sqrt`` is :math:`1/\sqrt{t_i}` and ``weights`` contains
    :math:`\sqrt{(t_j-\delta_{ij})/t_i}`. Thus ``terms`` implements the two summands.
    For a pure-state kernel :math:`g_n=\psi_n`; for a density kernel :math:`g_{(n,m)}=\rho_{n,m}`.
    These are Fock coefficients, not ordinary Taylor coefficients :math:`g_k/\sqrt{k!}`;
    no further factorial division or state normalization is needed here.

    See https://arxiv.org/pdf/2504.10455v3 Section A.3.1, Eqs. (54)-(55) and (57).
    """
    batch = mat_a.shape[0]
    if not bounds:
        return coeff_c.to(mat_a.dtype)
    _check_work(bounds, batch, mat_a.element_size())
    values = torch.cat((coeff_c.to(mat_a.dtype)[:, None], mat_a.new_zeros(batch, math.prod(bounds) - 1)), dim=-1)
    for ids, mode, previous, lower, weights, inverse_sqrt in _device_layers(bounds, mat_a):
        terms = (mat_a[:, mode, :] * weights * values[:, lower]).sum(-1)
        terms = terms + vec_b[:, mode] * inverse_sqrt * values[:, previous]
        values = values.index_copy(1, ids, terms)
    return values.reshape(batch, *bounds)


class BargmannPolyExp:
    r"""A finite sum of polynomial-exponential Bargmann functions.

    A :class:`BargmannPolyExp` represents the function

    .. math::

        F(z)=\sum_{l}\exp\!\left(\tfrac12 z^TA^{(l)}_{zz}z+b^{(l)T}_z z\right)
        \sum_k c^{(l)}_k\,G_k\!\left(A^{(l)}_{yy},\,b^{(l)}_y+A^{(l)}_{yz}z\right),

    where :math:`z` are the ``nfree`` free complex variables and :math:`y` are auxiliary
    variables that are differentiated at :math:`y=0`. :math:`G_k` is the renormalized Hermite
    tensor :math:`\exp(\tfrac12y^TAy+\beta^Ty)=\sum_kG_k(A,\beta)\,y^k/\sqrt{k!}`, so that
    ``coeff_c`` pairs directly with Fock coefficients. A Gaussian kernel is the special case with
    no auxiliary variables.

    Conventions:

    * ``A`` is complex symmetric; quadratic forms use the transpose, not the adjoint.
    * Variables are ordered as all free variables followed by all auxiliary variables.
    * Axis layout: ``mat_a (B, L, N, N)``, ``vec_b (B, L, N)``, ``coeff_c (B, L, *aux_shape)``, where ``B`` is
      the batch axis and ``L`` the axis of a coherent linear superposition (summed over).
    * Integrals use the Bargmann measure :math:`d\mu(z)=e^{-|z|^2}d^2z/\pi`.

    Algebraic operations do not modify input coefficients in place.
    Auxiliary axes of length one represent substitution at zero and are eliminated.

    Parameters must be finite. Symmetry is checked against the transpose (not the
    conjugate transpose), with absolute and relative tolerances of 100 dtype eps.

    Args:
        mat_a: Complex symmetric tensor of shape ``(B, L, N, N)``.
        vec_b: Complex tensor of shape ``(B, L, N)``.
        coeff_c: Complex tensor of shape ``(B, L, *aux_shape)`` with one axis per auxiliary variable.
    """

    def __init__(self, mat_a: torch.Tensor, vec_b: torch.Tensor, coeff_c: torch.Tensor) -> None:
        if mat_a.ndim != 4 or mat_a.shape[-1] != mat_a.shape[-2]:
            raise ValueError('mat_a must have shape (B, L, N, N)')
        nvar = mat_a.shape[-1]
        if vec_b.shape != mat_a.shape[:-1]:
            raise ValueError('vec_b must have shape (B, L, N), matching mat_a')
        if coeff_c.ndim < 2 or coeff_c.shape[:2] != mat_a.shape[:2]:
            raise ValueError('coeff_c must have shape (B, L, *aux_shape), matching mat_a')
        if coeff_c.ndim - 2 > nvar or any(size == 0 for size in coeff_c.shape):
            raise ValueError('coeff_c must have nonempty axes, with at most one auxiliary axis per variable')
        if not (mat_a.is_complex() and mat_a.dtype == vec_b.dtype == coeff_c.dtype):
            raise ValueError('mat_a, vec_b and coeff_c must share one complex dtype')
        if not mat_a.device == vec_b.device == coeff_c.device:
            raise ValueError('mat_a, vec_b and coeff_c must be on the same device')
        for name, value in (('mat_a', mat_a), ('vec_b', vec_b), ('coeff_c', coeff_c)):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'{name} must contain only finite values')
        tolerance = 100 * torch.finfo(mat_a.real.dtype).eps
        if not torch.allclose(mat_a, mat_a.mT, atol=tolerance, rtol=tolerance):
            raise ValueError('mat_a must be complex symmetric (mat_a == mat_a.T), not necessarily Hermitian')
        self._set_tensors(mat_a, vec_b, coeff_c)

    @classmethod
    def _from_tensors(cls, mat_a: torch.Tensor, vec_b: torch.Tensor, coeff_c: torch.Tensor) -> 'BargmannPolyExp':
        """Construct an internal result whose tensor layout and dtype are already established."""
        result = cls.__new__(cls)
        result._set_tensors(mat_a, vec_b, coeff_c)
        return result

    def _set_tensors(self, mat_a: torch.Tensor, vec_b: torch.Tensor, coeff_c: torch.Tensor) -> None:
        """Store coefficients, eliminating auxiliary variables with only a zeroth derivative."""
        aux_shape = coeff_c.shape[2:]
        if 1 in aux_shape:
            nfree = mat_a.shape[-1] - len(aux_shape)
            keep = list(range(nfree)) + [nfree + i for i, size in enumerate(aux_shape) if size > 1]
            mat_a = mat_a[..., keep, :][..., keep]
            vec_b = vec_b[..., keep]
            coeff_c = coeff_c[(slice(None), slice(None), *(0 if size == 1 else slice(None) for size in aux_shape))]
        self.mat_a, self.vec_b, self.coeff_c = mat_a, vec_b, coeff_c

    @property
    def nfree(self) -> int:
        """Number of free variables, excluding the auxiliary derivative variables."""
        return self.nvar - self.naux

    @property
    def batch_size(self) -> int:
        """Size of the batch axis."""
        return self.mat_a.shape[0]

    @property
    def nlin(self) -> int:
        """Number of terms in the linear superposition."""
        return self.mat_a.shape[1]

    @property
    def nvar(self) -> int:
        """Total number of variables."""
        return self.mat_a.shape[-1]

    @property
    def naux(self) -> int:
        """Number of auxiliary variables."""
        return self.coeff_c.ndim - 2

    @property
    def aux_shape(self) -> tuple[int, ...]:
        """Shape of the auxiliary axes of ``coeff_c``."""
        return tuple(self.coeff_c.shape[2:])

    @property
    def dtype(self) -> torch.dtype:
        """Complex dtype."""
        return self.mat_a.dtype

    @property
    def device(self) -> torch.device:
        """Device."""
        return self.mat_a.device

    def to(self, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> 'BargmannPolyExp':
        """Convert the complex dtype or device, returning self if neither changes."""
        if dtype is not None and not dtype.is_complex:
            raise ValueError('Bargmann coefficients require a complex dtype')
        if (dtype is None or dtype == self.dtype) and (device is None or torch.device(device) == self.device):
            return self
        return BargmannPolyExp._from_tensors(
            self.mat_a.to(device=device, dtype=dtype),
            self.vec_b.to(device=device, dtype=dtype),
            self.coeff_c.to(device=device, dtype=dtype),
        )

    def expand_batch(self, batch: int) -> 'BargmannPolyExp':
        """Broadcast a batch axis of size 1 to ``batch``."""
        if self.batch_size == batch:
            return self
        if self.batch_size != 1:
            raise ValueError(f'Cannot broadcast batch size {self.batch_size} to {batch}')
        return BargmannPolyExp._from_tensors(
            self.mat_a.expand(batch, *self.mat_a.shape[1:]),
            self.vec_b.expand(batch, *self.vec_b.shape[1:]),
            self.coeff_c.expand(batch, *self.coeff_c.shape[1:]),
        )

    def conj(self) -> 'BargmannPolyExp':
        r"""Complex-conjugate all coefficients, giving the kernel :math:`F^\#` of the bra side."""
        return BargmannPolyExp._from_tensors(self.mat_a.conj(), self.vec_b.conj(), self.coeff_c.conj())

    def reorder(self, perm: list[int]) -> 'BargmannPolyExp':
        """Reorder free variables: new variable ``i`` is old variable ``perm[i]``."""
        if sorted(perm) != list(range(self.nfree)):
            raise ValueError('perm must be a permutation of the free variables')
        if list(perm) == list(range(self.nfree)):
            return self
        idx = torch.tensor(list(perm) + list(range(self.nfree, self.nvar)), dtype=torch.long, device=self.device)
        mat_a = self.mat_a[..., idx, :][..., :, idx]
        vec_b = self.vec_b[..., idx]
        return BargmannPolyExp._from_tensors(mat_a, vec_b, self.coeff_c)

    def tensor_product(self, other: 'BargmannPolyExp') -> 'BargmannPolyExp':
        """Tensor product ``F(x) G(y)``.

        Free variables of ``self`` come first, then those of ``other``; auxiliary variables follow
        in the same order. Batches broadcast; superposition axes form an outer product.
        """
        if self.dtype != other.dtype or self.device != other.device:
            raise ValueError('Tensor product requires a common dtype and device')
        batch = max(self.batch_size, other.batch_size)
        f, g = self.expand_batch(batch), other.expand_batch(batch)
        l1, l2 = f.nlin, g.nlin
        n1, n2 = f.nvar, g.nvar
        mat_a = f.mat_a.new_zeros(batch, l1, l2, n1 + n2, n1 + n2)
        mat_a[..., :n1, :n1] = f.mat_a[:, :, None]
        mat_a[..., n1:, n1:] = g.mat_a[:, None]
        vec_b = torch.cat([f.vec_b[:, :, None].expand(-1, -1, l2, -1), g.vec_b[:, None].expand(-1, l1, -1, -1)], dim=-1)
        d1, d2 = f.aux_shape, g.aux_shape
        coeff_c = f.coeff_c.reshape(batch, l1, 1, *d1, *([1] * len(d2))) * g.coeff_c.reshape(
            batch, 1, l2, *([1] * len(d1)), *d2
        )
        perm = (
            list(range(f.nfree))
            + [n1 + j for j in range(g.nfree)]
            + list(range(f.nfree, n1))
            + [n1 + j for j in range(g.nfree, n2)]
        )
        mat_a = mat_a.reshape(batch, l1 * l2, n1 + n2, n1 + n2)
        vec_b = vec_b.reshape(batch, l1 * l2, n1 + n2)
        return BargmannPolyExp._from_tensors(
            mat_a[..., perm, :][..., perm], vec_b[..., perm], coeff_c.reshape(batch, l1 * l2, *d1, *d2)
        )

    def contract(self, pairs: list[tuple[int, int]]) -> 'BargmannPolyExp':
        """Integrate pairs of free variables with the Bargmann measure.

        The remaining free variables keep their relative order.
        """
        if any(i >= self.nfree or j >= self.nfree for i, j in pairs):
            raise ValueError('Only free variables can be integrated')
        batch, nlin, nvar = self.batch_size, self.nlin, self.nvar
        mat_a = self.mat_a.reshape(batch * nlin, nvar, nvar)
        vec_b = self.vec_b.reshape(batch * nlin, nvar)
        mat_a, vec_b, factor, _ = gaussian_contract(mat_a, vec_b, pairs)
        nkeep = mat_a.shape[-1]
        coeff_c = self.coeff_c * factor.reshape(batch, nlin, *([1] * self.naux))
        return BargmannPolyExp._from_tensors(
            mat_a.reshape(batch, nlin, nkeep, nkeep), vec_b.reshape(batch, nlin, nkeep), coeff_c
        )

    def to_fock(self, cutoffs: tuple[int, ...]) -> torch.Tensor:
        r"""Fock coefficients :math:`\psi_n=\sum_k c_k G_{(n,k)}(A,b)`, summed over the superposition axis.

        Args:
            cutoffs: Output dimension of each free variable.

        Returns:
            Tensor of shape ``(B, *cutoffs)``.
        """
        cutoffs = tuple(int(x) for x in cutoffs)
        if len(cutoffs) != self.nfree or any(x < 1 for x in cutoffs):
            raise ValueError('One positive cutoff is required for each free variable')
        batch, nlin, nvar = self.batch_size, self.nlin, self.nvar
        aux_shape = self.aux_shape
        bounds = cutoffs + aux_shape
        ones = self.mat_a.new_ones(batch * nlin)
        # Explicit sizes: with no variables left (nvar == 0) a reshape with -1 would be ambiguous.
        mat_a = self.mat_a.reshape(batch * nlin, nvar, nvar)
        vec_b = self.vec_b.reshape(batch * nlin, nvar)
        g = _fock_coefficients(mat_a, vec_b, ones, bounds)
        g = g.reshape(batch, nlin, *bounds)
        coeff_c = self.coeff_c.reshape(batch, nlin, *([1] * self.nfree), *aux_shape)
        g = g * coeff_c
        if aux_shape:
            g = g.sum(dim=tuple(range(2 + self.nfree, 2 + self.nfree + self.naux)))
        return g.sum(1)

    def evaluate(self) -> torch.Tensor:
        """Value after eliminating all free variables, shape ``(B,)``."""
        if self.nfree:
            raise ValueError('evaluate() requires all free variables to be integrated')
        return self.to_fock(())

    def __repr__(self) -> str:
        return (
            f'BargmannPolyExp(batch={self.batch_size}, nlin={self.nlin}, nfree={self.nfree}, '
            f'aux_shape={self.aux_shape}, dtype={self.dtype})'
        )


def fock_to_bargmann(tensor: torch.Tensor, nvar: int) -> BargmannPolyExp:
    r"""Represent a batched Fock tensor with ``nvar`` axes exactly.

    Uses :math:`A=\left(\begin{smallmatrix}0&I\\I&0\end{smallmatrix}\right)`, :math:`b=0` and
    ``c`` equal to the tensor, so that :math:`F(z)=\sum_n c_n z^n/\sqrt{n!}`.

    Args:
        tensor: Complex tensor of shape ``(B, d_1, ..., d_nvar)``.
        nvar: Number of Fock axes (free variables).
    """
    if tensor.ndim != nvar + 1:
        raise ValueError('The Fock tensor must have a batch axis plus one axis per variable')
    if not tensor.is_complex() or not bool(torch.isfinite(tensor).all()):
        raise ValueError('Fock coefficients must be finite complex values')
    eye = torch.eye(nvar, dtype=tensor.dtype, device=tensor.device)
    zero = torch.zeros_like(eye)
    mat_a = torch.cat([torch.cat([zero, eye], -1), torch.cat([eye, zero], -1)], -2)
    batch = tensor.shape[0]
    mat_a = mat_a.expand(batch, 1, 2 * nvar, 2 * nvar)
    vec_b = tensor.new_zeros(batch, 1, 2 * nvar)
    return BargmannPolyExp._from_tensors(mat_a, vec_b, tensor.unsqueeze(1))


def vacuum_bargmann(nvar: int, dtype: torch.dtype = torch.cfloat, device: torch.device | str | None = None):
    """Vacuum kernel ``F = 1`` on ``nvar`` free variables (a ket on ``nvar`` modes or a DM on ``nvar / 2``)."""
    mat_a = torch.zeros(1, 1, nvar, nvar, dtype=dtype, device=device)
    vec_b = torch.zeros(1, 1, nvar, dtype=dtype, device=device)
    coeff_c = torch.ones(1, 1, dtype=dtype, device=device)
    return BargmannPolyExp._from_tensors(mat_a, vec_b, coeff_c)


def gaussian_to_bargmann(
    cov: torch.Tensor, mean: torch.Tensor, den_mat: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Convert Gaussian moments to a ket or density kernel ``c exp(z.T A z / 2 + b.T z)``.

    Implements the Gaussian moment conversion with the unit and variable conventions below.

    With :math:`s=2\kappa^2/\hbar`, :math:`V=\mathrm{cov}` and :math:`d=\mathrm{mean}`, define
    :math:`W=2^{-1/2}\left(\begin{smallmatrix}I&iI\\I&-iI\end{smallmatrix}\right)`
    and :math:`J=\left(\begin{smallmatrix}0&I\\I&0\end{smallmatrix}\right)`:

    .. math::

        \Gamma=sV+I/2,\qquad Q=W\Gamma W^\dagger,\qquad \beta=\sqrt{s}Wd,

    .. math::

        A=(I-Q^{-1})J,\qquad b=Q^{-1}\beta,\qquad
        c=\frac{\exp(-\beta^\dagger Q^{-1}\beta/2)}{\sqrt{\det\Gamma}}.

    Here :math:`Q` is the Husimi covariance matrix, not the scalar Husimi Q function.
    For formal variables :math:`\xi=(z,w)`, the kernel is

    .. math::

        K_\rho(z,w)=c\exp(\xi^T A\xi/2+b^T\xi)
        =\sum_{n,m}\rho_{n,m}\frac{z^n w^m}{\sqrt{n!m!}}.

    On the coherent-state diagonal, :math:`z=\alpha^*` and :math:`w=\alpha`.
    Eq. (45) of the paper orders variables as :math:`(w,z)`; here they are :math:`(z,w)`.
    Thus :math:`A=J A_{\mathrm{paper}}J` and :math:`b=J b_{\mathrm{paper}}`.
    Use :math:`\hbar_{\mathrm{paper}}=\hbar/(2\kappa^2)` with the same moments.
    This explains the right-hand :math:`J` in :math:`A` and its absence in :math:`b`.
    The kernel form is Eq. (51) of the paper.

    See https://arxiv.org/pdf/2504.10455v3 Sections A.1, A.2 and A.4.1,
    Eqs. (45), (51) and (68)-(71).

    Args:
        cov: Real covariance tensor of shape ``(batch, 2M, 2M)`` in xxpp order.
        mean: Real mean tensor of shape ``(batch, 2M, 1)`` in the same units.
        den_mat: Return the density kernel. False requires pure inputs and fixes
            the ket's vacuum amplitude to be positive. Default: ``True``

    Returns:
        Complex symmetric ``A`` and complex ``b`` with shapes ``(batch, 2M, 2M)``
        and ``(batch, 2M)``, and real vacuum probabilities ``c`` of shape ``(batch,)``.
        The first M variables label ket indices; the last M label bra indices.
        For a ket, only the first M variables are retained and c is the vacuum amplitude.
    """
    if cov.ndim != 3 or cov.shape[-1] != cov.shape[-2] or cov.shape[-1] % 2 or cov.shape[-1] == 0:
        raise ValueError('cov must have shape (batch, 2M, 2M), with M > 0')
    if mean.ndim != 3 or mean.shape[-2:] != (cov.shape[-1], 1):
        raise ValueError('mean must have shape (batch, 2M, 1), matching cov')
    if cov.dtype not in (torch.float32, torch.float64) or mean.dtype != cov.dtype or mean.device != cov.device:
        raise ValueError('cov and mean must share a device and float32 or float64 dtype')
    if not bool(torch.isfinite(cov).all() & torch.isfinite(mean).all()):
        raise ValueError('Gaussian moments must be finite')
    try:
        batch = torch.broadcast_shapes(cov.shape[:1], mean.shape[:1])[0]
    except RuntimeError as exc:
        raise ValueError('cov and mean batch dimensions must be broadcastable') from exc
    cov = cov.expand(batch, -1, -1)
    mean = mean.expand(batch, -1, -1)
    identity = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
    # Q = W Gamma W^dagger with unitary W. Solve the real Gamma system:
    # this avoids a complex LU factorization (unsupported by PyTorch MPS).
    scale = 2 * dqp.kappa**2 / dqp.hbar
    gamma = scale * cov + identity / 2
    inverse = quadrature_to_ladder(torch.linalg.solve(gamma, identity.expand_as(gamma))) / scale
    beta = quadrature_to_ladder(mean)
    solved_mean = (inverse @ beta).squeeze(-1)
    nmode = cov.shape[-1] // 2
    exchange = identity.roll(nmode, dims=0).to(inverse.dtype)
    mat_a = (identity - inverse) @ exchange
    # A is complex symmetric (transpose, not conjugate transpose).
    mat_a = (mat_a + mat_a.mT) / 2
    log_c = -(torch.linalg.slogdet(gamma)[1] + (beta.squeeze(-1).conj() * solved_mean).sum(-1).real) / 2
    coeff_c = log_c.exp()
    if not den_mat:
        tolerance = 100 * torch.finfo(cov.dtype).eps
        if not bool((mat_a[:, :nmode, nmode:].detach().abs() <= tolerance).all()):
            raise ValueError('A ket requires a pure Gaussian input; set den_mat=True for mixed inputs')
        # The positive vacuum amplitude fixes the ket's otherwise undetermined global phase.
        mat_a, solved_mean, coeff_c = mat_a[:, :nmode, :nmode], solved_mean[:, :nmode], coeff_c.sqrt()
    return mat_a, solved_mean, coeff_c


def _gaussian_bargmann(mat_a: torch.Tensor, vec_b: torch.Tensor, coeff_c: torch.Tensor) -> BargmannPolyExp:
    """Add batch and superposition axes to an unbatched Gaussian triple."""
    nvar = mat_a.shape[-1]
    dtype = torch.promote_types(torch.promote_types(mat_a.dtype, vec_b.dtype), coeff_c.dtype)
    return BargmannPolyExp._from_tensors(
        mat_a.to(dtype).reshape(1, 1, nvar, nvar), vec_b.to(dtype).reshape(1, 1, nvar), coeff_c.to(dtype).reshape(1, 1)
    )


def passive_bargmann(unitary: torch.Tensor) -> BargmannPolyExp:
    r"""Passive linear-optical gate with :math:`Ua^\dagger U^\dagger=u^Ta^\dagger`: :math:`K=\exp(z^Tu\,w)`."""
    if unitary.ndim != 2 or unitary.shape[0] != unitary.shape[1]:
        raise ValueError('A passive mode transformation must be a square matrix')
    unitary = unitary + 0j
    nmode = unitary.shape[-1]
    zero = unitary.new_zeros(nmode, nmode)
    mat_a = torch.cat([torch.cat([zero, unitary], -1), torch.cat([unitary.mT, zero], -1)], -2)
    return _gaussian_bargmann(mat_a, unitary.new_zeros(2 * nmode), unitary.new_ones(()))


def _apply_bargmann(state: BargmannPolyExp, kernel: BargmannPolyExp, positions: list[int]) -> BargmannPolyExp:
    """Contract kernel inputs with the free variables at ``positions`` and put the outputs there.

    ``kernel`` has ``len(positions)`` output variables followed by as many input variables.
    """
    k = len(positions)
    if kernel.nfree != 2 * k:
        raise ValueError('The kernel must have one output and one input variable per position')
    dtype = torch.promote_types(state.dtype, kernel.dtype)
    state, kernel = state.to(dtype=dtype), kernel.to(dtype=dtype, device=state.device)
    joint = kernel.tensor_product(state)
    out = joint.contract([(k + j, 2 * k + p) for j, p in enumerate(positions)])
    current = list(positions) + [q for q in range(state.nfree) if q not in positions]
    return out.reorder([current.index(q) for q in range(state.nfree)])


def _project_pnr(
    polyexp: BargmannPolyExp, nmode: int, den_mat: bool, positions: tuple[int, ...], photons: tuple[int, ...]
) -> BargmannPolyExp:
    """Project validated modes onto Fock outcomes by converting free variables to auxiliary variables.

    Existing auxiliary axes precede the new measured axes. Density matrices project
    both ket and bra variables; retained variables keep their original order.
    No Fock expansion or normalization of retained modes is performed.
    """
    measured = list(positions)
    counts = photons
    if den_mat:
        measured += [p + nmode for p in positions]
        counts = photons * 2
    remaining = [p for p in range(polyexp.nfree) if p not in measured]
    source = polyexp
    perm = remaining + list(range(source.nfree, source.nvar)) + measured
    idx = torch.tensor(perm, device=source.device)
    bounds = tuple(h + 1 for h in counts)
    _check_work(source.aux_shape + bounds, source.batch_size * source.nlin, source.coeff_c.element_size())
    selector = source.coeff_c.new_zeros(bounds)
    selector[counts] = 1
    coeff_c = source.coeff_c.reshape(*source.coeff_c.shape, *([1] * len(counts))) * selector
    polyexp = BargmannPolyExp._from_tensors(source.mat_a[..., idx, :][..., idx], source.vec_b[..., idx], coeff_c)
    return polyexp


def _single_mode_density(
    mat_a: torch.Tensor,
    vec_b: torch.Tensor,
    coeff_c: torch.Tensor,
    wires: tuple[int, ...],
    herald: tuple[int, ...],
    cutoff: int,
) -> torch.Tensor:
    r"""Extract one remaining mode via a finite polynomial times a two-variable Gaussian.

    Let :math:`x=(z,w)` be the retained ket/bra variables and :math:`y` the measured ones.
    With :math:`K=A_{HH}`, :math:`L=A_{HR}` and :math:`G(x)=c e^{x^T A_{RR}x/2+b_R^Tx}`,

    .. math::

        K_\rho(x,y)=G(x)e^{y^TKy/2+(b_H+Lx)^Ty}
        =G(x)\sum_q P_q(x)\frac{y^q}{\sqrt{q!}}.

    The recurrence above, now with polynomial-valued linear terms, yields

    .. math::

        P_{q+e_i}(x)=\frac{[b_{H,i}+(Lx)_i]P_q(x)
        +\sum_j K_{ij}\sqrt{q_j}P_{q-e_j}(x)}{\sqrt{q_i+1}},\qquad P_0=1.

    PNR projection selects :math:`q=(h,h)`, so the conditional kernel is :math:`G(x)P_{(h,h)}(x)`.
    The polynomial has total degree at most :math:`D=2\sum_j h_j`.
    Derived by replacing the linear term in Eq. (57) of the paper, with a polynomial.
    Projection follows Eq. (25) on each density-kernel side. The polynomial
    recurrence and factorial convolution are local derivations, not numbered
    equations quoted from the paper.

    See https://arxiv.org/pdf/2504.10455v3 Sections 3.1 and A.3.1, Eqs. (25) and (57).
    """
    nmode = mat_a.shape[-1] // 2
    keep = next(i for i in range(nmode) if i not in wires)
    measured = list(wires) + [i + nmode for i in wires]
    output = [keep, keep + nmode]
    photons = herald * 2
    degree = sum(photons)
    bounds = tuple(h + 1 for h in photons)
    batch = mat_a.shape[0]
    _check_work(bounds, batch, mat_a.element_size(), (degree + 1) ** 2)
    _check_work((cutoff, cutoff), batch, mat_a.element_size())
    mat_a_measured = mat_a[:, measured][:, :, measured]
    mat_a_cross = mat_a[:, measured][:, :, output]
    vec_b_measured = vec_b[:, measured]
    # poly[..., u, v] stores ordinary monomial coefficients of z^u w^v, with P_0 = 1.
    poly = mat_a.new_ones(batch, 1, 1, 1)
    poly = functional.pad(poly, (0, degree, 0, degree))
    prev_ids = torch.zeros(1, dtype=torch.long, device=mat_a.device)
    old_ids, old_poly = prev_ids, poly
    layers = _device_layers(bounds, mat_a) if bounds else ()
    for ids, mode, previous, lower, weights, inverse_sqrt in layers:
        before = poly[:, torch.searchsorted(prev_ids, previous)]
        # Shifts multiply P by z or w in (b_H + L x) P.
        linear_z = functional.pad(before[:, :, :-1, :], (0, 0, 1, 0))
        linear_w = functional.pad(before[:, :, :, :-1], (1, 0, 0, 0))
        current = (
            mat_a_cross[:, mode, 0, None, None] * linear_z
            + mat_a_cross[:, mode, 1, None, None] * linear_w
            + vec_b_measured[:, mode, None, None] * before
        ) * inverse_sqrt[None, :, None, None]
        for j in range(len(photons)):
            positions = torch.searchsorted(old_ids, lower[:, j].contiguous()).clamp(max=len(old_ids) - 1)
            current = current + (mat_a_measured[:, mode, j] * weights[:, j])[..., None, None] * old_poly[:, positions]
        old_ids, old_poly, prev_ids, poly = prev_ids, poly, ids, current
    polynomial = poly[:, -1]
    base = _fock_coefficients(mat_a[:, output][:, :, output], vec_b[:, output], coeff_c, (cutoff, cutoff))
    # If P = sum p_uv z^u w^v and base_nm are normalized Fock coefficients of G,
    # result_nm = sum_{u<=n,v<=m} p_uv base_{n-u,m-v} sqrt(n! m! / ((n-u)! (m-v)!)).
    # lgamma(n + 1) = log(n!) supplies this change-of-basis factor stably.
    factorial = torch.lgamma(torch.arange(cutoff, dtype=mat_a.real.dtype, device=mat_a.device) + 1)
    result = mat_a.new_zeros(batch, cutoff, cutoff)
    for u in range(min(degree + 1, cutoff)):
        for v in range(min(degree - u + 1, cutoff)):
            # Do not skip terms based on a currently zero trainable displacement.
            factors = torch.exp(
                (factorial[u:] - factorial[: cutoff - u])[:, None] / 2
                + (factorial[v:] - factorial[: cutoff - v])[None, :] / 2
            )
            term = polynomial[:, u, v, None, None] * base[:, : cutoff - u, : cutoff - v] * factors
            result = result + functional.pad(term, (v, 0, u, 0))
    return result


def conditional_fock(
    cov: torch.Tensor,
    mean: torch.Tensor,
    wires: tuple[int, ...],
    herald: tuple[int, ...],
    cutoff: int,
    den_mat: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return an unnormalized PNR conditional tensor and its marginal event probability.

    PNR projection follows Eqs. (22), (25) and (28); for density matrices
    apply it on both sides, as in Eq. (8).

    With measured modes :math:`H`, retained modes :math:`R` and photon pattern :math:`h`, projection gives

    .. math::

        \widetilde\psi_n=\psi_{(h,n)},\qquad
        \widetilde\rho_{n,m}=\rho_{(h,n),(h,m)},\qquad
        p_h=\langle h|\operatorname{Tr}_R(\rho)|h\rangle.

    The tuples above illustrate an H,R reordering; returned axes retain their
    original mode order. Gaussian marginal moments are obtained by selecting
    the measured xxpp rows/columns, then converting that marginal afresh.
    Selecting a subblock of the full Bargmann A would instead project the other
    modes onto vacuum, and would not compute a partial trace.

    See https://arxiv.org/pdf/2504.10455v3 Sections 2.1 and 3.1, Eqs. (8), (22), (25) and (28).
    See https://the-walrus.readthedocs.io/en/latest/gbs.html for Gaussian marginals.

    Args:
        cov: Batched xxpp covariances, shape ``(batch, 2M, 2M)``.
        mean: Batched xxpp means, shape ``(batch, 2M, 1)``.
        wires: Validated, unique measured mode indices, in herald order.
        herald: Nonnegative integer photon numbers, one for each measured mode.
        cutoff: Positive output dimension of each remaining mode.
        den_mat: Return a density matrix when True, otherwise a ket.
            A ket requires every Gaussian input in the batch to be pure.

    Returns:
        Conditional Fock tensor (batch first, ket axes before bra axes), event
        probabilities of shape ``(batch,)``.
        Probabilities are computed from measured marginal moments and do not
        depend on the output cutoff. No normalization is applied.
    """
    mat_a, vec_b, coeff_c = gaussian_to_bargmann(cov, mean, den_mat=den_mat)
    nmode = cov.shape[-1] // 2
    selected = dict(zip(wires, herald, strict=True))
    bounds = tuple(selected[i] + 1 if i in selected else cutoff for i in range(nmode))
    indices = tuple(selected.get(i, slice(None)) for i in range(nmode))
    if not den_mat:
        values = _fock_coefficients(mat_a, vec_b, coeff_c, bounds)
        state = values[(slice(None), *indices)]
    elif nmode - len(wires) == 1:
        state = _single_mode_density(mat_a, vec_b, coeff_c, wires, herald, cutoff)
    else:
        values = _fock_coefficients(mat_a, vec_b, coeff_c, bounds * 2)
        state = values[(slice(None), *indices, *indices)]
    if not wires:
        probability = torch.ones_like(coeff_c)
    else:
        # p_h = (rho_H)_{h,h} sums over all retained occupations, independent of output cutoff.
        marginal = list(wires) + [i + nmode for i in wires]
        mat_a_marginal, vec_b_marginal, coeff_c_marginal = gaussian_to_bargmann(
            cov[:, marginal][:, :, marginal], mean[:, marginal]
        )
        coefficients = _fock_coefficients(
            mat_a_marginal, vec_b_marginal, coeff_c_marginal, tuple(h + 1 for h in herald) * 2
        )
        probability = coefficients[(slice(None), *herald, *herald)].real
    return state, probability
