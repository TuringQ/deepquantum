"""Common functions"""

import itertools
import math
import warnings
from collections import Counter
from collections.abc import Generator
from typing import Any

import torch
from numpy.polynomial.legendre import leggauss
from torch import vmap
from torch.distributions.multivariate_normal import MultivariateNormal

import deepquantum.photonic as dqp

from ..qmath import block_sample, decimal_to_list, is_unitary, list_to_decimal, partial_trace
from .draw import plot_wigner
from .utils import mem_to_chunksize


def _normalize_coefficients(value: torch.Tensor) -> torch.Tensor:
    """Normalize nonzero coefficient rows without squaring their original scale."""
    magnitude = torch.maximum(value.real.abs(), value.imag.abs()) if value.is_complex() else value.abs()
    scaled = value / magnitude.amax(dim=-1, keepdim=True)
    return scaled / torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)


def _spectrum_rows(value: Any, count: int, name: str) -> list[torch.Tensor]:
    """Read shared or per-input samples in double precision on CPU."""
    if isinstance(value, (list, tuple)) and value and torch.as_tensor(value[0], device='cpu').ndim:
        rows = value
    else:
        values = torch.as_tensor(value, device='cpu')
        rows = value if values.ndim == 2 else [value]
    result = []
    for row in rows:
        dtype = torch.complex128 if torch.as_tensor(row, device='cpu').is_complex() else torch.float64
        row = torch.as_tensor(row, dtype=dtype, device='cpu')
        if row.ndim > 1 or row.numel() == 0 or not torch.isfinite(row).all():
            raise ValueError(f'{name} must contain finite scalars or one-dimensional sample arrays.')
        result.append(row)
    if len(result) == 1:
        result *= count
    if len(result) != count:
        raise ValueError(f'{name} must be shared or have one row per delay.')
    return result


def _spectral_quadrature(knots: torch.Tensor, delay_span: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate linear spectral amplitudes with at most one radian of relative delay per interval."""
    widths = knots.diff()
    counts = (2 * torch.pi * delay_span * widths).ceil().clamp_min(1)
    if not torch.isfinite(counts).all() or counts.sum() > 1_000_000:
        raise ValueError('The frequency range and delay span require more than one million integration intervals.')
    counts = counts.long()
    intervals = torch.arange(len(widths), device=knots.device).repeat_interleave(counts)
    offsets = torch.arange(len(intervals), device=knots.device) - (counts.cumsum(0) - counts).repeat_interleave(counts)
    steps = widths[intervals] / counts[intervals]
    left = knots[intervals] + offsets * steps
    nodes, weights = (torch.as_tensor(value, device=knots.device) for value in leggauss(8))
    samples = left[:, None] + steps[:, None] * (nodes + 1) / 2
    weights = steps[:, None] * weights / 2
    return samples.flatten(), weights.flatten()


def delay_coefficients(
    delays: Any,
    *,
    frequency_hz: Any,
    intensity: Any = None,
    amplitude: Any = None,
    phase: Any = None,
) -> torch.Tensor:
    """Expand delayed spectral modes in a common orthogonal basis.

    Each mode has amplitude ``sqrt(intensity) * exp(1j * phase)`` or the supplied ``amplitude``.
    A delay multiplies it by ``exp(-2j * pi * frequency_hz * delay)``.
    Amplitudes are linearly interpolated, zero outside each grid, and normalized by frequency integration.
    The grids must resolve the spectra and phases. Transverse profiles and polarization are assumed matched.
    See https://arxiv.org/pdf/1909.04294v2 Eq.(1) and https://dlmf.nist.gov/1.14.E1 Eq.(1.14.1).

    Include all sources and local oscillators in one call to share their basis.
    The first input defines the first basis mode. Subsequent modes span orthogonal residuals in input order.
    Residual norms below ten times float64 epsilon are discarded.

    Args:
        delays: A scalar or one-dimensional array of delays in seconds.
        frequency_hz: A shared frequency grid or a list of per-input grids, sorted internally.
            Frequencies must be distinct within each grid and use the same reference across inputs.
            Use float64 for optical frequencies.
        intensity: Nonnegative spectral density per Hz, with shape ``(nsample,)`` or ``(ninput, nsample)``,
            or a list of per-input arrays. Specify exactly one of ``intensity`` and ``amplitude``.
            Default: ``None``
        amplitude: Real or complex spectral amplitudes with the same layout as ``intensity``. Default: ``None``
        phase: Spectral phase in radians, used with ``intensity``. A scalar, shared array or per-input arrays.
            Shape ``(ninput, 1)`` specifies constant per-input phases. Default: ``None`` (zero phase)

    Returns:
        Normalized coefficient rows of shape ``(ninput, nsubmode)``, in CPU complex128.
        ``coefficients.conj() @ coefficients.T`` gives the spectral overlaps.
    """
    if (intensity is None) == (amplitude is None):
        raise ValueError('Specify exactly one of intensity and amplitude.')
    if amplitude is not None and phase is not None:
        raise ValueError('phase is only used with intensity. Complex amplitude already includes its phase.')
    delays = _spectrum_rows(delays, 1, 'delays')[0].reshape(-1)
    if delays.is_complex():
        raise ValueError('delays must be real.')
    delays = delays - delays[0]
    count = len(delays)
    frequencies = _spectrum_rows(frequency_hz, count, 'frequency_hz')
    spectra = _spectrum_rows(intensity if amplitude is None else amplitude, count, 'spectrum')
    phases = _spectrum_rows(0.0 if phase is None else phase, count, 'phase')
    for j, (frequency, spectrum, phi) in enumerate(zip(frequencies, spectra, phases, strict=True)):
        if frequency.is_complex() or frequency.ndim != 1 or len(frequency) < 2:
            raise ValueError('Each frequency grid must contain at least two distinct real frequencies.')
        if spectrum.ndim != 1 or spectrum.shape != frequency.shape:
            raise ValueError('Each spectrum must match its frequency grid.')
        if amplitude is None:
            if spectrum.is_complex() or (spectrum < 0).any():
                raise ValueError('intensity must be real and nonnegative.')
            if phi.is_complex() or phi.numel() not in (1, len(frequency)):
                raise ValueError('Each phase must be real and scalar or match its frequency grid.')
            spectrum = torch.polar(spectrum.sqrt(), phi.expand_as(spectrum))
        if spectrum.abs().amax() == 0:
            raise ValueError('Each spectrum must be nonzero.')
        frequency, order = frequency.sort()
        if (frequency.diff() == 0).any():
            raise ValueError('Frequencies within each grid must be distinct.')
        frequencies[j] = frequency
        spectra[j] = _normalize_coefficients(spectrum[order])

    # Center frequencies before interpolation and retain their carrier phases separately.
    reference = frequencies[0][0]
    frequencies = [frequency - reference for frequency in frequencies]
    knots = torch.cat(frequencies).unique(sorted=True)
    grid, weights = _spectral_quadrature(knots, delays.max() - delays.min())
    samples = []
    for frequency, spectrum, delay in zip(frequencies, spectra, delays, strict=True):
        index = torch.searchsorted(frequency, grid).clamp(1, len(frequency) - 1)
        fraction = ((grid - frequency[index - 1]) / (frequency[index] - frequency[index - 1])).clamp(0, 1)
        value = spectrum[index - 1] * (1 - fraction) + spectrum[index] * fraction
        value = torch.where((grid >= frequency[0]) & (grid <= frequency[-1]), value, 0)
        value = value * torch.exp(-2j * torch.pi * grid * delay)
        samples.append(_normalize_coefficients(value * weights.sqrt()))

    # Two orthogonalization passes retain tiny residuals without subtracting 1 - |overlap|**2.
    basis, rows, origins = [], [], []
    for j, value in enumerate(samples):
        coefficients = value.new_zeros(len(basis))
        residual = value
        if basis:
            vectors = torch.stack(basis)
            for _ in range(2):
                correction = vectors.conj() @ residual
                coefficients = coefficients + correction
                residual = residual - correction @ vectors
        norm = torch.linalg.vector_norm(residual)
        if norm > 10 * torch.finfo(torch.float64).eps:
            basis.append(residual / norm)
            origins.append(j)
            coefficients = torch.cat([coefficients, norm.reshape(1)])
        rows.append(coefficients)
    coefficients = torch.stack([torch.nn.functional.pad(row, (0, len(basis) - len(row))) for row in rows])
    carrier = torch.exp(-2j * torch.pi * torch.fmod(reference * delays, 1))
    coefficients = coefficients * carrier[:, None] * carrier[origins].conj()[None, :]
    return _normalize_coefficients(coefficients)


def _fock_isometry(coefficients: torch.Tensor, cutoff: int) -> torch.Tensor:
    """Return the Fock isometry for a normalized orthogonal-basis coefficient vector.

    Output axes index occupations in each orthogonal basis mode, followed by the input photon number.
    Each axis has size ``cutoff``.

    See https://arxiv.org/pdf/quant-ph/0406127v1 Eq.(14), specialized to one occupied input mode.
    """
    nsubmode = len(coefficients)
    vacuum = coefficients.new_zeros([cutoff] * nsubmode)
    vacuum[(0,) * nsubmode] = 1
    ladder = torch.arange(1, cutoff, dtype=coefficients.real.dtype, device=coefficients.device).sqrt()
    columns = [vacuum]
    for n in range(1, cutoff):
        column = torch.zeros_like(vacuum)
        for k in range(nsubmode):
            shape = [1] * nsubmode
            shape[k] = cutoff - 1
            raised = columns[-1].narrow(k, 0, cutoff - 1) * ladder.reshape(shape)
            raised = torch.nn.functional.pad(raised, (0, 0) * (nsubmode - k - 1) + (1, 0))
            column = column + coefficients[k] * raised
        # |n_c> = sum_k c_k a_k^dagger |(n-1)_c> / sqrt(n).
        columns.append(column / math.sqrt(n))
    return torch.stack(columns, -1)


def dirac_rep(state: torch.Tensor, den_mat: bool = False, topk: int = 5) -> dict:
    """Convert the batched Fock state tensors to the dictionary of Dirac representation."""
    dirac_dict = {}
    for i in range(state.shape[0]):  # consider batch
        state_i = state[i]
        abs_state = abs(state_i).flatten()
        top_vals, top_indices = torch.topk(abs_state, k=min(len(abs_state), topk))
        coords = torch.stack(torch.unravel_index(top_indices, state_i.shape), dim=1)
        dirac_lst = []
        for val, idx_coords in zip(top_vals, coords, strict=True):
            if val <= 1e-5:
                continue
            idx = idx_coords.tolist()
            use_comma = any(x > 9 for x in idx)
            coeff = state_i[tuple(idx)].item()
            if den_mat:
                idx1 = idx[: len(idx) // 2]
                idx2 = idx[len(idx) // 2 :]
                state_b1 = ','.join(map(str, idx1)) if use_comma else ''.join(map(str, idx1))
                state_b2 = ','.join(map(str, idx2)) if use_comma else ''.join(map(str, idx2))
                state_str = f'{coeff:+6.3f}|{state_b1}><{state_b2}|'
            else:
                state_b = ','.join(map(str, idx)) if use_comma else ''.join(map(str, idx))
                state_str = f'{coeff:+6.3f}|{state_b}>'
            dirac_lst.append(state_str)
        dirac = ' '.join(dirac_lst)
        dirac_dict[f'state_{i}'] = dirac[1:] if dirac[0] == '+' else dirac
    return dirac_dict


def sort_dict_fock_basis(state_dict: dict, idx: int = 0) -> dict:
    """Sort the dictionary of Fock basis states in the descending order of probabilities."""
    sort_list = sorted(state_dict.items(), key=lambda t: abs(t[1][idx]), reverse=True)
    sorted_dict = {}
    for key, value in sort_list:
        sorted_dict[key] = value
    return sorted_dict


def group_probabilities(probabilities: dict | torch.Tensor, groups: list[list[int]]) -> dict:
    """Sum photon-count probabilities over unresolved orthogonal basis modes of each spatial mode.

    ``groups`` must partition all modes, in the desired order of spatial outputs.
    Input is the probability dictionary or dense probability tensor returned by
    ``cir(is_prob=True)``. Tensor input has shape ``(batch, cutoff, ..., cutoff)``
    or omits the batch dimension. Output keys are spatial Fock states and values
    retain the input batch dimensions and gradients. Probabilities are summed,
    never amplitudes. Truncation mass is preserved without renormalization.
    """
    wires = [wire for group in groups for wire in group]
    nmode = len(wires)
    if not groups or any(not group for group in groups) or sorted(wires) != list(range(nmode)):
        raise ValueError('groups must partition consecutive mode indices without duplicates.')
    if isinstance(probabilities, torch.Tensor):
        if probabilities.is_complex() or probabilities.ndim not in (nmode, nmode + 1):
            raise ValueError('Expected real probability tensors with one optional batch dimension.')
        if probabilities.ndim == nmode:
            probabilities = probabilities.unsqueeze(0)
        items = (
            (pattern, probabilities[(slice(None),) + pattern])
            for pattern in itertools.product(*(range(size) for size in probabilities.shape[1:]))
        )
    else:
        items = ((tuple(key.state.tolist()), value) for key, value in probabilities.items())
    result = {}
    for pattern, value in items:
        if isinstance(value, complex) or isinstance(value, torch.Tensor) and value.is_complex():
            raise ValueError('Group probabilities, not complex amplitudes.')
        if len(pattern) != nmode:
            raise ValueError('groups must cover every input mode.')
        counts = [sum(pattern[wire] for wire in group) for group in groups]
        key = dqp.FockState(counts)
        result[key] = result.get(key, 0) + value
    return result


def _pnr_condition(state, nmode, cutoff, den_mat, measured, counts, keep, *, normalize=True):
    """Project unresolved PNR counts and trace all unretained modes without a dense POVM.

    See https://arxiv.org/pdf/1902.02323v2 Eq.(24) and Eq.(109) for PNR projection and probability.
    Here each unresolved event sums resolved projectors, with unretained modes traced out.
    """
    rest = [wire for wire in range(nmode) if wire not in keep]
    order = keep + rest
    positions = [[rest.index(wire) for wire in group] for group in measured]
    mask = torch.tensor(
        [
            all(sum(pattern[i] for i in group) == count for group, count in zip(positions, counts, strict=True))
            for pattern in itertools.product(range(cutoff), repeat=len(rest))
        ],
        device=state.device,
    )
    nkeep, nrest = cutoff ** len(keep), cutoff ** len(rest)
    state = state.reshape([-1] + [cutoff] * nmode * (2 if den_mat else 1))
    axes = [0] + [w + 1 for w in order]
    if den_mat:
        axes += [w + nmode + 1 for w in order]
        state = state.permute(axes).reshape(-1, nkeep, nrest, nkeep, nrest)
        density = (state.diagonal(dim1=2, dim2=4) * mask).sum(-1)
    else:
        state = state.permute(axes).reshape(-1, nkeep, nrest)
        selected = state * mask
        density = selected @ selected.mH
    probability = density.diagonal(dim1=-2, dim2=-1).sum(-1).real
    if normalize:
        if (probability <= 0).any():
            raise ValueError('Cannot normalize a zero-probability PNR event.')
        density = density / probability[:, None, None]
    return density.reshape([-1] + [cutoff] * (2 * len(keep))), probability


def _fock_conditional_blocks(state, nmode, cutoff, den_mat, measured, keep):
    """Return unnormalized retained density blocks for every resolved PNR outcome.

    Output axes are (batch, measured photon counts..., retained ket, retained bra).
    Empty keep gives event probabilities in 1 x 1 blocks. No division by the
    event probability is performed, so impossible allocations remain valid zeros.

    See https://arxiv.org/pdf/1902.02323v2 Eq.(97) for the resolved PNR conditional operator.
    This implementation contracts Fock coefficients directly and traces unretained modes.
    """
    rest = [w for w in range(nmode) if w not in keep and w not in measured]
    order = keep + measured + rest
    k, p, r = cutoff ** len(keep), cutoff ** len(measured), cutoff ** len(rest)
    state = state.reshape([-1] + [cutoff] * nmode * (2 if den_mat else 1))
    axes = [0] + [w + 1 for w in order]
    if den_mat:
        axes += [w + nmode + 1 for w in order]
        state = state.permute(axes).reshape(-1, k, p, r, k, p, r)
        state = state.diagonal(dim1=3, dim2=6).sum(-1)
        density = state.diagonal(dim1=2, dim2=4).movedim(-1, 1)
    else:
        state = state.permute(axes).reshape(-1, k, p, r).transpose(1, 2)
        density = state @ state.mH
    return density.reshape([-1] + [cutoff] * len(measured) + [k, k])


def _group_count_axes(values, groups, limits=None, operators=False):
    """Sum resolved count axes per spatial port, preserving optional retained operators."""
    end = -2 if operators else None
    sizes = values.shape[1:end]
    shape = [1 + sum(sizes[k] - 1 for k in group) for group in groups]
    if limits is not None:
        shape = [min(size, count + 1) for size, count in zip(shape, limits, strict=True)]
    coordinates = torch.stack(
        torch.meshgrid(*(torch.arange(n, device=values.device) for n in sizes), indexing='ij'), -1
    ).reshape(-1, len(sizes))
    counts = torch.stack([coordinates[:, group].sum(-1) for group in groups], -1)
    valid = (counts < torch.tensor(shape, device=values.device)).all(-1)
    strides = torch.tensor([math.prod(shape[i + 1 :]) for i in range(len(shape))], device=values.device)
    index = torch.where(valid, (counts * strides).sum(-1), 0)
    flattened = values.reshape(values.shape[0], math.prod(sizes), -1) * valid[None, :, None]
    result = values.new_zeros(values.shape[0], math.prod(shape), flattened.shape[-1])
    result = result.scatter_add(1, index[None, :, None].expand_as(flattened), flattened)
    return result.reshape([values.shape[0]] + shape + (list(values.shape[-2:]) if operators else []))


def _convolve_counts(left, right, limits=None, operators=False):
    """Convolve count distributions, optionally tensoring their retained operators.

    A leading batch axis is followed by one count axis per detector. With
    operators=True, the last two axes are retained ket/bra axes. Optional limits
    discard counts above a target event. Direct products and scatter-add avoid
    FFT cancellation in very small herald probabilities.

    For independent blocks, ``P(n) = sum_{a+b=n} P_left(a) * P_right(b)``.
    Retained states replace each scalar product by ``A_left(a) tensor A_right(b)``.
    See https://arxiv.org/pdf/1012.0262v2, the convolution discussion following Eq.(19).
    """
    end = -2 if operators else None
    a, b = left.shape[1:end], right.shape[1:end]
    shape = [x + y - 1 for x, y in zip(a, b, strict=True)]
    if limits is not None:
        shape = [min(size, n + 1) for size, n in zip(shape, limits, strict=True)]
    device = left.device
    ca = torch.stack(torch.meshgrid(*(torch.arange(n, device=device) for n in a), indexing='ij'), -1).reshape(
        -1, len(a)
    )
    cb = torch.stack(torch.meshgrid(*(torch.arange(n, device=device) for n in b), indexing='ij'), -1).reshape(
        -1, len(b)
    )
    counts = ca[:, None] + cb[None, :]
    valid = (counts < torch.tensor(shape, device=device)).all(-1).flatten()
    strides = torch.tensor([math.prod(shape[i + 1 :]) for i in range(len(shape))], device=device)
    indices = (counts * strides).sum(-1).flatten()
    indices = torch.where(valid, indices, torch.zeros_like(indices))
    batch = torch.broadcast_shapes(left.shape[:1], right.shape[:1])[0]
    if operators:
        k, nright = left.shape[-1], right.shape[-1]
        left, right = left.reshape(-1, math.prod(a), k, k), right.reshape(-1, math.prod(b), nright, nright)
        values = torch.einsum('baij,bckl->bacikjl', left, right).reshape(batch, -1, (k * nright) ** 2)
    else:
        values = (left.reshape(-1, math.prod(a), 1) * right.reshape(-1, 1, math.prod(b))).reshape(batch, -1, 1)
    values = values * valid[None, :, None]
    result = values.new_zeros(batch, math.prod(shape), values.shape[-1])
    result = result.scatter_add(1, indices[None, :, None].expand_as(values), values)
    return result.reshape([batch] + shape + ([k * nright, k * nright] if operators else []))


def sub_matrix(u: torch.Tensor, input_state: torch.Tensor, output_state: torch.Tensor) -> torch.Tensor:
    """Get the submatrix for calculating the transfer amplitude and transfer probabilities.

    The rows are chosen according to the output state and the columns are chosen according to the input state.

    Args:
        u: The unitary matrix.
        input_state: The input state.
        output_state: The output state.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')  # local warning
        u1 = torch.repeat_interleave(u, output_state, dim=0)
        u2 = torch.repeat_interleave(u1, input_state, dim=-1)
    return u2


def permanent(mat: torch.Tensor) -> torch.Tensor:
    """Calculate the permanent."""
    shape = mat.shape
    if mat.numel() == 0:
        if shape[0] == shape[1] == 0:
            return torch.tensor(1, dtype=mat.dtype, device=mat.device)
        else:
            return torch.tensor(0, dtype=mat.dtype, device=mat.device)
    if len(mat.size()) == 0:
        return mat
    if shape[0] == 1:
        return mat[0, 0]
    if shape[0] == 2:
        return mat[0, 0] * mat[1, 1] + mat[0, 1] * mat[1, 0]
    if shape[0] == 3:
        return (
            mat[0, 2] * mat[1, 1] * mat[2, 0]
            + mat[0, 1] * mat[1, 2] * mat[2, 0]
            + mat[0, 2] * mat[1, 0] * mat[2, 1]
            + mat[0, 0] * mat[1, 2] * mat[2, 1]
            + mat[0, 1] * mat[1, 0] * mat[2, 2]
            + mat[0, 0] * mat[1, 1] * mat[2, 2]
        )
    return permanent_ryser(mat)


def create_subset(num_coincidence: int) -> Generator[torch.Tensor, None, None]:
    r"""Create all subsets from :math:`\{1,2,...,n\}`."""
    for k in range(1, num_coincidence + 1):
        comb_lst = []
        for comb in itertools.combinations(range(num_coincidence), k):
            comb_lst.append(list(comb))
        yield torch.tensor(comb_lst).reshape(len(comb_lst), k)


def get_powerset(n: int) -> list:
    r"""Get the powerset of :math:`\{0,1,...,n-1\}`."""
    powerset = []
    for k in range(n + 1):
        subset = []
        for i in itertools.combinations(range(n), k):
            subset.append(list(i))
        powerset.append(subset)
    return powerset


def permanent_ryser(mat: torch.Tensor) -> torch.Tensor:
    """Calculate the permanent by Ryser's formula."""

    def helper(subset: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
        num_elements = subset.numel()
        s = torch.sum(mat[:, subset], dim=-1)
        value_times = torch.prod(s) * (-1) ** num_elements
        return value_times

    num_coincidence = mat.size()[0]
    value_perm = 0
    chunk_size = mem_to_chunksize(mat.device, mat.dtype)
    for subset in create_subset(num_coincidence):
        temp_value = vmap(helper, in_dims=(0, None), chunk_size=chunk_size)(subset, mat)
        value_perm += temp_value.sum()
    value_perm *= (-1) ** num_coincidence
    return value_perm


def product_factorial(state: torch.Tensor, power: float = 1.0) -> torch.Tensor:
    """Return the factorial product raised to ``power``, evaluating the power in log space.

    Integer photon counts produce float64 results, or float32 on MPS.
    Floating inputs retain their dtype.

    See https://dlmf.nist.gov/5.4.E1 Eq.(5.4.1) for ``n! = Gamma(n + 1)``.
    Evaluate ``exp(power * sum(lgamma(n + 1)))`` without forming the unpowered factorial product.

    Args:
        state: Photon counts, with modes along the final axis.
        power: Exponent applied to the factorial product. Default: 1.0
    """
    log_value = torch.lgamma(state.cpu().double() + 1).sum(-1, keepdim=True)
    dtype = state.dtype
    if not state.is_floating_point():
        dtype = torch.float32 if state.device.type == 'mps' else torch.float64
    return torch.exp(power * log_value).to(device=state.device, dtype=dtype)


def fock_combinations(nmode: int, nphoton: int, cutoff: int | None = None, nancilla: int = 0) -> list[list[int]]:
    """Generate all possible combinations of Fock states for a given number of modes, photons, and cutoff.

    Args:
        nmode: The number of modes in the system.
        nphoton: The total number of photons in the system.
        cutoff: The Fock space truncation. Default: ``None``
        nancilla: The number of ancilla modes (NOT limited by ``cutoff``). Default: ``0``

    Returns:
        A list of all possible Fock states, each represented by a list of occupation numbers for each mode.

    Examples:
        >>> fock_combinations(2, 3)
        [[0, 3], [1, 2], [2, 1], [3, 0]]
        >>> fock_combinations(3, 2)
        [[0, 0, 2], [0, 1, 1], [0, 2, 0], [1, 0, 1], [1, 1, 0], [2, 0, 0]]
        >>> fock_combinations(4, 4, 2)
        [[1, 1, 1, 1]]
    """
    if cutoff is None:
        cutoff = nphoton + 1
    result = []

    def backtrack(state: list[int], length: int, num_sum: int) -> None:
        """A helper function that uses backtracking to generate all possible Fock states.

        Args:
            state: The current Fock state being constructed.
            length: The remaining number of modes to be filled.
            num_sum: The remaining number of photons to be distributed.
        """
        if length == 0:
            if num_sum == 0:
                result.append(state)
            return
        # Determine the effective length for cutoff
        effective_length = length - nancilla
        # skip iterations if remaining photons exceed the remaining cutoff
        if nancilla == 0 and num_sum > (cutoff - 1) * effective_length:
            return
        for i in range(min((num_sum + 1), cutoff) if effective_length > 0 else (num_sum + 1)):
            backtrack(state + [i], length - 1, num_sum - i)

    backtrack([], nmode, nphoton)
    return result


def ladder_ops(cutoff: int, dtype=torch.cfloat, device='cpu') -> tuple[torch.Tensor, torch.Tensor]:
    """Get the matrix representation of the annihilation and creation operators."""
    sqrt = torch.arange(1, cutoff, dtype=dtype, device=device) ** 0.5
    a = torch.diag(sqrt, diagonal=1)
    ad = a.mH  # share the memory
    return a, ad


def shift_func(lst: list, nstep: int) -> list:
    """Shift a list by a number of steps.

    If ``nstep`` is positive, it shifts to the left.
    """
    if len(lst) <= 1:
        return lst
    nstep = nstep % len(lst)
    return lst[nstep:] + lst[:nstep]


def xxpp_to_xpxp(matrix: torch.Tensor) -> torch.Tensor:
    """Transform the representation in ``xxpp`` ordering to the representation in ``xpxp`` ordering."""
    nmode = matrix.shape[-2] // 2
    idx = torch.arange(2 * nmode, device=matrix.device).reshape(2, nmode).T.flatten()
    if matrix.shape[-1] == 2 * nmode:
        return matrix[..., idx[:, None], idx]
    elif matrix.shape[-1] == 1:
        return matrix[..., idx, :]


def xpxp_to_xxpp(matrix: torch.Tensor) -> torch.Tensor:
    """Transform the representation in ``xpxp`` ordering to the representation in ``xxpp`` ordering."""
    nmode = matrix.shape[-2] // 2
    idx = torch.arange(2 * nmode, device=matrix.device).reshape(nmode, 2).T.flatten()
    if matrix.shape[-1] == 2 * nmode:
        return matrix[..., idx[:, None], idx]
    elif matrix.shape[-1] == 1:
        return matrix[..., idx, :]


def quadrature_to_ladder(tensor: torch.Tensor, symplectic: bool = False) -> torch.Tensor:
    """Transform the representation in ``xxpp`` ordering to the representation in ``aaa^+a^+`` ordering.

    Args:
        tensor: The input tensor in ``xxpp`` ordering.
        symplectic: Whether the transformation is applied for symplectic matrix or Gaussian state.
            Default: ``False`` (which means covariance matrix or displacement vector)
    """
    nmode = tensor.shape[-2] // 2
    tensor = tensor + 0j
    identity = torch.eye(nmode, dtype=tensor.dtype, device=tensor.device)
    omega = torch.cat([torch.cat([identity, identity * 1j], dim=-1), torch.cat([identity, identity * -1j], dim=-1)])
    if tensor.shape[-1] == 2 * nmode:
        if symplectic:
            return omega @ tensor @ omega.mH / 2  # inversed omega
        else:
            return omega @ tensor @ omega.mH * dqp.kappa**2 / dqp.hbar
    elif tensor.shape[-1] == 1:
        return omega @ tensor * dqp.kappa / dqp.hbar**0.5


def ladder_to_quadrature(tensor: torch.Tensor, symplectic: bool = False) -> torch.Tensor:
    """Transform the representation in ``aaa^+a^+`` ordering to the representation in ``xxpp`` ordering.

    Args:
        tensor: The input tensor in ``aaa^+a^+`` ordering.
        symplectic: Whether the transformation is applied for symplectic matrix or Gaussian state.
            Default: ``False`` (which means covariance matrix or displacement vector)
    """
    nmode = tensor.shape[-2] // 2
    tensor = tensor + 0j
    identity = torch.eye(nmode, dtype=tensor.dtype, device=tensor.device)
    omega = torch.cat([torch.cat([identity, identity], dim=-1), torch.cat([identity * -1j, identity * 1j], dim=-1)])
    if tensor.shape[-1] == 2 * nmode:
        if symplectic:
            return (omega @ tensor @ omega.mH).real / 2  # inversed omega
        else:
            return (omega @ tensor @ omega.mH).real * dqp.hbar / (4 * dqp.kappa**2)
    elif tensor.shape[-1] == 1:
        return (omega @ tensor).real * dqp.hbar**0.5 / (2 * dqp.kappa)


def _photon_number_mean_var_gaussian(cov: torch.Tensor, mean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the expectation value and variance of the photon number for single-mode Gaussian states."""
    coef = dqp.kappa**2 / dqp.hbar
    cov = cov.reshape(-1, 2, 2)
    mean = mean.reshape(-1, 2, 1)
    exp = coef * (vmap(torch.trace)(cov) + (mean.mT @ mean).squeeze()) - 1 / 2
    var = coef**2 * (vmap(torch.trace)(cov @ cov) + 2 * (mean.mT @ cov.to(mean.dtype) @ mean).squeeze()) * 2 - 1 / 4
    return exp, var


def _photon_number_mean_var_bosonic(
    cov: torch.Tensor, mean: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the expectation value and variance of the photon number for single-mode Bosonic states."""
    shape_cov = cov.shape
    shape_mean = mean.shape
    cov = cov.reshape(*shape_cov[:2], 2, 2).reshape(-1, 2, 2)
    mean = mean.reshape(*shape_mean[:2], 2, 1).reshape(-1, 2, 1)
    exp_gaussian, var_gaussian = _photon_number_mean_var_gaussian(cov, mean)
    exp_gaussian = exp_gaussian.reshape(shape_cov[:2])
    var_gaussian = var_gaussian.reshape(shape_cov[:2])
    exp = (weight * exp_gaussian).sum(-1)
    var = (weight * var_gaussian).sum(-1) + (weight * exp_gaussian**2).sum(-1) - exp**2
    zeros = cov.new_zeros(1)
    assert torch.allclose(exp.imag, zeros, atol=1e-6)
    assert torch.allclose(var.imag, zeros, atol=1e-6)
    return exp.real, var.real


def photon_number_mean_var_cv(
    cov: torch.Tensor, mean: torch.Tensor, weight: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the expectation value and variance of the photon number for single-mode Gaussian (Bosonic) states."""
    if weight is None:
        return _photon_number_mean_var_gaussian(cov, mean)
    else:
        return _photon_number_mean_var_bosonic(cov, mean, weight)


def photon_number_mean_var_fock(
    state: torch.Tensor, nmode: int, cutoff: int, wires: list[int], den_mat: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the expectation value and variance of the photon number for Fock state tensors."""
    if den_mat:
        rho = state.reshape(-1, cutoff**nmode, cutoff**nmode)
        prob = torch.diagonal(rho, dim1=1, dim2=2).reshape([-1] + [cutoff] * nmode).real
    else:
        if state.ndim == nmode:
            state = state.unsqueeze(0)
        prob = abs(state) ** 2
    num_op = torch.arange(cutoff, device=state.device)
    num_exp_list = []
    var_list = []
    for i in wires:
        p_i = torch.sum(prob, dim=[j + 1 for j in range(nmode) if j != i])
        num_exp = (num_op * p_i).sum(dim=-1)  # (batch,)
        num2_exp = ((num_op**2) * p_i).sum(dim=-1)  # (batch,)
        var = num2_exp - num_exp**2
        num_exp_list.append(num_exp)
        var_list.append(var)
    return torch.stack(num_exp_list), torch.stack(var_list)


def quadrature_mean_fock(
    state: torch.Tensor, nmode: int, cutoff: int, wires: list[int], den_mat: bool = False
) -> torch.Tensor:
    """Get the expectation value of the quadrature x for Fock state tensors."""
    coef = 2 * dqp.kappa**2 / dqp.hbar
    factor = torch.sqrt(torch.arange(1, cutoff, device=state.device, dtype=state.real.dtype) / 2)
    mean = []
    if den_mat:
        state = state.reshape(-1, cutoff**nmode, cutoff**nmode)
        for wire in wires:
            trace_lst = [i for i in range(nmode) if i != wire]
            reduced_dm = partial_trace(state, nmode, trace_lst, cutoff)  # (batch, cutoff, cutoff)
            reduced_dm = reduced_dm.reshape(-1, cutoff, cutoff)
            off_diag = reduced_dm.diagonal(offset=1, dim1=1, dim2=2)  # rho_{n, n+1}
            term = factor * 2 * off_diag.real  # only with real part contribution
            mean.append(term.sum(dim=1))
    else:
        if state.ndim == nmode:
            state = state.unsqueeze(0)
        factor = factor.view([1, -1] + [1] * (nmode - 1))
        for wire in wires:
            pm_shape = list(range(1, nmode + 1))
            pm_shape.remove(wire + 1)
            pm_shape = [0] + [wire + 1] + pm_shape
            state_i = state.permute(pm_shape)
            cn = state_i[:, :-1, ...]  # n
            cn1 = state_i[:, 1:, ...]  # n+1
            term = factor * 2 * (cn.conj() * cn1).real
            mean.append(term.sum(dim=tuple(range(1, nmode + 1))))
    return coef ** (-0.5) * torch.stack(mean)


def takagi(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Tagaki decomposition for a symmetric complex matrix.

    See https://math.stackexchange.com/questions/2026110/
    """
    size = a.size()[0]
    a_2 = torch.block_diag(-a.real, a.real)
    if torch.is_complex(a):
        a_2[:size, size:] = a.imag
        a_2[size:, :size] = a.imag
    s, u = torch.linalg.eigh(a_2)
    diag = s[size:]  # s already sorted
    v = u[:, size:][size:] + 1j * u[:, size:][:size]
    if is_unitary(v):
        return v, diag
    else:  # consider degeneracy case
        idx_zero = torch.where(abs(s) < 1e-5)[0]
        idx_max = max(idx_zero) + 1
        temp = abs(u[:size, idx_max:]) ** 2 + abs(u[size:, idx_max:]) ** 2
        sum_rhalf = temp.sum(1)
        idx_lt_1 = torch.where(abs(sum_rhalf - 1) > 1e-6)[0]
        r = size - (2 * size - idx_max)
        # find the correct combination
        for i in itertools.combinations(idx_zero, r):
            u_temp = u[:, list(i)]
            temp2 = abs(u_temp[idx_lt_1]) ** 2 + abs(u_temp[idx_lt_1 + size]) ** 2
            sum_lhalf = temp2.sum(1)
            sum_total = sum_lhalf + sum_rhalf[idx_lt_1]
            if torch.allclose(sum_total, torch.ones(len(idx_lt_1), dtype=sum_total.dtype, device=sum_total.device)):
                u_half = torch.cat([u[:, list(i)], u[:, idx_max:]], dim=1)
                v = u_half[size:] + 1j * u_half[:size]
                if is_unitary(v):
                    return v, diag


def sqrtm_herm(mat: torch.Tensor) -> torch.Tensor:
    """Compute the positive matrix square root of a Hermitian matrix using eigenvalue decomposition."""
    lambd, mat_q = torch.linalg.eigh(mat)
    return mat_q @ lambd.sqrt().diag_embed().to(mat_q.dtype) @ mat_q.mH


def schur_anti_symm_even(mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Schur decomposition for a real antisymmetric and even-dimensional matrix.

    This function decomposes a real antisymmetric matrix :math:`A` into the form :math:`A = O T O^T`,
    where :math:`O` is an orthogonal matrix and :math:`T` is a block-diagonal matrix
    with :math:`2 \times 2` antisymmetric blocks.
    """
    assert torch.allclose(mat, -mat.mT, rtol=1e-5, atol=1e-5)
    n = len(mat)
    hermitian = mat * -1j
    lambd, u = torch.linalg.eigh(hermitian)
    mat_t = torch.zeros_like(mat)
    idx1 = torch.arange(0, n, 2, device=mat.device)
    idx2 = torch.arange(1, n, 2, device=mat.device)
    # positive value is above the diagonal and in ascending order
    mat_t[idx1, idx2] = lambd[n // 2 :]
    mat_t[idx2, idx1] = -lambd[n // 2 :]
    mat_o = torch.zeros_like(mat)
    mat_o[:, ::2] = u[:, n // 2 :].real
    mat_o[:, 1::2] = u[:, n // 2 :].imag
    norm = torch.linalg.vector_norm(mat_o, dim=0, keepdim=True)
    mat_o = mat_o / norm
    return mat_t, mat_o


def williamson(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Williamson decomposition.

    This function decomposes a real symmetric and even-dimensional positive definite matrix :math:`V`
    into the form :math:`V = S D S^T`, where :math:`S` is a symplectic matrix and
    :math:`D` is a diagonal matrix with the symplectic eigenvalues.

    See https://arxiv.org/pdf/2403.04596 Section VII.
    """
    assert torch.allclose(cov, cov.mT, rtol=1e-5, atol=1e-5)
    nmode = cov.shape[-1] // 2
    omega = cov.new_ones(nmode)
    omega = torch.cat([-omega, omega]).diag_embed()
    omega = omega.reshape(2, nmode, 2 * nmode).flip(0).reshape(2 * nmode, 2 * nmode)  # symplectic form
    vals = torch.linalg.eigvalsh(cov)
    assert torch.all(vals > 0), 'Matrix must be positive definite.'
    cov_sqrt = sqrtm_herm(cov)
    cov_sqrt_inv = cov_sqrt.inverse()
    psi = cov_sqrt_inv @ omega @ cov_sqrt_inv  # antisymmetric
    mat_t, o_tilde = schur_anti_symm_even(psi)
    idx_perm = torch.arange(2 * nmode, device=cov.device).reshape(nmode, 2).T.flatten()
    mat_t_xxpp = mat_t[:, idx_perm][idx_perm]
    mat_o = o_tilde[:, idx_perm]
    idx = torch.arange(nmode, device=cov.device)
    phi = mat_t_xxpp[idx, idx + nmode]
    phi2 = torch.cat([phi, phi])
    diag = (1 / phi2).diag_embed()
    mat_s = cov_sqrt @ mat_o @ phi2.sqrt().diag_embed()
    return diag, mat_s


def measure_fock_tensor(
    state: torch.Tensor,
    shots: int = 1024,
    with_prob: bool = False,
    wires: int | list[int] | None = None,
    block_size: int = 2**24,
) -> dict | list[dict]:
    r"""Measure the batched Fock state tensors.

    Args:
        state: The quantum state to measure. It should be a tensor of shape
            :math:`(\text{batch}, \text{cutoff}, ..., \text{cutoff})`.
        shots: The number of times to sample from the quantum state. Default: 1024
        with_prob: A flag that indicates whether to return the probabilities along with the number of occurrences.
            Default: ``False``
        wires: The wires to measure. It can be an integer or a list of integers specifying
            the indices of the wires. Default: ``None`` (which means all wires are measured)
        block_size: The block size for sampling. Default: 2**24
    """
    from .state import FockState

    shape = state.shape
    batch = shape[0]
    cutoff = shape[-1]
    nmode = len(shape) - 1
    if wires is not None:
        if isinstance(wires, int):
            wires = [wires]
        assert isinstance(wires, list)
        wires = sorted(wires)
        pm_shape = list(range(nmode))
        for w in wires:
            pm_shape.remove(w)
        pm_shape = wires + pm_shape
    nwires = len(wires) if wires else nmode
    results_tot = []
    for i in range(batch):
        probs = torch.abs(state[i]) ** 2
        if wires is not None:
            probs = probs.permute(pm_shape).reshape([cutoff] * nwires + [-1]).sum(-1)
        probs = probs.reshape(-1)
        # Perform block sampling to reduce memory consumption
        samples = Counter(block_sample(probs, shots, block_size))
        results = {FockState(decimal_to_list(key, cutoff, nwires)): value for key, value in samples.items()}
        if with_prob:
            for k in results:
                index = list_to_decimal(k.state, cutoff)
                results[k] = results[k], probs[index]
        results_tot.append(results)
    if batch == 1:
        return results_tot[0]
    else:
        return results_tot


def sample_homodyne_fock(
    state: torch.Tensor,
    wire: int,
    nmode: int,
    cutoff: int,
    shots: int = 1,
    den_mat: bool = False,
    x_range: float = 15,
    nbin: int = 100000,
) -> torch.Tensor:
    """Get the samples of homodyne measurement for batched Fock state tensors on one mode.

    See https://arxiv.org/pdf/quant-ph/0511044v2 Eq.(26) and Eq.(43)-(44) for Fock
    wavefunctions and quadrature probabilities. The recurrence uses
    https://dlmf.nist.gov/18.9.E1 Eq.(18.9.1) and Table 18.9.1, rescaled by
    ``exp(-y**2/2) / sqrt(2**n * n!)``, with ``y = sqrt(2*kappa**2/hbar) * x``.
    Common normalization constants cancel in the discrete sampling weights.
    """
    coef = 2 * dqp.kappa**2 / dqp.hbar
    reduced_dm = _fock_conditional_blocks(state, nmode, cutoff, den_mat, [], [wire])
    xs = torch.linspace(-x_range, x_range, nbin, dtype=state.real.dtype, device=state.device)  # (nbin)
    scaled_x = coef**0.5 * xs
    # Normalized Hermite functions include the Gaussian envelope, avoiding large H_n and n!.
    functions = [torch.exp(-scaled_x.square() / 2)]
    for n in range(1, cutoff):
        value = math.sqrt(2 / n) * scaled_x * functions[-1]
        if n > 1:
            value = value - math.sqrt((n - 1) / n) * functions[-2]
        functions.append(value)
    h_vals = torch.stack(functions).to(reduced_dm.dtype)
    probs = torch.einsum('...ij,ix,jx->...x', reduced_dm, h_vals, h_vals).real.reshape(-1, nbin)
    probs = abs(probs).contiguous()
    probs[probs < 1e-10] = 0
    indices = torch.multinomial(probs.reshape(-1, nbin), num_samples=shots, replacement=True)  # (batch, shots)
    samples = xs[indices]
    return samples.unsqueeze(-1)  # (batch, shots, 1)


def sample_reject_bosonic(
    cov: torch.Tensor, mean: torch.Tensor, weight: torch.Tensor, cov_m: torch.Tensor, shots: int
) -> torch.Tensor:
    """Get the samples of the Bosonic states via rejection sampling.

    The covariance and mean may describe a marginal over selected quadratures.
    See https://arxiv.org/abs/2103.05530 Algorithm 1 in Section VI B
    """
    if cov.ndim == 3:
        cov = cov.unsqueeze(0)
    if mean.ndim == 3:
        mean = mean.unsqueeze(0)
    if weight.ndim == 1:
        weight = weight.unsqueeze(0)
    assert cov.ndim == mean.ndim == 4
    assert weight.ndim == 2
    batch = cov.shape[0]
    rst = [cov.new_empty(0)] * batch
    batches = list(range(batch))
    count_shots = [0] * batch
    shots_tmp = shots
    mask = (weight.real > 0) | (abs(weight.imag) > 1e-8) | (abs(mean.imag) > 1e-8).any(-2).squeeze(-1)
    exp_real = torch.exp(mean.imag.mT @ torch.linalg.solve(cov_m + cov, mean.imag) / 2).squeeze(-2, -1)
    c_tilde = mask * abs(weight) * exp_real
    while len(batches) > 0:
        cov_rest = cov[batches]
        mean_rest = mean[batches]
        cov_t = cov_m + cov_rest
        m0 = torch.multinomial(c_tilde[batches], 1).reshape(-1)  # (batch)
        cov_m0 = cov[batches, m0]
        mean_m0 = mean[batches, m0].squeeze(-1).real
        dist_g = MultivariateNormal(mean_rest.squeeze(-1).real, cov_t)  # (batch, ncomb, nquad)
        r0 = MultivariateNormal(mean_m0, cov_m + cov_m0).sample([shots_tmp])  # (shots, batch, nquad)
        prob_g = dist_g.log_prob(r0.unsqueeze(-2)).exp()  # (shots, batch, ncomb, nquad) -> (shots, batch, ncomb)
        g_r0 = (c_tilde[batches] * prob_g).sum(-1)  # (shots, batch)
        y0 = torch.rand_like(g_r0) * g_r0
        rm = r0.unsqueeze(-1).unsqueeze(-3)  # (shots, batch, nquad) -> (shots, batch, 1, nquad, 1)
        # (shots, batch, ncomb)
        exp_imag = torch.exp((rm - mean_rest.real).mT @ torch.linalg.solve(cov_t, mean_rest.imag) * 1j).squeeze()
        # Eq.(70-71)
        p_r0 = (weight[batches] * exp_real[batches] * prob_g * exp_imag).sum(-1)  # (shots, batch)
        assert torch.allclose(p_r0.imag, p_r0.imag.new_zeros(1))
        idx_shots, idx_batch = torch.where(y0 <= p_r0.real)
        batches_done = []
        for i in range(len(batches)):
            idx = batches[i]
            rst[idx] = torch.cat([rst[idx], r0[idx_shots[idx_batch == i], i]])  # (shots, nquad)
            count_shots[idx] = len(rst[idx])
            if count_shots[idx] >= shots:
                batches_done.append(idx)
                rst[idx] = rst[idx][:shots]
        for i in batches_done:
            batches.remove(i)
        shots_tmp = shots - min(count_shots)
    return torch.stack(rst)  # (batch, shots, nquad)


def align_shape(cov: torch.Tensor, mean: torch.Tensor, weight: torch.Tensor) -> list[torch.Tensor]:
    """Align the shape for Bosonic state."""
    ncomb = weight.shape[-1]
    if cov.ndim == mean.ndim == 4 and weight.ndim == 2:
        if cov.shape[1] == 1:
            cov = cov.expand(-1, ncomb, -1, -1)
        if mean.shape[1] == 1:
            mean = mean.expand(-1, ncomb, -1, -1)
        if weight.shape[0] == 1:
            weight = weight.expand(cov.shape[0], -1)
    elif cov.ndim == mean.ndim == 3 and weight.ndim == 1:
        if cov.shape[0] == 1:
            cov = cov.expand(ncomb, -1, -1)
        if mean.shape[0] == 1:
            mean = mean.expand(ncomb, -1, -1)
    return [cov, mean, weight]


def fock_to_wigner(
    state: torch.Tensor,
    wire: int,
    nmode: int,
    cutoff: int,
    den_mat: bool = False,
    xrange: int | list = 10,
    prange: int | list = 10,
    npoints: int | list = 100,
    plot: bool = True,
    k: int = 0,
) -> torch.Tensor:
    """Get the discretized Wigner function of the specified mode from a Fock state using the iterative method.

    See https://qutip.org/docs/4.7/modules/qutip/wigner.html

    Args:
        state: The input Fock state tensor or density matrix.
        wire: The Wigner function for the given wire.
        nmode: The mode number of the Fock state.
        cutoff: The Fock space truncation.
        den_mat: Whether to use density matrix representation. Only valid for Fock state tensor. Default: ``False``
        xrange: The range of quadrature x. Default: 10
        prange: The range of quadrature p. Default: 10
        npoints: The number of discretization points for quadratures. Default: 100
        plot: Whether to plot the Wigner function. Default: ``True``
        k: The index of the Wigner function within the batch to plot. Default: 0
    """
    if den_mat:
        rho = state.reshape(-1, cutoff**nmode, cutoff**nmode)
    else:
        state = state.reshape(-1, cutoff**nmode, 1)
        rho = state @ state.mH
    trace_lst = [i for i in range(nmode) if i != wire]
    reduced_dm = partial_trace(rho, nmode, trace_lst, cutoff)  # (batch, cutoff, cutoff)
    if reduced_dm.ndim == 2:
        reduced_dm = reduced_dm.unsqueeze(0)
    xlist = [-xrange, xrange] if isinstance(xrange, int) else xrange
    plist = [-prange, prange] if isinstance(prange, int) else prange
    if isinstance(npoints, int):
        xlist.append(npoints)
        plist.append(npoints)
    else:
        xlist.append(npoints[0])
        plist.append(npoints[1])
    assert len(xlist) == len(plist) == 3
    xvec = torch.linspace(*xlist, dtype=state.real.dtype, device=state.device)
    pvec = torch.linspace(*plist, dtype=state.real.dtype, device=state.device)
    coef = 2 * dqp.kappa**2 / dqp.hbar
    xlist, plist = torch.meshgrid(xvec, pvec, indexing='ij')
    # alpha = (sqrt(2) * kappa / sqrt(hbar)) * (q + i p) / sqrt(2)
    alpha = coef**0.5 * (xlist + 1.0j * plist) / 2**0.5
    w_list = xlist.new_zeros(cutoff, xlist.shape[-2], xlist.shape[-1]) * 1j
    w_00 = coef * torch.exp(-2 * abs(alpha) ** 2) / torch.pi
    w_list[0] = w_00
    w = reduced_dm[:, 0, 0].reshape(-1, 1, 1) * w_list[0]
    # First row: W_{0i}
    for i in range(1, cutoff):
        # For numerical stability, it is recommended to use cutoff < 80
        w_list[i] = 2 * alpha * w_list[i - 1] / rho.new_tensor(i).sqrt()
        w += 2 * (reduced_dm[:, 0, i].reshape(-1, 1, 1) * w_list[i]).real
    # Remaining rows: W_{ij}, i ≥ 1
    for i in range(1, cutoff):
        # Diagonal element W_{ii}
        sqrt_i = i**0.5
        temp = w_list[i].clone()
        w_list[i] = (2 * alpha.conj() * temp - sqrt_i * w_list[i - 1]) / sqrt_i
        w += reduced_dm[:, i, i].reshape(-1, 1, 1) * w_list[i]
        # Off-diagonal elements W_{ij}, j > i
        for j in range(i + 1, cutoff):
            sqrt_j = j**0.5
            temp2 = (2 * alpha * w_list[j - 1] - sqrt_i * temp) / sqrt_j
            temp = w_list[j].clone()
            w_list[j] = temp2
            w += 2 * (reduced_dm[:, i, j].reshape(-1, 1, 1) * w_list[j]).real
    if plot:
        plot_wigner(w.real, xvec, pvec, k)
    return w.real


def cv_to_wigner(
    state: list,
    wire: int,
    xrange: int | list = 10,
    prange: int | list = 10,
    npoints: int | list = 100,
    plot: bool = True,
    k: int = 0,
    normalize: bool = True,
):
    """Get the discretized Wigner function of the specified mode from a CV state.

    Args:
        state: The input ``Gaussianstate`` or ``BosonicState``.
        wire: The Wigner function for the given wire.
        xrange: The range of quadrature x. Default: 10
        prange: The range of quadrature p. Default: 10
        npoints: The number of discretization points for quadratures. Default: 100
        plot: Whether to plot the Wigner function. Default: ``True``
        k: The index of the Wigner function within the batch to plot. Default: 0
        normalize: Whether to normalize the Wigner function. Default: ``True``
    """
    cov, mean = state[:2]
    xlist = [-xrange, xrange] if isinstance(xrange, int) else xrange
    plist = [-prange, prange] if isinstance(prange, int) else prange
    if isinstance(npoints, int):
        xlist.append(npoints)
        plist.append(npoints)
    else:
        xlist.append(npoints[0])
        plist.append(npoints[1])
    assert len(xlist) == len(plist) == 3
    xvec = torch.linspace(*xlist, dtype=cov.dtype, device=cov.device)
    pvec = torch.linspace(*plist, dtype=cov.dtype, device=cov.device)
    grid_x, grid_y = torch.meshgrid(xvec, pvec, indexing='ij')
    coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)]).mT
    coords2 = coords.unsqueeze(1).unsqueeze(2)  # (npoints, 1, 1, 2)
    coords3 = coords.unsqueeze(-1).unsqueeze(-3)
    if not isinstance(wire, torch.Tensor):
        wire = torch.tensor(wire).reshape(1)
    if cov.ndim == 2:
        cov = cov.unsqueeze(0)
    if mean.ndim == 2:
        mean = mean.unsqueeze(0)
    if cov.ndim == 3:
        cov = cov.unsqueeze(1)
    if mean.ndim == 3:
        mean = mean.unsqueeze(1)
    if len(state) == 2:
        weight = cov.new_ones(1)
    elif len(state) == 3:
        weight = state[-1]
    cov, mean, weight = align_shape(cov, mean, weight)
    nmode = cov.shape[-1] // 2
    idx = torch.cat([wire, wire + nmode])  # xxpp order
    cov = cov[..., idx[:, None], idx]
    mean = mean[..., idx, :] + 0j  # for Gaussian state
    gauss_b = MultivariateNormal(mean.squeeze(-1).real, cov)  # mean shape: (batch, ncomb, 2)
    prob_g = gauss_b.log_prob(coords2).exp()  # (npoints, batch, ncomb)
    exp_real = torch.exp(mean.imag.mT @ torch.linalg.solve(cov, mean.imag) / 2).squeeze(-2, -1)  # (batch, ncomb)
    # (batch, npoints, ncomb)
    exp_imag = torch.exp(
        (coords3 - mean.real.unsqueeze(1)).mT @ torch.linalg.solve(cov, mean.imag).unsqueeze(1) * 1j
    ).squeeze(-2, -1)
    wigner_vals = exp_real.unsqueeze(-2) * prob_g.permute(1, 0, 2) * exp_imag * weight.unsqueeze(-2)
    wigner_vals = wigner_vals.sum(dim=2).reshape(-1, len(xvec), len(pvec)).real
    if normalize:
        # normalize the wigner function
        dx = xvec[1] - xvec[0]
        dp = pvec[1] - pvec[0]
        total_integral = torch.sum(wigner_vals, dim=[1, 2]) * dx * dp
        wigner_vals = wigner_vals / total_integral.reshape(-1, 1, 1)
    if plot:
        plot_wigner(wigner_vals, xvec, pvec, k)
    return wigner_vals
