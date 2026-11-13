"""PNR conditioning, cutoff diagnostics and continued evolution across photonic backends."""

import math

import pytest
import torch

import deepquantum as dq
import deepquantum.photonic as dqp
from deepquantum.photonic.bargmann import BargmannPolyExp, _fock_coefficients, conditional_fock, gaussian_to_bargmann
from deepquantum.photonic.state import BargmannState


def post_select_tensor(circuit, *args, **kwargs):
    state, *results = circuit.post_select(*args, **kwargs)
    return (state.state if isinstance(state, dq.FockState) else state, *results)


def coherent_ket(alpha, cutoff):
    alpha = torch.as_tensor(alpha, dtype=torch.complex128)
    n = torch.arange(cutoff, dtype=torch.float64, device=alpha.device)
    return torch.exp(-alpha.abs().square() / 2) * alpha**n / torch.exp(torch.lgamma(n + 1) / 2)


def moments_coherent(alphas):
    alphas = torch.as_tensor(alphas, dtype=torch.complex128)
    cov = torch.eye(2 * len(alphas), dtype=torch.float64)[None] * dqp.hbar / (4 * dqp.kappa**2)
    mean = torch.cat((alphas.real, alphas.imag)).reshape(1, -1, 1) * dqp.hbar**0.5 / dqp.kappa
    return cov, mean


@pytest.mark.parametrize('den_mat', [False, True])
@pytest.mark.parametrize('backend', ['fock', 'gaussian'])
def test_two_mode_squeezing(backend, den_mat):
    cir = dq.QumodeCircuit(2, 'vac', cutoff=9, backend=backend, basis=False, den_mat=den_mat)
    cir.s2([0, 1], 0.37, 0.2)
    cir.to(torch.float64)
    cir()
    for photons in (0, 1, 2):
        state, prob, info = post_select_tensor(cir, 0, photons, normalize=True, return_info=True)
        r = cir.operators[0].r.detach().double()
        expected_prob = (1 - r.tanh() ** 2) * r.tanh() ** (2 * photons)
        torch.testing.assert_close(prob, expected_prob.reshape(1), atol=1e-10, rtol=1e-8)
        expected = torch.zeros_like(state)
        if info['representation'] == 'ket':
            expected[0, photons] = 1
        else:
            expected[0, photons, photons] = 1
        torch.testing.assert_close(state.abs(), expected.abs(), atol=1e-10, rtol=1e-8)
        assert isinstance(info, dict)
        assert info['representation'] == ('dm' if backend == 'fock' or den_mat else 'ket')
        assert info['remaining_wires'] == (1,)


@pytest.mark.parametrize('den_mat', [False, True])
def test_coherent_complex_phase_and_cutoff(den_mat):
    cov, mean = moments_coherent([0.4 + 0.7j, 1.2 - 0.6j])
    cir = dq.QumodeCircuit(2, [cov, mean], cutoff=2, backend='gaussian', den_mat=den_mat)
    cir(stepwise=True)
    reference_prob = coherent_ket(0.4 + 0.7j, 4)[3].abs().square()
    masses = []
    for cutoff in (2, 5, 10):
        state, prob, info = post_select_tensor(cir, 0, 3, cutoff=cutoff, normalize=True, return_info=True)
        ket = coherent_ket(1.2 - 0.6j, cutoff)
        herald_amp = coherent_ket(0.4 + 0.7j, 4)[3]
        expected = ket * herald_amp / reference_prob.sqrt()
        if den_mat:
            expected = expected[:, None] * expected[None, :].conj()
        torch.testing.assert_close(state[0], expected)
        torch.testing.assert_close(prob[0], reference_prob)
        torch.testing.assert_close(info['retained_fraction'][0], ket.abs().square().sum())
        masses.append(info['probability_in_cutoff'].item())
    assert masses[0] < masses[1] < masses[2]


@pytest.mark.parametrize('density', [False, True])
def test_fock_unsorted_modes_and_source_probability(density):
    torch.manual_seed(13)
    ket = torch.randn(2, 3, 3, 3, 3, dtype=torch.complex128)
    ket = ket / ket.flatten(1).norm(dim=1).reshape(2, 1, 1, 1, 1)
    source = ket
    if density:
        vector = ket.reshape(2, -1)
        source = (vector[:, :, None] * vector[:, None, :].conj()).reshape(2, *([3] * 8))
    cir = dq.QumodeCircuit(4, source, cutoff=3, basis=False, den_mat=density)
    full = cir()
    state, prob, info = post_select_tensor(cir, [2, 0], [1, 2], cutoff=2, return_info=True, normalize=False)
    selected = ket[:, 2, :, 1, :]
    torch.testing.assert_close(prob, selected.abs().square().sum((1, 2)))
    selected = selected[:, :2, :2]
    vector = selected.reshape(2, -1)
    expected = (vector[:, :, None] * vector[:, None, :].conj()).reshape(2, 2, 2, 2, 2)
    torch.testing.assert_close(state, expected)
    assert info['remaining_wires'] == (1, 3)
    assert info['source_cutoff'] == 3
    assert cir.state is full
    torch.testing.assert_close(info['retained_fraction'], selected.abs().square().sum((1, 2)) / prob)


def test_thermal_mixed_batch_and_multiple_remaining_modes():
    occupations = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.7, 0.4]], dtype=torch.float64)
    diagonal = (2 * occupations + 1).repeat(1, 2)
    cov = diagonal.diag_embed()
    mean = torch.zeros(2, 6, 1, dtype=torch.float64)
    cir = dq.QumodeCircuit(3, [cov, mean], backend='gaussian', den_mat=True)
    cir(stepwise=True)
    state, prob, info = post_select_tensor(cir, 1, 0, cutoff=4, return_info=True, normalize=False)
    assert info['representation'] == 'dm'
    assert state.shape == (2, 4, 4, 4, 4)
    expected = torch.zeros_like(state)
    for batch in range(2):
        n0, n1, n2 = occupations[batch]
        for i in range(4):
            for j in range(4):
                expected[batch, i, j, i, j] = n0**i / (1 + n0) ** (i + 1) / (1 + n1) * n2**j / (1 + n2) ** (j + 1)
    torch.testing.assert_close(state, expected)
    torch.testing.assert_close(prob, 1 / (1 + occupations[:, 1]))
    ket_cir = dq.QumodeCircuit(3, [cov, mean], backend='gaussian', den_mat=False)
    ket_cir(stepwise=True)
    with pytest.raises(ValueError, match='den_mat=True'):
        post_select_tensor(ket_cir, 1, 0, normalize=False)


def optical_circuit(backend, cutoff, loss=False):
    cir = dq.QumodeCircuit(2, 'vac', cutoff=cutoff, backend=backend, basis=False, den_mat=True)
    cir.s(0, 0.22, 0.37)
    cir.s(1, 0.13, -0.23)
    cir.d(0, 0.17, 0.41)
    cir.bs([0, 1], [0.43, 0.29])
    if loss:
        cir.loss_t(0, 0.73)
        cir.loss_t(1, 0.81)
    cir.to(torch.float64)
    return cir


@pytest.mark.parametrize('loss', [False, True])
def test_gaussian_against_fock_evolution(loss):
    gaussian = optical_circuit('gaussian', 3, loss)
    fock = optical_circuit('fock', 12, loss)
    gaussian()
    fock()
    gs, gp = post_select_tensor(gaussian, 0, 1, cutoff=5, normalize=False)
    fs, fp = post_select_tensor(fock, 0, 1, cutoff=5, normalize=False)
    torch.testing.assert_close(gs, fs, atol=2e-8, rtol=2e-6)
    torch.testing.assert_close(gp, fp, atol=2e-8, rtol=2e-6)
    torch.testing.assert_close(gs, gs.mH, atol=1e-12, rtol=1e-10)
    assert torch.linalg.eigvalsh(gs).min() > -1e-12


@pytest.mark.parametrize('backend,density', [('gaussian', False), ('fock', False), ('fock', True)])
def test_forward_probability_cache_and_invalidation(backend, density):
    cir = dq.QumodeCircuit(2, 'vac', cutoff=4, backend=backend, basis=False, den_mat=density)
    cir.s2([0, 1], 0.2)
    cir.to(torch.float64)
    with pytest.raises(RuntimeError, match='forward'):
        post_select_tensor(cir, 0, 1, normalize=False)
    cir()
    expected = post_select_tensor(cir, 0, 1, normalize=False)
    probabilities = cir(is_prob=True)
    actual = post_select_tensor(cir, 0, 1, normalize=False)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b)
    assert cir.state is probabilities
    cir.ps(0, 0.1)
    cir.to(torch.float64)
    with pytest.raises(RuntimeError, match='forward'):
        post_select_tensor(cir, 0, 1, normalize=False)
    cir()
    cir.to(torch.float64)
    with pytest.raises(RuntimeError, match='forward'):
        post_select_tensor(cir, 0, 1, normalize=False)
    cir()
    cir.set_init_state('vac')
    with pytest.raises(RuntimeError, match='forward'):
        post_select_tensor(cir, 0, 1, normalize=False)


@pytest.mark.parametrize('backend', ['gaussian', 'fock'])
def test_encoded_batch_and_gradient(backend):
    cir = dq.QumodeCircuit(2, 'vac', cutoff=5, backend=backend, basis=False)
    cir.s2([0, 1], encode=True)
    cir.to(torch.float64)
    data = torch.tensor([[0.21, 0.13], [0.34, 0.27]], dtype=torch.float64, requires_grad=True)
    cir(data)
    state, prob = post_select_tensor(cir, 0, 1, normalize=False)
    assert state.shape == ((2, 5, 5) if backend == 'fock' else (2, 5))
    reference_prob = (1 - data[:, 0].tanh().square()) * data[:, 0].tanh().square()
    torch.testing.assert_close(prob, reference_prob)
    actual_grad = torch.autograd.grad(prob.sum(), data, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(reference_prob.sum(), data)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-8, rtol=1e-6)
    for i in range(2):
        cir(data[i])
        single_state, single_prob = post_select_tensor(cir, 0, 1, normalize=False)
        torch.testing.assert_close(single_state[0], state[i])
        torch.testing.assert_close(single_prob[0], prob[i])


@pytest.mark.parametrize('den_mat', [False, True])
def test_gradcheck_zero_displacement(den_mat):
    cov, mean = moments_coherent([0.3 + 0.2j, 0.0j])
    mean.requires_grad_()

    def result(displacement):
        state, prob = conditional_fock(cov, displacement, (0,), (1,), 3, den_mat=den_mat)
        return state, prob

    assert torch.autograd.gradcheck(result, (mean,), atol=1e-5, rtol=1e-4)
    state, _ = result(mean)
    value = state[0, 1] if not den_mat else state[0, 1, 0]
    gradient = torch.autograd.grad(value.real, mean)[0]
    assert gradient[0, 1, 0].abs() > 0.01


def test_gradcheck_mixed_loss_parameter():
    cir = optical_circuit('gaussian', 3)
    cov, mean = cir()
    eta = torch.tensor(0.68, dtype=torch.float64, requires_grad=True)

    def result(transmittance):
        lossy_cov = transmittance * cov + (1 - transmittance) * torch.eye(4, dtype=cov.dtype)
        lossy_mean = transmittance.sqrt() * mean
        state, prob = conditional_fock(lossy_cov, lossy_mean, (0,), (1,), 4, den_mat=True)
        return state / prob[:, None, None], prob

    assert torch.autograd.gradcheck(result, (eta,), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('backend', ['fock', 'gaussian'])
@pytest.mark.parametrize('den_mat', [False, True])
@pytest.mark.parametrize('nmode', [1, 2])
def test_requires_measured_and_remaining_modes(backend, den_mat, nmode):
    cir = dq.QumodeCircuit(nmode, 'vac', cutoff=3, backend=backend, basis=False, den_mat=den_mat)
    cir()
    with pytest.raises(ValueError, match='at least one measured mode'):
        cir.post_select([], [])
    with pytest.raises(ValueError, match='at least one remaining mode'):
        cir.post_select(list(reversed(range(nmode))), [0] * nmode)


@pytest.mark.parametrize('den_mat', [False, True])
@pytest.mark.parametrize('is_prob', [False, True])
def test_fock_post_select_options_preserve_state_type_and_normalization(den_mat, is_prob):
    cir = dq.QumodeCircuit(2, 'vac', cutoff=5, backend='fock', basis=False, den_mat=den_mat)
    cir.s2([0, 1], 0.3)
    cir.to(torch.float64)
    cir(is_prob=is_prob)
    reference, expected_probability = cir.post_select(0, 1)
    assert isinstance(reference, dq.FockState) and reference.den_mat
    for options in ({'return_info': True}, {'cutoff': 5}, {'keep': [1], 'return_info': True}):
        state, probability, *_ = cir.post_select(0, 1, **options)
        assert isinstance(state, dq.FockState) and state.den_mat
        torch.testing.assert_close(state.state, reference.state)
        torch.testing.assert_close(probability, expected_probability)
    torch.testing.assert_close(
        reference.state.diagonal(dim1=-2, dim2=-1).sum(-1).real, torch.ones(1, dtype=torch.float64)
    )
    unnormalized, probability = cir.post_select(0, 1, keep=[1], normalize=False)
    torch.testing.assert_close(unnormalized.state, reference.state * probability[:, None, None])


@pytest.mark.parametrize('den_mat', [False, True])
def test_submode_selection_with_cutoff_normalization_and_info(den_mat):
    circuit = dq.QumodeCircuit(2, [1, 1], cutoff=3, basis=False, den_mat=den_mat)
    circuit.bs([0, 1], [0.4, 0.1])
    circuit.set_hardware_model(dqp.HardwareModel().input(1).submodes([0.8j, 0.6]))
    circuit.to(torch.float64)
    probabilities = circuit(is_prob=True)
    reference, probability = circuit.post_select(1, 1, keep=0, lo=[0.8j, 0.6])
    state, actual_probability, info = circuit.post_select(
        1, 1, keep=0, lo=[0.8j, 0.6], cutoff=2, normalize=False, return_info=True
    )
    expected_probability = sum(p for key, p in probabilities.items() if key.state[1] == 1)
    torch.testing.assert_close(actual_probability, expected_probability)
    torch.testing.assert_close(state.state, reference.state[:, :2, :2] * probability[:, None, None])
    mass = state.state.diagonal(dim1=-2, dim2=-1).sum(-1).real
    torch.testing.assert_close(info['probability_in_cutoff'], mass)
    torch.testing.assert_close(info['retained_fraction'], mass / probability)
    assert info['remaining_wires'] == (0,)


@pytest.mark.parametrize('backend', ['fock', 'gaussian'])
def test_zero_probability_and_cutoff_excludes_output(backend):
    cir = dq.QumodeCircuit(2, 'vac', cutoff=3, backend=backend, basis=False)
    cir()
    state, prob, info = post_select_tensor(cir, 0, 1, return_info=True, normalize=False)
    assert torch.count_nonzero(state) == 0
    assert prob.item() == 0
    assert torch.isnan(info['retained_fraction']).all()
    with pytest.raises(ValueError, match='zero-probability'):
        post_select_tensor(cir, 0, 1, normalize=True)
    cir.s2([0, 1], 0.3)
    cir()
    state, prob, info = post_select_tensor(cir, 0, 1, cutoff=1, normalize=True, return_info=True)
    assert prob.item() > 0
    assert torch.count_nonzero(state) == 0
    assert info['retained_fraction'].item() == 0


@pytest.mark.parametrize(
    'wires,herald,cutoff',
    [
        ([0, 0], [1, 1], 3),
        ([2], [1], 3),
        ([-1], [1], 3),
        ([0], [-1], 3),
        ([0], [1, 2], 3),
        ([0.0], [1], 3),
        ([0], [1.0], 3),
        ([True], [1], 3),
        ([0], [1], 0),
        ([0], [1], 2.5),
        ([0], [1], True),
    ],
)
def test_invalid_arguments(wires, herald, cutoff):
    cir = dq.QumodeCircuit(2, 'vac', backend='gaussian')
    cir()
    with pytest.raises(ValueError):
        post_select_tensor(cir, wires, herald, cutoff=cutoff, normalize=False)


def test_unsupported_inputs():
    basis = dq.QumodeCircuit(2, [1, 0])
    with pytest.raises(NotImplementedError, match='basis=False'):
        post_select_tensor(basis, 0, 1, normalize=False)
    bosonic = dq.QumodeCircuit(2, 'vac', backend='bosonic')
    with pytest.raises(NotImplementedError):
        post_select_tensor(bosonic, 0, 1, normalize=False)
    mps = dq.QumodeCircuit(2, 'vac', cutoff=3, basis=False, mps=True)
    with pytest.raises(NotImplementedError, match='MPS'):
        post_select_tensor(mps, 0, 0, normalize=False)
    density = dq.QumodeCircuit(2, 'vac', cutoff=3, basis=False, den_mat=True)
    density()
    with pytest.raises(ValueError, match='simulated Fock'):
        post_select_tensor(density, 0, 3, normalize=False)
    with pytest.raises(ValueError, match='simulated Fock'):
        post_select_tensor(density, 0, 0, cutoff=4, normalize=False)


@pytest.mark.parametrize('hbar,kappa', [(1.0, 1.0), (3.0, 0.4)])
def test_nondefault_units(hbar, kappa):
    old = dqp.hbar, dqp.kappa
    try:
        dqp.set_hbar(hbar)
        dqp.set_kappa(kappa)
        cov, mean = moments_coherent([0.2 + 0.5j, 0.6 - 0.3j])
        cir = dq.QumodeCircuit(2, [cov, mean], backend='gaussian')
        cir(stepwise=True)
        state, prob = post_select_tensor(cir, 0, 1, cutoff=5, normalize=False)
        expected = coherent_ket(0.2 + 0.5j, 2)[1] * coherent_ket(0.6 - 0.3j, 5)
        torch.testing.assert_close(state[0], expected)
        torch.testing.assert_close(prob[0], coherent_ket(0.2 + 0.5j, 2)[1].abs().square())
        dqp.set_hbar(hbar + 1)
        with pytest.raises(RuntimeError, match='units changed'):
            post_select_tensor(cir, 0, 1, normalize=False)
    finally:
        dqp.set_hbar(old[0])
        dqp.set_kappa(old[1])


@pytest.mark.parametrize('loss', [False, True])
def test_against_thewalrus(loss):
    walrus = pytest.importorskip('thewalrus.quantum')
    cir = optical_circuit('gaussian', 5, loss)
    cov, mean = cir()
    state, prob = post_select_tensor(cir, 0, 1, cutoff=5, normalize=False)
    reference = walrus.density_matrix(
        mean[0].detach().numpy().flatten(), cov[0].detach().numpy(), post_select={0: 1}, cutoff=5
    )
    torch.testing.assert_close(state[0], torch.as_tensor(reference), atol=1e-10, rtol=1e-8)
    marginal = [0, 2]
    reference_prob = walrus.density_matrix_element(
        mean[0, marginal, 0].detach().numpy(), cov[0][marginal][:, marginal].detach().numpy(), [1], [1]
    )
    assert math.isclose(prob.item(), reference_prob.real, abs_tol=1e-10, rel_tol=1e-8)


def test_single_mode_polynomial_matches_full_density_recurrence():
    cir = dq.QumodeCircuit(3, 'vac', backend='gaussian', den_mat=True)
    cir.s(0, 0.25, 0.37)
    cir.s2([1, 2], 0.31, 0.21)
    cir.bs([0, 1], [0.4, 0.3])
    cir.d(2, 0.19, 0.7)
    cir.loss_t(1, 0.74)
    cir.to(torch.float64)
    cov, mean = cir()
    state, _ = post_select_tensor(cir, [1, 0], [1, 2], cutoff=4, normalize=False)
    a, b, c = gaussian_to_bargmann(cov, mean)
    full = _fock_coefficients(a, b, c, (3, 2, 4, 3, 2, 4))
    torch.testing.assert_close(state, full[:, 2, 1, :, 2, 1, :], atol=1e-12, rtol=1e-9)


def test_memory_guard():
    cir = dq.QumodeCircuit(3, 'vac', backend='gaussian', den_mat=True)
    cir()
    with pytest.raises(MemoryError, match='working-set'):
        post_select_tensor(cir, 0, 10000, cutoff=10000, normalize=False)


@pytest.mark.parametrize('device', ['cpu', 'cuda', 'mps'])
@pytest.mark.parametrize('backend', ['fock', 'gaussian'])
@pytest.mark.parametrize('den_mat', [False, True])
def test_device_dtype(device, backend, den_mat):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS is unavailable')
    cir = dq.QumodeCircuit(2, 'vac', cutoff=4, backend=backend, basis=False, den_mat=den_mat)
    cir.s2([0, 1], 0.2)
    cir.to(device=device, dtype=torch.float32)
    cir()
    state, prob = post_select_tensor(cir, 0, 1, cutoff=4, normalize=False)
    assert state.dtype == torch.complex64
    assert prob.dtype == torch.float32
    assert state.device.type == prob.device.type == device
    reference = (1 - math.tanh(0.2) ** 2) * math.tanh(0.2) ** 2
    assert math.isclose(prob.item(), reference, abs_tol=2e-6)


def test_batched_mixed_polynomial_and_density_forward():
    gaussian = dq.QumodeCircuit(2, 'vac', backend='gaussian', den_mat=True)
    gaussian.s2([0, 1], encode=True)
    gaussian.loss_t(1, 0.71)
    gaussian.to(torch.float64)
    data = torch.tensor([[0.21, 0.3], [0.4, -0.2]], dtype=torch.float64)
    gaussian(data)
    state, prob = post_select_tensor(gaussian, 0, 1, cutoff=4, normalize=False)
    assert state.shape == (2, 4, 4)
    for i in range(2):
        gaussian(data[i])
        single_state, single_prob = post_select_tensor(gaussian, 0, 1, cutoff=4, normalize=False)
        torch.testing.assert_close(single_state[0], state[i])
        torch.testing.assert_close(single_prob[0], prob[i])
    # Continue the conditional state in a dense density-matrix circuit, preserving batch.
    fock = dq.QumodeCircuit(1, state, cutoff=4, basis=False, den_mat=True)
    fock.ps(0, encode=True)
    fock.to(torch.float64)
    probabilities = fock(data[:, :1], is_prob=True)
    torch.testing.assert_close(probabilities, state.diagonal(dim1=-2, dim2=-1).real)


def test_correlated_multimode_density_against_walrus():
    walrus = pytest.importorskip('thewalrus.quantum')
    cir = dq.QumodeCircuit(3, 'vac', backend='gaussian', den_mat=True)
    cir.s(0, 0.23, 0.4)
    cir.s2([1, 2], 0.19, -0.3)
    cir.bs([0, 1], [0.43, 0.13])
    cir.d(2, 0.11, 0.51)
    cir.loss_t(0, 0.7)
    cir.to(torch.float64)
    cov, mean = cir()
    state, _ = post_select_tensor(cir, 1, 1, cutoff=3, normalize=False)
    reference = walrus.density_matrix(mean[0, :, 0].numpy(), cov[0].numpy(), post_select={1: 1}, cutoff=3)
    # Walrus: ket0,bra0,ket2,bra2. DeepQuantum: ket0,ket2,bra0,bra2.
    torch.testing.assert_close(state[0], torch.as_tensor(reference).permute(0, 2, 1, 3), atol=1e-10, rtol=1e-8)
    reversed_state, reversed_prob = post_select_tensor(cir, [2, 0], [1, 0], cutoff=3, normalize=False)
    ordered_state, ordered_prob = post_select_tensor(cir, [0, 2], [0, 1], cutoff=3, normalize=False)
    torch.testing.assert_close(reversed_state, ordered_state)
    torch.testing.assert_close(reversed_prob, ordered_prob)


@pytest.mark.parametrize('backend', ['gaussian', 'fock', 'bargmann'])
def test_repeated_forward_and_backward(backend):
    cir = dq.QumodeCircuit(2, 'vac', cutoff=5, backend=backend, basis=False)
    cir.s2([0, 1], encode=True)
    cir.to(torch.float64)
    parameters = torch.nn.Parameter(torch.tensor([0.18, 0.2], dtype=torch.float64))
    optimizer = torch.optim.Adam([parameters], lr=0.03)
    for _ in range(2):
        optimizer.zero_grad()
        cir(parameters)
        _, probability = cir.post_select(0, 1)
        (-probability.log().mean()).backward()
        assert torch.isfinite(parameters.grad).all()
        optimizer.step()


def test_no_rerun_of_noisy_gates():
    cir = dq.QumodeCircuit(2, 'vac', cutoff=4, basis=False)
    cir.set_hardware_model(dqp.HardwareModel(seed=7).gaussian(sigma=0.05))
    cir.s2([0, 1], 0.2)
    calls = []
    handle = cir.hardware_model.register_forward_hook(lambda *args: calls.append(1))
    try:
        cir(is_prob=True)
        first = post_select_tensor(cir, 0, 1, normalize=False)
        second = post_select_tensor(cir, 0, 1, normalize=False)
        assert len(calls) == 1
        for left, right in zip(first, second, strict=True):
            torch.testing.assert_close(left, right)
    finally:
        handle.remove()


def test_broadcast_covariance_with_batched_means():
    cov, mean1 = moments_coherent([0.2 + 0.3j, 0.4 - 0.1j])
    _, mean2 = moments_coherent([0.5 - 0.2j, 0.3 + 0.6j])
    cir = dq.QumodeCircuit(2, [cov, torch.cat((mean1, mean2))], backend='gaussian')
    cir(stepwise=True)
    state, prob = post_select_tensor(cir, 0, 1, cutoff=4, normalize=False)
    for i, mean in enumerate([mean1, mean2]):
        expected_state, expected_prob = conditional_fock(cov, mean, (0,), (1,), 4)
        torch.testing.assert_close(state[i], expected_state[0])
        torch.testing.assert_close(prob[i], expected_prob[0])


@pytest.mark.parametrize('density', [False, True])
def test_gaussian_reference_and_cutoff(density):
    circuits = []
    for backend in ['gaussian', 'bargmann']:
        cir = dq.QumodeCircuit(2, 'vac', backend=backend, den_mat=density, cutoff=5)
        cir.s2([0, 1], 0.31, 0.27)
        cir.to(torch.float64)
        cir()
        circuits.append(cir)
    reference, expected_p = circuits[0].post_select(0, 2, normalize=True)
    cir = circuits[1]
    original = cir.state
    state, p, info = cir.post_select(0, 2, cutoff=1, normalize=True, return_info=True)
    assert isinstance(state, BargmannState)
    assert cir.state is original
    assert state.cutoff == 1
    assert state.nmode == 1
    assert info['remaining_wires'] == (1,)
    assert info['retained_fraction'] is None
    assert info['probability_in_cutoff'] is None
    torch.testing.assert_close(p, expected_p)
    torch.testing.assert_close(state.to_fock(5), reference)
    torch.testing.assert_close(state.norm(), torch.ones_like(p))
    assert state.to_fock().abs().max() < 1e-12


@pytest.mark.parametrize('density', [False, True])
def test_existing_auxiliary_unsorted_batch_and_joint_weight(density):
    torch.manual_seed(7)
    psi = torch.randn(2, 3, 3, 3, dtype=torch.complex128)
    psi = psi / psi.flatten(1).norm(dim=1).reshape(2, 1, 1, 1)
    psi = psi * 0.4  # intentional prior event weight
    cir = dq.QumodeCircuit(3, BargmannState(psi, nmode=3, den_mat=density), backend='bargmann')
    cir()
    state, p, info = cir.post_select([2, 0], [1, 2], return_info=True, normalize=False)
    selected = psi[:, 2, :, 1]
    expected = selected[:, :, None] * selected.conj()[:, None, :] if density else selected
    torch.testing.assert_close(state.to_fock(3), expected)
    torch.testing.assert_close(p, selected.abs().square().sum(-1))
    torch.testing.assert_close(info['conditional_probability'], p / 0.16)
    assert info['remaining_wires'] == (1,)


def test_superposition_axis():
    # Two coherent exponentials, preserving complex interference between L terms.
    alpha = torch.tensor([0.2 + 0.1j, -0.2 - 0.1j], dtype=torch.complex128)
    a = torch.zeros(1, 2, 2, 2, dtype=torch.complex128)
    b = torch.stack([alpha, alpha * 0.5], -1).unsqueeze(0)
    c = torch.tensor([[0.7, 0.3j]], dtype=torch.complex128)
    state = BargmannState(BargmannPolyExp(a, b, c))
    cir = dq.QumodeCircuit(2, state, backend='bargmann')
    full = cir().to_fock(8)
    result, p = cir.post_select(0, 1, normalize=False)
    torch.testing.assert_close(result.to_fock(8), full[:, 1])
    torch.testing.assert_close(p, full[:, 1].abs().square().sum(-1), atol=1e-12, rtol=1e-9)


@pytest.mark.parametrize('density', [False, True])
def test_two_rounds_and_gradient(density):
    def run(r, theta):
        first = dq.QumodeCircuit(3, 'vac', backend='bargmann', den_mat=density)
        first.s2([0, 1], r, 0.0)
        first.to(torch.float64)
        first()
        state, p1 = first.post_select(0, 1, normalize=False)
        second = dq.QumodeCircuit(2, state, backend='bargmann')
        second.bs([0, 1], [theta, 0.0])
        second.to(torch.float64)
        second()
        result, p12, info = second.post_select(1, 0, return_info=True, normalize=False)
        torch.testing.assert_close(info['input_weight'], p1)
        torch.testing.assert_close(result.norm(), p12)
        expected = (1 - r.tanh() ** 2) * r.tanh() ** 2 * theta.cos() ** 2
        torch.testing.assert_close(p12, expected.reshape(1))
        return p12

    r = torch.tensor(0.29, dtype=torch.float64, requires_grad=True)
    theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(run, (r, theta), atol=1e-5)


def test_bargmann_zero_event_and_state_invalidation():
    cir = dq.QumodeCircuit(2, 'vac', backend='bargmann')
    with pytest.raises(RuntimeError):
        cir.post_select(0, 0, normalize=False)
    cir()
    zero, p = cir.post_select(0, 1, normalize=False)
    torch.testing.assert_close(p, torch.zeros_like(p))
    torch.testing.assert_close(zero.to_fock(2), torch.zeros(1, 2, dtype=zero.mat_a.dtype))
    with pytest.raises(ValueError, match='zero-probability'):
        cir.post_select(0, 1, normalize=True)
    cir.to(torch.float64)
    with pytest.raises(RuntimeError):
        cir.post_select(0, 0, normalize=False)


def test_mixed_loss_against_gaussian_and_no_fock_expansion(monkeypatch):
    results = []
    for backend in ['gaussian', 'bargmann']:
        cir = dq.QumodeCircuit(2, 'vac', backend=backend, den_mat=True, cutoff=5)
        cir.s2([0, 1], 0.3, 0.1)
        cir.loss_t(0, 0.7)
        cir.to(torch.float64)
        cir()
        if backend == 'bargmann':

            def reject(*args, **kwargs):
                raise AssertionError('No physical Fock readout during native heralding')

            with monkeypatch.context() as patch:
                patch.setattr(BargmannState, 'to_fock', reject)
                state, p = cir.post_select(0, 1, normalize=True)
            results.append((state.to_fock(5), p))
        else:
            results.append(cir.post_select(0, 1, normalize=True))
    torch.testing.assert_close(results[0][0], results[1][0])
    torch.testing.assert_close(results[0][1], results[1][1])
