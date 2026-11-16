"""Exact quadrature and photon-number moments, including non-Gaussian states."""

import math

import pytest
import torch

import deepquantum as dq
import deepquantum.photonic as dqp


@pytest.mark.parametrize('backend', ['fock', 'gaussian', 'bosonic', 'bargmann'])
@pytest.mark.parametrize('units', [(2.0, 2**-0.5), (1.3, 0.8)])
def test_gaussian_moments_and_mode_order(backend, units, monkeypatch):
    monkeypatch.setattr(dqp, 'hbar', units[0])
    monkeypatch.setattr(dqp, 'kappa', units[1])

    def build(name, cutoff=14):
        cir = dq.QumodeCircuit(2, 'vac', cutoff=cutoff, backend=name, basis=False, den_mat=True)
        cir.s(0, encode=True)
        cir.d(1, 0.18, -0.6)
        cir.bs([0, 1], [0.5, 0.2])
        cir.loss_t(1, 0.72)
        cir.double()
        return cir

    data = torch.tensor([[0.21, 0.32], [0.12, -0.4]], dtype=torch.double)
    reference, actual = build('gaussian'), build(backend)
    cov, mean = reference(data)
    actual(data)
    indices = [1, 0, 3, 2]
    expected = cov[:, indices, :][:, :, indices], mean[:, indices, :]
    for result, target in zip(actual.quadrature_moments([1, 0]), expected, strict=True):
        torch.testing.assert_close(result, target, atol=2e-8, rtol=2e-7)
    second, first = actual.quadrature_moments([1, 0], centered=False, normalize=False)
    torch.testing.assert_close(second, expected[0] + expected[1] @ expected[1].mT, atol=2e-8, rtol=2e-7)
    torch.testing.assert_close(first, expected[1], atol=2e-8, rtol=2e-7)
    if backend == 'bargmann':
        torch.testing.assert_close(
            actual.quadrature_mean([1, 0], [0.2, -0.3]), reference.quadrature_mean([1, 0], [0.2, -0.3])
        )
        torch.testing.assert_close(actual.photon_number_mean_var(), reference.photon_number_mean_var())
    if backend in ('fock', 'gaussian'):
        actual = build(backend, cutoff=3)
        actual(data)
        expected = actual.quadrature_moments([1, 0])
        actual(data, is_prob=True)
        torch.testing.assert_close(actual.quadrature_moments([1, 0]), expected)


@pytest.mark.parametrize('backend', ['fock', 'bargmann'])
@pytest.mark.parametrize('den_mat', [False, True])
def test_non_gaussian_moments_against_operator_matrices(backend, den_mat):
    torch.manual_seed(41)
    ket = torch.randn(2, 3, 3, dtype=torch.complex128)
    ket = ket / ket.flatten(1).norm(dim=-1)[:, None, None]
    vector = ket.flatten(1)
    density = vector[:, :, None] * vector[:, None, :].conj()
    if den_mat:
        density = 0.65 * density + 0.35 * density.flip((-1, -2))
    source = 0.2 * density.reshape(2, 3, 3, 3, 3) if den_mat else math.sqrt(0.2) * ket
    cir = dq.QumodeCircuit(2, source, cutoff=3, backend=backend, basis=False, den_mat=den_mat)
    state = cir()

    # One extra level keeps quadrature products exact at the highest occupied Fock level.
    padded = torch.zeros(2, 4, 4, 4, 4, dtype=torch.complex128)
    padded[:, :3, :3, :3, :3] = density.reshape(2, 3, 3, 3, 3)
    density = padded.reshape(2, 16, 16)
    annihilation = torch.diag(torch.arange(1, 4, dtype=torch.double).sqrt().to(torch.complex128), diagonal=1)
    x = (annihilation + annihilation.mH) * dqp.hbar**0.5 / (2 * dqp.kappa)
    p = -1j * (annihilation - annihilation.mH) * dqp.hbar**0.5 / (2 * dqp.kappa)
    eye = torch.eye(4, dtype=torch.complex128)
    operators = [torch.kron(x, eye), torch.kron(eye, x), torch.kron(p, eye), torch.kron(eye, p)]

    def expectation(operator):
        return torch.einsum('bij,ji->b', density, operator).real

    mean = torch.stack([expectation(operator) for operator in operators], dim=-1).unsqueeze(-1)
    second = torch.stack(
        [
            torch.stack([expectation((left @ right + right @ left) / 2) for right in operators], -1)
            for left in operators
        ],
        dim=-2,
    )
    torch.testing.assert_close(cir.quadrature_moments(), (second - mean @ mean.mT, mean))
    for centered, normalize in ((False, True), (True, False), (False, False)):
        expected_second = second - mean @ mean.mT if centered else second
        scale = 1 if normalize else 0.2
        torch.testing.assert_close(
            cir.quadrature_moments(centered=centered, normalize=normalize),
            (scale * expected_second, scale * mean),
        )
    if backend == 'bargmann':
        number = annihilation.mH @ annihilation
        numbers = [torch.kron(number, eye), torch.kron(eye, number)]
        expected_mean = torch.stack([expectation(operator) for operator in numbers], -1)
        expected_var = (
            torch.stack([expectation(operator @ operator) for operator in numbers], -1) - expected_mean.square()
        )
        torch.testing.assert_close(state.photon_number_mean_var(), (expected_mean, expected_var))


def test_bosonic_moments_include_between_component_covariance():
    cov = torch.eye(2, dtype=torch.double).expand(1, 2, 2, 2).clone()
    mean = torch.tensor([[[[1.0], [0.0]], [[-2.0], [0.0]]]], dtype=torch.double)
    weight = torch.tensor([[0.03, 0.07]], dtype=torch.double)
    cir = dq.QumodeCircuit(1, [cov, mean, weight], backend='bosonic')
    cir(stepwise=True)
    expected_mean = torch.tensor([[[-1.1], [0.0]]], dtype=torch.double)
    expected_cov = torch.tensor([[[2.89, 0.0], [0.0, 1.0]]], dtype=torch.double)
    torch.testing.assert_close(cir.quadrature_moments(), (expected_cov, expected_mean))
    second, first = cir.quadrature_moments(centered=False, normalize=False)
    torch.testing.assert_close(second, 0.1 * (expected_cov + expected_mean @ expected_mean.mT))
    torch.testing.assert_close(first, 0.1 * expected_mean)


@pytest.mark.parametrize('backend', ['fock', 'bargmann'])
def test_threshold_weighted_moments_add_over_events(backend):
    cir = dq.QumodeCircuit(2, 'vac', cutoff=7, backend=backend, basis=False)
    cir.s2([0, 1], encode=True)
    cir.d(1, 0.2, 0.4)
    cir.double()
    data = torch.tensor([[0.0, 0.0], [0.3, 0.2]], dtype=torch.double, requires_grad=True)
    cir(data)
    expected = cir.quadrature_moments(1, centered=False, normalize=False)
    contributions = []
    for click in (0, 1):
        state, probability = cir.post_select(0, click, detector='threshold', normalize=False)
        readout = dq.QumodeCircuit(1, state)
        readout()
        contributions.append(readout.quadrature_moments(centered=False, normalize=False))
        if click:
            assert probability[0] == 0
            for centered in (False, True):
                moments = readout.quadrature_moments(centered=centered, normalize=False)
                assert all(torch.count_nonzero(value[0]) == 0 for value in moments)
            with pytest.raises(ValueError, match='positive'):
                readout.quadrature_moments()
    actual = tuple(left + right for left, right in zip(*contributions, strict=True))
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(sum(value.sum() for value in actual), data, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(sum(value.sum() for value in expected), data)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize('backend', ['fock', 'bargmann'])
def test_moment_centering_at_small_state_weight(backend):
    ket = torch.ones(1, 2, dtype=torch.complex128) * (1e-100 / math.sqrt(2))
    cir = dq.QumodeCircuit(1, ket, backend=backend, cutoff=2, basis=False)
    cir()
    cov, mean = cir.quadrature_moments()
    torch.testing.assert_close(cov[0], torch.diag(torch.tensor([1.0, 2.0], dtype=torch.double)))
    torch.testing.assert_close(mean[0], torch.tensor([[1.0], [0.0]], dtype=torch.double))
    weighted_cov, weighted_mean = cir.quadrature_moments(normalize=False)
    torch.testing.assert_close(weighted_cov / 1e-200, cov)
    torch.testing.assert_close(weighted_mean / 1e-200, mean)


def test_threshold_moments_gradients_without_fock_readout(monkeypatch):
    def reject_readout(*args, **kwargs):
        raise AssertionError('Moment readout must not truncate the Bargmann state')

    monkeypatch.setattr(dq.BargmannState, 'to_fock', reject_readout)

    def evaluate(r, displacement, transmittance):
        cir = dq.QumodeCircuit(2, 'vac', cutoff=1, backend='bargmann')
        cir.s2([0, 1], r, 0.2)
        cir.double()
        cir()
        selected, probability = cir.post_select(0, 1, detector='threshold', normalize=False)
        continued = dq.QumodeCircuit(1, selected)
        continued.loss(0, 2 * transmittance.sqrt().acos())
        continued.d(0, displacement, 0.3)
        continued.double()
        state = continued()
        cov, mean = continued.quadrature_moments()
        number, variance = state.photon_number_mean_var()
        return torch.cat([probability, cov.flatten(), mean.flatten(), number.flatten(), variance.flatten()])

    inputs = tuple(torch.tensor(value, dtype=torch.double, requires_grad=True) for value in (0.31, 0.13, 0.76))
    assert torch.autograd.gradcheck(evaluate, inputs, atol=1e-6, rtol=1e-4)
    r, displacement, transmittance = inputs
    value = evaluate(*inputs)
    number = transmittance * r.cosh().square()
    variance = transmittance**2 * r.sinh().square() * r.cosh().square() + (1 - transmittance) * number
    torch.testing.assert_close(value[-2], number + displacement.square())
    torch.testing.assert_close(value[-1], variance + displacement.square() * (2 * number + 1))


def test_bargmann_moment_validation():
    state = dq.BargmannState([1, 0])
    for wires in ([], [-1], [0, 0], [2], [0.5], [True]):
        with pytest.raises(ValueError, match='wires'):
            state.quadrature_moments(wires)
    with pytest.raises(ValueError, match='phi'):
        state.quadrature_mean(phi=[0.0, 0.1, 0.2])
    state = dq.BargmannState(torch.zeros(1, 2, dtype=torch.complex128), nmode=1)
    with pytest.raises(ValueError, match='positive'):
        state.photon_number_mean_var()
