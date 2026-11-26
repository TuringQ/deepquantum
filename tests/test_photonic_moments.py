"""Exact quadrature and photon-number moments, including non-Gaussian states."""

import math

import pytest
import torch

import deepquantum as dq
import deepquantum.photonic as dqp


def _loss_observables(circuit, data=None):
    state = circuit(data)
    variance, mean = circuit.quadrature_moments(0, phi=0.0)
    if circuit.backend == 'gaussian':
        vacuum = circuit.get_prob([0])
    else:
        rho = state.to_fock(1) if circuit.backend == 'bargmann' else state
        vacuum = rho[:, 0, 0].real
    return torch.stack([mean.flatten().real, variance.flatten().real, vacuum.flatten().real])


@pytest.mark.parametrize('backend', ['gaussian', 'bargmann', 'fock'])
@pytest.mark.parametrize(
    'method,values',
    [('loss_t', [1.0, 0.64]), ('loss_db', [0.0, 1.0]), ('loss', [0.0, 4.0]), ('d_channel', [1.0, 0.64])],
)
def test_loss_gradients_at_unit_transmittance(backend, method, values):
    # A displaced thermal state has <x> = 1, Var(x) = 1 + 2*nbar before loss.
    nbar = 0.1
    cir = dq.QumodeCircuit(1, 'vac', cutoff=14, backend=backend, basis=False, den_mat=True)
    cir.thermal(0, nbar=torch.tensor(nbar, dtype=torch.double))
    cir.d(0, 0.5, 0.0)
    getattr(cir, method)(0, encode=True)
    cir.to(torch.double)
    control = torch.tensor(values, dtype=torch.double, requires_grad=True)
    if method == 'd_channel':
        dark = torch.zeros_like(control)
        data = torch.stack([dark, dark, control], dim=1)
    else:
        data = control[:, None]
    observed = _loss_observables(cir, data)
    if method == 'loss':
        field = (control / 2).cos()
        power = field.square()
    else:
        power = 10 ** (-control / 10) if method == 'loss_db' else control
        field = power.sqrt()
    thermal = 1 + nbar * power
    expected = [field, 1 + 2 * nbar * power, (-0.25 * power / thermal).exp() / thermal]
    for value, reference in zip(observed, expected, strict=True):
        torch.testing.assert_close(value, reference, atol=3e-9, rtol=1e-8)
        derivative = torch.autograd.grad(value.sum(), control, retain_graph=True)[0]
        target = torch.autograd.grad(reference.sum(), control, retain_graph=True)[0]
        torch.testing.assert_close(derivative, target, atol=3e-9, rtol=1e-8)


def test_small_loss_angle_adds_vacuum_noise_to_squeezed_quadrature():
    cov = torch.diag(torch.tensor([1e-3, 1e3], dtype=torch.double))
    angle = cov.new_tensor(1e-8)
    cir = dq.QumodeCircuit(1, [cov, cov.new_zeros(2, 1)], backend='gaussian')
    cir.loss(0, angle)
    cir.double()
    cir()
    variance, _ = cir.quadrature_moments(0, phi=0.0)
    vacuum = dqp.hbar / (4 * dqp.kappa**2)
    expected = cov[0, 0] + (vacuum - cov[0, 0]) * (angle / 2).sin().square()
    torch.testing.assert_close(variance.squeeze() - cov[0, 0], expected - cov[0, 0], atol=0, rtol=1e-12)


def test_loss_on_correlated_bosonic_components_matches_dense_affine_map():
    from deepquantum.photonic.operation import Channel

    generator = torch.Generator().manual_seed(14)
    matrix = torch.randn(2, 3, 6, 6, dtype=torch.double, generator=generator)
    cov = matrix @ matrix.mT + torch.eye(6)
    mean = torch.randn(2, 3, 6, 1, dtype=torch.cdouble, generator=generator)
    weights = torch.randn(2, 3, dtype=torch.cdouble, generator=generator)
    angle = torch.tensor(4.2, dtype=torch.double, requires_grad=True)
    loss = dqp.PhotonLoss(angle, nmode=3, wires=1)
    actual = loss([cov, mean, weights])
    dense = Channel.op_cv(loss, [cov, mean, weights])
    torch.testing.assert_close(actual, dense)
    assert actual[2] is weights
    for value, reference in zip(actual[:2], dense[:2], strict=True):
        torch.testing.assert_close(
            torch.autograd.grad(value.real.sum(), angle, retain_graph=True),
            torch.autograd.grad(reference.real.sum(), angle, retain_graph=True),
        )


@pytest.mark.parametrize('convention,value', [('t', 1.0), ('db', 0.0)])
def test_loss_control_and_angle_optimizer(convention, value):
    control = torch.tensor(value, dtype=torch.double, requires_grad=True)
    loss = dqp.PhotonLoss(control, convention=convention)
    state = [torch.eye(2, dtype=torch.double), torch.ones(2, 1, dtype=torch.double)]
    target = 0.5 if convention == 't' else -math.log(10) / 20
    for _ in range(2):
        mean = loss(state)[1][0, 0]
        torch.testing.assert_close(torch.autograd.grad(mean, control)[0], control.new_tensor(target))
        loss.load_state_dict(loss.state_dict())
    other = control.new_tensor(0.64 if convention == 't' else -10 * math.log10(0.64))
    updated = torch.func.functional_call(loss, {'gate.' + convention: other}, (state,), strict=True)
    torch.testing.assert_close(updated[1], torch.full_like(state[1], 0.8))
    restored = dqp.PhotonLoss(convention=convention).double()
    restored.load_state_dict(loss.state_dict())
    torch.testing.assert_close(restored(state), loss(state))

    # The optimizer uses the signed angle, including angles beyond pi.
    trainable = dqp.PhotonLoss(control, convention=convention, requires_grad=True)
    assert dict(trainable.named_parameters()) == {'gate.theta': trainable.theta}
    torch.testing.assert_close(trainable.theta, loss.theta)
    trainable.load_state_dict({'gate.theta': control.new_tensor(4.0)})
    optimizer = torch.optim.SGD(trainable.parameters(), lr=0.1)
    mean = trainable(state)[1][0, 0]
    mean.backward()
    torch.testing.assert_close(trainable.theta.grad, -(trainable.theta / 2).sin() / 2)
    optimizer.step()
    torch.testing.assert_close(trainable(state)[1], (trainable.theta / 2).cos() * state[1])
    assert trainable(state)[1][0, 0] < 0


@pytest.mark.parametrize('backend', ['fock', 'gaussian', 'bosonic', 'bargmann'])
def test_registered_quadratures_and_explicit_queries(backend):
    cir = dq.QumodeCircuit(3, 'vac', backend=backend, cutoff=6, basis=False)
    cir.d(0, encode=True)
    cir.d(2, 0.12, 0.5)
    theta = torch.tensor(-0.32, dtype=torch.double, requires_grad=True)
    cir.homodyne(2, phi=theta)
    cir.homodyne_x(0)
    cir.double()
    cir(torch.tensor([[0.1, 0.3], [-0.2, 0.6]], dtype=torch.double))

    # Build the independent projection from the complete xxpp pair in spatial order.
    cov, mean = cir.quadrature_moments(wires=[0, 1, 2])
    projected = mean.new_zeros(2, 6)
    projected[0, 2], projected[0, 5], projected[1, 0] = theta.cos(), theta.sin(), 1
    expected = projected @ mean
    torch.testing.assert_close(cir.quadrature_moments(), (projected @ cov @ projected.mT, expected))
    registered = cir.quadrature_mean().real
    torch.testing.assert_close(registered, expected)
    torch.testing.assert_close(
        torch.autograd.grad(registered.sum(), theta)[0], torch.autograd.grad(expected.sum(), theta)[0]
    )
    if backend in ('gaussian', 'bargmann'):
        powers = [(0, 1), (1, 0), (0, 1)]
        torch.testing.assert_close(cir.quadrature_moments(powers=powers), expected[:, [1, 0, 1], 0])

    with pytest.warns(UserWarning, match='override registered Homodyne'):
        torch.testing.assert_close(cir.quadrature_mean(wires=0).real, mean[:, :1])
    with pytest.warns(UserWarning, match='override registered Homodyne'):
        explicit = cir.quadrature_mean(phi=[theta.reshape(1), 0.0, 0.0]).real
    expected_x = mean[:, :3].clone()
    expected_x[:, 0] = mean[:, 0] * theta.cos() + mean[:, 3] * theta.sin()
    torch.testing.assert_close(explicit, expected_x)
    torch.testing.assert_close(cir.quadrature_moments(phi=[theta, 0.0, 0.0])[1], expected_x)
    torch.testing.assert_close(
        torch.autograd.grad(explicit.sum(), theta)[0], torch.autograd.grad(expected_x.sum(), theta)[0]
    )


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
    angles = torch.tensor([0.3, -0.7], dtype=torch.double)
    projection = torch.cat([angles.cos().diag(), angles.sin().diag()], dim=-1)
    torch.testing.assert_close(
        actual.quadrature_moments([1, 0], phi=angles),
        (projection @ expected[0] @ projection.mT, projection @ expected[1]),
        atol=2e-8,
        rtol=2e-7,
    )
    if backend in ('fock', 'bosonic'):
        with pytest.raises(NotImplementedError, match='powers'):
            actual.quadrature_moments([1, 0], phi=angles, powers=[(2, 2)])
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
    direction = torch.tensor([[math.cos(0.3), 0, math.sin(0.3), 0]], dtype=torch.double)
    torch.testing.assert_close(
        cir.quadrature_moments(0, phi=0.3, centered=False, normalize=False),
        (0.2 * direction @ second @ direction.mT, 0.2 * direction @ mean),
    )
    number = annihilation.mH @ annihilation
    numbers = [torch.kron(number, eye), torch.kron(eye, number)]
    expected_mean = torch.stack([expectation(operator) for operator in numbers], -1)
    expected_var = torch.stack([expectation(operator @ operator) for operator in numbers], -1) - expected_mean.square()
    phi = torch.tensor([0.2, -0.4], dtype=torch.double)
    quadrature = mean[:, :2, 0] * phi.cos() + mean[:, 2:, 0] * phi.sin()
    for normalize in (False, True):
        expected = quadrature if normalize else 0.2 * quadrature
        torch.testing.assert_close(cir.quadrature_mean(phi=phi[:, None], normalize=normalize), expected.unsqueeze(-1))
        if backend == 'bargmann':
            terms = [(0, 1), (0, 1)]
            for centered in (False, True):
                products = state.quadrature_moments(phi=phi, powers=terms, normalize=normalize, centered=centered)
                target = expected[:, [1, 1]]
                torch.testing.assert_close(products, torch.zeros_like(target) if centered else target)
    legacy = quadrature if backend == 'bargmann' else 0.2 * quadrature
    torch.testing.assert_close(cir.quadrature_mean(phi=phi), legacy.unsqueeze(-1))
    if backend == 'bargmann':
        torch.testing.assert_close(state.photon_number_mean_var(), (expected_mean, expected_var))
    else:
        first = 0.2 * expected_mean
        variance = 0.2 * (expected_var + expected_mean.square()) - first.square()
        torch.testing.assert_close(cir.photon_number_mean_var(), (first, variance))


@pytest.mark.parametrize('backend', ['fock', 'gaussian', 'bosonic', 'bargmann'])
@pytest.mark.parametrize('batch', [1, 3])
def test_scalar_readouts_preserve_batch_and_mode_axes(backend, batch):
    cir = dq.QumodeCircuit(2, 'vac', backend=backend, cutoff=4, basis=False, den_mat=True)
    cir.d(0, encode=True)
    cir.double()
    data = torch.tensor([[0.12, 0.3]], dtype=torch.double).expand(batch, -1)
    cir(data)
    expected = []
    for wires in ([0], [0, 1]):
        mean = cir.quadrature_mean(wires, phi=0.2)
        number, variance = cir.photon_number_mean_var(wires)
        assert mean.shape == (batch, len(wires), 1)
        assert number.shape == variance.shape == (batch, len(wires))
        expected.append((mean, number, variance))
    if backend in ('fock', 'gaussian'):
        cir(data, is_prob=True)
        for wires, values in zip(([0], [0, 1]), expected, strict=True):
            actual = cir.quadrature_mean(wires, phi=0.2), *cir.photon_number_mean_var(wires)
            torch.testing.assert_close(actual, values)


@pytest.mark.parametrize('den_mat', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('angle', [0.0, 0.4])
def test_fock_single_mode_moments_keep_batch_weight_and_phase_gradient(den_mat, dtype, angle):
    occupation = torch.tensor([0.1, 0.4, 0.8], dtype=dtype, requires_grad=True)
    weight = torch.tensor([0.2, 0.6, 0.1], dtype=dtype, requires_grad=True)
    ket = torch.stack([(1 - occupation).sqrt(), 1j * occupation.sqrt()], -1) * weight.sqrt()[:, None]
    source = ket[:, :, None] * ket[:, None, :].conj() if den_mat else ket
    cir = dq.QumodeCircuit(1, source, cutoff=2, basis=False, den_mat=den_mat)
    phi = torch.tensor(angle, dtype=dtype, requires_grad=True)
    first = weight * occupation
    expected_number = first[:, None], (first - first.square())[:, None]
    expected_quadrature = weight * dqp.hbar**0.5 / dqp.kappa * (occupation * (1 - occupation)).sqrt() * phi.sin()
    for is_prob in (False, True):
        cir(is_prob=is_prob)
        torch.testing.assert_close(cir.photon_number_mean_var(), expected_number)
        actual = cir.quadrature_mean(phi=phi)
        torch.testing.assert_close(actual, expected_quadrature[:, None, None])
    actual_grad = torch.autograd.grad(actual.sum(), phi, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected_quadrature.sum(), phi)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    number = cir.photon_number_mean_var()[0].sum()
    grad = torch.autograd.grad(number, (occupation, weight))
    torch.testing.assert_close(grad[0], weight)
    torch.testing.assert_close(grad[1], occupation)
    cir(state=torch.zeros_like(source))
    assert torch.count_nonzero(cir.quadrature_mean()) == 0
    assert all(torch.count_nonzero(value) == 0 for value in cir.photon_number_mean_var())


@pytest.mark.parametrize('den_mat', [False, True])
@pytest.mark.parametrize('layout', ['independent', 'blocks', 'joint'])
def test_fock_spatial_photon_moments_retain_probability_mass(den_mat, layout):
    profiles = {
        'independent': None,
        'blocks': [[0.8, 0, 0.6j], [0, 1, 0], [0.6j, 0, 0.8]],
        'joint': [[1, 1, 1], [1, complex(-0.5, 3**0.5 / 2), complex(-0.5, -(3**0.5) / 2)]],
    }
    ratios = [1, 0.5] if layout == 'joint' else [1, 0.5, 0.3]
    cir = dq.QumodeCircuit(1, 'vac', cutoff=3, basis=False, den_mat=den_mat)
    cir.s(0, encode=True)
    cir.set_hardware_model(dqp.HardwareModel().s(0).submodes(profiles[layout], relative_r=ratios))
    cir.double()
    probabilities = cir(torch.tensor([[0.5, 0.2], [0.7, 0.1]], dtype=torch.double), is_prob=True)
    probability = torch.stack(list(probabilities.values()))
    weight = probability.sum(0)
    assert (weight < 0.99).all()
    numbers = torch.tensor([pattern.state.item() for pattern in probabilities], dtype=torch.double)[:, None]
    mean = (numbers * probability).sum(0)
    variance = (numbers.square() * probability).sum(0) - mean.square()
    torch.testing.assert_close(cir.photon_number_mean_var(), (mean[:, None], variance[:, None]))


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
    torch.testing.assert_close(cir.quadrature_mean(normalize=True).real, expected_mean[:, :1])
    torch.testing.assert_close(cir.quadrature_mean().real, first[:, :1])


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
            assert torch.count_nonzero(readout.quadrature_mean(normalize=False)[0]) == 0
            with pytest.raises(ValueError, match='positive'):
                readout.quadrature_mean(normalize=True)
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
    with pytest.raises(ValueError, match='distinct'):
        state.quadrature_moments([1, 1], [0, math.pi / 2], powers=[(1, 1)])
    with pytest.raises(ValueError, match='phi'):
        state.quadrature_moments(phi=[[0, 0.2]], powers=[(2, 2)])
    for invalid in ([], [4, 0], [(4,)], [(2, 0.2)], [(1, -1)], [(torch.tensor(True), 0)]):
        with pytest.raises(ValueError, match='powers'):
            state.quadrature_moments(powers=invalid)
    state = dq.BargmannState(torch.zeros(1, 2, dtype=torch.complex128), nmode=1)
    with pytest.raises(ValueError, match='positive'):
        state.photon_number_mean_var()


@pytest.mark.parametrize('backend', ['gaussian', 'bargmann', 'fock'])
@pytest.mark.parametrize('auxiliary', [None, 0.0, 0.31])
def test_loss_observable_endpoint_expansion(backend, auxiliary):
    # T=0 pure loss or T=1 coupling to a coherent auxiliary. Expand the whole observable in the signed field.
    cir = dq.QumodeCircuit(1, 'vac', cutoff=14, backend=backend, basis=False, den_mat=True)
    cir.thermal(0, nbar=torch.tensor(0.1, dtype=torch.double))
    cir.d(0, 0.5, 0.0)
    theta = torch.tensor(math.pi if auxiliary is None else 0.0, dtype=torch.double, requires_grad=True)
    if auxiliary is None:
        cir.loss(0, theta)
    else:
        cir.d_channel(0, amplitude=auxiliary, phase=0.0, t=theta, convention='loss_theta')
    cir.double()
    values = _loss_observables(cir).squeeze(-1)
    first = torch.stack([torch.autograd.grad(y, theta, create_graph=True)[0] for y in values])
    second = torch.stack([torch.autograd.grad(g, theta, retain_graph=True)[0] for g in first])
    # T = cos²(theta/2). At theta=pi the sqrt(T) and T coefficients are -2 f' and 2 f''.
    sign = -2 if auxiliary is None else 2
    root, slope = sign * first, -sign * second
    if auxiliary is None:
        torch.testing.assert_close(root, theta.new_tensor([1.0, 0.0, 0.0]), atol=3e-9, rtol=1e-8)
        torch.testing.assert_close(slope, theta.new_tensor([0.0, 0.2, -0.35]), atol=3e-9, rtol=1e-8)
    else:
        # Independent power series in q=sqrt(1-T) for a displaced thermal state.
        q = theta.new_zeros((), requires_grad=True)
        power = 1 - q.square()
        alpha = 0.5 * power.sqrt() + auxiliary * q
        thermal = 1 + 0.1 * power
        for i, value in enumerate([2 * alpha, 1 + 0.2 * power, (-alpha.square() / thermal).exp() / thermal]):
            (g,) = torch.autograd.grad(value, q, create_graph=True, retain_graph=True)
            (h,) = torch.autograd.grad(g, q, retain_graph=True)
            torch.testing.assert_close(root[i], g, atol=5e-8, rtol=1e-7)
            torch.testing.assert_close(slope[i], -h / 2, atol=5e-8, rtol=1e-7)


@pytest.mark.parametrize('backend', ['gaussian', 'bargmann', 'fock'])
def test_power_derivative_of_displaced_thermal_state_at_complete_loss(backend):
    cir = dq.QumodeCircuit(1, 'vac', cutoff=14, backend=backend, basis=False, den_mat=True)
    cir.thermal(0, nbar=torch.tensor(0.1, dtype=torch.double))
    cir.d(0, 0.5, 0.0)
    cir.loss_t(0, encode=True)
    cir.double()
    power = torch.tensor([[0.0], [0.36]], dtype=torch.double, requires_grad=True)
    mean, variance, vacuum = _loss_observables(cir, power)
    for observed, expected in [(variance, [0.2, 0.2]), (mean, [math.inf, 1 / 1.2])]:
        (gradient,) = torch.autograd.grad(observed.sum(), power, retain_graph=True)
        torch.testing.assert_close(gradient[:, 0], power.new_tensor(expected), atol=4e-9, rtol=1e-8)
    # P(0) = exp[-0.25 T/(1+0.1 T)] / (1+0.1 T).
    (vacuum_gradient,) = torch.autograd.grad(vacuum[0], power)
    torch.testing.assert_close(vacuum_gradient[:, 0], power.new_tensor([-0.35, 0.0]), atol=4e-9, rtol=1e-8)


@pytest.mark.parametrize('backend', ['gaussian', 'bargmann', 'fock'])
def test_bright_displacement_endpoint_variance(backend):
    cir = dq.QumodeCircuit(1, 'vac', cutoff=6, backend=backend, basis=False, den_mat=True)
    cir.thermal(0, nbar=torch.tensor(0.1, dtype=torch.double))
    cir.d(0, 0.5, 0.0)
    cir.d_channel(0, encode=True)
    cir.double()
    data = torch.tensor([[0.31, 0, 0], [0.31, 0, 1]], dtype=torch.double, requires_grad=True)
    cir(data)
    covariance, mean = cir.quadrature_moments(0, phi=0.0)
    (dmean,) = torch.autograd.grad(mean.sum(), data, retain_graph=True)
    (dcov,) = torch.autograd.grad(covariance.sum(), data)
    torch.testing.assert_close(dmean[:, -1], data.new_tensor([math.inf, -math.inf]))
    # At finite cutoff, projection after displacement changes even centered moments.
    # The projected coherent state at T=0 and the signed-angle expansion at T=1
    # both have a resolved nonzero first-order field coefficient at cutoff six.
    expected = [-math.inf, math.inf] if backend == 'fock' else [0.2, 0.2]
    torch.testing.assert_close(dcov[:, -1], data.new_tensor(expected), atol=1e-12, rtol=1e-12)


def test_power_derivative_of_user_objective_across_batch_rows():
    cir = dq.QumodeCircuit(1, 'vac', backend='gaussian')
    cir.d(0, 0.5, 0.0)
    cir.loss_t(0, encode=True)
    cir.double()
    power = torch.zeros(2, 1, dtype=torch.double, requires_grad=True)
    cir(power)
    # (sqrt(T0) + sqrt(T1))² has partial derivative 1 along each axis at (0, 0).
    objective = cir.quadrature_mean(0).sum().square()
    (gradient,) = torch.autograd.grad(objective, power, create_graph=True)
    torch.testing.assert_close(gradient, torch.ones_like(power))


@pytest.fixture(
    params=[
        ('cpu', torch.double),
        pytest.param(
            ('mps', torch.float), marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS')
        ),
    ]
)
def gradient_device(request):
    return request.param


def _require_complex_solve(device):
    if device == 'mps':
        matrix = torch.eye(1, device=device, dtype=torch.complex64)
        try:
            torch.linalg.solve(matrix, matrix)
        except RuntimeError as error:
            if 'MPS only supports floats' not in str(error):
                raise
            pytest.skip('This PyTorch MPS backend requires a real matrix for torch.linalg.solve')


@pytest.mark.parametrize('backend', ['gaussian', 'bargmann', 'fock'])
@pytest.mark.parametrize('nloss', [1, 2])
def test_transmittance_grad_of_complete_objective(backend, nloss, gradient_device):
    # One field squared and two transmitted fields in succession both give T.
    device, dtype = gradient_device
    if backend == 'bargmann':
        _require_complex_solve(device)
    cir = dq.QumodeCircuit(1, 'vac', cutoff=10, backend=backend, basis=False, den_mat=True)
    cir.d(0, 0.5, 0.0)
    for _ in range(nloss):
        cir.loss(0, encode=True)
    cir.to(dtype).to(device)
    for value in (0.0, 1.2, math.pi):
        angle = torch.tensor(value, dtype=dtype, device=device, requires_grad=True)
        cir(angle.expand(nloss))
        mean = cir.quadrature_mean(0).sum()
        observable = mean.square() if nloss == 1 else mean
        torch.testing.assert_close(dqp.transmittance_grad(observable, angle), angle.new_ones(()))


@pytest.mark.parametrize('backend', ['gaussian', 'bargmann', 'fock'])
def test_transmittance_grad_bright_displacement(backend):
    cir = dq.QumodeCircuit(1, 'vac', cutoff=6, backend=backend, basis=False, den_mat=True)
    cir.thermal(0, nbar=torch.tensor(0.1, dtype=torch.double))
    cir.d(0, 0.5, 0.0)
    cir.d_channel(0, convention='loss_theta', encode=True)
    cir.double()
    for value in (0.0, math.pi):
        angle = torch.tensor(value, dtype=torch.double, requires_grad=True)
        cir(torch.stack([angle.new_tensor(0.31), angle.new_zeros(()), angle]))
        variance, mean = cir.quadrature_moments(0, phi=0.0)
        slope = -math.inf if value == 0 else math.inf
        torch.testing.assert_close(dqp.transmittance_grad(mean.sum(), angle), angle.new_tensor(slope))
        expected = -slope if backend == 'fock' else 0.2
        torch.testing.assert_close(dqp.transmittance_grad(variance.sum(), angle), angle.new_tensor(expected))


def test_transmittance_grad_of_fock_basis_survival():
    cir = dq.QumodeCircuit(1, [1], cutoff=2)
    cir.loss(0, encode=True)
    cir.double()
    angle = torch.tensor(torch.pi, dtype=torch.double, requires_grad=True)
    survival = cir(angle.reshape(1), is_prob=True)[dq.FockState([1])].sum()
    torch.testing.assert_close(dqp.transmittance_grad(survival, angle), angle.new_ones(()))


@pytest.mark.parametrize('detector', ['pnrd', 'threshold', 'click'])
@pytest.mark.parametrize('coordinate', ['power', 'angle'])
def test_full_loss_gaussian_count_gradients(detector, coordinate, gradient_device):
    device, dtype = gradient_device
    if detector != 'click':
        _require_complex_solve(device)
    cir = dq.QumodeCircuit(1, 'vac', backend='gaussian', cutoff=3, detector=detector)
    cir.thermal(0, nbar=torch.tensor(0.1, dtype=dtype))
    cir.d(0, 0.5, 0.0)
    if coordinate == 'power':
        cir.loss_t(0, encode=True)
    else:
        cir.loss(0, encode=True)
    cir.to(dtype).to(device)
    control = torch.tensor(0.0 if coordinate == 'power' else torch.pi, dtype=dtype, device=device, requires_grad=True)
    table = cir(control.reshape(1), is_prob=True)
    patterns = torch.arange(2 if detector == 'threshold' else 3).unsqueeze(-1)
    queried = cir.get_prob(patterns)
    listed = torch.stack([table[dq.FockState(row)].squeeze() for row in patterns])
    # A transmitted photon comes from nbar=0.1 or |alpha|²=0.25. Two photons are O(T²).
    expected = control.new_tensor([-0.35, 0.35, 0.0])[: len(patterns)]
    for probabilities in (listed, queried):
        if coordinate == 'power':
            slopes = [torch.autograd.grad(p, control, retain_graph=True)[0] for p in probabilities]
        else:
            slopes = [dqp.transmittance_grad(p, control) for p in probabilities]
        torch.testing.assert_close(torch.stack(slopes), expected)
    if coordinate == 'power' and dtype == torch.double:
        with torch.no_grad():
            step = 1e-6
            cir((control + step).reshape(1))
            torch.testing.assert_close((cir.get_prob(patterns) - queried) / step, expected, atol=3e-7, rtol=1e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS is not available')
@pytest.mark.parametrize('backend', ['bargmann', 'fock'])
def test_loss_channel_mps_dtype_and_gradient(backend):
    def evaluate(device):
        if backend == 'bargmann':
            power = torch.tensor(0.4, device=device, requires_grad=True)
            kernel = dqp.PhotonLoss(power, cutoff=4, convention='t').get_bargmann().mat_a
            observable = kernel.real.sum()
            (gradient,) = torch.autograd.grad(observable, power)
        else:
            circuit = dq.QumodeCircuit(1, 'vac', cutoff=5, basis=False, den_mat=True)
            circuit.d_channel(0, encode=True)
            circuit.to(device)
            controls = torch.tensor([0.31, 0.4, 0.7], device=device, requires_grad=True)
            circuit(controls)
            observable = circuit.quadrature_mean(0, phi=0.2).sum()
            (gradient,) = torch.autograd.grad(observable, controls)
        return observable.detach().cpu(), gradient.cpu()

    torch.testing.assert_close(evaluate('mps'), evaluate('cpu'), atol=3e-6, rtol=3e-5)
