from copy import deepcopy

import pytest
import torch

import deepquantum.photonic as dqp


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


@pytest.mark.parametrize('backend', ['gaussian', 'bosonic', 'fock'])
@pytest.mark.parametrize('phi', [None, 0.3])
def test_terminal_detector_matches_manual_optics_and_seed(backend, phi):
    cir = dqp.QumodeCircuit(2, 'vac', backend=backend, cutoff=5, basis=False, den_mat=True)
    cir.s(1, r=torch.tensor(0.2))
    cir.d(1, r=torch.tensor(0.15))
    ref = deepcopy(cir)
    model = dqp.HardwareModel(seed=14)
    model.output(1).loss(t=0.9)
    model.homodyne(1).efficiency(0.7)
    model.homodyne(1).phi.gaussian(mu=0.1, sigma=0)
    model.homodyne(1).readout(gain=-1.3, bias=0.2, noise_var=0.07)
    cir.set_hardware_model(model)
    ref.loss_t(1, torch.tensor(0.9))
    ref.loss_t(1, torch.tensor(0.7))
    ref.r(1, torch.tensor(-0.1))
    cir.double()
    ref.double()
    cir()
    ref()
    saved = deepcopy(cir.state)
    torch.manual_seed(4)
    ideal = ref.measure_homodyne(16, wires=1, phi=phi)
    nquad = 2 if phi is None and backend != 'fock' else 1
    values = ideal.reshape(1, 16, nquad)
    generator = torch.Generator().manual_seed(14)
    expected = torch.stack(
        [-1.3 * values[..., i] + 0.2 + 0.07**0.5 * torch.randn((1, 16), generator=generator) for i in range(nquad)],
        -1,
    ).reshape(ideal.shape)
    torch.manual_seed(4)
    actual = cir.measure_homodyne(16, wires=1, phi=phi)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(cir.state, saved)
    assert cir.state_measured is None
    model.manual_seed(14)
    torch.manual_seed(4)
    torch.testing.assert_close(cir.measure_homodyne(16, wires=1, phi=phi), actual)


@pytest.mark.parametrize('backend', ['gaussian', 'bosonic'])
@pytest.mark.parametrize('eta', [0.0, 0.7, 1.0])
def test_noisy_conditional_state_matches_observed_record(backend, eta, monkeypatch):
    monkeypatch.setattr(dqp, 'hbar', 1.3)
    monkeypatch.setattr(dqp, 'kappa', 0.8)
    cir = dqp.QumodeCircuit(2, 'vac', backend=backend)
    cir.s2([0, 1], r=torch.tensor(0.6))
    cir.d(0, r=torch.tensor(0.2))
    cir.homodyne(0, phi=torch.tensor(0.0))
    gain, bias, noise = -1.4, 0.3, 0.25
    model = dqp.HardwareModel().homodyne(0).efficiency(eta)
    model.homodyne(0).readout(gain, bias, noise)
    cir.set_hardware_model(model)
    cir.double()
    cov, mean = cir()[:2]
    cov, mean = cov.reshape(4, 4), mean.reshape(4, 1).real
    vacuum = dqp.hbar / (4 * dqp.kappa**2)
    var_y = gain**2 * (eta * cov[0, 0] + (1 - eta) * vacuum) + noise
    cross = gain * eta**0.5 * cov[[1, 3], 0]
    mean_y = gain * eta**0.5 * mean[0, 0] + bias
    expected_cov = cov[[1, 3]][:, [1, 3]] - torch.outer(cross, cross) / var_y
    torch.manual_seed(42)
    samples = cir.measure_homodyne(12)
    actual_cov, actual_mean = cir.state_measured[:2]
    actual_cov = actual_cov.reshape(12, 4, 4)[:, [1, 3]][:, :, [1, 3]]
    actual_mean = actual_mean.reshape(12, 4, 1)[:, [1, 3], 0].real
    expected_mean = mean[[1, 3], 0] + (samples - mean_y).unsqueeze(-1) * cross / var_y
    torch.testing.assert_close(actual_cov, expected_cov.expand_as(actual_cov), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual_mean, expected_mean, atol=1e-10, rtol=1e-10)
    if backend == 'bosonic':
        torch.testing.assert_close(cir.state_measured[2], torch.ones_like(cir.state_measured[2]))


@pytest.mark.parametrize('conditional', [False, True])
def test_calibrated_readout_statistics(conditional):
    squeezing, displacement = 0.5, 0.4
    cir = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
    cir.s(0, r=torch.tensor(squeezing))
    cir.d(0, r=torch.tensor(displacement))
    if conditional:
        cir.homodyne(0, phi=0.0)
    model = dqp.HardwareModel(seed=4).homodyne().efficiency(0.8)
    model.homodyne().readout(gain=1.2, bias=-0.1, noise_var=0.3)
    cir.set_hardware_model(model)
    cir.double()
    cir()
    torch.manual_seed(9)
    samples = cir.measure_homodyne(20000, phi=0.0)
    vacuum = dqp.hbar / (4 * dqp.kappa**2)
    expected_mean = 1.2 * 0.8**0.5 * displacement * dqp.hbar**0.5 / dqp.kappa - 0.1
    expected_var = 1.2**2 * vacuum * (0.8 * torch.exp(torch.tensor(-2 * squeezing)) + 0.2) + 0.3
    assert abs(samples.mean() - expected_mean) < 0.035
    torch.testing.assert_close(samples.var(), expected_var, atol=0.035, rtol=0.03)


@pytest.mark.parametrize('backend', ['gaussian', 'bosonic', 'fock'])
def test_lo_readout_uses_spatial_ports(backend):
    cir = dqp.QumodeCircuit(2, 'vac', backend=backend, cutoff=3, basis=False, den_mat=True)
    cir.d(1, r=torch.tensor(0.1))
    model = dqp.HardwareModel(seed=8)
    model.input().submodes([1.0, 0.0])
    model.d(1).submodes([0.0, 1.0])
    model.homodyne(0).readout(bias=100)
    model.homodyne(1).efficiency(0.8)
    model.homodyne(1).readout(gain=1.5, bias=-0.2)
    cir.set_hardware_model(model)
    cir.double()
    cir()
    lo = [0.0, 1.0]
    # Orthogonal mode 1 contains the displacement, independently of prepared input mode 0.
    ref = dqp.QumodeCircuit(1, 'vac', backend=backend, cutoff=3, basis=False, den_mat=True)
    ref.d(0, r=torch.tensor(0.1))
    ref.loss_t(0, torch.tensor(0.8))
    ref.double()
    ref()
    torch.manual_seed(15)
    expected = 1.5 * ref.measure_homodyne(10, phi=0.4) - 0.2
    torch.manual_seed(15)
    actual = cir.measure_homodyne(10, wires=1, lo=lo, phi=0.4)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_rule_validation_overrides_and_noiseless_limit():
    cir = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
    with pytest.raises(ValueError, match='matches no operations'):
        cir.set_hardware_model(dqp.HardwareModel().bs().before.loss(t=0.8))
    with pytest.raises(ValueError, match='outside the circuit'):
        cir.set_hardware_model(dqp.HardwareModel().homodyne(1).efficiency(0.8))
    model = dqp.HardwareModel().homodyne(0).efficiency(0.8)
    model.homodyne(0).before.loss(t=1.0)
    assert len(model.bindings) == 1
    model.homodyne().readout()
    ref = deepcopy(cir)
    ref.r(0, torch.tensor(0.0))
    cir.set_hardware_model(model)
    cir.double()
    ref.double()
    cir()
    ref()
    for phi in (None, 0.0):
        torch.manual_seed(21)
        actual = cir.measure_homodyne(5, phi=phi)
        torch.manual_seed(21)
        torch.testing.assert_close(actual, ref.measure_homodyne(5, phi=phi))
    for kw in ({'gain': 0}, {'noise_var': -1}, {'bias': float('nan')}, {'gain': [1, 2]}):
        with pytest.raises(ValueError):
            model.homodyne().readout(**kw)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_readout_parameter_registration_and_conditional_gradients(dtype):
    torch.set_default_dtype(dtype)
    gain, bias, noise = [torch.nn.Parameter(torch.tensor(v)) for v in (1.1, 0.2, 0.3)]
    cir = dqp.QumodeCircuit(2, 'vac', backend='gaussian')
    cir.s2([0, 1], r=0.5)
    cir.homodyne(0, phi=0.2)
    model = dqp.HardwareModel().homodyne().readout(gain, bias, noise)
    model.homodyne().efficiency(0.8, requires_grad=True)
    cir.set_hardware_model(model)
    assert {gain, bias, noise} <= set(cir.parameters())
    cir.to(dtype)
    cir()
    records = cir.measure_homodyne(4)
    loss = records.square().mean() + sum(x.real.square().sum() for x in cir.state_measured)
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert records.dtype == dtype


def test_fock_noisy_condition_uses_mixed_state():
    cir = dqp.QumodeCircuit(2, [(2**-0.5, [0, 0]), (2**-0.5, [1, 1])], cutoff=12, basis=False, den_mat=True)
    cir.homodyne(0, phi=0.0)
    model = dqp.HardwareModel().homodyne().readout(noise_var=0.4)
    cir.set_hardware_model(model)
    cir.double()
    cir()
    torch.manual_seed(8)
    cir.measure_homodyne(3)
    reduced = cir.state_measured[:, 0, :, 0, :]
    torch.testing.assert_close(reduced.diagonal(dim1=-2, dim2=-1).sum(-1), torch.ones(3, dtype=torch.cdouble))
    torch.testing.assert_close(reduced, reduced.mH)
    assert (torch.linalg.eigvalsh(reduced) >= -1e-12).all()
    assert ((reduced @ reduced).diagonal(dim1=-2, dim2=-1).sum(-1).real < 1 - 1e-3).all()
    # Convolution of the n=0,1 Hermite wavefunctions gives an independent analytic posterior.
    noise = torch.tensor(0.4, requires_grad=True)
    conditional = cir.measurements[0](cir.tensor_rep(cir.state), samples=0.4, noise_var=noise)
    scalar_noise = cir.measurements[0](cir.tensor_rep(cir.state), samples=0.4, noise_var=0.4)
    torch.testing.assert_close(conditional, scalar_noise)
    q = 0.4 / (1 + noise)
    second_moment = noise / (1 + noise) + q**2
    expected = torch.stack([q.new_ones(()), q, q, second_moment]).reshape(2, 2) / (1 + second_moment)
    torch.testing.assert_close(conditional[0, 0, :2, 0, :2].real, expected, atol=1e-9, rtol=1e-9)
    actual_grad = torch.autograd.grad(conditional[0, 0, 0, 0, 0].real, noise)[0]
    expected_grad = torch.autograd.grad(expected[0, 0], noise)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-9, rtol=1e-9)


def test_terminal_tdm_rules_survive_unrolling():
    cir = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
    cir.s(0, r=torch.tensor(0.2))
    cir.delay(0, ntau=2, inputs=torch.tensor([0.3, 0.1]))
    model = dqp.HardwareModel().homodyne(0).efficiency(0.7)
    model.homodyne(0).readout(gain=1.2, bias=0.4)
    cir.set_hardware_model(model)
    cir.double()
    expanded = cir.global_circuit(2)
    expanded()
    output = cir.tdm_output_wires(2).flatten().tolist()
    ref = deepcopy(expanded)
    ref.hardware_model = None
    state = ref.state
    for wire in output:
        state = dqp.PhotonLoss(torch.tensor(0.7), ref.nmode, wire, convention='t')(state)
    ref.state = state
    torch.manual_seed(6)
    expected = 1.2 * ref.measure_homodyne(5, output, phi=0.3) + 0.4
    torch.manual_seed(6)
    torch.testing.assert_close(expanded.measure_homodyne(5, output, phi=0.3), expected)


def test_conditional_kernel_noise_gradient_and_vmap():
    cir = dqp.QumodeCircuit(2, 'vac', backend='gaussian')
    cir.s2([0, 1], r=torch.tensor(0.4))
    cir.double()
    state = cir()
    measurement = dqp.Homodyne(torch.tensor(0.2), nmode=2, wires=0)

    def condition(noise):
        return measurement(state, samples=0.3, noise_var=noise)

    noise = torch.tensor([0.1, 0.2, 0.3])
    actual = torch.vmap(condition)(noise)
    expected = [torch.stack(values) for values in zip(*(condition(n) for n in noise), strict=True)]
    torch.testing.assert_close(actual, expected)
    assert torch.autograd.gradcheck(condition, (noise[0].clone().requires_grad_(),))


def test_bosonic_complex_components_match_gaussian_convolution():
    cir = dqp.QumodeCircuit(2, 'vac', backend='bosonic')
    cir.cat(0, r=0.7, theta=0.6)
    cir.bs([0, 1], torch.tensor([0.4, 0.2]))
    cir.double()
    state = cir()
    measurement = dqp.Homodyne(torch.tensor(0.0), nmode=2, wires=0)
    variance, record = 0.3, 0.2
    actual = measurement(state, samples=record, noise_var=torch.tensor(variance))
    # Integrate the ideal quadrature instrument against the electronic-noise likelihood.
    grid = torch.linspace(-8, 8, 4001)
    cov, mean, weight = state
    location = mean[..., 0, 0]
    var = cov[..., 0, 0]
    delta = grid[:, None, None] - location
    density = torch.exp(-delta.square() / (2 * var)) / (2 * torch.pi * var).sqrt()
    electronics = torch.exp(-(grid - record).square() / (2 * variance)) / (2 * torch.pi * variance) ** 0.5
    likelihood = torch.trapezoid(density * electronics[:, None, None], grid, dim=0)
    posterior = weight * likelihood
    posterior /= posterior.sum(-1, keepdim=True)
    torch.testing.assert_close(actual[2], posterior, atol=1e-10, rtol=1e-10)
    assert torch.isfinite(actual[1]).all()


def test_batched_terminal_readout_phase_sampling_and_queries():
    cir = dqp.QumodeCircuit(2, 'vac', backend='gaussian')
    cir.d(0, encode=True)
    cir.d(1, r=torch.tensor(0.3))
    model = dqp.HardwareModel(seed=3)
    model.homodyne().phi.gaussian(sigma=0.1)
    model.homodyne(0).readout(gain=torch.tensor([[2.0]]), bias=0.3, noise_var=0.05)
    cir.set_hardware_model(model)
    cir.double()
    cir(torch.tensor([[0.1, 0.2], [0.2, 0.4]]))
    rng = model._generator.get_state()
    cir.quadrature_mean(wires=0, phi=0.2)
    assert torch.equal(rng, model._generator.get_state())
    torch.manual_seed(4)
    samples = cir.measure_homodyne(8, wires=[1, 0], phi=[0.2, 0.3])
    assert samples.shape == (2, 8, 2)
    generator = torch.Generator().manual_seed(3)
    torch.randn((), generator=generator)
    torch.randn((), generator=generator)
    torch.randn((2, 8), generator=generator)
    assert torch.equal(generator.get_state(), model._generator.get_state())


@pytest.mark.parametrize('backend', ['gaussian', 'bosonic'])
@pytest.mark.parametrize('readout', ['none', 'float_zero', 'tensor_zero'])
def test_exact_ideal_conditioning_and_zero_readout(backend, readout, monkeypatch):
    monkeypatch.setattr(dqp, 'hbar', 1.3)
    monkeypatch.setattr(dqp, 'kappa', 0.8)
    cir = dqp.QumodeCircuit(2, 'vac', backend=backend)
    cir.s2([0, 1], r=0.6, theta=0.3)
    cir.d(0, r=0.2, theta=0.4)
    cir.homodyne(0, phi=torch.tensor(0.4))
    if readout != 'none':
        noise = 0.0 if readout == 'float_zero' else torch.tensor(0.0)
        cir.set_hardware_model(dqp.HardwareModel().homodyne().readout(noise_var=noise))
    cir.double()
    state = cir()
    cov, mean = state[:2]
    cov, mean = cov.reshape(4, 4), mean.reshape(4, 1).real
    angle = torch.tensor(0.4)
    u = torch.stack([angle.cos(), angle.sin()])
    measured, rest = [0, 2], [1, 3]
    variance = u @ cov[measured][:, measured] @ u
    cross = cov[rest][:, measured] @ u
    location = u @ mean[measured, 0]
    torch.manual_seed(52)
    samples = cir.measure_homodyne(4)
    actual_cov, actual_mean = [value.reshape(4, *value.shape[-2:]) for value in cir.state_measured[:2]]
    expected_cov = cov[rest][:, rest] - torch.outer(cross, cross) / variance
    expected_mean = mean[rest, 0] + (samples - location)[:, None] * cross / variance
    torch.testing.assert_close(actual_cov[:, rest][:, :, rest], expected_cov.expand(4, 2, 2), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual_mean[:, rest, 0].real, expected_mean, atol=1e-12, rtol=1e-12)
    vacuum = dqp.hbar / (4 * dqp.kappa**2)
    torch.testing.assert_close(actual_cov[:, 0, 0], torch.full((4,), vacuum))
    assert cir.measurements[0].samples.shape == (4, 1)


@pytest.mark.parametrize('noise', [0.0, torch.tensor(0.0)])
def test_zero_readout_keeps_ideal_pure_fock_measurement(noise):
    cir = dqp.QumodeCircuit(2, [(2**-0.5, [0, 0]), (2**-0.5, [1, 1])], cutoff=10, basis=False)
    cir.homodyne_x(0)
    ref = deepcopy(cir)
    cir.set_hardware_model(dqp.HardwareModel().homodyne().readout(noise_var=noise))
    for circuit in (cir, ref):
        circuit.double()
        circuit()
    torch.manual_seed(31)
    records = cir.measure_homodyne(3)
    torch.manual_seed(31)
    torch.testing.assert_close(records, ref.measure_homodyne(3))
    torch.testing.assert_close(cir.state_measured, ref.state_measured)


@pytest.mark.parametrize('backend', ['gaussian', 'bosonic'])
@pytest.mark.parametrize('eps', [1.0, 0.02])
def test_generaldyne_squeezed_seed_conditioning(backend, eps):
    # Two-mode squeezed vacuum with local variance 2 and correlation sqrt(3).
    root3 = 3**0.5
    cov = torch.tensor([[2, root3, 0, 0], [root3, 2, 0, 0], [0, 0, 2, -root3], [0, 0, -root3, 2]])
    state = [cov[None], torch.zeros(1, 4, 1)]
    if backend == 'bosonic':
        state = [state[0][:, None], state[1][:, None].to(torch.complex128), torch.ones(1, 1, dtype=torch.complex128)]
    seed = torch.diag(torch.tensor([eps**2, eps**-2]))
    measurement = dqp.Generaldyne(seed, nmode=2, wires=1)
    q, p = 0.7, -0.4
    cov_out, mean_out = measurement(state, samples=[q, p])[:2]
    cov_out, mean_out = cov_out.reshape(4, 4), mean_out.reshape(4)
    expected_cov = torch.diag(torch.tensor([2 - 3 / (2 + eps**2), 2 - 3 / (2 + eps**-2)]))
    expected_mean = torch.tensor([root3 * q / (2 + eps**2), -root3 * p / (2 + eps**-2)])
    torch.testing.assert_close(cov_out[[0, 2]][:, [0, 2]], expected_cov)
    torch.testing.assert_close(mean_out[[0, 2]].real, expected_mean)


def test_general_dyne_complex_likelihood_matches_bosonic_measurement():
    cir = dqp.QumodeCircuit(2, 'vac', backend='bosonic')
    cir.cat(0, r=0.7, theta=0.6)
    cir.bs([0, 1], torch.tensor([0.4, 0.2]))
    cir.double()
    state = cir()
    state[0] = state[0].expand(-1, state[1].shape[1], -1, -1)
    original = [value.clone() for value in state]
    seed = torch.tensor([[1.5, 0.3], [0.3, 1.2]])
    actual = dqp.Generaldyne(seed, nmode=2, wires=0)(state, samples=[0.3, -0.1])
    reference = dqp.GeneralBosonic(seed, torch.ones(1, dtype=torch.complex128), nmode=2, wires=0)
    expected = reference(state, samples=[0.3, -0.1])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state, original)


@pytest.mark.parametrize('backend', ['gaussian', 'bosonic'])
def test_multiwire_homodyne_limit_matches_sequential_conditioning(backend):
    cir = dqp.QumodeCircuit(3, 'vac', backend=backend)
    cir.s2([0, 1], r=0.4)
    cir.bs([1, 2], [0.3, 0.2])
    cir.double()
    state = cir()
    simultaneous = dqp.Generaldyne(nmode=3, wires=[0, 2])(state, samples=[0.2, -0.4])
    sequential = dqp.Homodyne(0.0, nmode=3, wires=0).double()(state, samples=0.2)
    sequential = dqp.Homodyne(0.0, nmode=3, wires=2).double()(sequential, samples=-0.4)
    torch.testing.assert_close(simultaneous, sequential)


@pytest.mark.parametrize('backend', ['gaussian', 'bosonic', 'fock'])
@pytest.mark.parametrize('batch', [1, 2])
def test_homodyne_postselection_shape_and_precision(backend, batch):
    cir = dqp.QumodeCircuit(2, 'vac', backend=backend, cutoff=4, basis=False, den_mat=True)
    cir.s2([0, 1], r=0.2)
    cir.double()
    state = cir()
    if backend == 'fock':
        state = cir.tensor_rep(state)
        state = state.repeat(batch, *([1] * (state.ndim - 1)))
    else:
        state = [value.repeat(batch, *([1] * (value.ndim - 1))) for value in state]
    measurement = dqp.Homodyne(0.2, nmode=2, wires=1, cutoff=4, den_mat=True).double()
    records = [0.3] * batch
    expected = measurement(state, samples=torch.tensor(records).reshape(batch, 1))
    actual = measurement(state, samples=0.3 if batch == 1 else records)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(measurement.samples, torch.full((batch, 1), 0.3), atol=0, rtol=0)
    for invalid in [torch.zeros(batch, 2), torch.zeros(batch, 1, 1), torch.zeros(batch + 1, 1)]:
        with pytest.raises(ValueError, match='samples must have shape'):
            measurement(state, samples=invalid)
    with pytest.raises(ValueError, match='real quadrature'):
        measurement(state, samples=torch.zeros(batch, 1, dtype=torch.complex128))
    if backend == 'fock':
        actual = measurement(state, samples=records, noise_var=0.1)
        expected = measurement(state, samples=torch.tensor(records)[:, None], noise_var=0.1)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_generaldyne_postselection_shape():
    state = [torch.eye(4)[None].repeat(2, 1, 1), torch.zeros(2, 4, 1)]
    measurement = dqp.Generaldyne(torch.eye(2), nmode=2, wires=1)
    measurement(state, samples=[[0.1, 0.2], [0.3, 0.4]])
    for invalid in [0.1, [0.1, 0.2, 0.3, 0.4], torch.zeros(2, 1)]:
        with pytest.raises(ValueError, match='samples must have shape'):
            measurement(state, samples=invalid)
    single = [value[:1] for value in state]
    torch.testing.assert_close(measurement(single, samples=[0.1, 0.2]), measurement(single, samples=[[0.1, 0.2]]))


def test_generaldyne_float32_squeezed_seed_precision():
    torch.set_default_dtype(torch.float32)
    cir = dqp.QumodeCircuit(2, 'vac', backend='bosonic')
    cir.cat(0, r=1.1, theta=0.4)
    cir.s(1, r=1.5, theta=0.2)
    cir.bs([0, 1], [0.6, 0.3])
    state = cir()
    seed = torch.diag(torch.tensor([4e-8, 2.5e7]))
    actual = dqp.Generaldyne(seed, nmode=2, wires=1)(state, samples=[0.4, 5000.0])
    state64 = [value.to(torch.complex128 if value.is_complex() else torch.float64) for value in state]
    expected = dqp.Generaldyne(seed.double(), nmode=2, wires=1)(state64, samples=[0.4, 5000.0])
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-6, check_dtype=False)


def test_bosonic_conditional_homodyne_samples_match_quadrature_marginal():
    import numpy as np
    from scipy.special import erf

    cir = dqp.QumodeCircuit(2, 'vac', backend='bosonic')
    cir.cat(0, r=0.7, theta=0.6)
    cir.bs([0, 1], [0.4, 0.2])
    phi = 0.37
    cir.homodyne(0, phi=phi)
    cir.double()
    cov, mean, weight = cir()
    direction = torch.tensor([np.cos(phi), np.sin(phi)])
    variance = direction @ cov[0, :, [0, 2]][:, :, [0, 2]] @ direction
    location = mean[0, :, [0, 2], 0] @ direction.to(mean.dtype)
    torch.manual_seed(170)
    samples = cir.measure_homodyne(4096).sort().values.numpy()
    # The exact CDF integrates each complex Gaussian term in the selected quadrature.
    cdf = 0.5 * (1 + erf((samples[:, None] - location.numpy()) / np.sqrt(2 * variance.numpy())))
    cdf = (cdf * weight[0].numpy()).sum(-1)
    assert abs(cdf.imag).max() < 1e-12
    empirical = np.arange(1, len(samples) + 1) / len(samples)
    error = max(abs(cdf.real - empirical).max(), abs(cdf.real - empirical + 1 / len(samples)).max())
    assert error < 0.04
