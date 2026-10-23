from copy import deepcopy

import pytest
import torch
from torch.func import functional_call

import deepquantum.photonic as dqp


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def circuit():
    cir = dqp.QumodeCircuit(2, 'vac', backend='gaussian', cutoff=3)
    cir.s(0, r=0.4, theta=0.2)
    cir.d(1, r=0.15, theta=-0.1)
    cir.bs([0, 1], [0.3, -0.2])
    cir.ps(1, torch.tensor(0.2))
    return cir.to(torch.double)


def assert_state_close(actual, expected):
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('t', [0.0, 0.4, 1.0])
def test_loss_formula(t):
    a = torch.arange(16.0).reshape(4, 4) / 10
    cov = (a @ a.mT + torch.eye(4)).unsqueeze(0)
    mean = torch.arange(4.0).reshape(1, 4, 1) / 10
    cir = dqp.QumodeCircuit(2, [cov, mean], backend='gaussian')
    cir.set_hardware_model(dqp.HardwareModel().output(1).loss(t=t))
    scale = torch.diag(torch.tensor([1, t**0.5, 1, t**0.5]))
    vacuum = dqp.hbar / (4 * dqp.kappa**2)
    expected = [scale @ cov @ scale + (torch.eye(4) - scale @ scale) * vacuum, scale @ mean]
    assert_state_close(cir(), expected)
    if 0 < t < 1:
        cir.set_hardware_model(dqp.HardwareModel().output(1).loss(db=-10 * torch.tensor(t).log10()))
        assert_state_close(cir(), expected)


def test_bindings_and_use_match_manual_circuit():
    cir = circuit()
    model = dqp.HardwareModel()
    model.s(0).after.loss(t=0.8)
    model.bs([0, 1]).before.loss(t=0.9)
    model.ps(1).theta.use(dqp.GaussianNoise(mu=0.03, sigma=0))
    model.output(1).loss(db=0.1)
    cir.set_hardware_model(model)
    ref = dqp.QumodeCircuit(2, 'vac', backend='gaussian')
    ref.s(0, r=0.4, theta=0.2)
    ref.loss_t(0, torch.tensor(0.8))
    ref.d(1, r=0.15, theta=-0.1)
    for wire in [0, 1]:
        ref.loss_t(wire, torch.tensor(0.9))
    ref.bs([0, 1], [0.3, -0.2])
    ref.ps(1, torch.tensor(0.23))
    ref.loss_db(1, torch.tensor(0.1))
    assert_state_close(cir(), ref.to(torch.double)())


def test_seed_ideal_queries_and_detach():
    mean = torch.arange(4.0).reshape(1, 4, 1) / 10
    cir = dqp.QumodeCircuit(2, [2 * torch.eye(4), mean], backend='gaussian')
    cir.bs([0, 1], [0.3, 0.1])
    cir.ps(0, 0.2)
    cir.to(torch.double)
    ideal = cir()
    matrices = [cir.get_unitary(), cir.get_symplectic(), cir.get_displacement(mean)]
    model = dqp.HardwareModel(seed=8).ps(0).theta.gaussian(sigma=0.1)
    cir.set_hardware_model(model)
    rng = torch.get_rng_state()
    first, second = cir(), cir()
    assert not torch.allclose(first[1], second[1])
    model.manual_seed(8)
    assert_state_close(cir(), first)
    assert_state_close([cir.get_unitary(), cir.get_symplectic(), cir.get_displacement(mean)], matrices)
    assert_state_close(cir(), second)
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(cir.operators[-1].theta, torch.tensor(0.2))
    cir.set_hardware_model(None)
    assert cir.state is None
    assert_state_close(cir(), ideal)


def test_batch_parameter_registration_and_gradients():
    cir = circuit()
    cir.ps(0, encode=True)
    mu, sigma, t = [torch.nn.Parameter(torch.tensor(x)) for x in [0.01, 0.02, 0.8]]
    model = dqp.HardwareModel(seed=5)
    model.ps(0).theta.gaussian(mu=mu, sigma=sigma, name='phase')
    model.output().loss(t=t, name='loss')
    cir.set_hardware_model(model)
    cir.to(torch.double)
    names = ['phase.plugin.mu', 'phase.plugin.sigma', 'loss.plugin.t']
    for name, parameter in zip(names, [mu, sigma, t], strict=True):
        assert dict(cir.named_parameters())[f'hardware_model.bindings.{name}'] is parameter
    data = torch.tensor([[0.1], [0.2]], requires_grad=True)
    actual = cir(data)
    rows = []
    for row in data:
        model.manual_seed(5)
        rows.append(cir(row))
    assert_state_close(actual, [torch.cat([row[i] for row in rows]) for i in range(2)])

    def evaluate(data, mu, sigma, t):
        model.manual_seed(5)
        parameters = dict(zip([f'hardware_model.bindings.{name}' for name in names], [mu, sigma, t], strict=True))
        state = functional_call(cir, parameters, (data,))
        return torch.cat([value.flatten() for value in state])

    assert torch.autograd.gradcheck(evaluate, (data, mu, sigma, t))


def test_measurements_use_the_noisy_state_once():
    cir = circuit()
    ref = deepcopy(cir)
    cir.set_hardware_model(dqp.HardwareModel().output().loss(t=0.8))
    for wire in [0, 1]:
        ref.loss_t(wire, torch.tensor(0.8))
    for target in [cir, ref]:
        target.homodyne(0, phi=0.3, eps=0.1)
        target.to(torch.double)
        target()
    optical = [value.clone() for value in cir.state]
    torch.testing.assert_close(cir.get_prob([0, 0]), ref.get_prob([0, 0]))
    torch.manual_seed(12)
    expected = ref.measure_homodyne(8)
    torch.manual_seed(12)
    torch.testing.assert_close(cir.measure_homodyne(8), expected)
    torch.manual_seed(13)
    expected = ref.measure(shots=8)
    torch.manual_seed(13)
    assert cir.measure(shots=8) == expected
    assert_state_close(cir.state, optical)


@pytest.mark.parametrize('use_deepcopy', [False, True])
def test_global_circuit_maps_bindings_and_copies_parameters(use_deepcopy):
    def template(manual=False):
        cir = dqp.QumodeCircuit(2, 'vac', backend='gaussian')
        if manual:
            cir.loss_t(1, torch.tensor(0.85))
        cir.s(0, r=0.3)
        if manual:
            cir.loss_t(0, torch.tensor(0.8))
        cir.ps(0, torch.tensor(0.22 if manual else 0.2))
        cir.bs([0, 1], [0.3, 0.1])
        cir.delay(0, ntau=2, inputs=[0.4, 0.05])
        if manual:
            cir.loss_t(1, torch.tensor(0.9))
        return cir.to(torch.double)

    cir = template()
    mu, t = [torch.nn.Parameter(torch.tensor(x)) for x in [0.02, 0.8]]
    model = dqp.HardwareModel(seed=7)
    model.input(1).loss(t=0.85)
    model.s(0).after.loss(t=t, name='source')
    model.ps(0).theta.gaussian(mu=mu, sigma=0, name='phase')
    model.output(1).loss(t=0.9)
    cir.set_hardware_model(model)
    actual = cir.global_circuit(3, use_deepcopy=use_deepcopy).to(torch.double)
    expected = template(manual=True).global_circuit(3).to(torch.double)
    parameters = dict(actual.named_parameters())
    mapped_mu = parameters['hardware_model.bindings.phase.plugin.mu']
    mapped_t = parameters['hardware_model.bindings.source.plugin.t']
    assert (mapped_mu is mu) == (not use_deepcopy)
    assert (mapped_t is t) == (not use_deepcopy)
    assert (actual.operators[0].r is cir.operators[0].r) == (not use_deepcopy)
    assert (actual.operators[0].r is actual.operators[6].r) == (not use_deepcopy)
    n = 2 * actual.nmode
    state = [2 * torch.eye(n), torch.linspace(0.1, 0.3, n).reshape(n, 1)]
    result = actual(state=state)
    assert_state_close(result, expected(state=state))
    (result[0][0, 0, 1] + result[1].sum()).backward()
    assert mapped_mu.grad is not None and torch.isfinite(mapped_mu.grad)
    assert mapped_t.grad is not None and torch.isfinite(mapped_t.grad)
    if use_deepcopy:
        assert mu.grad is None and t.grad is None
    with pytest.raises(NotImplementedError, match='global_circuit'):
        cir()


def test_global_circuit_samples_each_occurrence():
    cir = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
    cir.s(0, r=0.4)
    cir.ps(0, torch.tensor(0.2))
    cir.to(torch.double)
    cir.set_hardware_model(dqp.HardwareModel(seed=7).ps().theta.gaussian(sigma=0.1))
    actual = cir.global_circuit(3).to(torch.double)
    ref = dqp.QumodeCircuit(3, 'vac', backend='gaussian')
    generator = torch.Generator().manual_seed(7)
    for wire in range(3):
        ref.s(wire, r=0.4)
        ref.ps(wire, 0.2 + 0.1 * torch.randn((), generator=generator))
    assert_state_close(actual(), ref.to(torch.double)())


@pytest.mark.parametrize('convention', ['bs', 'mzi'])
def test_delay_bindings_match_internal_gates(convention):
    def template(manual=False):
        selected = int(convention == 'mzi')  # DelayBS already contains a phase shifter.
        loop_gates = [
            dqp.PhaseShift(torch.tensor(value + (0.03 if manual and i == selected else 0)))
            for i, value in enumerate([0.12, -0.2])
        ]
        if manual:
            loop_gates.insert(selected + 1, dqp.PhotonLoss(torch.tensor(0.98), convention='t'))
        phase = 0.02 if manual else 0
        cir = dqp.QumodeCircuit(2, 'vac', backend='gaussian')
        cir.s(0, r=0.4)
        cir.d(1, r=0.15)
        cir.delay(0, ntau=1, inputs=torch.tensor([0.35, 0.06 + phase]))
        cir.delay(
            0,
            ntau=2,
            inputs=torch.tensor([0.41 if manual else 0.4, 0.07 + phase]),
            convention=convention,
            loop_gates=loop_gates,
        )
        cir.delay(1, ntau=1, inputs=torch.tensor([0.3, 0.08 + phase]), convention='mzi')
        cir.ps(0, torch.tensor(0.24 if manual else 0.2))
        return cir.to(torch.double)

    cir = template()
    model = dqp.HardwareModel()
    model.delay(0, index=1).theta.gaussian(mu=0.01, sigma=0)
    model.delay().phi.gaussian(mu=0.02, sigma=0)
    model.delay(0, index=1).ps(index=1).theta.gaussian(mu=0.03, sigma=0, name='inner')
    assert model.delay(0, index=1).ps(index=1).after.loss(t=0.98) is model
    model.ps(0).theta.gaussian(mu=0.04, sigma=0)
    cir.set_hardware_model(model)
    expected = template(manual=True)
    # Inserting a non-delay gate must not change which delay the selector addresses.
    for target, phase in [(cir, 0.1), (expected, 0.14)]:
        target.operators.insert(3, dqp.PhaseShift(torch.tensor(phase), nmode=2, wires=0))
    assert model.bindings.inner.resolve(cir.operators) == [(4, 2)]
    unrolled = cir.global_circuit(3)
    assert len(unrolled.hardware_model.bindings.inner.op_indices) == 3
    assert_state_close(unrolled(), expected.global_circuit(3)())


@pytest.mark.parametrize('use_deepcopy', [False, True])
def test_encoded_delay_noise_shares_plugins_not_samples(use_deepcopy):
    cir = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
    cir.s(0, r=0.4)
    cir.delay(0, ntau=1, convention='mzi', encode=True)
    cir.to(torch.double)
    ref = cir.global_circuit(3)
    mu, sigma = [torch.nn.Parameter(torch.tensor(value)) for value in [0.01, 0.02]]
    plugin = dqp.GaussianNoise(mu=mu, sigma=sigma)
    model = dqp.HardwareModel(seed=7)
    model.delay(0).theta.use(plugin, name='theta')
    model.delay(0).phi.use(plugin, name='phi')
    cir.set_hardware_model(model)
    unrolled = cir.global_circuit(3, use_deepcopy=use_deepcopy)
    mapped = unrolled.hardware_model.bindings.theta.plugin
    assert mapped is unrolled.hardware_model.bindings.phi.plugin
    assert (mapped is plugin) == (not use_deepcopy)
    data = torch.linspace(0.1, 0.6, 12).reshape(2, 6).requires_grad_()
    actual = unrolled(data)
    generator = torch.Generator().manual_seed(7)
    epsilon = torch.stack([torch.randn((), generator=generator) for _ in range(6)])
    expected = ref(data + mapped.mu + mapped.sigma * epsilon)
    assert_state_close(actual, expected)
    inputs = (data, mapped.mu, mapped.sigma)
    actual_grad = torch.autograd.grad(sum(value.square().sum() for value in actual), inputs)
    expected_grad = torch.autograd.grad(sum(value.square().sum() for value in expected), inputs)
    assert_state_close(actual_grad, expected_grad)


def test_invalid_parameters_bindings_and_backend():
    with pytest.raises(ValueError):
        dqp.HardwareModel().output().loss(t=1.1)
    with pytest.raises(ValueError):
        dqp.HardwareModel().ps().theta.gaussian(sigma=-0.1)
    with pytest.raises(ValueError, match='matches no operations'):
        circuit().set_hardware_model(dqp.HardwareModel().ps(0).theta.gaussian())
    for backend in ['fock', 'bosonic']:
        with pytest.raises(NotImplementedError, match='Gaussian backend'):
            dqp.QumodeCircuit(1, 'vac', backend=backend).set_hardware_model(dqp.HardwareModel())
    with pytest.raises(ValueError, match='index'):
        dqp.HardwareModel().delay(0, index=-1)
    cir = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
    cir.delay(0, inputs=[0.3, 0.1])
    for config in [dqp.HardwareModel().delay(0, index=1).theta, dqp.HardwareModel().delay(0).ps(index=1).theta]:
        with pytest.raises(ValueError, match='matches no operations'):
            cir.set_hardware_model(config.gaussian())
