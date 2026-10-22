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
def test_global_circuit_maps_bindings_and_shares_parameters(use_deepcopy):
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
    assert parameters['hardware_model.bindings.phase.plugin.mu'] is mu
    assert parameters['hardware_model.bindings.source.plugin.t'] is t
    n = 2 * actual.nmode
    state = [2 * torch.eye(n), torch.linspace(0.1, 0.3, n).reshape(n, 1)]
    result = actual(state=state)
    assert_state_close(result, expected(state=state))
    (result[0][0, 0, 1] + result[1].sum()).backward()
    assert mu.grad is not None and torch.isfinite(mu.grad)
    assert t.grad is not None and torch.isfinite(t.grad)
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
