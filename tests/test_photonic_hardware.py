from copy import deepcopy

import pytest
import torch
from torch.func import functional_call

import deepquantum.photonic as dqp
from deepquantum.photonic.hardware import _PARAMETERS


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


@pytest.mark.parametrize('kind', ['gaussian', 'bosonic', 'density', 'basis'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_loss_backends_match_channels_and_gradients(kind, dtype, device):
    torch.set_default_dtype(dtype)
    t, db = [torch.nn.Parameter(torch.tensor(value, device=device)) for value in [0.8, 0.4]]
    backend = kind if kind in ('gaussian', 'bosonic') else 'fock'
    initial = [2 * torch.eye(4), torch.arange(4.0).reshape(4, 1) / 10]
    if kind == 'bosonic':
        initial = [torch.stack([initial[0], 1.5 * initial[0]]), torch.stack([initial[1], -initial[1]]), [0.4, 0.6]]
    elif kind == 'density':
        initial = [1, 0]
    elif kind == 'basis':
        initial = torch.tensor([[1, 0], [0, 2]])

    def make_circuit(hardware):
        cir = dqp.QumodeCircuit(2, initial, cutoff=3, backend=backend, basis=kind == 'basis', den_mat=kind == 'density')
        if not hardware:
            cir.add(dqp.PhotonLoss(t, nmode=2, wires=0, cutoff=3, convention='t'))
        cir.bs([0, 1], encode=True)
        if not hardware:
            for wire in [0, 1]:
                cir.loss_t(wire, torch.tensor(0.85))
        cir.ps(1, 0.2)
        cir.loss_t(0, torch.tensor(0.95))
        if hardware:
            model = dqp.HardwareModel().input(0).loss(t=t)
            model.bs().theta.gaussian(mu=0.03, sigma=0)
            model.bs().after.loss(t=0.85)
            model.output(1).loss(db=db)
            cir.set_hardware_model(model)
        else:
            cir.add(dqp.PhotonLoss(db, nmode=2, wires=1, cutoff=3, convention='db'))
        return cir.to(device=device, dtype=dtype)

    cir, ref = make_circuit(True), make_circuit(False)
    data = torch.tensor([[0.2, 0.3], [0.4, 0.1]], device=device, requires_grad=True)
    actual = cir(data, is_prob=True if kind == 'basis' else None)
    expected = ref(data + data.new_tensor([0.03, 0]), is_prob=True if kind == 'basis' else None)
    if kind == 'basis':
        assert actual.keys() == expected.keys()
        keys = list(expected)
        actual, expected = [actual[key] for key in keys], [expected[key] for key in keys]
    elif kind == 'density':
        actual, expected = [actual], [expected]
    tolerance = 1e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert all(x.real.dtype == dtype and x.device.type == device for x in actual)
    variables = (data, t, db)
    actual_grad = torch.autograd.grad(sum(x.real.square().sum() for x in actual), variables)
    expected_grad = torch.autograd.grad(sum(x.real.square().sum() for x in expected), variables)
    torch.testing.assert_close(actual_grad, expected_grad, atol=tolerance, rtol=tolerance)


def test_fock_loss_updates_environment_modes():
    cir = dqp.QumodeCircuit(1, [1], cutoff=2)
    model = dqp.HardwareModel().output().loss(t=1)
    cir.set_hardware_model(model)
    cir.set_fock_basis()
    for t in [1, 0.4, 0]:
        model.output().loss(t=t)
        result = cir(is_prob=True)
        torch.testing.assert_close(result[dqp.FockState([1])], torch.tensor([t], dtype=torch.double))
        assert cir._nloss == 1
        unitary = cir()
        torch.testing.assert_close(cir.get_prob([1]), torch.ones(1), check_dtype=False)
        torch.testing.assert_close(cir.get_prob([1], unitary=unitary), torch.tensor([t], dtype=torch.double))
    model.input().loss(t=0.5)
    with pytest.raises(AssertionError, match='set_fock_basis'):
        cir(is_prob=True)
    cir.set_fock_basis(reset_in_forward=True)
    assert cir().shape[-2:] == (3, 3)
    cir.set_hardware_model(None)
    assert cir().shape[-2:] == (1, 1)
    assert cir._nloss == 0


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


def test_parameter_noise_override_order_and_sampling():
    class DoubleParameter(torch.nn.Module):
        def forward(self, value, run):
            return 2 * value

    gate = dqp.BeamSplitter(torch.tensor([0.3, 0.2]), nmode=2, wires=[0, 1])
    old_mu, mu, sigma = [torch.nn.Parameter(torch.tensor(x)) for x in [0.01, 0.07, 0.02]]
    model = dqp.HardwareModel(seed=7).gaussian(mu=old_mu, sigma=0.5)
    model.bs().theta.use(DoubleParameter())
    model.on(gate).gaussian(mu=mu, sigma=sigma)
    model.bs().phi.gaussian(mu=-0.04, sigma=0)
    model.bs().before.loss(t=0.9)
    model.bs().after.loss(t=0.7)
    model.on(gate).after.loss(t=0.8)
    state = [torch.diag(torch.tensor([2.0, 1.0, 0.5, 1.0])), torch.arange(4.0).reshape(4, 1) / 10]

    actual = model(state, gate, 'gaussian')
    generator = torch.Generator().manual_seed(7)
    theta = 2 * gate.theta + mu + sigma * torch.randn((), generator=generator)
    ref = dqp.BeamSplitter([theta, gate.phi - 0.04], nmode=2, wires=[0, 1])
    cov, mean = ref(state)
    vacuum = dqp.hbar / (4 * dqp.kappa**2)
    expected = [0.72 * cov + 0.28 * vacuum * torch.eye(4), 0.72**0.5 * mean]
    assert_state_close(actual, expected)
    assert torch.equal(model._generator.get_state(), generator.get_state())
    actual_grad = torch.autograd.grad(sum(x.square().sum() for x in actual), (mu, sigma, old_mu), allow_unused=True)
    expected_grad = torch.autograd.grad(sum(x.square().sum() for x in expected), (mu, sigma))
    assert_state_close(actual_grad[:2], expected_grad)
    assert actual_grad[2] is None
    torch.testing.assert_close(gate.theta, torch.tensor(0.3))


def test_repeated_rules_replace_parameters_and_keep_order():
    old, new = [torch.nn.Parameter(torch.tensor(value)) for value in [0.01, 0.02]]
    model = dqp.HardwareModel()
    model.bs([0, 1]).theta.gaussian(mu=old, name='theta')
    model.bs([0, 1]).phi.gaussian(name='phi')
    model.bs([1, 0]).theta.gaussian(mu=new, name='theta')
    assert list(model.bindings) == ['phi', 'theta']
    assert set(model.parameters()) == {new}
    model.output(0).loss(t=old, name='loss')
    model.output(0).loss(t=new)
    assert list(model.bindings) == ['phi', 'theta', 'loss']
    assert set(model.parameters()) == {new}
    with pytest.raises(ValueError, match='Duplicate binding name'):
        model.ps(0).theta.gaussian(name='theta')
    assert model.bindings.theta.plugin.mu is new
    model.input(0).loss(t=1, name='binding_4')
    model.input(1).loss(t=1)
    assert len(model.bindings) == 5


@pytest.mark.parametrize('where', ['input', 'output'])
def test_io_loss_overrides_per_mode(where):
    cir = circuit()
    ref = deepcopy(cir)
    model = dqp.HardwareModel()
    getattr(model, where)([0, 1]).loss(t=0.9)
    getattr(model, where)(0).loss(t=0.8)
    cir.set_hardware_model(model)
    for wire, t in [(0, 0.8), (1, 0.9)]:
        loss = dqp.PhotonLoss(torch.tensor(t), nmode=2, wires=wire, convention='t')
        if where == 'input':
            ref.operators.insert(0, loss)
        else:
            ref.add(loss)
    state = [2 * torch.eye(4), torch.arange(4.0).reshape(4, 1) / 10]
    assert_state_close(cir(state=state), ref(state=state))


def test_noise_overrides_in_streaming_unrolling_and_measurement():
    def template(manual=False):
        cir = dqp.QumodeCircuit(2, 'vac', backend='gaussian')
        cir.s(0, r=0.3)
        cir.ps(0, torch.tensor(0.14 if manual else 0.1))
        cir.ps(1, torch.tensor(0.22 if manual else 0.2))
        cir.delay(
            0,
            ntau=2,
            inputs=torch.tensor([0.36, 0.16] if manual else [0.35, 0.12]),
            convention='mzi',
            loop_gates=[dqp.PhotonLoss(torch.tensor(0.8), convention='t')] if manual else None,
        )
        if manual:
            cir.loss_t(0, torch.tensor(0.8))
            cir.loss_t(0, torch.tensor(0.85))
            cir.loss_t(1, torch.tensor(0.9))
        cir.homodyne(0, phi=torch.tensor(0.25 if manual else 0.2), eps=0.1)
        return cir.to(torch.double)

    cir, ref = template(), template(manual=True)
    model = dqp.HardwareModel().ps().theta.gaussian(mu=0.02, sigma=0)
    model.ps(0).theta.gaussian(mu=0.04, sigma=0)
    model.delay().mzi().gaussian(mu=0.01, sigma=0)
    model.delay(0).phi.gaussian(mu=0.04, sigma=0)
    model.delay().mzi().after.loss(t=0.9)
    model.delay(0).mzi().after.loss(t=0.8)
    model.output().loss(t=0.9)
    model.output(0).loss(t=0.85)
    model.homodyne().phi.gaussian(mu=0.01, sigma=0)
    model.on(cir.measurements[0]).gaussian(mu=0.05, sigma=0)
    cir.set_hardware_model(model)
    assert_state_close(cir(), ref())
    actual = cir.tdm_substate(4, [1, 3])
    expanded = ref.global_circuit(4)
    expected = expanded()
    outputs = cir.tdm_output_wires(4)[[1, 3]].flatten()
    indices = torch.cat([outputs, outputs + expanded.nmode])
    assert_state_close(actual, [expected[0][:, indices[:, None], indices], expected[1][:, indices]])
    assert_state_close(cir.global_circuit(4)(), expected)
    torch.manual_seed(7)
    samples = cir.measure_homodyne(3)
    torch.manual_seed(7)
    torch.testing.assert_close(samples, ref.measure_homodyne(3))
    model.ps().theta.gaussian(mu=0.08, sigma=0)
    ref.operators[1].init_para(torch.tensor(0.18))
    ref.operators[2].init_para(torch.tensor(0.28))
    assert_state_close(cir(), ref())


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
    assert_state_close(cir(), template(manual=True)())


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
    assert len(unrolled.hardware_model.bindings.inner.bound_ops) == 3
    assert_state_close(unrolled(), expected.global_circuit(3)())
    assert_state_close(cir(), expected(stepwise=True))


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
    for mps in [False, True]:
        with pytest.raises(NotImplementedError, match='density matrix'):
            dqp.QumodeCircuit(1, 'vac', cutoff=2, backend='fock', basis=False, mps=mps).set_hardware_model(
                dqp.HardwareModel().output().loss(t=0.8)
            )
    with pytest.raises(ValueError, match='index'):
        dqp.HardwareModel().delay(0, index=-1)
    cir = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
    cir.delay(0, inputs=[0.3, 0.1])
    for config in [dqp.HardwareModel().delay(0, index=1).theta, dqp.HardwareModel().delay(0).ps(index=1).theta]:
        with pytest.raises(ValueError, match='matches no operations'):
            cir.set_hardware_model(config.gaussian())


def test_unrolled_bindings_follow_objects_and_reject_removal():
    cir = circuit()
    cir.set_hardware_model(dqp.HardwareModel().ps(1).theta.gaussian(mu=0.07, sigma=0, name='phase'))
    expanded = cir.global_circuit(2)
    cloned = deepcopy(expanded)
    assert cloned.hardware_model.bindings.phase.bound_ops <= set(cloned.operators)
    assert_state_close(cloned.global_circuit(1)(), cloned())
    ref = cir.global_circuit(2, use_deepcopy=True)
    for target in [expanded, ref]:
        target.operators.insert(0, dqp.PhaseShift(torch.tensor(0.1), nmode=4, wires=1))
        target.operators[0], target.operators[1] = target.operators[1], target.operators[0]
    for gate in ref.hardware_model.bindings.phase.bound_ops:
        gate.init_para(gate.theta + 0.07)
    ref.set_hardware_model(None)
    assert_state_close(expanded(), ref())
    selected = next(iter(expanded.hardware_model.bindings.phase.bound_ops))
    index = next(i for i, gate in enumerate(expanded.operators) if gate is selected)
    expanded.operators[index] = dqp.PhaseShift(torch.tensor(0.2), nmode=4, wires=selected.wires)
    with pytest.raises(ValueError, match='removed or replaced'):
        expanded()


@pytest.mark.parametrize('gate_type', [op for op in _PARAMETERS if op is not dqp.Homodyne])
def test_standalone_gate_noise_and_gradients(gate_type):
    names = _PARAMETERS[gate_type]
    inputs = torch.linspace(0.1, 0.2, len(names)).requires_grad_()
    gate = gate_type(inputs=inputs.squeeze() if len(names) == 1 else inputs, nmode=2, cutoff=3).to(torch.double)
    mu, sigma = [torch.nn.Parameter(torch.tensor(value)) for value in [0.02, 0.03]]
    model = dqp.HardwareModel(seed=9).on(gate).gaussian(mu, sigma)
    state = torch.arange(1, 10, dtype=torch.double).reshape(1, 3, 3).to(torch.cdouble)
    state = state / state.norm()
    actual = model(state, gate, backend='fock')
    generator = torch.Generator().manual_seed(9)
    overrides = {name: getattr(gate, name) + mu + sigma * torch.randn((), generator=generator) for name in names}
    expected = functional_call(gate, overrides, (state,))
    torch.testing.assert_close(actual, expected)
    variables = (inputs, mu, sigma)
    actual_grad = torch.autograd.grad(actual.real.sum(), variables, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.real.sum(), variables)
    assert_state_close(actual_grad, expected_grad)


@pytest.mark.parametrize('kind', ['basis', 'tensor', 'density', 'mps', 'gaussian', 'bosonic'])
def test_noise_backend_batch_and_encoding(kind):
    backend = kind if kind in ('gaussian', 'bosonic') else 'fock'
    initial = [1, 0] if backend == 'fock' else [2 * torch.eye(4), torch.arange(4.0).reshape(4, 1) / 10]
    if backend == 'bosonic':
        initial.append(torch.ones(1))
    cir = dqp.QumodeCircuit(
        2, initial, cutoff=3, backend=backend, basis=kind == 'basis', den_mat=kind == 'density', mps=kind == 'mps'
    )
    cir.bs([0, 1], encode=True)
    cir.ps(0, encode=True)
    cir.to(torch.double)
    ref = deepcopy(cir)
    mu, sigma = [torch.nn.Parameter(torch.tensor(value)) for value in [0.01, 0.02]]
    cir.set_hardware_model(dqp.HardwareModel(seed=3).gaussian(mu, sigma))
    data = torch.tensor([[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]], requires_grad=True)
    generator = torch.Generator().manual_seed(3)
    epsilon = torch.stack([torch.randn((), generator=generator) for _ in range(3)])
    actual = cir(data)
    expected = ref(data + mu + sigma * epsilon)
    actual = actual if isinstance(actual, list) else [actual]
    expected = expected if isinstance(expected, list) else [expected]
    assert_state_close(actual, expected)
    variables = (data, mu, sigma)
    actual_grad = torch.autograd.grad(sum(x.real.sum() for x in actual), variables)
    expected_grad = torch.autograd.grad(sum(x.real.sum() for x in expected), variables)
    assert_state_close(actual_grad, expected_grad)


def test_streaming_noise_matches_global_state_and_gradients():
    cir = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
    cir.s(0, r=0.3)
    cir.delay(0, ntau=2, encode=True, convention='mzi', loop_gates=[dqp.PhaseShift(torch.tensor(0.2))])
    mu, sigma, t = [torch.nn.Parameter(torch.tensor(value)) for value in [0.01, 0.03, 0.9]]
    model = dqp.HardwareModel(seed=4)
    model.delay().theta.gaussian(mu, sigma)
    model.delay().ps().after.loss(t=t)
    model.input().loss(t=0.95)
    model.output().loss(t=0.85)
    cir.set_hardware_model(model)
    cir.to(torch.double)
    global_cir = cir.global_circuit(4)
    encoded_buffers = dict(cir.encoders[0].named_buffers())
    data = torch.linspace(0.1, 0.5, 24).reshape(2, 12).requires_grad_()
    actual = cir.tdm_substate(4, [1, 3], data=data)
    assert all(value is encoded_buffers[name] for name, value in cir.encoders[0].named_buffers())
    full = global_cir(data)
    outputs = cir.tdm_output_wires(4)[[1, 3]].flatten()
    xp = torch.cat([outputs, outputs + global_cir.nmode])
    expected = [full[0][:, xp[:, None], xp], full[1][:, xp]]
    assert_state_close(actual, expected)
    variables = (data, mu, sigma, t)
    actual_grad = torch.autograd.grad(sum(x.square().sum() for x in actual), variables)
    expected_grad = torch.autograd.grad(sum(x.square().sum() for x in expected), variables)
    assert_state_close(actual_grad, expected_grad)


@pytest.mark.parametrize('backend', ['fock', 'gaussian', 'bosonic'])
def test_homodyne_phase_noise(backend):
    cir = dqp.QumodeCircuit(1, 'vac', cutoff=5, backend=backend, basis=False)
    cir.s(0, r=0.2)
    cir.d(0, r=0.1)
    ref = deepcopy(cir)
    cir.homodyne(0, phi=0.2, eps=0.1)
    ref.homodyne(0, phi=0.23, eps=0.1)
    cir.set_hardware_model(dqp.HardwareModel().homodyne(0).phi.gaussian(mu=0.03, sigma=0))
    cir.to(torch.double)()
    ref.to(torch.double)()
    torch.manual_seed(5)
    actual = cir.measure_homodyne(2)
    torch.manual_seed(5)
    expected = ref.measure_homodyne(2)
    torch.testing.assert_close(actual, expected)
    assert_state_close(cir.state_measured, ref.state_measured)
    torch.testing.assert_close(cir.measurements[0].phi, torch.tensor([0.2]))


@pytest.mark.parametrize('backend', ['gaussian', 'bosonic'])
def test_tdm_homodyne_and_global_mapping(backend):
    def template(offset):
        cir = dqp.QumodeCircuitTDM(1, 'vac', backend=backend)
        cir.s(0, r=torch.tensor(0.2 + offset), theta=torch.tensor(offset))
        cir.delay(
            0,
            ntau=2,
            inputs=torch.tensor([0.3 + offset, 0.1 + offset]),
            loop_gates=[dqp.PhotonLoss(torch.tensor(0.9), convention='t')] if offset else None,
        )
        cir.homodyne(0, phi=torch.tensor(0.2 + offset), eps=0.1)
        return cir.to(torch.double)

    cir, ref = template(0), template(0.03)
    cir.set_hardware_model(dqp.HardwareModel().gaussian(mu=0.03, sigma=0).delay().ps().after.loss(t=0.9))
    expanded = cir.global_circuit(3)
    binding = next(iter(expanded.hardware_model.bindings.values()))
    assert set(expanded.measurements) <= binding.bound_ops
    torch.manual_seed(6)
    actual = cir(nstep=3)
    torch.manual_seed(6)
    expected = ref(nstep=3)
    assert_state_close(actual, expected)
    torch.testing.assert_close(cir.samples, ref.samples)
