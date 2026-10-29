from copy import deepcopy
from decimal import Decimal, localcontext
from math import log
from unittest.mock import patch

import pytest
import torch
from torch.func import functional_call, stack_module_state, vmap

import deepquantum.photonic as dqp


@pytest.mark.parametrize('dtype, large', [(torch.float32, 90.0), (torch.float64, 710.0)])
def test_exprel_values_and_gradients(dtype, large):
    x = torch.tensor(
        [-1000, -0.050001, -0.05, -0.049999, -1e-8, 0, 1e-8, 0.049999, 0.05, 0.050001, large],
        dtype=dtype,
        requires_grad=True,
    )
    actual = dqp.OPA._exprel(x)
    gradient = torch.autograd.grad(actual.sum(), x)[0]
    with localcontext() as context:
        context.prec = 80
        reference = []
        for value in x.detach().tolist():
            z = Decimal(value)
            reference.append([1, 0.5] if z == 0 else [(z.exp() - 1) / z, ((z - 1) * z.exp() + 1) / z**2])
    expected = torch.tensor([[float(v) for v in row] for row in reference], dtype=dtype)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected[:, 0], atol=0, rtol=tolerance)
    torch.testing.assert_close(gradient, expected[:, 1], atol=0, rtol=tolerance)


def reference_opa(state, wire, r, theta, attenuation):
    """Integrate the joint drift/diffusion generator using a block matrix exponential."""
    cov, mean = state[:2]
    size = cov.shape[-1]
    indices = torch.tensor([wire, wire + size // 2], device=r.device)
    drift = r.new_zeros(size, size)
    diffusion = torch.zeros_like(drift)
    cos, sin = theta.cos(), theta.sin()
    drift[indices[:, None], indices] = torch.stack(
        [-attenuation / 2 - r * cos, -r * sin, -r * sin, -attenuation / 2 + r * cos]
    ).reshape(2, 2)
    diffusion[indices, indices] = attenuation * dqp.hbar / (4 * dqp.kappa**2)
    generator = torch.cat(
        [torch.cat([drift, diffusion], dim=1), torch.cat([torch.zeros_like(drift), -drift.mT], dim=1)]
    )
    evolution = torch.matrix_exp(generator)
    x = evolution[:size, :size]
    y = evolution[:size, size:] @ x.mT
    return [x @ cov @ x.mT + y, x.to(mean.dtype) @ mean] + state[2:]


@pytest.mark.parametrize('backend', ['gaussian', 'bosonic'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('convention', ['theta', 't', 'db', 'gain_db'])
def test_opa_generator_values_gradients_and_external_loss(backend, dtype, convention):
    cov = torch.tensor([[2.0, 0.2], [0.2, 1.5]], dtype=dtype)
    mean = torch.tensor([[0.1], [-0.2]], dtype=dtype)
    initial = [cov, mean]
    if backend == 'bosonic':
        initial = dqp.CatState(r=0.4, theta=0.3).to(dtype=dtype)
    cir = dqp.QumodeCircuit(1, initial, backend=backend)
    cir.opa(0, convention=convention, encode=True)
    assert cir.ndata == 3
    model = dqp.HardwareModel()
    model.opa().after.loss(t=torch.tensor(0.8, dtype=dtype))
    model.opa().r.gaussian(mu=torch.tensor(0.03, dtype=dtype), sigma=0)
    model.opa().theta.gaussian(mu=torch.tensor(-0.04, dtype=dtype), sigma=0)
    model.opa().loss_theta.gaussian(mu=torch.tensor(0.02, dtype=dtype), sigma=0)
    cir.set_hardware_model(model)
    cir.to(dtype=dtype)
    data = torch.tensor([[0.3, 0.2, 0.7], [-0.2, 0.7, 0.8]], dtype=dtype, requires_grad=True)
    actual = cir(data)
    state = [cir.init_state.cov, cir.init_state.mean]
    if backend == 'bosonic':
        state.append(cir.init_state.weight)
    expected_rows = []
    for r, theta, loss in data:
        if convention == 'gain_db':
            r = (r + loss) * (log(10) / 20)
        if convention == 'theta':
            angle = loss
        elif convention == 't':
            angle = 2 * torch.acos(loss.sqrt())
        else:
            angle = 2 * torch.acos(10 ** (-loss / 20))
        attenuation = -2 * torch.log(torch.cos((angle + 0.02) / 2))
        row = reference_opa(state, 0, r + 0.03, theta - 0.04, attenuation)
        row[0] = 0.8 * row[0] + 0.2 * torch.eye(2, dtype=dtype) * dqp.hbar / (4 * dqp.kappa**2)
        row[1] = 0.8**0.5 * row[1]
        expected_rows.append(row)
    expected = [torch.cat([row[i] for row in expected_rows]) for i in range(2)] + state[2:]
    tolerance = 2e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert all(value.real.dtype == dtype for value in actual)

    def objective(values):
        return sum(value.abs().square().sum() for value in values)

    torch.testing.assert_close(
        torch.autograd.grad(objective(actual), data),
        torch.autograd.grad(objective(expected), data),
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize('r, attenuation', [(0.0, 0.0), (0.3, 0.0), (0.0, 0.4), (0.3, 0.6), (-0.3, 0.6)])
def test_opa_limits_and_balanced_gain_loss(r, attenuation):
    cir = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
    cir.opa(0, encode=True)
    cir.double()
    angle = 2 * torch.acos(torch.exp(torch.tensor(-attenuation / 2, dtype=torch.double)))
    data = torch.tensor([r, 0.4, angle], dtype=torch.double, requires_grad=True)
    actual = cir(data)
    expected = reference_opa(
        [cir.init_state.cov, cir.init_state.mean], 0, data[0], data[1], -2 * torch.log(torch.cos(data[2] / 2))
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        torch.autograd.grad(actual[0].square().sum(), data),
        torch.autograd.grad(expected[0].square().sum(), data),
        atol=1e-11,
        rtol=1e-11,
    )
    vacuum = dqp.hbar / (4 * dqp.kappa**2)
    assert actual[0].det().item() >= vacuum**2 - 1e-12


@pytest.mark.parametrize('convention, loss', [('theta', torch.pi), ('t', 0.0), ('db', 1e20)])
@pytest.mark.parametrize('requires_grad', [False, True])
def test_opa_complete_loss_limit(convention, loss, requires_grad):
    op = dqp.OPA([0.4, 0.3, loss], convention=convention, requires_grad=requires_grad).double()
    x, y = op.update_transform_xy()
    torch.testing.assert_close(x, torch.zeros(2, 2, dtype=torch.double))
    torch.testing.assert_close(y, torch.eye(2, dtype=torch.double) * dqp.hbar / (4 * dqp.kappa**2))
    if requires_grad:
        gradients = torch.autograd.grad(x.sum() + y.sum(), tuple(op.parameters()))
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
    torch.testing.assert_close(op.float().double().update_transform_xy(), (x, y))


def test_opa_split_step_convergence_on_entangled_displaced_input():
    preparation = dqp.QumodeCircuit(2, 'vac', backend='gaussian').double()
    preparation.s2([0, 1], r=torch.tensor(0.3, dtype=torch.double))
    preparation.d(0, r=torch.tensor(0.2, dtype=torch.double))
    initial = preparation()
    r, theta, attenuation = [torch.tensor(x, dtype=torch.double) for x in [0.5, 0.7, 0.3]]
    cir = dqp.QumodeCircuit(2, initial, backend='gaussian')
    cir.opa(1, r=r, theta=theta, loss=attenuation * 10 / log(10), convention='db')
    actual = cir()
    reference = reference_opa(initial, 1, r, theta, attenuation)
    torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)
    errors = []
    for nstep in [8, 32, 128]:
        step = dqp.QumodeCircuit(2, initial, backend='gaussian')
        step.s(1, r=r / (2 * nstep), theta=theta)
        step.loss_db(1, attenuation * 10 / (log(10) * nstep))
        step.s(1, r=r / (2 * nstep), theta=theta)
        state = initial
        for _ in range(nstep):
            state = step(state=state)
        errors.append((state[0] - actual[0]).abs().max().item())
        torch.testing.assert_close(state[1], actual[1], atol=1e-12, rtol=1e-12)
    assert errors[1] < errors[0] / 10
    assert errors[2] < errors[1] / 10
    assert errors[2] < 1e-6


def test_opa_noise_realization():
    cir = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
    cir.opa(0, *torch.tensor([0.4, 0.2, 0.7], dtype=torch.double))
    model = dqp.HardwareModel(seed=17).opa().gaussian(sigma=0.03)
    cir.set_hardware_model(model)
    cir.double()
    actual = cir()
    generator = torch.Generator().manual_seed(17)
    r, theta, loss_theta = torch.tensor(
        [0.4, 0.2, 0.7], dtype=torch.double
    ) + model.bindings.binding_0.plugin.sigma * torch.randn(3, generator=generator, dtype=torch.double)
    reference = reference_opa(
        [cir.init_state.cov, cir.init_state.mean], 0, r, theta, -2 * torch.log(torch.cos(loss_theta / 2))
    )
    torch.testing.assert_close(actual, reference)
    assert torch.equal(model._generator.get_state(), generator.get_state())
    torch.testing.assert_close(cir.operators[0].r, torch.tensor(0.4, dtype=torch.double))
    model.manual_seed(17)
    torch.testing.assert_close(cir(), actual)


def test_opa_model_vmap_and_gradients():
    circuits = []
    for _ in range(2):
        cir = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
        cir.opa(0)
        cir.opa(0, encode=True)
        circuits.append(cir.double())
    parameters, buffers = stack_module_state(circuits)
    data = torch.tensor([[0.3, 0.2, 0.7], [0.4, 0.5, 0.8]], dtype=torch.double, requires_grad=True)

    def evaluate(parameters, buffers):
        return functional_call(circuits[0], (parameters, buffers), (data,))[0]

    actual = vmap(evaluate)(parameters, buffers)
    expected = torch.stack(
        [evaluate({k: v[i] for k, v in parameters.items()}, {k: v[i] for k, v in buffers.items()}) for i in range(2)]
    )
    torch.testing.assert_close(actual, expected)
    variables = (data, *parameters.values())
    torch.testing.assert_close(
        torch.autograd.grad(actual.square().sum(), variables),
        torch.autograd.grad(expected.square().sum(), variables),
    )


@pytest.mark.parametrize('encode_loop', [False, True])
@pytest.mark.parametrize('gain_db', [False, True])
def test_opa_tdm_streaming_unrolling_and_parameter_sharing(encode_loop, gain_db):
    cir = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
    if gain_db:
        cir.opa_db(0, encode=True)
    else:
        cir.opa(0, encode=True)
    loop_opa = dqp.OPA([0.1, 0.2, 0.7], requires_grad=not encode_loop)
    cir.delay(0, ntau=2, inputs=[0.3, 0.1], loop_gates=[loop_opa], encode=encode_loop)
    model = dqp.HardwareModel(seed=9).opa().theta.gaussian(sigma=0.02)
    model.delay().opa().loss_theta.gaussian(sigma=0.02)
    cir.set_hardware_model(model)
    cir.double()
    data = torch.linspace(0.1, 0.8, 2 * 3 * cir.ndata, dtype=torch.double).reshape(2, -1).requires_grad_()
    streamed = cir.tdm_substate(3, [0, 2], data=data)
    expanded = cir.global_circuit(3).double()
    expanded.hardware_model.manual_seed(9)
    cov, mean = expanded(data)
    outputs = cir.tdm_output_wires(3)[[0, 2]].flatten()
    indices = torch.cat([outputs, outputs + expanded.nmode])
    expected = [cov[:, indices[:, None], indices], mean[:, indices]]
    torch.testing.assert_close(streamed, expected)
    variables = (data, *loop_opa.parameters())
    torch.testing.assert_close(
        torch.autograd.grad(streamed[0].square().sum(), variables),
        torch.autograd.grad(expected[0].square().sum(), variables),
    )
    if not encode_loop:
        expanded_opa = [op for op in expanded.operators if isinstance(op, dqp.OPA)][1::2]
        for name, parameter in loop_opa.named_parameters():
            assert all(getattr(op, name) is parameter for op in expanded_opa)
        copied = cir.global_circuit(3, use_deepcopy=True)
        copied_opa = next(op for op in copied.operators if isinstance(op, dqp.OPA) and op.requires_grad)
        assert all(getattr(copied_opa, name) is not value for name, value in loop_opa.named_parameters())


@pytest.mark.parametrize('backend', ['gaussian', 'bosonic'])
def test_opa_tdm_measured_evolution_against_generator(backend):
    initial = 'vac' if backend == 'gaussian' else dqp.CatState(r=0.3, theta=0.2)
    cir = dqp.QumodeCircuitTDM(1, initial, backend=backend)
    cir.opa_db(0, gain_db=3.0, theta=0.2, loss_db=0.4)
    cir.delay(0, ntau=2, inputs=[0.3, 0.1], loop_gates=[dqp.OPA([0.1, 0.4, 0.2], convention='db')])
    cir.homodyne(0, phi=0.1)
    cir.double()
    reference = deepcopy(cir)
    torch.manual_seed(25)
    actual = cir(nstep=3)

    def reference_forward(op, state):
        return reference_opa(state, op.wires[0], op.r, op.theta, -torch.log(op.t))

    with patch.object(dqp.OPA, 'forward', reference_forward):
        torch.manual_seed(25)
        expected = reference(nstep=3)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(cir.samples, reference.samples, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize(
    'convention, loss',
    [
        ('db', -0.1),
        ('db', float('inf')),
        ('theta', float('nan')),
        ('t', torch.ones(2)),
        ('theta', 1j),
        ('t', -0.1),
        ('t', 1.1),
    ],
)
def test_opa_rejects_invalid_loss(convention, loss):
    with pytest.raises((ValueError, TypeError)):
        dqp.OPA([0.3, 0.2, loss], convention=convention)


@pytest.mark.parametrize('basis, den_mat', [(True, False), (False, False), (False, True)])
def test_opa_rejects_fock(basis, den_mat):
    cir = dqp.QumodeCircuit(1, 'vac', basis=basis, den_mat=den_mat)
    with pytest.raises(NotImplementedError, match='OPA supports Gaussian and Bosonic'):
        cir.opa(0, r=0.3, theta=0, loss=0.2, convention='db')
    op = dqp.OPA([0.3, 0, 0.2], convention='db')
    with pytest.raises(NotImplementedError, match='OPA supports Gaussian and Bosonic'):
        op(torch.ones(1, 2, 2))
    cir.operators.append(op)
    with pytest.raises(NotImplementedError, match='OPA'):
        cir()


@pytest.mark.parametrize('method', ['opa', 'opa_db'])
def test_opa_channel_api_training_and_moments(method):
    cir = dqp.QumodeCircuit(2, 'vac', backend='gaussian')
    cir.d(0, r=0.2, theta=0.1)
    cir.s2([0, 1], r=0.3)
    getattr(cir, method)(1)
    cir.double()
    opa = cir.operators[-1]
    assert isinstance(opa, dqp.operation.Channel)
    assert set(dict(opa.named_parameters())) == {'r', 'theta', 'loss_theta'}
    cov, mean = cir()
    torch.testing.assert_close(cir.get_displacement(cir.init_state.mean), mean)
    gradients = torch.autograd.grad(cov.square().sum() + mean.square().sum(), tuple(opa.parameters()))
    assert all(torch.isfinite(g).all() for g in gradients)
    with pytest.raises(ValueError, match='channels'):
        cir.get_symplectic()
    from deepquantum.photonic.draw import DrawCircuit

    drawing = DrawCircuit('opa', cir.nmode, cir.operators, cir.measurements)
    drawing.draw()
    assert 'OPA' in drawing.draw_.tostring()
    assert 'dB=' in drawing.draw_.tostring()


def test_opa_complete_positivity_and_vacuum_normalization():
    vacuum = dqp.hbar / (4 * dqp.kappa**2)
    omega = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.double)
    for r in [-0.8, 0.0, 0.7]:
        for db in [0.0, 0.3, 4.0]:
            op = dqp.OPA([r, 0.6, db], convention='db').double()
            x, y = op.update_transform_xy()
            cp = y + 1j * vacuum * (omega - x @ omega @ x.mT)
            assert torch.linalg.eigvalsh(cp).min() >= -1e-12
    for hbar, kappa in [(1.0, 1.0), (2.0, 2**-0.5)]:
        with patch.object(dqp, 'hbar', hbar), patch.object(dqp, 'kappa', kappa):
            op = dqp.OPA([0.0, 0.7, 1.2], convention='db').double()
            loss = dqp.PhotonLoss(op.db, convention='db').double()
            torch.testing.assert_close(op.update_transform_xy(), loss.update_transform_xy())
            squeeze = dqp.Squeezing([0.4, 0.7]).double()
            op = dqp.OPA([0.4, 0.7, 0.0]).double()
            x, y = op.update_transform_xy()
            torch.testing.assert_close(x, squeeze.update_transform_xp()[0])
            torch.testing.assert_close(y, torch.zeros_like(y))


@pytest.mark.parametrize('convention', ['theta', 't', 'db'])
def test_opa_separate_parameters_and_defaults(convention):
    cir = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
    cir.opa(0, r=0.3, convention=convention)
    op = cir.operators[0]
    assert not list(op.parameters())
    assert set(dict(op.named_buffers())) == {'r', 'theta', 'loss_theta'}
    torch.testing.assert_close(op.theta, torch.zeros(()))
    torch.testing.assert_close(op.t, torch.ones(()))
    torch.testing.assert_close(cir()[0][0], torch.diag(torch.exp(torch.tensor([-0.6, 0.6]))))


def test_opa_db_net_coherent_gain():
    gain, phase, loss = torch.tensor([6.0, 0.4, 1.2], dtype=torch.double, requires_grad=True)
    cir = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
    cir.opa_db(0, gain_db=gain, theta=phase, loss_db=loss)
    cir.double()
    assert not list(cir.parameters())
    op = cir.operators[0]
    torch.testing.assert_close(op.r, (gain + loss) * (log(10) / 20))
    x, y = op.update_transform_xy()
    amplified_axis = torch.stack([-torch.sin(phase / 2), torch.cos(phase / 2)])
    net_gain = (x @ amplified_axis).square().sum()
    torch.testing.assert_close(net_gain, 10 ** (gain / 10))
    torch.testing.assert_close(
        torch.autograd.grad(net_gain, (gain, phase, loss), retain_graph=True),
        (net_gain * log(10) / 10, torch.zeros_like(phase), torch.zeros_like(loss)),
        atol=1e-12,
        rtol=1e-12,
    )
    theoretical = dqp.OPA([op.r, phase, loss], convention='db')
    torch.testing.assert_close((x, y), theoretical.update_transform_xy())


@pytest.mark.parametrize('backend', ['gaussian', 'bosonic'])
@pytest.mark.parametrize('tdm', [False, True])
def test_opa_quadrature_normalization_invariance(backend, tdm):
    results = []
    for hbar, kappa in [(2.0, 2**-0.5), (1.0, 2**-0.5), (2.0, 1.0), (1.0, 1.0)]:
        with patch.object(dqp, 'hbar', hbar), patch.object(dqp, 'kappa', kappa):
            vacuum = hbar / (4 * kappa**2)
            initial = 'vac' if backend == 'gaussian' else dqp.CatState(r=0.3, theta=0.2)
            cir = dqp.QumodeCircuit(1, initial, backend=backend, cutoff=3)
            cir.d(0, r=0.2, theta=0.3)
            cir.opa(0, convention='db', encode=True)
            if tdm:
                cir.delay(0, ntau=1, inputs=[0.3, 0.1], loop_gates=[dqp.OPA([0.1, 0.2, 0.4])])
                cir = cir.global_circuit(2)
            cir.double()
            data = torch.tensor([0.4, 0.6, 0.7] * (2 if tdm else 1), dtype=torch.double, requires_grad=True)
            cov, mean, *weight = cir(data)
            normalized = [cov / vacuum, mean / vacuum**0.5, *weight]
            objective = sum(value.abs().square().sum() for value in normalized)
            gradients = torch.autograd.grad(objective, data)
            op = dqp.OPA(torch.tensor([0.4, 0.6, 0.7], dtype=torch.double), convention='db')
            x, y = op.update_transform_xy()
            values = [normalized, x, y / vacuum, gradients, cir.photon_number_mean_var()]
            if backend == 'gaussian':
                values.append(cir.get_prob([0] * cir.nmode))
            results.append(values)
    for values in results[1:]:
        torch.testing.assert_close(values, results[0], atol=2e-6, rtol=2e-6)
