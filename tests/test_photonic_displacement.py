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
@pytest.mark.parametrize('loss, convention', [(0.0, 't'), (0.7, 't'), (1.0, 't'), (-0.8, 'theta'), (4.0, 'theta')])
def test_displacement_channel_matches_coherent_ancilla_beam_splitter(backend, loss, convention, monkeypatch):
    monkeypatch.setattr(dqp, 'hbar', 1.3)
    monkeypatch.setattr(dqp, 'kappa', 0.8)
    cutoff = 10
    magnitude, phase = torch.tensor(0.3), torch.tensor(0.4)
    init = [1, 0] if backend == 'fock' else 'vac'
    cir = dqp.QumodeCircuit(2, init, backend=backend, cutoff=cutoff, basis=False, den_mat=True)
    ref = dqp.QumodeCircuit(3, [1, 0, 0] if backend == 'fock' else 'vac', backend=backend, cutoff=cutoff, basis=False)
    for circuit in (cir, ref):
        if backend == 'fock':
            circuit.bs([0, 1], [0.3, 0.2])
        else:
            circuit.s2([0, 1], r=0.3)
        circuit.d(1, r=0.07, theta=0.2)
    cir.d_channel(1, amplitude=magnitude, theta=phase, loss=loss, convention=convention)
    ref.d(2, r=magnitude, theta=phase)
    angle = torch.acos(torch.tensor(loss).sqrt()) if convention == 't' else torch.tensor(loss / 2)
    ref.bs([1, 2], [angle, torch.tensor(torch.pi)])
    cir.double()
    ref.double()
    actual, expected = cir(), ref()
    if backend == 'fock':
        ket = expected.reshape(cutoff**2, cutoff)
        torch.testing.assert_close(actual.reshape(cutoff**2, cutoff**2), ket @ ket.mH, atol=3e-8, rtol=3e-8)
    else:
        cov, mean = expected[:2]
        keep = [0, 1, 3, 4]
        expected = [cov[..., keep, :][..., keep], mean[..., keep, :]] + expected[2:]
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('coefficients', [[1.0, 0.0], [0.0, 1.0], [0.8, 0.6j]])
def test_encoded_coupling_gradients_and_submodes(dtype, coefficients):
    torch.set_default_dtype(dtype)
    cir = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
    cir.s(0, r=0.3)
    cir.d_channel(0, loss=0.9, convention='t', encode=True)
    model = dqp.HardwareModel().input().submodes([1.0, 0.0])
    model.s().submodes([1.0, 0.0])
    model.d_channel().submodes(coefficients)
    offset = torch.nn.Parameter(torch.tensor(0.02))
    model.d_channel().loss_theta.gaussian(mu=offset, sigma=0)
    cir.set_hardware_model(model)
    cir.to(dtype)
    data = torch.tensor([[0.2, 0.4, 0.9], [0.3, -0.1, 0.8]], requires_grad=True)
    result = cir(data)
    cov, mean = result.to_joint()
    angle = 2 * torch.acos(data[:, 2].sqrt()) + offset
    t = torch.cos(angle / 2).square()
    expected_xvar = t * torch.exp(torch.tensor(-0.6)) + (1 - t)
    torch.testing.assert_close(cov[:, 0, 0], expected_xvar)
    torch.testing.assert_close(cov[:, 1, 1], torch.ones(2))
    profile = torch.as_tensor(coefficients, dtype=torch.complex64 if dtype == torch.float32 else torch.complex128)
    profile = profile / profile.abs().square().sum().sqrt()
    alpha = torch.sin(angle / 2) * data[:, 0] * torch.exp(1j * data[:, 1])
    shift = alpha[:, None] * profile.to(alpha)
    expected_mean = torch.cat([shift.real, shift.imag], -1).unsqueeze(-1) * dqp.hbar**0.5 / dqp.kappa
    torch.testing.assert_close(mean, expected_mean)
    gradients = torch.autograd.grad(cov.square().sum() + mean.square().sum(), (data, offset))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert gradients[1].abs() > 0
    assert cir.ndata == 3


def test_polar_zero_drive_gradients_and_parameter_dtype():
    channel = dqp.DisplacementChannel([0.0, 0.3, 0.8], cutoff=6, convention='t', requires_grad=True)
    channel.float()
    channel.double()
    assert all(p.dtype == torch.float64 for p in channel.parameters())
    assert len(list(channel.parameters())) == 3
    state = dqp.FockState('vac', nmode=1, cutoff=6, basis=False, den_mat=True).double().state
    state = channel.tensor_rep(state)
    out = channel(state).reshape(6, 6)
    # <1|rho|0> = alpha to first order at vacuum.
    real_grad = torch.autograd.grad(out[1, 0].real, channel.amplitude, retain_graph=True)[0]
    imag_grad = torch.autograd.grad(out[1, 0].imag, channel.amplitude, retain_graph=True)[0]
    phase_grad = torch.autograd.grad(out[1, 0].real, channel.theta)[0]
    expected = torch.sin(channel.loss_theta.detach() / 2) * torch.exp(1j * channel.theta.detach())
    torch.testing.assert_close(real_grad, expected.real)
    torch.testing.assert_close(imag_grad, expected.imag)
    torch.testing.assert_close(phase_grad, torch.zeros_like(phase_grad))


def test_displacement_channel_validation():
    for t in (-0.1, 1.1, float('nan')):
        cir = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
        with pytest.raises(ValueError):
            cir.d_channel(0, amplitude=1, loss=t, convention='t')
        assert len(cir.operators) == 0
    with pytest.raises(NotImplementedError, match='den_mat=True'):
        dqp.QumodeCircuit(1, 'vac', basis=False).d_channel(0, amplitude=0.1, loss=0.9, convention='t')
    with pytest.raises(NotImplementedError, match='den_mat=True'):
        dqp.QumodeCircuit(1, 'vac').add(dqp.DisplacementChannel([0.1, 0.0, 0.9], convention='t'))


def test_displacement_channel_in_tdm_substate():
    cir = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
    cir.s(0, r=torch.tensor(0.3))
    cir.delay(0, ntau=2, inputs=torch.tensor([0.4, 0.2]))
    cir.d_channel(0, amplitude=0.3, theta=0.2, loss=0.9, convention='t')
    model = dqp.HardwareModel().d_channel().amplitude.gaussian(mu=0.03, sigma=0)
    model.d_channel().loss_theta.gaussian(mu=0.02, sigma=0)
    cir.set_hardware_model(model)
    cir.double()
    full = cir.global_circuit(4)
    cov, mean = full()
    wires = cir.tdm_output_wires(4).flatten()
    selected = wires[[1, 3]]
    idx = torch.cat([selected, selected + full.nmode])
    expected = [cov[:, idx[:, None], idx], mean[:, idx]]
    actual = cir.tdm_substate(4, [1, 3])
    torch.testing.assert_close(actual, expected)


def test_affine_channel_drive_vmap_and_gradcheck():
    channel = dqp.DisplacementChannel([0.0, 0.2, 0.8], nmode=2, wires=1, convention='t')
    state = [torch.eye(4)[None], torch.zeros(1, 4, 1)]

    def displaced(amplitude):
        return torch.func.functional_call(channel, {'amplitude': amplitude}, (state,))

    amplitudes = torch.tensor([0.0, 0.1, 0.3])
    actual = torch.vmap(displaced)(amplitudes)
    expected = [torch.stack(values) for values in zip(*(displaced(a) for a in amplitudes), strict=True)]
    torch.testing.assert_close(actual, expected)
    assert torch.autograd.gradcheck(displaced, (amplitudes[0].clone().requires_grad_(),))


def test_displacement_channel_circuit_parameters_and_conventions():
    torch.manual_seed(71)
    trainable = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
    trainable.d_channel(0)
    trainable.double()
    op = trainable.operators[0]
    assert set(dict(op.named_parameters())) == {'amplitude', 'theta', 'loss_theta'}
    _, mean = trainable()
    (mean[0, 0, 0] + 0.7 * mean[0, 1, 0]).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in op.parameters())

    angle = torch.tensor(0.8)
    states = []
    for loss, convention in [(angle, 'theta'), (torch.cos(angle / 2).square(), 't')]:
        cir = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
        cir.d_channel(0, amplitude=0.3, theta=0.2, loss=loss, convention=convention)
        assert not list(cir.parameters())
        cir.double()
        states.append(cir())
        torch.testing.assert_close(cir.operators[0].loss_theta, angle)
    torch.testing.assert_close(*states)

    encoded = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
    encoded.d_channel(0, encode=True)
    assert encoded.ndata == 3 and not list(encoded.parameters())
    encoded.double()
    actual = encoded(data=torch.tensor([0.3, 0.2, 0.8]))
    torch.testing.assert_close(actual, states[0])
