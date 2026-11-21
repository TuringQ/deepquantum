"""Gaussian channel bridges, native sources and conditional loop evolution."""

import pytest
import torch

import deepquantum as dq
import deepquantum.photonic as dqp
from deepquantum.photonic.bargmann import gaussian_channel_to_bargmann


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('angle', [0.0, 0.7, torch.pi, 4.0])
def test_gaussian_channel_bridge_loss_and_displacement(dtype, angle):
    loss = dqp.PhotonLoss(torch.tensor(angle, dtype=dtype), cutoff=4)
    kernel = gaussian_channel_to_bargmann(*loss.update_transform_xyd())
    torch.testing.assert_close(kernel.to_fock((4,) * 4)[0], loss.get_superoperator())
    channel = dqp.DisplacementChannel(torch.tensor([0.3, 0.4, angle], dtype=dtype), convention='loss_theta', cutoff=4)
    # Loss cannot increase occupation, so composing its finite Kraus operators
    # with the analytic displacement matrix is exact for these input/output indices.
    kraus = channel.update_matrix_state()
    expected = torch.einsum('aik,ajl->ijkl', kraus, kraus.conj())
    torch.testing.assert_close(channel.get_superoperator(), expected, atol=2e-6, rtol=2e-5)


def test_multimode_affine_channel_bridge():
    gaussian = dq.QumodeCircuit(2, 'vac', backend='gaussian')
    gaussian.s2([0, 1], r=0.3, theta=0.4)
    gaussian.d(1, r=0.2, theta=-0.7)
    gaussian.double()
    cov, mean = gaussian()
    optical = dq.QumodeCircuit(2, 'vac', backend='gaussian')
    optical.bs([1, 0], [0.4, 0.6])
    optical.s(1, r=0.2, theta=-0.3)
    optical.double()
    x = 0.8 * optical.get_symplectic()
    y = torch.eye(4, dtype=torch.double) * 0.36
    d = torch.tensor([[0.1], [0.2], [-0.1], [0.0]], dtype=torch.double)
    kernel = gaussian_channel_to_bargmann(x, y, d)
    initial = dq.BargmannState([cov, mean], den_mat=True)
    actual = initial.apply_channel(kernel, [1, 0])
    order = [1, 0, 3, 2]
    xx, yy, dd = x[order][:, order], y[order][:, order], d[order]
    torch.testing.assert_close(actual.quadrature_moments(), (xx @ cov @ xx.mT + yy, xx @ mean + dd))
    torch.testing.assert_close(actual.norm(), torch.ones(1, dtype=torch.double))


@pytest.mark.parametrize('r, loss', [(0.0, 0.7), (0.3, 0.0), (0.3, 0.7), (0.3, torch.pi)])
def test_opa_fock_lindblad_reference(r, loss):
    # A larger truncated master equation is an independent non-Gaussian reference.
    cutoff, reference_cutoff = 5, 16
    values = torch.tensor([r, 0.4, loss], dtype=torch.double)
    channel = dqp.OPA(values, cutoff=cutoff)
    initial = dq.BargmannState(torch.tensor([0.6, 0.8j], dtype=torch.cdouble), nmode=1, den_mat=True)
    actual = channel(initial.to_fock(cutoff))[0]
    torch.testing.assert_close(actual, channel(initial).to_fock(cutoff)[0])
    if loss == torch.pi:
        expected = torch.zeros_like(actual)
        expected[0, 0] = 1
    else:
        a = torch.diag(torch.arange(1, reference_cutoff, dtype=torch.double).sqrt(), 1).to(torch.cdouble)
        identity = torch.eye(reference_cutoff, dtype=torch.cdouble)
        z = r * torch.exp(1j * values[1])
        generator = (z.conj() * (a @ a) - z * (a.mH @ a.mH)) / 2
        rate = -torch.log(channel.t)
        number = a.mH @ a
        liouvillian = torch.kron(generator, identity) + torch.kron(identity, generator.conj())
        liouvillian += rate * (
            torch.kron(a, a.conj()) - (torch.kron(number, identity) + torch.kron(identity, number.mT.contiguous())) / 2
        )
        rho = initial.to_fock(reference_cutoff)[0]
        expected = (torch.matrix_exp(liouvillian) @ rho.flatten()).reshape(reference_cutoff, reference_cutoff)
        expected = expected[:cutoff, :cutoff]
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-9)
    torch.testing.assert_close(actual, actual.mH)
    assert torch.linalg.eigvalsh(actual).min() > -1e-12


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_channel_encoding_and_hardware_gradients(dtype):
    outputs, gradients = [], []
    for backend in ['fock', 'bargmann']:
        cir = dq.QumodeCircuit(2, [0, 1], cutoff=4, backend=backend, basis=False, den_mat=True)
        cir.opa(1, encode=True)
        model = dqp.HardwareModel().opa().r.gaussian(mu=0.03, sigma=0)
        cir.set_hardware_model(model)
        cir.to(dtype)
        data = torch.tensor([[0.2, 0.3, 0.7], [0.3, -0.2, 0.4]], dtype=dtype, requires_grad=True)
        out = cir(data)
        if backend == 'bargmann':
            out = out.to_fock()
        outputs.append(out)
        gradients.append(torch.autograd.grad(out.real.sum(), data)[0])
    torch.testing.assert_close(*outputs)
    torch.testing.assert_close(*gradients, atol=2e-6, rtol=2e-5)
    assert torch.isfinite(gradients[0]).all() and (gradients[0].abs() > 1e-6).all()


def test_bargmann_sources_and_replacement_preserve_weight():
    nbar = torch.tensor(0.3, dtype=torch.double, requires_grad=True)
    initial = dq.BargmannState([(0.5, [0, 0]), (0.5j, [1, 1])], den_mat=True).double()
    cir = dq.QumodeCircuit(2, initial, backend='bargmann', den_mat=True)
    cir.thermal(0, nbar=nbar)
    cir.single_photon(1)
    cir.double()
    state = cir()
    torch.testing.assert_close(state.norm(), torch.tensor([0.5], dtype=torch.double))
    photons, _ = state.photon_number_mean_var()
    torch.testing.assert_close(photons, torch.stack([nbar, nbar.new_tensor(1)]).unsqueeze(0))
    torch.testing.assert_close(torch.autograd.grad(photons.sum(), nbar)[0], torch.ones_like(nbar))
    # Thermal preparation is analytic even when its Fock projection has a large missing tail.
    thermal = dq.QumodeCircuit(1, 'vac', backend='bargmann', den_mat=True, cutoff=2)
    thermal.thermal(0, nbar=2)
    out = thermal.double()()
    torch.testing.assert_close(out.norm(), torch.ones(1, dtype=torch.double))
    torch.testing.assert_close(out.to_fock()[0].trace().real, torch.tensor(5 / 9, dtype=torch.double))
    pure = dq.QumodeCircuit(2, 'vac', backend='bargmann')
    pure.single_photon([0, 1])
    torch.testing.assert_close(pure().to_fock(2)[0, 1, 1], torch.tensor(1 + 0j))
    prepared = dq.QumodeCircuit(1, 'vac', backend='bargmann')
    prepared.single_photon(0, coefficients=[1j], requires_grad=True)
    photon = prepared().to_fock(2)[0, 1]
    torch.testing.assert_close(photon, torch.tensor(1j))
    photon.real.backward()
    assert prepared.sources['0'].components.grad is not None


def test_bargmann_ket_source_checks_vacuum_relative_to_input_weight():
    amplitudes = torch.tensor([1e-3, 0], dtype=torch.cfloat)
    vacuum = dq.BargmannState(amplitudes, nmode=1)
    cir = dq.QumodeCircuit(1, vacuum, backend='bargmann')
    cir.single_photon(0)
    torch.testing.assert_close(cir().to_fock(2)[0], amplitudes.flip(0), atol=0, rtol=1e-6)
    cir.set_init_state(dq.BargmannState(amplitudes.flip(0), nmode=1))
    with pytest.raises(ValueError, match='requires vacuum'):
        cir()


def test_bargmann_loop_join_preserves_interleaved_density_axes_and_gradients():
    rng = torch.Generator().manual_seed(31)
    spatial_ket = torch.randn(2, 2, dtype=torch.cdouble, generator=rng, requires_grad=True)
    memory_ket = torch.randn(2, 2, 2, dtype=torch.cdouble, generator=rng, requires_grad=True)
    spatial_dm = torch.einsum('ij,kl->ijkl', spatial_ket, spatial_ket.conj())
    spatial = dq.BargmannState(spatial_dm, nmode=2, cutoff=2, den_mat=True)
    memory = dq.BargmannState(memory_ket, nmode=2, cutoff=7)
    cir = dq.QumodeCircuit(2, spatial, backend='bargmann')
    for wire in (0, 1):
        cir.delay(wire, ntau=1, inputs=[0.0, 0.0])
    cir.double()

    # Each identity delay places its loop mode immediately before its spatial mode.
    joint = cir(memory=memory)
    actual = joint.to_fock().reshape(2, 16, 16)
    ket = torch.einsum('ij,bkl->bkilj', spatial_ket, memory_ket).flatten(1)
    expected = ket.unsqueeze(-1) * ket.conj().unsqueeze(-2)
    assert joint.cutoff == 2 and memory.cutoff == 7
    torch.testing.assert_close(actual, expected)
    inputs = (spatial_ket, memory_ket)
    actual_grad = torch.autograd.grad(actual[:, 1, 4].imag.sum(), inputs)
    expected_grad = torch.autograd.grad(expected[:, 1, 4].imag.sum(), inputs)
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize('density, detector', [(False, 'pnrd'), (True, 'pnrd'), (True, 'threshold')])
def test_bargmann_loop_conditioning_matches_unrolled(density, detector):
    cir = dq.QumodeCircuit(1, 'vac', backend='bargmann', den_mat=density)
    cir.s(0, encode=True)
    cir.delay(0, ntau=2, inputs=[0.4, 0.1])
    if density:
        cir.opa(0, r=0.15, theta=0.2, loss=0.3)
        cir.set_hardware_model(dqp.HardwareModel().delay().bs_theta().after.loss(t=0.9))
    cir.double()
    data = torch.tensor([[0.3, 0.2], [0.25, -0.1], [0.2, 0.4]], dtype=torch.double, requires_grad=True)
    memory = None
    outcomes = [1, 0, 1]
    for row, outcome in zip(data, outcomes, strict=True):
        out = cir(row, memory=memory)
        assert out.nmode == 3
        memory, probability = cir.post_select(0, outcome, detector=detector, normalize=False)
    unrolled = cir.global_circuit(3)
    unrolled(data.flatten())
    outputs = cir.tdm_output_wires(3).flatten().tolist()
    expected, expected_probability = unrolled.post_select(outputs, outcomes, detector=detector, normalize=False)
    # Match the fixed loop wires in the unrolled circuit to the mode order after three time steps.
    expected = expected.reorder([1, 0])
    torch.testing.assert_close(memory.to_fock(3), expected.to_fock(3), atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(probability, expected_probability)
    grad = torch.autograd.grad(probability.sum(), data, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected_probability.sum(), data)[0]
    torch.testing.assert_close(grad, expected_grad, atol=1e-10, rtol=1e-8)


def test_native_loop_sources_batch_and_unconditional_memory():
    cir = dq.QumodeCircuit(2, 'vac', backend='bargmann', den_mat=True)
    cir.thermal(0, nbar=0.15)
    cir.d_channel(0, amplitude=0.3, phase=0.2, t=0.9)
    cir.s(1, encode=True)
    cir.bs([0, 1], [0.3, 0.1])
    cir.delay(0, ntau=2, inputs=[0.4, 0.1])
    cir.delay(1, ntau=1, inputs=[0.2, -0.1])
    cir.set_hardware_model(dqp.HardwareModel().output(1).loss(t=0.8))
    cir.double()
    unrolled = cir.global_circuit(3)
    data = torch.tensor([[0.2, 0.1], [0.3, -0.2]], dtype=torch.double, requires_grad=True)
    memory = None
    for _ in range(3):
        state = cir(data, memory=memory)
        memory = state.partial_trace([2, 4])
    reference = unrolled(data.repeat(1, 3))
    expected = reference.partial_trace([2, 4, 5, 6]).reorder([1, 0, 3, 2, 4])
    torch.testing.assert_close(state.quadrature_moments(), expected.quadrature_moments())
    torch.testing.assert_close(cir.quadrature_moments(), state.quadrature_moments([2, 4]))
    readout = {'phi': [0.1, 0.3], 'powers': [(2, 2)]}
    torch.testing.assert_close(cir.quadrature_moments([1, 0], **readout), state.quadrature_moments([4, 2], **readout))
    torch.testing.assert_close(cir.quadrature_mean(), state.quadrature_mean([2, 4]))
    actual_grad = torch.autograd.grad(state.photon_number_mean_var()[0].sum(), data, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.photon_number_mean_var()[0].sum(), data)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize('backend', ['gaussian', 'bosonic', 'fock'])
@pytest.mark.parametrize('profile', [False, True])
def test_opa_relative_r_keeps_uniform_distributed_loss(backend, profile):
    cir = dq.QumodeCircuit(1, 'vac', backend=backend, cutoff=4, basis=False, den_mat=True)
    cir.d(0, r=0.35, theta=0.1)
    cir.opa(0, r=0.25, theta=-0.3, loss=0.7)
    # Populate a zero-gain mode as well, so a missing attenuation is observable.
    model = dqp.HardwareModel().d().submodes([0, 0, 1])
    coefficients = torch.eye(3, dtype=torch.cdouble) if profile else None
    model.opa().submodes(coefficients, relative_r=[2, 1, 0])
    cir.set_hardware_model(model)
    cir.double()
    actual = cir().to_joint()
    reference = dq.QumodeCircuit(3, 'vac', backend=backend, cutoff=4, basis=False, den_mat=True)
    reference.d(2, r=cir.operators[0].r, theta=cir.operators[0].theta)
    opa = cir.operators[1]
    for wire, ratio in enumerate([1, 0.5, 0]):
        reference.opa(wire, r=opa.r * ratio, theta=opa.theta, loss=opa.loss_theta)
    reference.double()
    expected = reference()
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)


def test_bargmann_loop_normalized_event_weights_and_single_photon_source():
    def circuit():
        cir = dq.QumodeCircuit(1, 'vac', backend='bargmann', den_mat=True, cutoff=1)
        cir.single_photon(0)
        cir.delay(0, ntau=1, inputs=[0.4, 0.1])
        cir.double()
        return cir

    normalized, weighted = circuit(), circuit()
    memory_n = memory_w = None
    product = torch.ones(1, dtype=torch.double)
    for _ in range(3):
        normalized(memory=memory_n)
        weighted(memory=memory_w)
        memory_n, p = normalized.post_select(0, 1)
        memory_w, joint, info = weighted.post_select(0, 1, normalize=False, return_info=True)
        product = product * p
        torch.testing.assert_close(product, joint)
        torch.testing.assert_close(info['conditional_probability'], p)
        torch.testing.assert_close(memory_n.to_fock(4) * joint[:, None, None], memory_w.to_fock(4))


@pytest.mark.parametrize('channel_type', [dqp.OPA, dqp.DisplacementChannel])
def test_affine_channel_gradcheck_and_cutoff_projection(channel_type):
    values = torch.tensor([0.4, 0.2, 0.7], dtype=torch.double, requires_grad=True)

    def superoperator(controls):
        return channel_type(controls, cutoff=3).get_superoperator()

    assert torch.autograd.gradcheck(superoperator, (values,))
    small = channel_type(values, cutoff=4)(dq.BargmannState([1], den_mat=True).double().to_fock(4))
    large = channel_type(values, cutoff=12)(dq.BargmannState([1], den_mat=True).double().to_fock(12))
    torch.testing.assert_close(small, large[:, :4, :4])
    assert small[0].trace().real < large[0].trace().real
    assert large[0].trace().real > 0.999


@pytest.mark.parametrize('backend', ['gaussian', 'fock'])
def test_opa_factorized_submode_kernel_and_gradients(backend):
    cir = dq.QumodeCircuit(1, 'vac', backend=backend, basis=False, den_mat=True, cutoff=3)
    cir.opa(0, encode=True)
    cir.set_hardware_model(dqp.HardwareModel().opa().submodes(relative_r=[1, 0.5, 0]))
    cir.double()
    data = torch.tensor([0.2, 0.3, 0.7], dtype=torch.double, requires_grad=True)
    result = cir(data)
    assert result.factorized
    reference = dq.QumodeCircuit(3, 'vac', backend=backend, basis=False, den_mat=True, cutoff=3)
    for wire, ratio in enumerate([1, 0.5, 0]):
        reference.opa(wire, r=data[0] * ratio, theta=data[1], loss=data[2])
    reference.double()
    expected, actual = reference(), result.to_joint()
    torch.testing.assert_close(actual, expected)
    if backend == 'gaussian':
        actual, expected = actual[0], expected[0]
    grad = torch.autograd.grad(actual.real.sum(), data, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.real.sum(), data)[0]
    torch.testing.assert_close(grad, expected_grad)


@pytest.mark.parametrize('hbar, kappa', [(2.0, 2**-0.5), (1.0, 0.5)])
def test_opa_multiple_complex_gain_modes(monkeypatch, hbar, kappa):
    monkeypatch.setattr(dqp, 'hbar', hbar)
    monkeypatch.setattr(dqp, 'kappa', kappa)
    cir = dq.QumodeCircuit(1, 'vac', backend='gaussian')
    cir.thermal(0, nbar=0.1)
    cir.d(0, r=0.2, theta=0.1)
    cir.opa(0, r=0.3, theta=0.4, loss=0.7)
    modes = torch.tensor([[1, 1j], [1j, 1]], dtype=torch.cdouble) / 2**0.5
    cir.set_hardware_model(dqp.HardwareModel().opa().submodes(modes, relative_r=[1, 0.4]))
    cir.double()
    actual = cir().to_joint()
    # Rotate an independent two-mode reference into the fixed gain basis.
    reference = dq.QumodeCircuit(2, 'vac', backend='gaussian')
    reference.thermal(0, nbar=0.1)
    reference.d(0, r=0.2, theta=0.1)
    reference.any(modes.conj(), wires=[0, 1])
    reference.opa(0, r=0.3, theta=0.4, loss=0.7)
    reference.opa(1, r=0.12, theta=0.4, loss=0.7)
    reference.any(modes.mT, wires=[0, 1])
    reference.double()
    expected = reference()
    torch.testing.assert_close(actual, expected)
    # The same channel bridge must respect the quadrature unit convention.
    vacuum = hbar / (4 * kappa**2)
    cov = torch.diag(torch.tensor([1.2, 1, 1.2, 1], dtype=torch.double)) * vacuum
    native = dq.BargmannState([cov, torch.zeros(4, 1, dtype=torch.double)], den_mat=True)
    for op in reference.operators:
        native = op(native)
    torch.testing.assert_close(native.quadrature_moments(), tuple(expected))
