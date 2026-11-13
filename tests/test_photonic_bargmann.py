"""Tests for the Bargmann backend: state representation, gate kernels and evolution."""

import cmath
import math

import pytest
import torch

import deepquantum as dq
import deepquantum.photonic as dqp
from deepquantum.photonic.bargmann import BargmannPolyExp, _fock_coefficients, gaussian_contract, gaussian_to_bargmann
from deepquantum.photonic.state import BargmannState

dtype = torch.float64
cdtype = torch.complex128


def polyexp(mat_a, vec_b, coeff_c):
    mat_a = torch.as_tensor(mat_a, dtype=cdtype)
    nvar = mat_a.shape[-1]
    vec_b = torch.as_tensor(vec_b, dtype=cdtype)
    coeff_c = torch.as_tensor(coeff_c, dtype=cdtype)
    return BargmannPolyExp(
        mat_a.reshape(1, 1, nvar, nvar), vec_b.reshape(1, 1, nvar), coeff_c.reshape(1, 1, *coeff_c.shape)
    )


def test_complex_gaussian_contraction_keeps_determinant_phase():
    phase = torch.tensor(math.pi / 3, dtype=dtype)
    z = torch.exp(1j * phase)
    coupling = 1 - z
    matrix = torch.zeros(1, 4, 4, dtype=cdtype)
    matrix[0, 0, 1] = matrix[0, 1, 0] = coupling
    matrix[0, 2, 3] = matrix[0, 3, 2] = coupling
    linear = torch.zeros(1, 4, dtype=cdtype)
    contracted, displacement, factor, keep = gaussian_contract(matrix, linear, [(0, 1), (2, 3)])
    assert keep == [] and contracted.shape == (1, 0, 0) and displacement.shape == (1, 0)
    torch.testing.assert_close(factor, (1 / z**2).reshape(1))


def test_identity_contraction():
    # K(z, w) = exp(z w) contracted with F(u) = exp(a u^2 / 2) gives exp(a z^2 / 2).
    a0 = 0.3 + 0.2j
    joint = polyexp([[0, 1, 0], [1, 0, 0], [0, 0, a0]], [0, 0, 0], 1)
    out = joint.contract([(1, 2)])
    torch.testing.assert_close(out.mat_a[0, 0], torch.tensor([[a0]], dtype=cdtype))
    torch.testing.assert_close(out.coeff_c[0, 0], torch.tensor(1, dtype=cdtype))


def test_thermal_trace():
    q = 0.3
    rho = polyexp([[0, q], [q, 0]], [0, 0], 1 - q)
    value = rho.contract([(0, 1)]).evaluate()
    torch.testing.assert_close(value, torch.ones(1, dtype=cdtype))


def test_fock_tensor_roundtrip_and_tensor_product():
    psi = torch.tensor([[0.6, 0.0, 0.8j], [0.0, 0.0, 0.0]], dtype=cdtype)
    state = BargmannState(psi.unsqueeze(0), nmode=2)
    torch.testing.assert_close(state.to_fock(3)[0, :2], psi)
    joint = state.polyexp.tensor_product(state.polyexp.conj())
    rho = joint.to_fock((3,) * 4)[0]
    full = torch.zeros(3, 3, dtype=cdtype)
    full[:2] = psi
    torch.testing.assert_close(rho, torch.einsum('ab,cd->abcd', full, full.conj()))


@pytest.mark.parametrize('batched', [False, True])
def test_single_mode_ket_converts_to_density(batched):
    ket = torch.tensor([0.6, 0.8j], dtype=cdtype)
    if batched:
        ket = torch.stack([ket, ket.conj()])
        initial = dq.FockState(ket, nmode=1, basis=False)
    else:
        initial = ket
        ket = ket.unsqueeze(0)
    state = BargmannState(initial, nmode=1, den_mat=True)
    torch.testing.assert_close(state.to_fock(2), ket[:, :, None] * ket.conj()[:, None, :])


@pytest.mark.parametrize(
    'gate_type',
    [
        dqp.PhaseShift,
        dqp.BeamSplitter,
        dqp.BeamSplitterTheta,
        dqp.BeamSplitterPhi,
        dqp.BeamSplitterSingle,
        dqp.MZI,
        dqp.Squeezing,
        dqp.Squeezing2,
        dqp.Displacement,
        dqp.DisplacementPosition,
        dqp.DisplacementMomentum,
        dqp.UAnyGate,
    ],
)
def test_gate_bargmann_matches_fock_matrix(gate_type):
    two_mode = issubclass(gate_type, dqp.BeamSplitter) or gate_type in (dqp.Squeezing2, dqp.UAnyGate)
    nmode = 2 if two_mode else 1
    kwargs = dict(nmode=nmode, wires=list(range(nmode)), cutoff=5)
    if gate_type is dqp.UAnyGate:
        gate = gate_type(torch.tensor([[1, 1j], [1j, 1]], dtype=cdtype) / 2**0.5, **kwargs)
    else:
        gate = gate_type(**kwargs).to(dtype)
        gate.init_para(torch.tensor([0.31, 0.17] if gate.npara == 2 else 0.31, dtype=dtype))
    kernel = gate.get_bargmann()
    actual = kernel.to_fock((gate.cutoff,) * kernel.nfree)[0]
    torch.testing.assert_close(actual, gate.update_matrix_state().to(cdtype), atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize('theta', [0.9, 4.0])
def test_channel_bargmann_matches_kraus_map(theta):
    channel = dqp.PhotonLoss(torch.tensor(theta, dtype=dtype), cutoff=3)
    kraus = channel.update_matrix_state().to(cdtype)
    expected = torch.einsum('lac,lbd->abcd', kraus, kraus.conj())
    torch.testing.assert_close(channel.get_bargmann().to_fock((3,) * 4)[0], expected)


def test_barrier_and_unsupported_operations():
    assert dqp.Barrier(nmode=2).get_bargmann() is None
    for operation in (dqp.Kerr(cutoff=3), dqp.OPA(cutoff=3)):
        with pytest.raises(NotImplementedError, match='not supported by the Bargmann backend'):
            operation.get_bargmann()


def test_direct_gate_and_channel_apply_to_bargmann_state():
    ket = BargmannState([1], cutoff=2).to(dtype)
    shifted = dqp.PhaseShift(0.3)(ket)
    torch.testing.assert_close(shifted.to_fock(2)[0, 1], torch.exp(torch.tensor(0.3j, dtype=cdtype)))
    loss = dqp.PhotonLoss(0.25, cutoff=2, convention='t')
    density = loss(ket.to_dm()).to_fock(2).reshape(2, 2)
    torch.testing.assert_close(density.diagonal().real, torch.tensor([0.75, 0.25], dtype=dtype))


def test_bargmann_hardware_output_loss_and_control_offset():
    circuit = dq.QumodeCircuit(1, [1], cutoff=2, backend='bargmann', den_mat=True)
    circuit.set_hardware_model(dqp.HardwareModel().output().loss(t=0.25, requires_grad=True))
    density = circuit.to(dtype)().to_fock(2).reshape(2, 2)
    torch.testing.assert_close(density.diagonal().real, torch.tensor([0.75, 0.25], dtype=dtype))
    theta = next(circuit.hardware_model.parameters())
    assert torch.autograd.grad(density[1, 1].real, theta)[0].abs() > 0

    phase = dq.QumodeCircuit(1, [1], cutoff=2, backend='bargmann')
    phase.ps(0, 0.0)
    mu = torch.nn.Parameter(torch.tensor(0.3, dtype=dtype))
    phase.set_hardware_model(dqp.HardwareModel().ps(0).theta.gaussian(mu=mu, sigma=0))
    amplitude = phase.to(dtype)().to_fock(2)[0, 1]
    torch.testing.assert_close(amplitude, torch.exp(1j * mu))
    assert torch.autograd.grad(amplitude.real, mu)[0].abs() > 0


def test_bargmann_hardware_seeded_control_noise():
    circuit = dq.QumodeCircuit(1, [1], cutoff=2, backend='bargmann')
    circuit.ps(0, 0.2)
    circuit.set_hardware_model(dqp.HardwareModel(seed=7).ps(0).theta.gaussian(mu=0.03, sigma=0.1))
    generator = torch.Generator().manual_seed(7)
    sampled_theta = 0.23 + 0.1 * torch.randn((), generator=generator)
    torch.testing.assert_close(circuit().to_fock(2)[0, 1], torch.exp(1j * sampled_theta))


def test_bargmann_hardware_input_and_gate_boundary_loss_match_explicit_channels():
    circuit = dq.QumodeCircuit(2, [1, 0], cutoff=2, backend='bargmann', den_mat=True)
    circuit.ps(0, 0.1)
    model = dqp.HardwareModel().input(0).loss(t=0.8)
    model.ps(0).after.loss(t=0.5)
    model.output(1).loss(t=0.3)
    circuit.set_hardware_model(model)

    reference = dq.QumodeCircuit(2, [1, 0], cutoff=2, backend='bargmann', den_mat=True)
    reference.loss_t(0, 0.8)
    reference.ps(0, 0.1)
    reference.loss_t(0, 0.5)
    reference.loss_t(1, 0.3)
    torch.testing.assert_close(circuit.to(dtype)().to_fock(2), reference.to(dtype)().to_fock(2))


def test_bargmann_hardware_encoded_batch_retains_sample_gradients():
    circuit = dq.QumodeCircuit(1, [1], cutoff=2, backend='bargmann')
    circuit.ps(0, encode=True)
    circuit.set_hardware_model(dqp.HardwareModel().ps(0).theta.gaussian(mu=0.3, sigma=0))
    data = torch.tensor([[0.1], [0.2]], dtype=dtype, requires_grad=True)
    amplitude = circuit.to(dtype)(data).to_fock(2)[:, 1]
    torch.testing.assert_close(amplitude, torch.exp(1j * (data[:, 0] + 0.3)))
    torch.testing.assert_close(torch.autograd.grad(amplitude.real.sum(), data)[0], -torch.sin(data + 0.3))


@pytest.mark.parametrize('den_mat', [False, True])
def test_bargmann_hardware_batch_shares_noise_with_fock(den_mat):
    initial = [(1 / math.sqrt(2), [0]), (1 / math.sqrt(2), [1])]
    data = torch.tensor([[0.2], [0.4], [0.6]], dtype=dtype)

    def build(backend):
        circuit = dq.QumodeCircuit(1, initial, cutoff=3, backend=backend, basis=False, den_mat=den_mat)
        circuit.ps(0, encode=True)
        mu = torch.nn.Parameter(torch.tensor(0.01, dtype=dtype))
        sigma = torch.nn.Parameter(torch.tensor(0.2, dtype=dtype))
        model = dqp.HardwareModel(seed=7).ps(0).theta.gaussian(mu=mu, sigma=sigma)
        if den_mat:
            model.output().loss(t=0.8)
        circuit.set_hardware_model(model)
        return circuit.to(dtype), model, mu, sigma

    results = []
    for backend in ('bargmann', 'fock'):
        circuit, model, mu, sigma = build(backend)
        inputs = data.clone().requires_grad_()
        output = circuit(inputs)
        tensor = output.to_fock(3) if backend == 'bargmann' else output
        generator = torch.Generator().manual_seed(7)
        torch.randn((), generator=generator, dtype=dtype)
        assert torch.equal(model._generator.get_state(), generator.get_state())
        rows = []
        for row in data:
            model.manual_seed(7)
            single = circuit(row)
            rows.append(single.to_fock(3)[0] if backend == 'bargmann' else single[0])
        torch.testing.assert_close(tensor, torch.stack(rows))
        following = circuit(data)
        following = following.to_fock(3) if backend == 'bargmann' else following
        assert not torch.allclose(following, tensor)
        torch.randn((), generator=generator, dtype=dtype)
        assert torch.equal(model._generator.get_state(), generator.get_state())
        weights = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
        entry = tensor[:, 0, 1] if den_mat else tensor[:, 1]
        gradients = torch.autograd.grad((entry.real * weights).sum(), (inputs, mu, sigma))
        results.append((tensor, gradients))

    torch.testing.assert_close(results[0][0], results[1][0])
    for bargmann_grad, fock_grad in zip(results[0][1], results[1][1], strict=True):
        torch.testing.assert_close(bargmann_grad, fock_grad)


def test_replacing_hardware_model_invalidates_output():
    circuit = dq.QumodeCircuit(2, [1, 0], cutoff=2, backend='bargmann', den_mat=True)
    circuit()
    circuit.set_hardware_model(dqp.HardwareModel().output(0).loss(t=0.5))
    with pytest.raises(RuntimeError, match='Run the circuit forward'):
        circuit.post_select(0, 1, normalize=False)
    circuit()
    _, probability = circuit.post_select(0, 1, normalize=False)
    torch.testing.assert_close(probability, torch.tensor([0.5]))


def test_bargmann_hardware_rejects_unsupported_state_layouts():
    pure = dq.QumodeCircuit(1, [1], cutoff=2, backend='bargmann')
    with pytest.raises(NotImplementedError, match='den_mat=True'):
        pure.set_hardware_model(dqp.HardwareModel().output().loss(t=0.25))
    with pytest.raises(NotImplementedError, match='Orthogonal-basis submodes'):
        pure.set_hardware_model(dqp.HardwareModel().input(0).submodes([0.8, 0.6]))


@pytest.mark.parametrize('den_mat', [False, True])
@pytest.mark.parametrize('hardware', [False, True])
def test_forward_cache_is_not_part_of_bargmann_checkpoint(den_mat, hardware):
    def build():
        circuit = dq.QumodeCircuit(1, [1], cutoff=3, backend='bargmann', den_mat=den_mat)
        circuit.ps(0, 0.2)
        if hardware:
            model = dqp.HardwareModel()
            if den_mat:
                model.output().loss(t=0.8)
            else:
                model.ps(0).theta.gaussian(mu=0.1, sigma=0)
            circuit.set_hardware_model(model)
        return circuit

    circuit = build()
    expected_keys = set(circuit.state_dict())
    output = circuit()
    assert circuit.state is output
    assert 'state' not in dict(circuit.named_modules())
    checkpoint = circuit.state_dict()
    assert set(checkpoint) == expected_keys
    restored = build()
    restored.load_state_dict(checkpoint)
    torch.testing.assert_close(restored().to_fock(3), output.to_fock(3))
    circuit.to(dtype)
    assert circuit.state is None


def gaussian_circuit(backend, den_mat=False, loss=False, cutoff=6, **kwargs):
    cir = dq.QumodeCircuit(3, 'vac', cutoff=cutoff, backend=backend, den_mat=den_mat, **kwargs)
    cir.s(0, 0.3, 0.2)
    cir.s2([2, 1], 0.25, 0.7)
    cir.bs([1, 0], [0.4, 0.3])
    cir.d(2, 0.2, 1.1)
    cir.ps(1, 0.5)
    if loss:
        cir.loss_t(0, 0.7)
        cir.d(0, 0.15, 0.3)
        cir.loss_t(2, 0.85)
    cir.to(dtype)
    return cir


def moments_density(cov, mean, cutoff):
    mat_a, vec_b, coeff_c = gaussian_to_bargmann(cov, mean)
    return _fock_coefficients(mat_a, vec_b, coeff_c, (cutoff,) * mat_a.shape[-1])


@pytest.mark.parametrize('den_mat', [False, True])
@pytest.mark.parametrize('loss', [False, True])
def test_gaussian_circuit_matches_gaussian_backend(den_mat, loss):
    if loss and not den_mat:
        pytest.skip('Loss requires density matrices')
    cutoff = 4
    cov, mean = gaussian_circuit('gaussian', loss=loss)()
    reference = moments_density(cov, mean, cutoff)
    state = gaussian_circuit('bargmann', den_mat=den_mat, loss=loss, cutoff=cutoff)()
    assert isinstance(state, BargmannState)
    assert state.mat_a.dtype == cdtype
    assert state.coeff_c.shape[2:] == ()  # Gaussian states carry no derived variables
    rho = state.to_dm().to_fock(cutoff)
    torch.testing.assert_close(rho, reference, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(state.norm(), torch.ones(1, dtype=dtype), atol=1e-10, rtol=1e-10)


def test_ket_global_phase_matches_fock_backend():
    big, cutoff = 16, 4
    reference = gaussian_circuit('fock', cutoff=big, basis=False)()
    state = gaussian_circuit('bargmann', cutoff=cutoff)()
    torch.testing.assert_close(state.to_fock(cutoff), reference[:, :cutoff, :cutoff, :cutoff], atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('den_mat', [False, True])
def test_fock_input_passive_and_loss_match_fock_backend(den_mat):
    def build(backend):
        cir = dq.QumodeCircuit(3, [1, 0, 1], cutoff=3, backend=backend, basis=False, den_mat=den_mat)
        cir.bs([0, 1], [0.6, 0.2])
        cir.ps(2, 0.4)
        cir.bs([1, 2], [0.3, 1.1])
        if den_mat:
            cir.loss_t(1, 0.6)
        cir.bs([0, 2], [0.9, -0.5])
        cir.to(dtype)
        return cir

    reference = build('fock')()
    state = build('bargmann')()
    assert state.polyexp.aux_shape == ((2, 2) * (2 if den_mat else 1))
    torch.testing.assert_close(state.to_fock(3), reference, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(state.norm(), torch.ones(1, dtype=dtype), atol=1e-12, rtol=1e-12)


def test_non_gaussian_input_with_gaussian_gates_matches_fock_backend():
    def build(backend, cutoff):
        cir = dq.QumodeCircuit(2, [(0.6, [1, 0]), (0.8j, [0, 2])], cutoff=cutoff, backend=backend, basis=False)
        cir.s(0, 0.2, 0.3)
        cir.bs([0, 1], [0.5, 0.1])
        cir.d(1, 0.15, 0.4)
        cir.s2([1, 0], 0.1, 0.2)
        cir.to(dtype)
        return cir

    big, cutoff = 30, 5
    reference = build('fock', big)()
    state = build('bargmann', cutoff)()
    torch.testing.assert_close(state.to_fock(cutoff), reference[:, :cutoff, :cutoff], atol=1e-8, rtol=1e-6)
    # The input amplitudes are float32 literals, so the norm is one only to float32 precision.
    torch.testing.assert_close(state.norm(), torch.ones(1, dtype=dtype), atol=1e-6, rtol=1e-6)


def test_mixed_gaussian_input_and_state_conversions():
    nbar = 0.4
    cov = torch.eye(2, dtype=dtype) * (2 * nbar + 1) * dqp.hbar / (4 * dqp.kappa**2)
    mean = torch.tensor([[0.3], [0.1]], dtype=dtype)
    with pytest.raises(ValueError, match='pure Gaussian'):
        BargmannState([cov, mean], nmode=1)
    state = BargmannState([cov, mean], nmode=1, den_mat=True)
    torch.testing.assert_close(state.norm(), torch.ones(1, dtype=dtype))
    vac = BargmannState('vac', nmode=2).to(dtype)
    torch.testing.assert_close(vac.to_dm().to_fock(2)[0, 0, 0, 0, 0], torch.tensor(1, dtype=cdtype))


def test_gradcheck_through_gates_and_loss():
    def evaluate(r, theta, displacement):
        state = BargmannState([1, 0]).to(dtype)
        state = dqp.Squeezing([r, theta], nmode=2, wires=0)(state)
        state = dqp.Displacement([displacement, theta], nmode=2, wires=1)(state)
        state = dqp.BeamSplitter([r, theta], nmode=2, wires=[0, 1])(state)
        state = dqp.PhotonLoss(theta, nmode=2, wires=1)(state.to_dm())
        return torch.view_as_real(state.to_fock(3))

    inputs = tuple(torch.tensor(x, dtype=dtype, requires_grad=True) for x in (0.3, 0.4, 0.0))
    assert torch.autograd.gradcheck(evaluate, inputs, eps=1e-6, atol=1e-6, rtol=1e-4)


def test_circuit_parameters_receive_gradients():
    torch.manual_seed(0)
    cir = dq.QumodeCircuit(2, [1, 0], cutoff=4, backend='bargmann', den_mat=True)
    cir.s(0)
    cir.bs([0, 1])
    cir.d(1)
    cir.loss(1)
    cir.to(dtype)
    rho = cir().to_fock()
    loss = 1 - rho[0, 1, 1, 1, 1].real
    loss.backward()
    params = list(cir.parameters())
    assert params and all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)


def test_batched_encoding_matches_single_samples():
    cir = dq.QumodeCircuit(2, 'vac', cutoff=4, backend='bargmann')
    cir.s(0, encode=True)
    cir.bs([0, 1], [0.3, 0.2])
    cir.d(1, encode=True)
    cir.to(dtype)
    data = torch.rand(3, 4, dtype=dtype)
    batched = cir(data=data).to_fock()
    assert batched.shape == (3, 4, 4)
    for i in range(3):
        torch.testing.assert_close(batched[i], cir(data=data[i]).to_fock()[0])


@pytest.mark.parametrize(
    'method,args',
    [
        ('get_prob', ([1],)),
        ('get_amplitude', ([1],)),
        ('measure', ()),
        ('photon_number_mean_var', ()),
        ('quadrature_mean', ()),
        ('measure_homodyne', ()),
    ],
)
def test_unsupported_bargmann_readout_raises_explicitly(method, args):
    circuit = dq.QumodeCircuit(1, [1], cutoff=3, backend='bargmann')
    state = circuit()
    torch.testing.assert_close(state.to_fock(3)[0], torch.tensor([0, 1, 0], dtype=torch.complex64))
    with pytest.raises(NotImplementedError, match=rf'{method}.*to_fock'):
        getattr(circuit, method)(*args)


def test_unsupported_operations():
    cir = dq.QumodeCircuit(1, 'vac', cutoff=4, backend='bargmann')
    with pytest.raises(AssertionError):
        cir.loss_t(0, 0.5)
    with pytest.raises(NotImplementedError):
        dq.QumodeCircuit(1, 'vac', cutoff=4, backend='bargmann')(is_prob=True)


def test_initial_state_types():
    for init in ('vac', [0, 1], [(1.0, [0, 1])], dq.GaussianState('vac', nmode=2)):
        cir = dq.QumodeCircuit(2, init, cutoff=3, backend='bargmann')
        assert cir.backend == 'bargmann'
        assert isinstance(cir.init_state, BargmannState)
        torch.testing.assert_close(cir().norm(), torch.ones(1))
    state = BargmannState([0, 1], cutoff=3)
    cir = dq.QumodeCircuit(2, state, backend='fock')
    assert cir.backend == 'bargmann' and cir.cutoff == 3


def test_complex_phase_of_displacement():
    alpha = 0.2 * cmath.exp(0.7j)
    cir = dq.QumodeCircuit(1, 'vac', cutoff=5, backend='bargmann')
    cir.d(0, abs(alpha), cmath.phase(alpha))
    cir.to(dtype)
    ket = cir().to_fock()[0]
    expected = torch.tensor(
        [math.exp(-(abs(alpha) ** 2) / 2) * alpha**n / math.sqrt(math.factorial(n)) for n in range(5)], dtype=cdtype
    )
    torch.testing.assert_close(ket, expected, atol=1e-7, rtol=1e-6)


def polyexp_tensors(dtype=torch.complex128, coefficient=1):
    mat_a = torch.zeros(1, 1, 2, 2, dtype=dtype)
    vec_b = torch.zeros(1, 1, 2, dtype=dtype)
    coeff_c = torch.full((1, 1), coefficient, dtype=dtype)
    return mat_a, vec_b, coeff_c


@pytest.mark.parametrize('field', [0, 1, 2])
@pytest.mark.parametrize('invalid', [float('nan'), float('inf')])
def test_nonfinite_parameters_rejected(field, invalid):
    values = polyexp_tensors()
    values[field].reshape(-1)[0] = invalid
    with pytest.raises(ValueError, match='finite'):
        BargmannPolyExp(*values)


def test_asymmetric_matrix_rejected():
    mat_a, vec_b, coeff_c = polyexp_tensors()
    mat_a[0, 0, 0, 1] = 1
    with pytest.raises(ValueError, match='symmetric'):
        BargmannPolyExp(mat_a, vec_b, coeff_c)


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_complex_symmetric_matrix_and_roundoff_accepted(dtype):
    mat_a, vec_b, coeff_c = polyexp_tensors(dtype)
    mat_a[..., 0, 1] = 0.1j
    mat_a[..., 1, 0] = 0.1j + torch.finfo(mat_a.real.dtype).eps
    state = BargmannPolyExp(mat_a, vec_b, coeff_c)
    assert torch.isfinite(state.to_fock((3, 3))).all()


@pytest.mark.parametrize('invalid', [-0.1, 1.1, float('nan'), float('inf'), 0.2j])
def test_invalid_loss_transmittance_rejected(invalid):
    with pytest.raises(ValueError, match='transmittance'):
        dqp.PhotonLoss(torch.tensor(invalid), convention='t')


@pytest.mark.parametrize('transmittance', [0.0, 1.0])
def test_loss_endpoints_accepted(transmittance):
    assert torch.isfinite(dqp.PhotonLoss(torch.tensor(transmittance), convention='t').get_bargmann().mat_a).all()


def test_loss_interior_gradient():
    t = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    coefficient = dqp.PhotonLoss(t, convention='t').get_bargmann().mat_a[0, 0, 0, 2].real
    derivative = torch.autograd.grad(coefficient, t)[0]
    torch.testing.assert_close(derivative, 0.5 / t.sqrt())
    torch.autograd.gradcheck(lambda x: dqp.PhotonLoss(inputs=x, convention='t').t, (t,))


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_norm_rejects_significant_imaginary_trace(dtype):
    state = BargmannState(BargmannPolyExp(*polyexp_tensors(dtype, 1 + 0.01j)), den_mat=True)
    with pytest.raises(ValueError, match='imaginary'):
        state.norm()


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_norm_allows_roundoff_and_preserves_gradient(dtype):
    eps = torch.finfo(torch.empty((), dtype=dtype).real.dtype).eps
    mat_a, vec_b, coeff_c = polyexp_tensors(dtype, 1 + 1j * eps)
    coeff_c.requires_grad_()
    state = BargmannState(BargmannPolyExp(mat_a, vec_b, coeff_c), den_mat=True)
    value = state.norm()
    torch.testing.assert_close(value, torch.ones_like(value))
    gradient = torch.autograd.grad(value.sum(), coeff_c)[0]
    torch.testing.assert_close(gradient, torch.ones_like(coeff_c))


@pytest.mark.parametrize('density', [False, True])
def test_zero_order_auxiliary_variables_are_eliminated(density):
    initial = BargmannState([0, 0, 0], den_mat=density)
    assert initial.polyexp.naux == 0
    circuit = dq.QumodeCircuit(3, initial, backend='bargmann')
    circuit.s2([0, 1], 0.3, 0.2)
    circuit.to(dtype)
    circuit()
    state, probability = circuit.post_select(0, 0)
    assert state.polyexp.naux == 0
    expected = torch.zeros_like(state.to_fock(3))
    expected.reshape(1, -1)[:, 0] = 1
    torch.testing.assert_close(state.to_fock(3), expected)
    r = circuit.operators[0].r
    torch.testing.assert_close(probability, (1 / r.cosh() ** 2).reshape(1))


def test_thermal_source_rejects_bargmann_at_configuration():
    circuit = dq.QumodeCircuit(1, 'vac', backend='bargmann', den_mat=True)
    with pytest.raises(NotImplementedError, match='Thermal source preparation'):
        circuit.thermal(0, nbar=0.2)
