"""Tests for the Bargmann backend: state representation, gate kernels and evolution."""

import cmath
import math

import pytest
import torch

import deepquantum as dq
import deepquantum.photonic as dqp
from deepquantum.photonic.bargmann import (
    PolyExpAnsatz,
    _fock_coefficients,
    displacement_kernel,
    gaussian_contract,
    gaussian_to_bargmann,
    loss_kernel,
    passive_kernel,
    squeezing_kernel,
    two_mode_squeezing_kernel,
)
from deepquantum.photonic.state import BargmannState

DT = torch.float64
CT = torch.complex128


def t(x):
    return torch.tensor(x, dtype=DT)


def ansatz(a, b, c, n_cv):
    a = torch.as_tensor(a, dtype=CT)
    nvar = a.shape[-1]
    b = torch.as_tensor(b, dtype=CT)
    c = torch.as_tensor(c, dtype=CT)
    return PolyExpAnsatz(a.reshape(1, 1, nvar, nvar), b.reshape(1, 1, nvar), c.reshape(1, 1, *c.shape), n_cv)


# ---------------------------------------------------------------------- algebra


def test_complex_gaussian_contraction_keeps_determinant_phase():
    phase = torch.tensor(math.pi / 3, dtype=DT)
    z = torch.exp(1j * phase)
    coupling = 1 - z
    matrix = torch.zeros(1, 4, 4, dtype=CT)
    matrix[0, 0, 1] = matrix[0, 1, 0] = coupling
    matrix[0, 2, 3] = matrix[0, 3, 2] = coupling
    linear = torch.zeros(1, 4, dtype=CT)
    contracted, displacement, factor, keep = gaussian_contract(matrix, linear, [(0, 1), (2, 3)])
    assert keep == [] and contracted.shape == (1, 0, 0) and displacement.shape == (1, 0)
    torch.testing.assert_close(factor, (1 / z**2).reshape(1))


def test_identity_contraction():
    # K(z, w) = exp(z w) contracted with F(u) = exp(a u^2 / 2) gives exp(a z^2 / 2).
    a0 = 0.3 + 0.2j
    joint = ansatz([[0, 1, 0], [1, 0, 0], [0, 0, a0]], [0, 0, 0], 1, 3)
    out = joint.contract([(1, 2)])
    torch.testing.assert_close(out.a[0, 0], torch.tensor([[a0]], dtype=CT))
    torch.testing.assert_close(out.c[0, 0], torch.tensor(1, dtype=CT))


def test_thermal_trace():
    q = 0.3
    rho = ansatz([[0, q], [q, 0]], [0, 0], 1 - q, 2)
    value = rho.contract([(0, 1)]).evaluate()
    torch.testing.assert_close(value, torch.ones(1, dtype=CT))


def test_fock_tensor_roundtrip_and_tensor_product():
    psi = torch.tensor([[0.6, 0.0, 0.8j], [0.0, 0.0, 0.0]], dtype=CT)
    state = BargmannState(psi.unsqueeze(0), nmode=2)
    torch.testing.assert_close(state.to_fock(3)[0, :2], psi)
    joint = state.ansatz.tensor_product(state.ansatz.conj())
    rho = joint.to_fock((3,) * 4)[0]
    full = torch.zeros(3, 3, dtype=CT)
    full[:2] = psi
    torch.testing.assert_close(rho, torch.einsum('ab,cd->abcd', full, full.conj()))


# ---------------------------------------------------------------------- kernels vs DeepQuantum gates

C = 7


def _kernel_cases():
    th = t(0.37)
    ps = dqp.PhaseShift(cutoff=C)
    yield 'ps', passive_kernel(ps.get_matrix(th)), ps.get_matrix_state(ps.get_matrix(th))
    bs = dqp.BeamSplitter(nmode=2, wires=[0, 1], cutoff=C)
    u = bs.get_matrix(t(0.53), t(0.21))
    yield 'bs', passive_kernel(u), bs.get_matrix_state(u)
    d = dqp.Displacement(cutoff=C)
    yield 'd', displacement_kernel(t(0.3), t(0.7)), d.get_matrix_state(t(0.3), t(0.7))
    s = dqp.Squeezing(cutoff=C)
    yield 's', squeezing_kernel(t(0.4), t(0.31)), s.get_matrix_state(t(0.4), t(0.31))
    s2 = dqp.Squeezing2(nmode=2, wires=[0, 1], cutoff=C)
    yield 's2', two_mode_squeezing_kernel(t(0.35), t(0.6)), s2.get_matrix_state(t(0.35), t(0.6))
    loss = dqp.PhotonLoss(cutoff=C)
    theta = t(0.9)
    kraus = loss.get_matrix_state(theta).to(CT)
    yield 'loss', loss_kernel(torch.cos(theta / 2) ** 2), torch.einsum('lac,lbd->abcd', kraus, kraus.conj())


@pytest.mark.parametrize('case', list(_kernel_cases()), ids=lambda case: case[0])
def test_kernels_match_fock_matrices(case):
    _, kernel, reference = case
    value = kernel.to_fock((C,) * kernel.n_cv)[0]
    torch.testing.assert_close(value, reference.to(CT), atol=1e-12, rtol=1e-10)


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
def test_operation_kernel_matches_fock_matrix(gate_type):
    two_mode = issubclass(gate_type, dqp.BeamSplitter) or gate_type in (dqp.Squeezing2, dqp.UAnyGate)
    nmode = 2 if two_mode else 1
    kwargs = dict(nmode=nmode, wires=list(range(nmode)), cutoff=3)
    if gate_type is dqp.UAnyGate:
        gate = gate_type(torch.tensor([[1, 1j], [1j, 1]], dtype=CT) / 2**0.5, **kwargs)
    else:
        gate = gate_type(**kwargs).to(DT)
    kernel = gate.get_bargmann_kernel()
    actual = kernel.to_fock((3,) * kernel.n_cv)[0]
    torch.testing.assert_close(actual, gate.update_matrix_state().to(CT), atol=1e-10, rtol=1e-9)


def test_channel_kernel_matches_kraus_map():
    channel = dqp.PhotonLoss(t(0.9), cutoff=3)
    kraus = channel.update_matrix_state().to(CT)
    expected = torch.einsum('lac,lbd->abcd', kraus, kraus.conj())
    torch.testing.assert_close(channel.get_bargmann_kernel().to_fock((3,) * 4)[0], expected)


def test_bargmann_exports_and_unsupported_operations():
    assert dq.BargmannState is dqp.BargmannState is BargmannState
    assert dqp.PolyExpAnsatz is PolyExpAnsatz
    assert dqp.Barrier(nmode=2).get_bargmann_kernel() is None
    for operation in (dqp.Kerr(cutoff=3), dqp.OPA(cutoff=3)):
        with pytest.raises(NotImplementedError, match='not supported by the Bargmann backend'):
            operation.get_bargmann_kernel()


def test_direct_gate_and_channel_apply_to_bargmann_state():
    ket = BargmannState([1], cutoff=2).to(DT)
    shifted = dqp.PhaseShift(0.3)(ket)
    torch.testing.assert_close(shifted.to_fock(2)[0, 1], torch.exp(torch.tensor(0.3j, dtype=CT)))
    loss = dqp.PhotonLoss(0.25, cutoff=2, convention='t')
    density = loss(ket.to_dm()).to_fock(2).reshape(2, 2)
    torch.testing.assert_close(density.diagonal().real, torch.tensor([0.75, 0.25], dtype=DT))


def test_bargmann_hardware_output_loss_and_control_offset():
    circuit = dq.QumodeCircuit(1, [1], cutoff=2, backend='bargmann', den_mat=True)
    circuit.set_hardware_model(dqp.HardwareModel().output().loss(t=0.25, requires_grad=True))
    density = circuit.to(DT)().to_fock(2).reshape(2, 2)
    torch.testing.assert_close(density.diagonal().real, torch.tensor([0.75, 0.25], dtype=DT))
    theta = next(circuit.hardware_model.parameters())
    assert torch.autograd.grad(density[1, 1].real, theta)[0].abs() > 0

    phase = dq.QumodeCircuit(1, [1], cutoff=2, backend='bargmann')
    phase.ps(0, 0.0)
    mu = torch.nn.Parameter(torch.tensor(0.3, dtype=DT))
    phase.set_hardware_model(dqp.HardwareModel().ps(0).theta.gaussian(mu=mu, sigma=0))
    amplitude = phase.to(DT)().to_fock(2)[0, 1]
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
    torch.testing.assert_close(circuit.to(DT)().to_fock(2), reference.to(DT)().to_fock(2))


def test_bargmann_hardware_encoded_batch_retains_sample_gradients():
    circuit = dq.QumodeCircuit(1, [1], cutoff=2, backend='bargmann')
    circuit.ps(0, encode=True)
    circuit.set_hardware_model(dqp.HardwareModel().ps(0).theta.gaussian(mu=0.3, sigma=0))
    data = torch.tensor([[0.1], [0.2]], dtype=DT, requires_grad=True)
    amplitude = circuit.to(DT)(data).to_fock(2)[:, 1]
    torch.testing.assert_close(amplitude, torch.exp(1j * (data[:, 0] + 0.3)))
    torch.testing.assert_close(torch.autograd.grad(amplitude.real.sum(), data)[0], -torch.sin(data + 0.3))


def test_replacing_hardware_model_invalidates_bargmann_herald_cache():
    circuit = dq.QumodeCircuit(2, [1, 0], cutoff=2, backend='bargmann', den_mat=True)
    circuit()
    circuit.set_hardware_model(dqp.HardwareModel().output(0).loss(t=0.5))
    with pytest.raises(RuntimeError, match='Run the circuit forward'):
        circuit.heralded_state(0, 1)
    circuit()
    _, probability = circuit.heralded_state(0, 1)
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
    assert circuit.state is output and circuit._herald_state is output
    assert 'state' not in dict(circuit.named_modules())
    assert '_herald_state' not in dict(circuit.named_modules())
    checkpoint = circuit.state_dict()
    assert set(checkpoint) == expected_keys
    restored = build()
    restored.load_state_dict(checkpoint)
    legacy = checkpoint.copy()
    for cache in ('state', '_herald_state'):
        for field in ('a', 'b', 'c'):
            legacy[f'{cache}.{field}'] = getattr(output, field).clone()
    build().load_state_dict(legacy)
    legacy['unrelated_cache_key'] = torch.zeros(())
    with pytest.raises(RuntimeError, match='unrelated_cache_key'):
        build().load_state_dict(legacy)
    torch.testing.assert_close(restored().to_fock(3), output.to_fock(3))
    circuit.to(DT)
    assert circuit.state is None and circuit._herald_state is None


# ---------------------------------------------------------------------- evolution


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
    cir.to(DT)
    return cir


def moments_density(cov, mean, cutoff):
    a, b, c = gaussian_to_bargmann(cov, mean)
    return _fock_coefficients(a, b, c, (cutoff,) * a.shape[-1])


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
    assert state.a.dtype == CT
    assert state.c.shape[2:] == ()  # Gaussian states carry no derived variables
    rho = state.to_dm().to_fock(cutoff)
    torch.testing.assert_close(rho, reference, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(state.norm(), torch.ones(1, dtype=DT), atol=1e-10, rtol=1e-10)


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
        cir.to(DT)
        return cir

    reference = build('fock')()
    state = build('bargmann')()
    assert state.c.shape[2:] == ((2, 1, 2) * (2 if den_mat else 1))
    torch.testing.assert_close(state.to_fock(3), reference, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(state.norm(), torch.ones(1, dtype=DT), atol=1e-12, rtol=1e-12)


def test_non_gaussian_input_with_gaussian_gates_matches_fock_backend():
    def build(backend, cutoff):
        cir = dq.QumodeCircuit(2, [(0.6, [1, 0]), (0.8j, [0, 2])], cutoff=cutoff, backend=backend, basis=False)
        cir.s(0, 0.2, 0.3)
        cir.bs([0, 1], [0.5, 0.1])
        cir.d(1, 0.15, 0.4)
        cir.s2([1, 0], 0.1, 0.2)
        cir.to(DT)
        return cir

    big, cutoff = 30, 5
    reference = build('fock', big)()
    state = build('bargmann', cutoff)()
    torch.testing.assert_close(state.to_fock(cutoff), reference[:, :cutoff, :cutoff], atol=1e-8, rtol=1e-6)
    # The input amplitudes are float32 literals, so the norm is one only to float32 precision.
    torch.testing.assert_close(state.norm(), torch.ones(1, dtype=DT), atol=1e-6, rtol=1e-6)


def test_mixed_gaussian_input_and_state_conversions():
    nbar = 0.4
    cov = torch.eye(2, dtype=DT) * (2 * nbar + 1) * dqp.hbar / (4 * dqp.kappa**2)
    mean = torch.tensor([[0.3], [0.1]], dtype=DT)
    with pytest.raises(ValueError, match='pure Gaussian'):
        BargmannState([cov, mean], nmode=1)
    state = BargmannState([cov, mean], nmode=1, den_mat=True)
    torch.testing.assert_close(state.norm(), torch.ones(1, dtype=DT))
    vac = BargmannState('vac', nmode=2).to(DT)
    torch.testing.assert_close(vac.to_dm().to_fock(2)[0, 0, 0, 0, 0], torch.tensor(1, dtype=CT))


# ---------------------------------------------------------------------- gradients, batches, errors


def test_gradcheck_through_kernels():
    def f(r, theta, alpha_r):
        state = BargmannState([1, 0], den_mat=False).to(DT)
        state = state.apply_unitary(squeezing_kernel(r, theta), [0])
        state = state.apply_unitary(displacement_kernel(alpha_r, theta), [1])
        rotation = torch.stack([torch.cos(r), -torch.sin(r), torch.sin(r), torch.cos(r)]).reshape(2, 2)
        state = state.apply_unitary(passive_kernel(rotation), [0, 1])
        state = state.to_dm().apply_channel(loss_kernel(torch.sigmoid(theta)), [1])
        return torch.view_as_real(state.to_fock(3))

    inputs = (t(0.3).requires_grad_(), t(0.4).requires_grad_(), t(0.0).requires_grad_())
    assert torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-6, rtol=1e-4)


def test_circuit_parameters_receive_gradients():
    torch.manual_seed(0)
    cir = dq.QumodeCircuit(2, [1, 0], cutoff=4, backend='bargmann', den_mat=True)
    cir.s(0)
    cir.bs([0, 1])
    cir.d(1)
    cir.loss(1)
    cir.to(DT)
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
    cir.to(DT)
    data = torch.rand(3, 4, dtype=DT)
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
    cir.k(0, 0.1)
    with pytest.raises(NotImplementedError, match='not supported by the Bargmann backend'):
        cir()
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
    cir.to(DT)
    ket = cir().to_fock()[0]
    expected = torch.tensor(
        [math.exp(-(abs(alpha) ** 2) / 2) * alpha**n / math.sqrt(math.factorial(n)) for n in range(5)], dtype=CT
    )
    torch.testing.assert_close(ket, expected, atol=1e-7, rtol=1e-6)
