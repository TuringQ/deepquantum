"""Tests for the Bargmann backend: state representation, gate kernels and evolution."""

import cmath
import math

import pytest
import torch

import deepquantum as dq
import deepquantum.photonic as dqp
from deepquantum.photonic.bargmann import (
    BargmannState,
    PolyExpAnsatz,
    _fock_coefficients,
    displacement_kernel,
    gaussian_to_bargmann,
    loss_kernel,
    passive_kernel,
    squeezing_kernel,
    two_mode_squeezing_kernel,
)

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
        [math.exp(-abs(alpha) ** 2 / 2) * alpha**n / math.sqrt(math.factorial(n)) for n in range(5)], dtype=CT
    )
    torch.testing.assert_close(ket, expected, atol=1e-7, rtol=1e-6)
