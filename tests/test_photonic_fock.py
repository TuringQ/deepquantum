import math

import pytest
import torch

import deepquantum as dq
import deepquantum.photonic as dqp
from deepquantum.photonic.qmath import product_factorial, sample_homodyne_fock
from deepquantum.qmath import evolve_state


@pytest.mark.parametrize('convention,value', [('t', 0.64), ('theta', 4.0)])
def test_photon_loss_density_coherences_match_kraus(convention, value):
    cutoff = 4
    gen = torch.Generator().manual_seed(24)
    ket = torch.randn(2, cutoff**2, dtype=torch.cdouble, generator=gen)
    ket = ket / ket.norm(dim=1, keepdim=True)
    rho = (ket[:, :, None] * ket.conj()[:, None, :]).reshape(2, *([cutoff] * 4)).requires_grad_()
    control = torch.tensor(value, dtype=torch.double, requires_grad=True)
    loss = dqp.PhotonLoss(control, nmode=2, wires=1, cutoff=cutoff, convention=convention)
    direct = loss(rho)
    operators = loss.update_matrix_state()
    expected = torch.einsum('lam,bimjn,lcn->biajc', operators, rho, operators.conj())
    torch.testing.assert_close(direct, expected)
    probe = torch.randn(rho.shape, dtype=rho.dtype, generator=gen)
    torch.testing.assert_close(
        torch.autograd.grad((probe * direct).real.sum(), (control, rho)),
        torch.autograd.grad((probe * expected).real.sum(), (control, rho)),
    )


@pytest.mark.parametrize('gate_type,wires,inputs', [(dqp.BeamSplitter, [2, 0], [0.41, -0.57]), (dqp.Kerr, [1], 0.23)])
def test_local_gate_superoperator_matches_density_evolution(gate_type, wires, inputs):
    nmode, cutoff = 3, 3
    controls = torch.tensor(inputs, dtype=torch.double, requires_grad=True)
    gate = gate_type(controls, nmode=nmode, wires=wires, cutoff=cutoff, den_mat=True)
    ket = torch.randn(2, cutoff**nmode, dtype=torch.cdouble, generator=torch.Generator().manual_seed(13))
    ket = ket / torch.linalg.vector_norm(ket, dim=1, keepdim=True)
    rho = (ket.unsqueeze(-1) * ket.conj().unsqueeze(-2)).reshape((2,) + (cutoff,) * (2 * nmode))
    expected = gate(rho)

    superoperator = gate.get_superoperator()
    assert superoperator.shape == (cutoff,) * (4 * len(wires))
    targets = wires + [w + nmode for w in wires]
    dim = cutoff ** len(targets)
    actual = evolve_state(rho, superoperator.reshape(dim, dim), 2 * nmode, targets, cutoff)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.real.sum(), controls), torch.autograd.grad(expected.real.sum(), controls)
    )


def test_barrier_local_fock_superoperator():
    barrier = dqp.Barrier(nmode=3, wires=[2, 0], cutoff=2).double()
    identity = torch.eye(16, dtype=torch.cdouble).reshape((2,) * 8)
    torch.testing.assert_close(barrier.get_superoperator(), identity)


def test_large_photon_normalization_and_homodyne(monkeypatch):
    cir = dq.QumodeCircuit(1, [100], cutoff=101)
    counts = torch.tensor([100])
    norm = cir._get_permanent_norms(counts, counts)
    torch.testing.assert_close(norm, torch.tensor([float(math.factorial(100))], dtype=torch.double))

    probabilities = []

    def capture(weights, num_samples, **_kwargs):
        assert torch.isfinite(weights).all()
        probabilities.append(weights / weights.sum(-1, keepdim=True))
        return torch.zeros(weights.shape[0], num_samples, dtype=torch.long, device=weights.device)

    monkeypatch.setattr(torch, 'multinomial', capture)
    state = torch.zeros(1, 41, dtype=torch.cfloat)
    state[0, 40] = 1
    sample_homodyne_fock(state, 0, 1, 41, nbin=4097)
    xs = torch.linspace(-15, 15, 4097)
    variance = (probabilities[0] * xs.square()).sum()
    expected = torch.tensor(81 * dqp.hbar / (4 * dqp.kappa**2), dtype=variance.dtype)
    torch.testing.assert_close(variance, expected, atol=1e-3, rtol=1e-4)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS is not available')
def test_product_factorial_mps_dtype():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        counts = torch.tensor([3], dtype=torch.long, device='mps')
        expected = torch.tensor([6.0], dtype=torch.float32, device='mps')
        torch.testing.assert_close(product_factorial(counts), expected)
        torch.testing.assert_close(product_factorial(counts.half()), expected.half())
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize('batch_data', [False, True], ids=['shared-controls', 'per-sample-controls'])
def test_batched_fock_basis_states(batch_data):
    init_state = torch.tensor([[1, 0, 1], [2, 1, 0], [0, 0, 1]])
    data = torch.randn((len(init_state), 8) if batch_data else (8,))
    cir = dq.QumodeCircuit(nmode=3, init_state=init_state, basis=True)
    cir.ps([0], encode=True)
    cir.ps([1], encode=True)
    cir.ps([2], encode=True)
    cir.bs_theta([0, 1], encode=True)
    cir.bs_theta([1, 2], encode=True)
    cir.ps([0], encode=True)
    cir.ps([1], encode=True)
    cir.ps([2], encode=True)

    unitaries = cir(data=data, state=init_state)
    amplitudes = cir(data=data, state=init_state, is_prob=False)
    probabilities = cir(data=data, state=init_state, is_prob=True)
    for i, state in enumerate(init_state):
        row = data[i] if batch_data else data
        unitary = cir(data=row, state=state)
        if batch_data:
            torch.testing.assert_close(unitary, unitaries[i], rtol=1e-5, atol=1e-6)
        else:
            assert torch.equal(unitary, unitaries[i])
        for event, amplitude in cir(data=row, state=state, is_prob=False).items():
            torch.testing.assert_close(amplitude, amplitudes[event][i], rtol=1e-5, atol=1e-5 if batch_data else 1e-6)
        for event, probability in cir(data=row, state=state, is_prob=True).items():
            torch.testing.assert_close(probability, probabilities[event][i], rtol=1e-5, atol=1e-6)


def test_loss_batched_fock_basis_states():
    batch = 3
    # set nphoton = max_photon_in_one_mode + 1
    nphoton = 3
    init_state = torch.randint(nphoton, (batch, 3))
    data = torch.randn(8)
    transmittance = torch.rand(6)

    cir = dq.QumodeCircuit(nmode=3, init_state=init_state, basis=True)
    cir.loss([0], transmittance[0])
    cir.ps([0], encode=True)
    cir.ps([1], encode=True)
    cir.ps([2], encode=True)
    cir.bs_theta([0, 1], encode=True)
    cir.loss([0], transmittance[1])
    cir.loss([1], transmittance[2])
    cir.bs_theta([1, 2], encode=True)
    cir.ps([0], encode=True)
    cir.ps([1], encode=True)
    cir.ps([2], encode=True)
    cir.loss(0, transmittance[3])
    cir.loss(1, transmittance[4])
    cir.loss(2, transmittance[5])

    re1 = cir(data=data, state=init_state, is_prob=None)
    re2 = cir(data=data, state=init_state, is_prob=True)

    for i in range(init_state.shape[0]):
        res1 = cir(data=data, state=init_state[i], is_prob=None)
        res2 = cir(data=data, state=init_state[i], is_prob=True)

        # test is_prob=None
        assert torch.equal(res1, re1[i])

        for key in res2:
            # test is prob=True
            assert torch.allclose(res2[key], re2[key][i], atol=1e-6)
