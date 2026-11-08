import math

import pytest
import torch

import deepquantum as dq
import deepquantum.photonic as dqp
from deepquantum.photonic.qmath import product_factorial, sample_homodyne_fock


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


def test_batched_fock_basis_states():
    batch = 3
    # set nphoton = max_photon_in_one_mode + 1
    nphoton = 3
    init_state = torch.randint(nphoton, (batch, 3))
    data = torch.randn(8)

    cir = dq.QumodeCircuit(nmode=3, init_state=init_state, basis=True)
    cir.ps([0], encode=True)
    cir.ps([1], encode=True)
    cir.ps([2], encode=True)
    cir.bs_theta([0, 1], encode=True)
    cir.bs_theta([1, 2], encode=True)
    cir.ps([0], encode=True)
    cir.ps([1], encode=True)
    cir.ps([2], encode=True)

    re1 = cir(data=data, state=init_state, is_prob=None)
    re2 = cir(data=data, state=init_state, is_prob=False)
    re3 = cir(data=data, state=init_state, is_prob=True)

    for i in range(init_state.shape[0]):
        res1 = cir(data=data, state=init_state[i], is_prob=None)
        res2 = cir(data=data, state=init_state[i], is_prob=False)
        res3 = cir(data=data, state=init_state[i], is_prob=True)

        # test is_prob=None
        assert torch.equal(res1, re1[i])

        for key in res2:
            # test is_prob=False
            assert torch.allclose(res2[key], re2[key][i], atol=1e-6)

        for key in res3:
            # test is prob=True
            assert torch.allclose(res3[key], re3[key][i], atol=1e-6)


def test_batched_fock_basis_states_and_data():
    batch = 3
    # set nphoton = max_photon_in_one_mode + 1
    nphoton = 3
    init_state = torch.randint(nphoton, (batch, 3))
    data = torch.randn(batch * 8).reshape(batch, 8)

    cir = dq.QumodeCircuit(nmode=3, init_state=init_state, basis=True)
    cir.ps([0], encode=True)
    cir.ps([1], encode=True)
    cir.ps([2], encode=True)
    cir.bs_theta([0, 1], encode=True)
    cir.bs_theta([1, 2], encode=True)
    cir.ps([0], encode=True)
    cir.ps([1], encode=True)
    cir.ps([2], encode=True)

    re1 = cir(data=data, state=init_state, is_prob=None)
    re2 = cir(data=data, state=init_state, is_prob=False)
    re3 = cir(data=data, state=init_state, is_prob=True)

    for i in range(init_state.shape[0]):
        res1 = cir(data=data[i], state=init_state[i], is_prob=None)
        res2 = cir(data=data[i], state=init_state[i], is_prob=False)
        res3 = cir(data=data[i], state=init_state[i], is_prob=True)

        # test is_prob=None
        assert torch.allclose(res1, re1[i], atol=1e-6)

        for key in res2:
            # test is_prob=False
            assert torch.allclose(res2[key], re2[key][i], atol=1e-5)

        for key in res3:
            # test is_prob=True
            assert torch.allclose(res3[key], re3[key][i], atol=1e-6)


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
