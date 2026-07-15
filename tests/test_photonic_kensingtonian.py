from math import comb

import pytest
import torch

import deepquantum as dq
import deepquantum.photonic as dqp
from deepquantum.photonic import kensingtonian, torontonian


def test_kensingtonian_reduces_to_torontonian():
    torch.manual_seed(1)
    nmode = 3
    identity = torch.eye(2 * nmode, dtype=torch.double)
    raw = torch.randn(2 * nmode, 2 * nmode, dtype=torch.double)
    sigma_q = raw @ raw.mT + 2 * identity
    matrix = identity - torch.linalg.solve(sigma_q, identity)
    gamma = torch.randn(2 * nmode, dtype=torch.double) / 10
    clicks = torch.tensor([1, 0, 1])
    clicked = (clicks > 0).nonzero(as_tuple=False).flatten()
    indices = torch.cat([clicked, clicked + len(clicks)])
    submatrix = matrix[indices[:, None], indices]

    assert torch.allclose(kensingtonian(matrix, clicks, 1), torontonian(submatrix), atol=1e-10)
    assert torch.allclose(
        kensingtonian(matrix, clicks, 1, gamma),
        torontonian(submatrix, 2**0.5 * gamma[indices]),
        atol=1e-10,
    )


def test_kensingtonian_batch_chunk_and_edge_cases():
    torch.manual_seed(2)
    batch_size = 3
    nmode = 2
    identity = torch.eye(2 * nmode, dtype=torch.double)
    raw = torch.randn(batch_size, 2 * nmode, 2 * nmode, dtype=torch.double)
    sigma_q = raw @ raw.mT + 2 * identity
    matrices = identity - torch.linalg.solve(sigma_q, identity)
    gammas = torch.randn(batch_size, 2 * nmode, dtype=torch.double) / 10
    clicks = torch.tensor([2, 1])

    expected = torch.stack(
        [kensingtonian(matrix, clicks, 4, gamma) for matrix, gamma in zip(matrices, gammas, strict=True)]
    )
    actual = kensingtonian(matrices, clicks, 4, gammas)
    chunked = kensingtonian(matrices, clicks, 4, gammas, chunk_size=3)
    shared_gamma = gammas[0]

    assert torch.allclose(actual, expected, atol=1e-10)
    assert torch.allclose(chunked, expected, atol=1e-10)
    assert torch.equal(
        kensingtonian(matrices, clicks, 4, shared_gamma),
        kensingtonian(matrices, clicks, 4, shared_gamma.unsqueeze(0)),
    )
    assert torch.equal(kensingtonian(matrices, [0, 0], 4), torch.ones(batch_size, dtype=torch.double))


def test_qumode_circuit_click_matches_coherent_binomial_distribution():
    cutoff = 5
    num_detectors = cutoff - 1
    displacement = 0.6
    cir = dq.QumodeCircuit(1, 'vac', cutoff=cutoff, backend='gaussian', detector='click')
    cir.d(0, displacement, theta=0.3)
    cir.to(torch.double)
    cir()
    click_prob = 1 - torch.exp(torch.tensor(-(displacement**2) / num_detectors, dtype=torch.double))

    for num_clicks in range(cutoff):
        expected = (
            comb(num_detectors, num_clicks) * click_prob**num_clicks * (1 - click_prob) ** (num_detectors - num_clicks)
        )
        assert torch.allclose(cir.get_prob([num_clicks]), expected, atol=1e-10)

    probs = torch.stack(list(cir(is_prob=True).values()))
    assert torch.allclose(probs.sum(), torch.ones((), dtype=torch.double), atol=1e-10)


def test_qumode_circuit_click_probabilities_are_normalized():
    cir = dq.QumodeCircuit(2, 'vac', cutoff=4, backend='gaussian', detector='click')
    cir.s(0, r=0.35, theta=0.2)
    cir.s(1, r=0.2, theta=-0.1)
    cir.bs([0, 1], [0.4, 0.3])
    cir.to(torch.double)

    probs = torch.stack(list(cir(is_prob=True).values()))

    assert torch.all((probs >= 0) & (probs <= 1))
    assert torch.allclose(probs.sum(), torch.ones((), dtype=torch.double), rtol=1e-9, atol=1e-10)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS is not available')
def test_qumode_circuit_click_matches_cpu_on_mps():
    identity = torch.eye(4)
    sigma_q = torch.tensor(
        [
            [1.5, 0.1, 0.0, 0.0],
            [0.1, 1.6, 0.0, 0.0],
            [0.0, 0.0, 1.7, 0.1],
            [0.0, 0.0, 0.1, 1.8],
        ]
    )
    alpha = torch.tensor([[0.1], [-0.1], [0.05], [-0.05]])
    cov = (sigma_q - identity / 2) * dqp.hbar / (2 * dqp.kappa**2)
    mean = alpha * dqp.hbar**0.5 / dqp.kappa
    final_states = torch.tensor([[0, 2], [2, 0]])
    cir = dq.QumodeCircuit(2, 'vac', cutoff=3, backend='gaussian', detector='click')

    expected = cir._get_probs_gaussian_helper(final_states, cov, mean, detector='click', loop=True)
    actual = cir._get_probs_gaussian_helper(
        final_states,
        cov.to('mps'),
        mean.to('mps'),
        detector='click',
        loop=True,
    ).cpu()

    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)
