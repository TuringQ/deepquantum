import networkx as nx
import numpy as np
import pytest
import torch
from scipy.stats import unitary_group

import deepquantum as dq
import deepquantum.photonic as dqp
from deepquantum.photonic import Squeezing2, ladder_to_quadrature, quadrature_to_ladder, xpxp_to_xxpp, xxpp_to_xpxp


def test_delay_coefficients_independent_grids_and_inputs():
    grids = [[2.0, 0.0, 1.0], [1.0, 1.2, 2.0, 3.0], [0.0, 0.25, 2.0]]
    magnitudes = [
        torch.ones(3, dtype=torch.double),
        torch.tensor([0, 0.2, 1, 2], dtype=torch.double),
        torch.ones(3, dtype=torch.double),
    ]
    phases = [0.0, 0.0, 0.7]
    amplitudes = [a * np.exp(1j * p) for a, p in zip(magnitudes, phases, strict=True)]
    coefficients = dqp.delay_coefficients(
        [0, 0, 0], frequency_hz=grids, amplitude=[a * s for a, s in zip(amplitudes, [1e-250, 1e250, 3], strict=True)]
    )
    from_intensity = dqp.delay_coefficients(
        [0, 0, 0],
        frequency_hz=grids,
        intensity=[a.square() * s for a, s in zip(magnitudes, [1e250, 1e-250, 2], strict=True)],
        phase=[[p] for p in phases],
    )
    torch.testing.assert_close(coefficients, from_intensity, atol=1e-14, rtol=1e-14)
    # Constant amplitude on [0, 2] and a linear ramp on [1, 3] have overlap sqrt(3) / 8.
    overlap, phase = np.sqrt(3) / 8, np.exp(0.7j)
    expected = torch.tensor(
        [[1, overlap, phase], [overlap, 1, overlap * phase], [phase.conjugate(), overlap * phase.conjugate(), 1]],
        dtype=torch.cdouble,
    )
    assert coefficients.shape == (3, 2)
    torch.testing.assert_close(coefficients.conj() @ coefficients.T, expected, atol=1e-14, rtol=1e-14)
    disjoint = dqp.delay_coefficients([0, 0], frequency_hz=[[0, 1], [2, 3]], intensity=[1, 1])
    torch.testing.assert_close(disjoint, torch.eye(2, dtype=torch.cdouble))


def test_delay_coefficients_resolves_phase_between_samples():
    delay = 37.25
    coefficients = dqp.delay_coefficients([0, delay], frequency_hz=[2e11 + 0.25, 2e11 + 1.25], intensity=[1, 1])
    # A unit-width flat spectrum has overlap exp(-2i*pi*center*tau) * sinc(tau).
    # The lower endpoint contributes exactly 0.3125 cycles modulo an integer.
    expected = torch.tensor(np.exp(-1j * np.pi * delay - 2j * np.pi * 0.3125) * np.sinc(delay), dtype=torch.cdouble)
    torch.testing.assert_close(coefficients[1, 0], expected, atol=1e-14, rtol=1e-13)


@pytest.mark.parametrize(
    'overrides,match',
    [
        ({'intensity': [1, 1]}, 'exactly one'),
        ({'phase': 0.0}, 'only used with intensity'),
        ({'amplitude': None, 'intensity': [1, -1]}, 'nonnegative'),
        ({'amplitude': [0, 0]}, 'nonzero'),
        ({'frequency_hz': [0, 0]}, 'distinct'),
        ({'frequency_hz': [0, float('nan')]}, 'finite'),
        ({'amplitude': [1, 2, 3]}, 'match its frequency grid'),
        ({'frequency_hz': [[0, 1]] * 3}, 'one row per delay'),
    ],
)
def test_delay_coefficients_validation(overrides, match):
    inputs = {'delays': [0, 0], 'frequency_hz': [0, 1], 'amplitude': [1, 1]} | overrides
    with pytest.raises(ValueError, match=match):
        dqp.delay_coefficients(**inputs)


def test_quadrature_ladder_transform():
    batch = 2
    nmode = 4
    vector = torch.randn(batch, 2 * nmode, 1)
    matrix = torch.randn(batch, 2 * nmode, 2 * nmode)
    vector2 = ladder_to_quadrature(quadrature_to_ladder(vector))
    matrix2 = ladder_to_quadrature(quadrature_to_ladder(matrix))
    assert torch.allclose(vector, vector2, atol=1e-5)
    assert torch.allclose(matrix, matrix2, atol=1e-5)


def test_gaussian_ordering():
    batch = 2
    nmode = 4
    vector = torch.randn(batch, 2 * nmode, 1)
    matrix = torch.randn(batch, 2 * nmode, 2 * nmode)
    vector2 = xpxp_to_xxpp(xxpp_to_xpxp(vector))
    matrix2 = xpxp_to_xxpp(xxpp_to_xpxp(matrix))
    assert torch.allclose(vector, vector2)
    assert torch.allclose(matrix, matrix2)


def test_takagi():
    size = 8
    for _ in range(10):
        graph = nx.erdos_renyi_graph(size, 0.5)
        a = torch.tensor(nx.to_numpy_array(graph), dtype=torch.float)
        u, diag = dq.takagi(a)
        s_diag = torch.diag(diag).to(u.dtype)
        assert torch.allclose(u @ u.mH, torch.eye(size) + 0j, rtol=1e-5, atol=1e-5)
        assert torch.allclose(u @ s_diag @ u.mT, a + 0j, rtol=1e-5, atol=1e-5)


def test_quadrature_ladder_transform_sq2():
    gate = Squeezing2()
    mat_ladder = gate.update_matrix()
    mat_xxpp = gate.update_transform_xp()[0]
    assert torch.allclose(ladder_to_quadrature(mat_ladder, True), mat_xxpp)
    assert torch.allclose(quadrature_to_ladder(mat_xxpp, True), mat_ladder)


def test_schur_anti_symm_even():
    n = np.random.randint(1, 6) * 2
    m = torch.randn(n, n)
    a = m - m.mT
    t, o = dqp.schur_anti_symm_even(a)
    err1 = abs(o @ t @ o.mT - a).sum()
    err2 = abs(o @ o.mT - torch.eye(n)).sum()
    assert err1 + err2 < 1e-4


def test_williamson():
    nmode = np.random.randint(2, 10)
    u = unitary_group.rvs(nmode)
    sqs = torch.rand(nmode)
    cir = dq.QumodeCircuit(nmode, init_state='vac', cutoff=4, backend='gaussian')
    for i in range(nmode):
        cir.s(i, sqs[i])
    cir.any(unitary=u, wires=list(range(nmode)))
    cov, _ = cir()
    t, s = dq.williamson(cov[0])
    err1 = abs((s @ t @ s.mT) - cov[0]).sum()  # 验证分解正确性
    omega = cov.new_ones(nmode)
    omega = torch.cat([-omega, omega]).diag_embed()
    omega = omega.reshape(2, nmode, 2 * nmode).flip(0).reshape(2 * nmode, 2 * nmode)  # symplectic form
    err2 = abs((s.mT @ omega @ s) - omega).sum()  # 验证辛形式
    assert err1 + err2 < 5e-4
