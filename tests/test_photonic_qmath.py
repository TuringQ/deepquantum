import networkx as nx
import numpy as np
import torch
from scipy.stats import unitary_group

import deepquantum as dq
import deepquantum.photonic as dqp
from deepquantum.photonic import ladder_to_quadrature, quadrature_to_ladder, Squeezing2, xpxp_to_xxpp, xxpp_to_xpxp


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
    err1 = abs((s @ t @ s.mT) - cov[0]).sum() # 验证分解正确性
    omega = cov.new_ones(nmode)
    omega = torch.cat([-omega, omega]).diag_embed()
    omega = omega.reshape(2, nmode, 2 * nmode).flip(0).reshape(2 * nmode, 2 * nmode) # symplectic form
    err2 = abs((s.mT @ omega @ s) - omega).sum() # 验证辛形式
    assert err1 + err2 < 5e-4
