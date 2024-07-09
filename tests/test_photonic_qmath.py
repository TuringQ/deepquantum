import deepquantum.photonic as dqp
import networkx as nx
import numpy as np
import pytest
import thewalrus
import torch
from deepquantum.photonic.qmath import quadrature_to_ladder, ladder_to_quadrature, takagi, hafnian, torontonian


def test_quadrature_ladder_transform():
    nmode = 4
    vector = torch.randn(2 * nmode, 1)
    matrix = torch.randn(2 * nmode, 2 * nmode)
    vector2 = ladder_to_quadrature(quadrature_to_ladder(vector))
    matrix2 = ladder_to_quadrature(quadrature_to_ladder(matrix))
    assert torch.allclose(vector, vector2, atol=1e-5)
    assert torch.allclose(matrix, matrix2, atol=1e-5)


def test_takagi():
    size = 8
    for _ in range(10):
        graph = nx.erdos_renyi_graph(size, 0.5)
        a = torch.tensor(nx.to_numpy_array(graph), dtype=torch.float)
        u, diag = takagi(a)
        s_diag = torch.diag(diag).to(u.dtype)
        assert torch.allclose(u @ u.mH, torch.eye(size) + 0j, rtol=1e-5, atol=1e-5)
        assert torch.allclose(u @ s_diag @ u.mT, a + 0j, rtol=1e-5, atol=1e-5)

def test_hafnian():
    n = 12
    temp = np.random.rand(n, n)
    A = temp + temp.transpose()
    haf1 = thewalrus.hafnian(A, loop=False)
    haf2 = hafnian(A, if_loop=False)
    assert abs(haf1-haf2) < 1e-6

def test_hafnian_loop():
    n = 12
    temp = np.random.rand(n, n)
    A = temp + temp.transpose()
    haf1 = thewalrus.hafnian(A, loop=True)
    haf2 = hafnian(A, if_loop=True)
    assert abs(haf1-haf2) < 1e-6


def test_torontonian():
    nmode = 15
    cir = dqp.QumodeCircuit(nmode=nmode, init_state='vac', cutoff=5, backend='gaussian')
    for i in range(nmode):
        cir.s(wires=[i])
        cir.d(wires=[i])
    for i in range(nmode-1):
        cir.bs(wires=[i,i+1])
    cir.to(torch.double)

    covs, means = cir()
    cov_ladder = quadrature_to_ladder(covs[0])
    mean_ladder = quadrature_to_ladder(means[0])
    q = cov_ladder + torch.eye(2 * nmode) / 2
    o_mat = torch.eye(2 * nmode) - torch.inverse(q)
    tor1 = torontonian(o_mat)
    tor2 =thewalrus.tor(o_mat.detach().numpy())
    assert abs(tor1-tor2) < 1e-6


