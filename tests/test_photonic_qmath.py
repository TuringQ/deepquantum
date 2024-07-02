import deepquantum.photonic as dqp
import networkx as nx
import numpy as np
import pytest
import torch
from deepquantum.photonic.qmath import quadrature_to_ladder, ladder_to_quadrature, takagi


def test_quadrature_ladder_transform():
    nmode = 4
    vector = torch.randn(2 * nmode, 1)
    matrix = torch.randn(2 * nmode, 2 * nmode)
    vector2 = ladder_to_quadrature(quadrature_to_ladder(vector))
    matrix2 = ladder_to_quadrature(quadrature_to_ladder(matrix))
    assert torch.allclose(vector, vector2)
    assert torch.allclose(matrix, matrix2)


def test_takagi():
    size = 8
    for _ in range(10):
        graph = nx.erdos_renyi_graph(size, 0.5)
        a = torch.tensor(nx.to_numpy_array(graph), dtype=torch.float)
        u, diag = takagi(a)
        s_diag = torch.diag(diag).to(u.dtype)
        assert torch.allclose(u @ u.mH, torch.eye(size) + 0j, rtol=1e-5, atol=1e-5)
        assert torch.allclose(u @ s_diag @ u.mT, a + 0j, rtol=1e-5, atol=1e-5)
