import deepquantum.photonic as dqp
import numpy as np
import pytest
import thewalrus
import torch
from deepquantum.photonic.qmath import quadrature_to_ladder
from deepquantum.photonic.hafnian import hafnian
from deepquantum.photonic.torontonian import torontonian


def test_hafnian():
    n = 12
    temp = np.random.rand(n, n)
    mat = temp + temp.transpose()
    haf1 = thewalrus.hafnian(mat, loop=False)
    mat = torch.tensor(mat)
    haf2 = hafnian(mat, loop=False)
    assert abs(haf1 - haf2) < 1e-6


def test_hafnian_loop():
    n = 12
    temp = np.random.rand(n, n)
    mat = temp + temp.transpose()
    haf1 = thewalrus.hafnian(mat, loop=True)
    mat = torch.tensor(mat)
    haf2 = hafnian(mat, loop=True)
    assert abs(haf1 - haf2) < 1e-6


def test_torontonian():
    nmode = 15
    cir = dqp.QumodeCircuit(nmode=nmode, init_state='vac', cutoff=5, backend='gaussian')
    for i in range(nmode):
        cir.s(wires=i)
        cir.d(wires=i)
    for i in range(nmode - 1):
        cir.bs(wires=[i,i+1])
    cir.to(torch.double)

    covs, means = cir()
    cov_ladder = quadrature_to_ladder(covs[0])
    mean_ladder = quadrature_to_ladder(means[0])
    q = cov_ladder + torch.eye(2 * nmode) / 2
    o_mat = torch.eye(2 * nmode) - torch.inverse(q)
    tor1 = torontonian(o_mat)
    tor2 = thewalrus.tor(o_mat.detach().numpy())
    assert abs(tor1 - tor2) < 1e-6


def test_torontonian_loop():
    nmode = 10
    cir = dqp.QumodeCircuit(nmode=nmode, init_state='vac', cutoff=5, backend='gaussian')
    for i in range(nmode):
        cir.s(wires=i)
        cir.d(wires=i)
    for i in range(nmode - 1):
        cir.bs(wires=[i,i+1])
    cir.to(torch.double)

    covs, means = cir()
    cov_ladder = quadrature_to_ladder(covs[0])
    mean_ladder = quadrature_to_ladder(means[0])
    q = cov_ladder + torch.eye(2 * nmode) / 2
    gamma = mean_ladder.conj().mT @ torch.inverse(q)
    gamma = gamma.squeeze()
    o_mat = torch.eye(2 * nmode) - torch.inverse(q)
    tor1 = torontonian(o_mat, gamma)
    tor2 = thewalrus.ltor(o_mat.detach().numpy(), gamma.detach().numpy())
    assert abs(tor1 - tor2) < 1e-6
