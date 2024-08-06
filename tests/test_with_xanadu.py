import deepquantum as dq
import numpy as np
import pytest
import thewalrus
import torch
from deepquantum.photonic import quadrature_to_ladder, hafnian, torontonian


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
    cir = dq.QumodeCircuit(nmode=nmode, init_state='vac', cutoff=5, backend='gaussian')
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
    cir = dq.QumodeCircuit(nmode=nmode, init_state='vac', cutoff=5, backend='gaussian')
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


def test_gaussian_prob_random_circuit():
    para_r = np.random.uniform(0, 1, [1, 4])[0]
    para_theta = np.random.uniform(0, 2 * np.pi, [1, 6])[0]

    cir = dq.QumodeCircuit(nmode=2, init_state='vac', cutoff=5, backend='gaussian')
    cir.s(0, para_r[0], para_theta[0])
    cir.s(1, para_r[1], para_theta[1])
    cir.d(0, para_r[2], para_theta[2])
    cir.d(1, para_r[3], para_theta[3])
    cir.bs([0,1], [para_theta[4], para_theta[5]])

    cir.to(torch.double)
    cov, mean = cir(is_prob=False)
    state = cir(is_prob=True)

    test_prob = thewalrus.quantum.probabilities(mu=mean[0].squeeze().numpy(), cov=cov[0].numpy(), cutoff=5)
    error = []
    for i in state.keys():
        idx = i.state.tolist()
        error.append(abs(test_prob[tuple(idx)] - state[i].item()))
    assert sum(error) < 1e-10
