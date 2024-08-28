import deepquantum as dq
import numpy as np
import pytest
import strawberryfields as sf
import thewalrus
import torch
from deepquantum.photonic import quadrature_to_ladder, hafnian, torontonian
from strawberryfields.ops import Sgate, BSgate, Rgate, MeasureHomodyne, Dgate


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

def test_measure_homodyne():
    n = 3
    r1 = np.random.rand(1)
    r2 = np.random.rand(1)
    r3 = np.random.rand(1)
    bs1 = np.random.rand(2)
    bs2 = np.random.rand(2)
    meas_angle = np.random.rand(2)

    prog0 = sf.Program(n)
    with prog0.context as q:
        Sgate(r1) | q[0]
        Dgate(r2) | q[1]
        Sgate(r3) | q[1]
        BSgate(*bs1) | (q[0], q[1])
        BSgate(*bs2) | (q[1], q[2])
        MeasureHomodyne(meas_angle[0]) | q[0]
        MeasureHomodyne(meas_angle[1]) | q[1]
    eng0 = sf.Engine("gaussian")
    result0 = eng0.run(prog0)

    circ = dq.QumodeCircuit(nmode=n, init_state='vac', cutoff=3, backend='gaussian', basis=True)
    circ.s([0], r=r1)
    circ.d([1], r=r2)
    circ.s([1], r=r3)
    circ.bs([0,1], inputs=bs1)
    circ.bs([1,2], inputs=bs2)
    circ.homodyne(wires=[0], inputs=meas_angle[0])
    circ.homodyne(wires=[1], inputs=meas_angle[1])
    circ.to(torch.double)
    st = circ()
    mea = circ.measure_homodyne()
    st2 = circ.state_measured
    err = abs(st2[0] - result0.state.cov()).max() # comparing the covariance matrix after the measurement
    assert err < 1e-6

