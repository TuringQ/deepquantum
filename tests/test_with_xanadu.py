import numpy as np
import strawberryfields as sf
import thewalrus
import torch
from strawberryfields.ops import BSgate, Dgate, Fock, MeasureHomodyne, Rgate, Sgate

import deepquantum as dq
from deepquantum.photonic import hafnian, quadrature_to_ladder, torontonian


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
        cir.bs(wires=[i, i + 1])
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
        cir.bs(wires=[i, i + 1])
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
    cir.bs([0, 1], [para_theta[4], para_theta[5]])

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
    phi = np.random.rand(2)

    prog = sf.Program(n)
    with prog.context as q:
        Sgate(r1) | q[0]
        Dgate(r2) | q[1]
        Sgate(r3) | q[1]
        BSgate(*bs1) | (q[0], q[1])
        BSgate(*bs2) | (q[1], q[2])
        MeasureHomodyne(phi[0]) | q[0]
        MeasureHomodyne(phi[1]) | q[1]
    eng = sf.Engine('gaussian')
    result = eng.run(prog)

    cir = dq.QumodeCircuit(nmode=n, init_state='vac', cutoff=3, backend='gaussian')
    cir.s(0, r=r1)
    cir.d(1, r=r2)
    cir.s(1, r=r3)
    cir.bs([0, 1], inputs=bs1)
    cir.bs([1, 2], inputs=bs2)
    cir.homodyne(wires=0, phi=phi[0])
    cir.homodyne(wires=1, phi=phi[1])
    cir.to(torch.double)
    cir()
    sample = cir.measure_homodyne()
    state = cir.state_measured
    err = abs(state[0] - result.state.cov()).max()  # compare the covariance matrix after the measurement
    assert err < 1e-6


def test_non_adjacent_bs_fock():
    n = 3
    angles = np.random.rand(6)
    prog = sf.Program(n)
    with prog.context as q:
        Fock(1) | q[0]
        Fock(1) | q[1]
        Fock(1) | q[2]
        Rgate(angles[0]) | q[0]
        Rgate(angles[1]) | q[1]
        BSgate(angles[2], angles[3]) | (q[0], q[2])
        BSgate(angles[4], angles[5]) | (q[1], q[2])
    eng = sf.Engine('fock', backend_options={'cutoff_dim': 4})
    result = eng.run(prog)

    nmode = n
    cir = dq.QumodeCircuit(nmode=nmode, init_state=[1, 1, 1], cutoff=4, backend='fock', basis=True)
    cir.ps(0, angles[0])
    cir.ps(1, angles[1])
    cir.bs([0, 2], [angles[2], angles[3]])
    cir.bs([1, 2], [angles[4], angles[5]])
    state = cir(is_prob=True)
    err = 0
    for key in state.keys():
        dq_prob = state[key]
        fock_st = key.state.tolist()
        sf_prob = result.state.fock_prob(fock_st)
        err = err + abs(dq_prob - sf_prob)
    assert err < 1e-6


def test_non_adjacent_bs_gaussian():
    n = 4
    angles = np.random.rand(12)
    prog = sf.Program(n)
    with prog.context as q:
        Sgate(angles[0]) | q[0]
        Sgate(angles[1]) | q[1]
        Sgate(angles[2]) | q[2]
        Sgate(angles[3]) | q[3]
        Dgate(angles[4]) | q[0]
        Dgate(angles[5]) | q[1]
        Dgate(angles[6]) | q[2]
        Dgate(angles[7]) | q[3]
        BSgate(angles[8], angles[9]) | (q[0], q[2])
        BSgate(angles[10], angles[11]) | (q[1], q[3])
    eng = sf.Engine('gaussian')
    result = eng.run(prog)
    cov_sf = result.state.cov()
    mean_sf = result.state.means()

    nmode = n
    cir = dq.QumodeCircuit(nmode=nmode, init_state='vac', cutoff=3, backend='gaussian')
    cir.s(0, r=angles[0])
    cir.s(1, r=angles[1])
    cir.s(2, r=angles[2])
    cir.s(3, r=angles[3])
    cir.d(0, r=angles[4])
    cir.d(1, r=angles[5])
    cir.d(2, r=angles[6])
    cir.d(3, r=angles[7])

    cir.bs([0, 2], [angles[8], angles[9]])
    cir.bs([1, 3], [angles[10], angles[11]])

    state = cir()
    err = abs(state[0].squeeze() - cov_sf).sum() + abs(state[1].squeeze() - mean_sf).sum()
    assert err < 1e-5
