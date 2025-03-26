import itertools

import deepquantum as dq
import numpy as np
import strawberryfields as sf
import torch
from strawberryfields.ops import Fock, Rgate, BSgate, LossChannel


def test_loss_fock_basis_True():
    n = 3
    angles = np.random.rand(6) * np.pi
    transmittance = np.random.rand(6)
    prog = sf.Program(n)
    with prog.context as q:
        Fock(1) | q[0]
        Fock(1) | q[1]
        Fock(1) | q[2]
        LossChannel(transmittance[0]) | q[0]
        Rgate(angles[0]) | q[0]
        Rgate(angles[1]) | q[1]
        BSgate(angles[2], angles[3]) | (q[0], q[2])
        LossChannel(transmittance[1]) | q[0]
        LossChannel(transmittance[2]) | q[2]
        BSgate(angles[4], angles[5]) | (q[1], q[2])
        LossChannel(transmittance[3]) | q[0]
        LossChannel(transmittance[4]) | q[1]
        LossChannel(transmittance[5]) | q[2]
    eng = sf.Engine('fock', backend_options={'cutoff_dim': 4})
    result = eng.run(prog)

    nmode = n
    cir = dq.QumodeCircuit(nmode=nmode, init_state=[1,1,1], cutoff=4, backend='fock', basis=True)
    cir.loss_t(0, transmittance[0])
    cir.ps(0, angles[0])
    cir.ps(1, angles[1])
    cir.bs([0,2], [angles[2], angles[3]])
    cir.loss_t(0, transmittance[1])
    cir.loss_t(2, transmittance[2])
    cir.bs([1,2], [angles[4], angles[5]])
    cir.loss_t(0, transmittance[3])
    cir.loss_t(1, transmittance[4])
    cir.loss_t(2, transmittance[5])
    # cir.to(torch.float64)
    state = cir(is_prob=True)
    err = 0
    for key in state.keys():
        dq_prob = state[key]
        fock_st = key.state.tolist()
        sf_prob = result.state.fock_prob(fock_st)
        err += abs(dq_prob - sf_prob)
    assert err < 1e-6


def test_loss_fock_basis_False():
    n = 3
    angles = np.random.rand(6) * np.pi
    transmittance = np.random.rand(6)
    prog = sf.Program(n)
    with prog.context as q:
        Fock(1) | q[0]
        Fock(1) | q[1]
        Fock(1) | q[2]
        LossChannel(transmittance[0]) | q[0]
        Rgate(angles[0]) | q[0]
        Rgate(angles[1]) | q[1]
        BSgate(angles[2], angles[3]) | (q[0], q[2])
        LossChannel(transmittance[1]) | q[0]
        LossChannel(transmittance[2]) | q[2]
        BSgate(angles[4], angles[5]) | (q[1], q[2])
        LossChannel(transmittance[3]) | q[0]
        LossChannel(transmittance[4]) | q[1]
        LossChannel(transmittance[5]) | q[2]
    eng = sf.Engine('fock', backend_options={'cutoff_dim': 4})
    result = eng.run(prog)

    nmode = n
    cir = dq.QumodeCircuit(nmode=nmode, init_state=[(1, [1,1,1])], cutoff=4, backend='fock', basis=False, den_mat=True)
    cir.loss_t(0, transmittance[0])
    cir.ps(0, angles[0])
    cir.ps(1, angles[1])
    cir.bs([0,2], [angles[2], angles[3]])
    cir.loss_t(0, transmittance[1])
    cir.loss_t(2, transmittance[2])
    cir.bs([1,2], [angles[4], angles[5]])
    cir.loss_t(0, transmittance[3])
    cir.loss_t(1, transmittance[4])
    cir.loss_t(2, transmittance[5])
    cir.to(torch.float64)
    state = cir(is_prob=True)
    err = 0
    for key in itertools.product(range(4), repeat=3):
        dq_prob = state[0][key]
        sf_prob = result.state.fock_prob(key)
        err += abs(dq_prob - sf_prob)
    assert err < 1e-6


def test_loss_gaussian():
    n = 3
    angles = np.random.rand(6) * np.pi
    transmittance = np.random.rand(6)
    prog = sf.Program(n)
    with prog.context as q:
        LossChannel(transmittance[0]) | q[0]
        Rgate(angles[0]) | q[0]
        Rgate(angles[1]) | q[1]
        BSgate(angles[2], angles[3]) | (q[0], q[1])
        LossChannel(transmittance[1]) | q[0]
        LossChannel(transmittance[2]) | q[2]
        BSgate(angles[4], angles[5]) | (q[1], q[2])
        LossChannel(transmittance[3]) | q[0]
        LossChannel(transmittance[4]) | q[1]
        LossChannel(transmittance[5]) | q[2]
    eng = sf.Engine('gaussian')
    result = eng.run(prog)
    sf_cov = result.state.cov()
    sf_mean = result.state.means()

    nmode = n
    cir = dq.QumodeCircuit(nmode=nmode, init_state='vac', cutoff=4, backend='gaussian')
    cir.loss_t(0, transmittance[0])
    cir.ps(0, angles[0])
    cir.ps(1, angles[1])
    cir.bs([0,1], [angles[2], angles[3]]) # only support adjacent wires for gaussian loss channel
    cir.loss_t(0, transmittance[1])
    cir.loss_t(2, transmittance[2])
    cir.bs([1,2], [angles[4], angles[5]])
    cir.loss_t(0, transmittance[3])
    cir.loss_t(1, transmittance[4])
    cir.loss_t(2, transmittance[5])
    cir.to(torch.float64)
    state = cir()
    cov, mean = state
    err = (abs(cov[0] - sf_cov).flatten()).sum() + (abs(mean[0].flatten() - sf_mean).flatten()).sum()
    assert err < 1e-6
