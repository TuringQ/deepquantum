import strawberryfields as sf
from strawberryfields.ops import *
import deepquantum as dq
import numpy as np
import pytest

import torch

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
    cir.loss(0, transmittance[0])
    cir.ps(0, angles[0])
    cir.ps(1, angles[1])
    cir.bs([0,2], [angles[2], angles[3]])
    cir.loss(0, transmittance[1])
    cir.loss(2, transmittance[2])
    cir.bs([1,2], [angles[4], angles[5]])
    cir.loss(0, transmittance[3])
    cir.loss(1, transmittance[4])
    cir.loss(2, transmittance[5])
    state = cir(is_prob=True)
    err = 0
    for key in state.keys():
        dq_prob = state[key]
        fock_st = key.state.tolist()
        sf_prob = result.state.fock_prob(fock_st)
        err += abs(dq_prob - sf_prob)
    assert err < 1e-6