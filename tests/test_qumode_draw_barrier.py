import deepquantum as dq
import pytest
import torch
import numpy as np
import sys
import sys

def test_fock():
    nmode = np.random.randint(2, 11)
    init_state = [1] + [0]*(nmode-1)
    cir1 = dq.QumodeCircuit(nmode=nmode, backend='fock', init_state=init_state, basis=True)
    for i in range(nmode-1):
        cir1.bs([i, i+1], encode=True)
        cir1.ps(i, encode=True)
        cir1.barrier()
    data = torch.rand(2, cir1.ndata)
    state1 = cir1(data=data, is_prob=False)
    cir2 = dq.QumodeCircuit(nmode=nmode, backend='fock', init_state=init_state, basis=True)
    for i in range(nmode-1):
        cir2.bs([i, i+1], encode=True)
        cir2.ps(i, encode=True)
    state2 = cir2(data=data, is_prob=False)
    err = []
    for s in state1:
        err.append(abs(state1[s] - state2[s]))
    assert sum(torch.stack(err).flatten()) < 1e-6

    nmode = np.random.randint(2, 11)
    init_state = [1] + [0]*(nmode-1)
    cir1 = dq.QumodeCircuit(nmode=nmode, backend='fock', init_state=init_state, basis=False)
    for i in range(nmode-1):
        cir1.bs([i, i+1], encode=True)
        cir1.ps(i, encode=True)
        cir1.barrier()
    data = torch.rand(2, cir1.ndata)
    state1 = cir1(data=data, is_prob=False)
    cir2 = dq.QumodeCircuit(nmode=nmode, backend='fock', init_state=init_state, basis=False)
    for i in range(nmode-1):
        cir2.bs([i, i+1], encode=True)
        cir2.ps(i, encode=True)
    state2 = cir2(data=data, is_prob=False)
    assert (abs(state1-state2)).sum() < 1e-6

def test_gaussian():
    nmode = np.random.randint(2, 11)
    cir1 = dq.QumodeCircuit(nmode=nmode, backend='gaussian', init_state='vac', basis=True, cutoff=2)
    for i in range(nmode-1):
        cir1.s(i, encode=True)
        cir1.bs([i, i+1], encode=True)
        cir1.ps(i, encode=True)
        cir1.barrier()
    data = torch.rand(2, cir1.ndata)
    cov1, mean1 = cir1(data=data)
    p1 = cir1(data=data, is_prob=True)
    cir2 = dq.QumodeCircuit(nmode=nmode, backend='gaussian', init_state='vac', basis=True, cutoff=2)
    for i in range(nmode-1):
        cir2.s(i, encode=True)
        cir2.bs([i, i+1], encode=True)
        cir2.ps(i, encode=True)
    cov2, mean2 = cir2(data=data)
    p2 = cir2(data=data, is_prob=True)
    err1 = abs(cov1-cov2).sum()
    err2 = [ ]
    for s in p1:
        err2.append(abs(p1[s]-p2[s]))
    assert err1 + sum(torch.stack(err2).flatten()) < 1e-6






