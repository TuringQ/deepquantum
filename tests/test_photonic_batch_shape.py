import deepquantum as dq
import pytest
import torch


def test_gaussian_shape():
    cir = dq.QumodeCircuit(nmode=1, init_state='vac', cutoff=3, backend='gaussian')
    cir.s(0, 0., encode=True)

    data2 = torch.tensor([[0,0], [0,1]])
    state = cir()
    assert tuple(state[0].shape) == (1, 2, 2) and tuple(state[1].shape) == (1, 2, 1)
    state = cir(data=data2)
    assert tuple(state[0].shape) == (2, 2, 2) and tuple(state[1].shape) == (2, 2, 1)


def test_gaussian_batch_shape():
    batch = torch.randint(1, 10, size=[1])[0]
    covs = torch.stack([torch.eye(2)] * batch)
    means = torch.tensor([[0, 0]] * batch)
    cir = dq.QumodeCircuit(nmode=1, init_state=[covs, means], cutoff=3, backend='gaussian')
    cir.s(0, 0., encode=True)

    data2 = torch.tensor([[0,0]] * batch)
    state = cir()
    assert tuple(state[0].shape) == (batch, 2, 2) and tuple(state[1].shape) == (batch, 2, 1)
    state = cir(data=data2)
    assert tuple(state[0].shape) == (batch, 2, 2) and tuple(state[1].shape) == (batch, 2, 1)


def test_bosonic_shape():
    cir = dq.QumodeCircuit(nmode=2, init_state='vac', cutoff=3, backend='bosonic')
    cir.cat(0, r=1, theta=0.)
    cir.gkp(1, theta=0., phi=0.)
    cir.s(0, 0., encode=True)

    data2 = torch.tensor([[0,0], [0,1]])
    state = cir()
    assert (tuple(state[0].shape) == (1, 1, 4, 4) and
            tuple(state[1].shape) == (1, 356, 4, 1) and
            tuple(state[2].shape) == (1, 356))
    state = cir(data=data2)
    assert (tuple(state[0].shape) == (2, 1, 4, 4) and
            tuple(state[1].shape) == (2, 356, 4, 1) and
            tuple(state[2].shape) == (1, 356))


def test_bosonic_batch_shape():
    batch = torch.randint(1, 10, size=[1])[0]
    cat = dq.CatState(r=1., theta=0., p=1)
    cov_in = cat.cov.expand(batch, 1, 2, 2)
    mean_in = cat.mean.expand(batch, 4, 2, 1)
    weight_in = cat.weight.expand(batch, 4)
    cir = dq.QumodeCircuit(nmode=1, init_state=[cov_in, mean_in, weight_in], cutoff=3, backend='bosonic')
    cir.s(0, 0., encode=True)

    data2 = torch.tensor([[0,0]] * batch)
    state = cir()
    assert (tuple(state[0].shape) == (batch, 1, 2, 2) and
            tuple(state[1].shape) == (batch, 4, 2, 1) and
            tuple(state[2].shape) == (batch, 4))
    state = cir(data=data2)
    assert (tuple(state[0].shape) == (batch, 1, 2, 2) and
            tuple(state[1].shape) == (batch, 4, 2, 1) and
            tuple(state[2].shape) == (batch, 4))
