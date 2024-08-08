import deepquantum as dq
import pytest
import torch

batch = 3
# set nphoton = max_photon_in_one_mode + 1
nphoton = 3

def test_batch_init():
    init_state1 = torch.randint(nphoton,(batch,3))

    circ1 = dq.QumodeCircuit(nmode=3, init_state=init_state1, basis=True)
    circ1.ps([0])
    circ1.ps([1])
    circ1.ps([2])
    circ1.bs_theta([0, 1])
    circ1.bs_theta([1, 2])
    circ1.ps([0])
    circ1.ps([1])
    circ1.ps([2])

    data = torch.randn(8)

    re1 = circ1(data=data, state=init_state1, is_prob=None)
    re2 = circ1(data=data, state=init_state1, is_prob=False)
    re3 = circ1(data=data, state=init_state1, is_prob=True)

    for i in range(init_state1.shape[0]):

        res1 = circ1(data=data, state=init_state1[i], is_prob=None)
        res2 = circ1(data=data, state=init_state1[i], is_prob=False)
        res3 = circ1(data=data, state=init_state1[i], is_prob=True)

        # test is_prob=None
        assert torch.equal(res1,re1[i])

        for key in res2.keys():
            # test is_prob=False
            assert torch.allclose(res2[key], re2[key][i], atol=1e-5)

        for key in res3.keys():
            # test is prob = True
            assert torch.allclose(res3[key], re3[key][i], atol=1e-5)

def test_batch_init_data():
    init_state1 = torch.randint(nphoton,(batch,3))

    circ1 = dq.QumodeCircuit(nmode=3, init_state=init_state1, basis=True)
    circ1.ps([0])
    circ1.ps([1])
    circ1.ps([2])
    circ1.bs_theta([0, 1])
    circ1.bs_theta([1, 2])
    circ1.ps([0])
    circ1.ps([1])
    circ1.ps([2])

    data = torch.randn(8*batch).reshape(batch,8)

    re1 = circ1(data=data, state=init_state1, is_prob=None)
    re2 = circ1(data=data, state=init_state1, is_prob=False)
    re3 = circ1(data=data, state=init_state1, is_prob=True)

    for i in range(init_state1.shape[0]):

        res1 = circ1(data=data[i], state=init_state1[i], is_prob=None)
        res2 = circ1(data=data[i], state=init_state1[i], is_prob=False)
        res3 = circ1(data=data[i], state=init_state1[i], is_prob=True)

        # test is_prob=None
        assert torch.equal(res1,re1[i])

        for key in res2.keys():
            # test is_prob=False
            assert torch.allclose(res2[key], re2[key][i], atol=1e-5)

        for key in res3.keys():
            # test is prob = True
            assert torch.allclose(res3[key], re3[key][i], atol=1e-5)