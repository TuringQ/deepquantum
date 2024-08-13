import deepquantum as dq
import pytest
import torch


def test_batched_fock_basis_states():
    batch = 3
    # set nphoton = max_photon_in_one_mode + 1
    nphoton = 3
    init_state = torch.randint(nphoton, (batch, 3))
    data = torch.randn(8)

    cir = dq.QumodeCircuit(nmode=3, init_state=init_state, basis=True)
    cir.ps([0], encode=True)
    cir.ps([1], encode=True)
    cir.ps([2], encode=True)
    cir.bs_theta([0, 1], encode=True)
    cir.bs_theta([1, 2], encode=True)
    cir.ps([0], encode=True)
    cir.ps([1], encode=True)
    cir.ps([2], encode=True)

    re1 = cir(data=data, state=init_state, is_prob=None)
    re2 = cir(data=data, state=init_state, is_prob=False)
    re3 = cir(data=data, state=init_state, is_prob=True)

    for i in range(init_state.shape[0]):
        res1 = cir(data=data, state=init_state[i], is_prob=None)
        res2 = cir(data=data, state=init_state[i], is_prob=False)
        res3 = cir(data=data, state=init_state[i], is_prob=True)

        # test is_prob=None
        assert torch.equal(res1, re1[i])

        for key in res2.keys():
            # test is_prob=False
            assert torch.allclose(res2[key], re2[key][i], atol=1e-6)

        for key in res3.keys():
            # test is prob = True
            assert torch.allclose(res3[key], re3[key][i], atol=1e-6)


def test_batched_fock_basis_states_and_data():
    batch = 3
    # set nphoton = max_photon_in_one_mode + 1
    nphoton = 3
    init_state = torch.randint(nphoton, (batch, 3))
    data = torch.randn(batch * 8).reshape(batch, 8)

    cir = dq.QumodeCircuit(nmode=3, init_state=init_state, basis=True)
    cir.ps([0], encode=True)
    cir.ps([1], encode=True)
    cir.ps([2], encode=True)
    cir.bs_theta([0, 1], encode=True)
    cir.bs_theta([1, 2], encode=True)
    cir.ps([0], encode=True)
    cir.ps([1], encode=True)
    cir.ps([2], encode=True)

    re1 = cir(data=data, state=init_state, is_prob=None)
    re2 = cir(data=data, state=init_state, is_prob=False)
    re3 = cir(data=data, state=init_state, is_prob=True)

    for i in range(init_state.shape[0]):
        res1 = cir(data=data[i], state=init_state[i], is_prob=None)
        res2 = cir(data=data[i], state=init_state[i], is_prob=False)
        res3 = cir(data=data[i], state=init_state[i], is_prob=True)

        # test is_prob=None
        assert torch.equal(res1, re1[i])

        for key in res2.keys():
            # test is_prob=False
            assert torch.allclose(res2[key], re2[key][i], atol=1e-6)

        for key in res3.keys():
            # test is prob = True
            assert torch.allclose(res3[key], re3[key][i], atol=1e-6)
