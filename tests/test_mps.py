import random

import torch

import deepquantum as dq
from deepquantum.qmath import get_prob_mps, slice_state_vector


def test_cir_get_prob():
    n = 10
    data = torch.randn(3 * n)
    num_bits = random.randint(1, n)
    bits_value = random.randint(0, 2**num_bits - 1)
    bits = bin(bits_value)[2:].zfill(num_bits)
    wires = random.sample(range(n), k=num_bits)
    wires.sort()

    cir1 = dq.QubitCircuit(nqubit=n, mps=False)
    cir1.hlayer()
    cir1.rxlayer(encode=True)
    cir1.rylayer(encode=True)
    cir1.rzlayer(encode=True)
    cir1.cnot_ring()
    cir1(data=data)
    prob1 = cir1.get_prob(bits, wires)

    cir2 = dq.QubitCircuit(nqubit=n, mps=True)
    cir2.hlayer()
    cir2.rxlayer(encode=True)
    cir2.rylayer(encode=True)
    cir2.rzlayer(encode=True)
    cir2.cnot_ring()
    cir2(data=data)
    prob2 = cir2.get_prob(bits, wires)

    assert torch.allclose(prob1, prob2)


def test_get_prob_mps():
    n = 10
    data = torch.randn(2 * n)
    num_bits = random.randint(1, n)
    bits_value = random.randint(0, 2**num_bits - 1)
    bits = bin(bits_value)[2:].zfill(num_bits)
    wires = random.sample(range(n), k=num_bits)
    wires.sort()

    cir = dq.QubitCircuit(n, mps=True)
    cir.rylayer(encode=True)
    cir.cnot_ring()
    cir.rxlayer(encode=True)
    mps = cir(data=data)

    cir2 = dq.QubitCircuit(n)
    cir2.rylayer(encode=True)
    cir2.cnot_ring()
    cir2.rxlayer(encode=True)
    sv = cir2(data=data).reshape([2] * n)

    for offset, i, b in enumerate(zip(wires, bits, strict=True)):
        prob0_sv = (slice_state_vector(sv, n - offset, [i - offset], '0', False).abs() ** 2).sum()
        prob1_sv = (slice_state_vector(sv, n - offset, [i - offset], '1', False).abs() ** 2).sum()
        probs_mps = get_prob_mps(mps, i)
        assert torch.allclose(prob0_sv, probs_mps[0])
        assert torch.allclose(prob1_sv, probs_mps[1])
        sv = slice_state_vector(sv, n - offset, [i - offset], b, False)
        mps[i] = mps[i][:, [int(b)], :]
