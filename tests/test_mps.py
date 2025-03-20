import deepquantum as dq
import pytest
import torch
import random

def test_get_prob_mps():
    n = 10
    data = torch.randn(3 * n)
    num_bits = random.randint(1,10)
    bits_value = random.randint(0, 2**num_bits-1)
    bits = bin(bits_value)[2:].zfill(num_bits)
    wires = random.sample(range(n), k=num_bits)
    wires.sort()

    cir1 = dq.QubitCircuit(nqubit=n, mps=False)
    for i in range(n):
        cir1.h(i)
        cir1.rx(i, encode=True)
        cir1.ry(i, encode=True)
        cir1.rz(i, encode=True)
    cir1.cnot_ring()
    state = cir1(data=data).reshape(-1)
    pm_shape = list(range(n))
    for w in wires:
        pm_shape.remove(w)
    pm_shape = wires + pm_shape
    num_bits = len(wires) if wires else n
    probs = torch.abs(state) ** 2
    prob1 = probs.reshape([2] * n).permute(pm_shape).reshape([2] * len(wires) + [-1]).sum(-1).reshape(-1)[bits_value]

    cir2 = dq.QubitCircuit(nqubit=n, mps=True)
    for i in range(n):
        cir2.h(i)
        cir2.rx(i, encode=True)
        cir2.ry(i, encode=True)
        cir2.rz(i, encode=True)
    cir2.cnot_ring()
    cir2(data=data)
    prob2 = cir2.get_prob_mps(bits, wires)

    assert torch.allclose(prob1, prob2)
