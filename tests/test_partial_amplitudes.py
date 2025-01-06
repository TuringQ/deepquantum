import deepquantum as dq
import pytest
import torch


def test_get_amplitude():
    n = 10
    data = torch.randn(4, 3)
    bits = '0101010101'

    cir1 = dq.QubitCircuit(nqubit=n, mps=False)
    for i in range(n):
        cir1.h(i)
        cir1.rx(i, encode=True)
        cir1.ry(i, encode=True)
        cir1.rz(i, encode=True)
    cir2.cnot_ring()
    cir1(data=data)
    amp1 = cir1.get_amplitude(bits)

    cir2 = dq.QubitCircuit(nqubit=n, mps=True, chi=8)
    for i in range(n):
        cir2.h(i)
        cir2.rx(i, encode=True)
        cir2.ry(i, encode=True)
        cir2.rz(i, encode=True)
    cir2.cnot_ring()
    cir2(data=data)
    amp2 = cir2.get_amplitude(bits)
    assert abs(amp1 - amp2).max() < 1e-5
