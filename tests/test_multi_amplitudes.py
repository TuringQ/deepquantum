import deepquantum as dq
import pytest
import torch

def test_multi_amplitudes():
    n = 10
    cir1 = dq.QubitCircuit(nqubit=n, mps=False)
    angles = 2*torch.pi*torch.rand(3)
    for i in range(n):
        cir1.h(i)
        cir1.rx(i, angles[0])
        cir1.ry(i, angles[1])
        cir1.rz(i, angles[2])
    for i in range(n-1):
        cir1.cnot(i, i+1)
    cir1()
    amps_1 = cir1.get_amplitudes(['0000000000', '1111111111', '0101010101'])

    cir2 = dq.QubitCircuit(nqubit=n, mps=True, chi=5)
    for i in range(n):
        cir2.h(i)
        cir2.rx(i, angles[0])
        cir2.ry(i, angles[1])
        cir2.rz(i, angles[2])
    for i in range(n-1):
        cir2.cnot(i, i+1)
    cir2()
    amps_2 = cir2.get_amplitudes(['0000000000', '1111111111', '0101010101'])
    assert abs(amps_1 - amps_2).max() < 1e-6
