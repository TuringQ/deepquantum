import deepquantum as dq
import pytest
import torch


def test_qubit_mps():
    nqubit = 3
    cir = dq.QubitCircuit(nqubit, mps=True, chi=64)
    cir.rx(0, 0.1)
    cir.cnot(0, 1)
    cir.ry(1, 0.2)
    cir.cnot(1, 2)
    state1 = dq.MatrixProductState(3, cir()).full_tensor().reshape(-1)

    cir = dq.QubitCircuit(nqubit)
    cir.rx(0, 0.1)
    cir.cnot(0, 1)
    cir.ry(1, 0.2)
    cir.cnot(1, 2)
    state2 = cir().reshape(-1)
    assert torch.allclose(state1, state2, rtol=1e-5, atol=1e-5)


def test_fock_mps():
    nmode = 3
    cutoff = 8
    cir = dq.QumodeCircuit(nmode, init_state='zeros', cutoff=cutoff, backend='fock', basis=False, mps=True, chi=64)
    cir.s(0, [0.1,0])
    cir.s(1, [0.2,0])
    cir.s(2, [0.3,0])
    cir.bs([0,1], [0.1,0.2])
    cir.bs([1,2], [0.3,0.4])
    state1 = dq.MatrixProductState(nmode, cir()).full_tensor().reshape(-1)

    cir = dq.QumodeCircuit(nmode, init_state='zeros', cutoff=cutoff, backend='fock', basis=False)
    cir.s(0, [0.1,0])
    cir.s(1, [0.2,0])
    cir.s(2, [0.3,0])
    cir.bs([0,1], [0.1,0.2])
    cir.bs([1,2], [0.3,0.4])
    state2 = cir().reshape(-1)
    assert torch.allclose(state1, state2, rtol=1e-5, atol=1e-5)
