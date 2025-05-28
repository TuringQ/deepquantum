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
    cir.s(0, 0.1)
    cir.s(1, 0.2)
    cir.s(2, 0.3)
    cir.bs([0,1], [0.1,0.2])
    cir.bs([1,2], [0.3,0.4])
    state1 = dq.MatrixProductState(nmode, cir()).full_tensor().reshape(-1)

    cir = dq.QumodeCircuit(nmode, init_state='zeros', cutoff=cutoff, backend='fock', basis=False)
    cir.s(0, 0.1)
    cir.s(1, 0.2)
    cir.s(2, 0.3)
    cir.bs([0,1], [0.1,0.2])
    cir.bs([1,2], [0.3,0.4])
    state2 = cir().reshape(-1)
    assert torch.allclose(state1, state2, rtol=1e-5, atol=1e-5)


def test_qubit_dist():
    data = torch.randn(10)
    cir = dq.DistritubutedQubitCircuit(4, reupload=True)
    cir.rxlayer(encode=True)
    cir.rylayer(encode=True)
    cir.rzlayer(encode=True)
    cir.hlayer()
    cir.cnot_ring()
    cir.toffoli(0,1,2)
    cir.fredkin(2,1,0)
    cir.rx(0, controls=[1,2,3], encode=True)
    cir.ry(1, controls=[0,2,3], encode=True)
    cir.rz(2, controls=[0,1,3], encode=True)
    cir.rxx([0,1], controls=[2,3], encode=True)
    cir.ryy([1,2], controls=[0,3], encode=True)
    cir.rzz([2,3], controls=[0,1], encode=True)
    cir.rxy([3,0], controls=[1,2], encode=True)
    state1 = cir(data=data).amps

    cir = dq.QubitCircuit(4, reupload=True)
    cir.rxlayer(encode=True)
    cir.rylayer(encode=True)
    cir.rzlayer(encode=True)
    cir.hlayer()
    cir.cnot_ring()
    cir.toffoli(0,1,2)
    cir.fredkin(2,1,0)
    cir.rx(0, controls=[1,2,3], encode=True)
    cir.ry(1, controls=[0,2,3], encode=True)
    cir.rz(2, controls=[0,1,3], encode=True)
    cir.rxx([0,1], controls=[2,3], encode=True)
    cir.ryy([1,2], controls=[0,3], encode=True)
    cir.rzz([2,3], controls=[0,1], encode=True)
    cir.rxy([3,0], controls=[1,2], encode=True)
    state2 = cir(data=data).reshape(-1)
    assert torch.allclose(state1, state2)
