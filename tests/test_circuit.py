import torch

import deepquantum as dq


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
    cir.bs([0, 1], [0.1, 0.2])
    cir.bs([1, 2], [0.3, 0.4])
    state1 = dq.MatrixProductState(nmode, cir()).full_tensor().reshape(-1)

    cir = dq.QumodeCircuit(nmode, init_state='zeros', cutoff=cutoff, backend='fock', basis=False)
    cir.s(0, 0.1)
    cir.s(1, 0.2)
    cir.s(2, 0.3)
    cir.bs([0, 1], [0.1, 0.2])
    cir.bs([1, 2], [0.3, 0.4])
    state2 = cir().reshape(-1)
    assert torch.allclose(state1, state2, rtol=1e-5, atol=1e-5)


def test_qubit_dist():
    data = torch.randn(10)
    cir = dq.DistributedQubitCircuit(4, reupload=True)
    cir.rxlayer(encode=True)
    cir.rylayer(encode=True)
    cir.rzlayer(encode=True)
    cir.u3layer(encode=True)
    cir.hlayer()
    cir.cnot_ring()
    cir.toffoli(0, 1, 2)
    cir.fredkin(2, 1, 0)
    cir.swap([2, 3])
    cir.rx(0, controls=[1, 2, 3], encode=True)
    cir.ry(1, controls=[0, 2, 3], encode=True)
    cir.rz(2, controls=[0, 1, 3], encode=True)
    cir.rxx([0, 1], controls=[2, 3], encode=True)
    cir.ryy([1, 2], controls=[0, 3], encode=True)
    cir.rzz([2, 3], controls=[0, 1], encode=True)
    cir.rxy([3, 0], controls=[1, 2], encode=True)
    state1 = cir(data=data).amps

    cir = dq.QubitCircuit(4, reupload=True)
    cir.rxlayer(encode=True)
    cir.rylayer(encode=True)
    cir.rzlayer(encode=True)
    cir.u3layer(encode=True)
    cir.hlayer()
    cir.cnot_ring()
    cir.toffoli(0, 1, 2)
    cir.fredkin(2, 1, 0)
    cir.swap([2, 3])
    cir.rx(0, controls=[1, 2, 3], encode=True)
    cir.ry(1, controls=[0, 2, 3], encode=True)
    cir.rz(2, controls=[0, 1, 3], encode=True)
    cir.rxx([0, 1], controls=[2, 3], encode=True)
    cir.ryy([1, 2], controls=[0, 3], encode=True)
    cir.rzz([2, 3], controls=[0, 1], encode=True)
    cir.rxy([3, 0], controls=[1, 2], encode=True)
    state2 = cir(data=data).reshape(-1)
    assert torch.allclose(state1, state2)


def test_qubit_expectation_and_differentiation_dist():
    data1 = torch.arange(10, dtype=torch.float, requires_grad=True)
    cir1 = dq.DistributedQubitCircuit(4, reupload=True)
    cir1.rxlayer(encode=True)
    cir1.rylayer(encode=True)
    cir1.rzlayer(encode=True)
    cir1.u3layer(encode=True)
    cir1.hlayer()
    cir1.cnot_ring()
    cir1.toffoli(0, 1, 2)
    cir1.fredkin(2, 1, 0)
    cir1.swap([2, 3])
    cir1.rx(0, controls=[1, 2, 3], encode=True)
    cir1.ry(1, controls=[0, 2, 3], encode=True)
    cir1.rz(2, controls=[0, 1, 3], encode=True)
    cir1.rxx([0, 1], controls=[2, 3], encode=True)
    cir1.ryy([1, 2], controls=[0, 3], encode=True)
    cir1.rzz([2, 3], controls=[0, 1], encode=True)
    cir1.rxy([3, 0], controls=[1, 2], encode=True)
    cir1.observable(0)
    cir1.observable(1, 'x')
    cir1.observable([2, 3], 'xy')
    cir1(data=data1)
    exp1 = cir1.expectation().sum()
    exp1.backward()

    data2 = torch.arange(10, dtype=torch.float, requires_grad=True)
    cir2 = dq.QubitCircuit(4, reupload=True)
    cir2.rxlayer(encode=True)
    cir2.rylayer(encode=True)
    cir2.rzlayer(encode=True)
    cir2.u3layer(encode=True)
    cir2.hlayer()
    cir2.cnot_ring()
    cir2.toffoli(0, 1, 2)
    cir2.fredkin(2, 1, 0)
    cir2.swap([2, 3])
    cir2.rx(0, controls=[1, 2, 3], encode=True)
    cir2.ry(1, controls=[0, 2, 3], encode=True)
    cir2.rz(2, controls=[0, 1, 3], encode=True)
    cir2.rxx([0, 1], controls=[2, 3], encode=True)
    cir2.ryy([1, 2], controls=[0, 3], encode=True)
    cir2.rzz([2, 3], controls=[0, 1], encode=True)
    cir2.rxy([3, 0], controls=[1, 2], encode=True)
    cir2.observable(0)
    cir2.observable(1, 'x')
    cir2.observable([2, 3], 'xy')
    cir2(data=data2)
    exp2 = cir2.expectation().sum()
    exp2.backward()

    assert torch.allclose(exp1, exp2)
    assert torch.allclose(data1.grad, data2.grad)


def test_qumode_dist():
    nmode = 5
    cutoff = 6
    shots = 10000
    data = torch.randn(20)
    key = dq.FockState([0])

    cir = dq.DistributedQumodeCircuit(nmode, [0] * nmode, cutoff)
    for i in range(nmode):
        cir.s(i, encode=True)
    for i in range(nmode - 1):
        cir.bs([i, i + 1], encode=True)
    state1 = cir(data).amps
    rst1 = cir.measure(shots=shots, with_prob=True, wires=[0])

    cir = dq.QumodeCircuit(nmode, [0] * nmode, cutoff, basis=False)
    for i in range(nmode):
        cir.s(i, encode=True)
    for i in range(nmode - 1):
        cir.bs([i, i + 1], encode=True)
    state2 = cir(data)
    rst2 = cir.measure(shots=shots, with_prob=True, wires=[0])

    assert torch.allclose(state1, state2)
    assert torch.allclose(rst1[key][1], rst2[key][1])
