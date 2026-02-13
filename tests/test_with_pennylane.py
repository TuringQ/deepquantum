import numpy as np
import pennylane as qml
import torch

import deepquantum as dq


def test_number_operator_exp():
    nmode = 4
    cutoff = 4
    sq = np.random.rand(nmode)
    bs_angles = np.random.rand(nmode - 1, 2) * 2 * np.pi

    dev = qml.device('strawberryfields.fock', wires=nmode, cutoff_dim=cutoff)

    @qml.qnode(dev)
    def circuit():
        for i in range(nmode):
            qml.Squeezing(sq[i], 0.0, wires=i)
            if i < nmode - 1:
                qml.Beamsplitter(theta=bs_angles[i][0], phi=bs_angles[i][1], wires=[i, i + 1])
        exp = qml.expval(qml.NumberOperator(0))  # number operator
        exp1 = qml.expval(qml.NumberOperator(1))
        exp2 = qml.expval(qml.NumberOperator(2))
        exp3 = qml.expval(qml.NumberOperator(3))
        return exp, exp1, exp2, exp3

    exp_qml = circuit()

    cir = dq.QumodeCircuit(nmode=nmode, backend='fock', basis=False, cutoff=cutoff, init_state='vac', den_mat=True)
    for i in range(nmode):
        cir.s(i, sq[i], encode=True)
        if i < nmode - 1:
            cir.bs(wires=[i, i + 1], inputs=bs_angles[i])
    cir()
    exp_dq, _ = cir.photon_number_mean_var()
    assert (exp_dq.flatten() - exp_qml.numpy()).sum() < 1e-6


def test_quadrature_operator_exp():
    nmode = 4
    cutoff = 4
    sq = np.random.rand(nmode)
    bs_angles = np.random.rand(nmode - 1, 2) * 2 * np.pi
    phi = np.random.rand(nmode) * 2 * np.pi

    dev = qml.device('strawberryfields.fock', wires=nmode, cutoff_dim=cutoff)

    @qml.qnode(dev)
    def circuit():
        for i in range(nmode):
            qml.Squeezing(sq[i], 0.0, wires=i)
            if i < nmode - 1:
                qml.Beamsplitter(theta=bs_angles[i][0], phi=bs_angles[i][1], wires=[i, i + 1])
        exp = []
        for i in range(nmode):
            exp.append(qml.expval(qml.QuadOperator(phi=phi[i], wires=i)))  # Quadrature X, P
            return exp

    exp_qml = circuit()

    cir = dq.QumodeCircuit(nmode=nmode, backend='fock', basis=False, cutoff=cutoff, init_state='vac', den_mat=True)
    for i in range(nmode):
        cir.s(i, sq[i], encode=True)
        if i < nmode - 1:
            cir.bs(wires=[i, i + 1], inputs=bs_angles[i])
    cir.to(torch.double)
    cir()
    exp_dq = cir.quadrature_mean(wires=list(range(nmode)), phi=list(phi))
    assert (exp_dq.flatten() - exp_qml.numpy()).sum() < 1e-6
