import deepquantum as dq
import numpy as np
import pytest
import thewalrus
import torch


def test_random_circuit_two_approaches():
    """Compare two approaches for the photonic quantum circuit."""
    nmode = np.random.randint(2, 7)
    ndevice = np.random.randint(10, 100)
    ini_state_pre = np.random.randn(nmode)
    ini_state_pre = ini_state_pre - ini_state_pre.min()
    ini_state_pre = np.round(ini_state_pre / ini_state_pre.sum() * nmode)
    ini_state_pre[ini_state_pre < 0] = 0
    if ini_state_pre.sum() != nmode:
        ini_state_pre[ini_state_pre.argmax()] += nmode - ini_state_pre.sum()
    ini_state_1 = [int(i) for i in ini_state_pre]
    ini_state_2 = [(1, ini_state_1)]

    dq_gate_1 = dq.QumodeCircuit(nmode=nmode, init_state=ini_state_1, basis=True)
    dq_gate_2 = dq.QumodeCircuit(nmode=nmode, init_state=ini_state_2, basis=False)

    for _ in range(ndevice): # take the random circuit
        j = np.random.uniform(-2, 5)
        if j > 4:
            temp_1 = int(np.random.choice(np.arange(nmode)))
            angle_1 = np.random.uniform(0, 2 * np.pi)
            dq_gate_1.ps([temp_1], angle_1)
            dq_gate_2.ps([temp_1], angle_1)
        if 3 < j < 4:
            k = int(np.random.choice(np.arange(nmode - 1)))
            angle_1 = np.random.uniform(0, 2 * np.pi)
            dq_gate_1.bs_theta([k, k+1], angle_1)
            dq_gate_2.bs_theta([k, k+1], angle_1)
        if 2 < j < 3:
            k = int(np.random.choice(np.arange(nmode - 1)))
            angle_1 = np.random.uniform(0, 2 * np.pi)
            dq_gate_1.bs_rx([k, k+1], angle_1)
            dq_gate_2.bs_rx([k, k+1], angle_1)
        if 1 < j < 2:
            k = int(np.random.choice(np.arange(nmode - 1)))
            angle_1 = np.random.uniform(0, 2 * np.pi)
            dq_gate_1.bs_ry([k, k+1], angle_1)
            dq_gate_2.bs_ry([k, k+1], angle_1)
        if 0 < j < 1:
            k = int(np.random.choice(np.arange(nmode - 1)))
            angle_1 = np.random.uniform(0, 2 * np.pi)
            dq_gate_1.bs_h([k, k+1], angle_1)
            dq_gate_2.bs_h([k, k+1], angle_1)
        if j < 0:
            k = int(np.random.choice(np.arange(nmode - 1)))
            angle_1 = np.random.uniform(0, 2 * np.pi, 2)
            dq_gate_1.mzi([k, k+1], angle_1)
            dq_gate_2.mzi([k, k+1], angle_1)
    re1 = dq_gate_1()
    re2 = dq_gate_2()
    max_error = -1.0
    for key, value in re1.items():
        temp = key.state.tolist()
        tmp_error = abs(re2[0][tuple(temp)] - value)
        if tmp_error > max_error:
            max_error = tmp_error
    assert max_error < 1e-4


def test_gaussian_prob_random_circuit():
    para_r = np.random.uniform(0, 1, [1, 4])[0]
    para_theta = np.random.uniform(0, 2 * np.pi, [1, 6])[0]

    cir = dq.QumodeCircuit(nmode=2, init_state='vac', cutoff=5, backend='gaussian')
    cir.s(wires=0, inputs=[para_r[0], para_theta[0]])
    cir.s(wires=1, inputs=[para_r[1], para_theta[1]])
    cir.d(wires=0, inputs=[para_r[2], para_theta[2]])
    cir.d(wires=1, inputs=[para_r[3], para_theta[3]])
    cir.bs(wires=[0,1], inputs=[para_theta[4], para_theta[5]])

    cir.to(torch.double)
    cov, mean = cir(is_prob=False)
    state = cir(is_prob=True)

    test_prob = thewalrus.quantum.probabilities(mu=mean[0].squeeze().numpy(), cov=cov[0].numpy(), cutoff=5)
    error = []
    for i in state.keys():
        idx = i.state.tolist()
        error.append(abs(test_prob[tuple(idx)] - state[i].item()))
    assert sum(error) < 1e-10
