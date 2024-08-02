import deepquantum as dq
import numpy as np
import pytest


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
    re1 = dq_gate_1(is_prob=False)
    re2 = dq_gate_2()
    max_error = -1.0
    for key, value in re1.items():
        temp = key.state.tolist()
        tmp_error = abs(re2[0][tuple(temp)] - value)
        if tmp_error > max_error:
            max_error = tmp_error
    assert max_error < 1e-4
