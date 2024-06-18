import deepquantum as dq
import numpy as np
import perceval as pcvl
import perceval.components as comp
import pytest
from perceval.components import BS


def test_random_circuit():
    """Compare with Perceval."""
    max_mode = 7
    max_device = 100
    nmode = np.random.randint(2, max_mode)
    ndevice = np.random.randint(10, max_device)
    ini_state_pre = np.random.randn(nmode)
    ini_state_pre = ini_state_pre - ini_state_pre.min()
    ini_state_pre = np.round(ini_state_pre / ini_state_pre.sum() * nmode)
    ini_state_pre[ini_state_pre < 0] = 0
    if ini_state_pre.sum() != nmode:
        ini_state_pre[ini_state_pre.argmax()] += nmode - ini_state_pre.sum()
    ini_state = [int(i) for i in ini_state_pre]
    assert np.sum(ini_state_pre) == nmode
    test_gate = pcvl.Circuit(nmode, name='test1')
    dq_gate = dq.QumodeCircuit(nmode=nmode, init_state=ini_state, name='test',
                               cutoff=sum(ini_state)+1, basis=True)
    encode = True
    for _ in range(ndevice): # take the random circuit
        j = np.random.uniform(-1, 5)
        if 4 < j < 5: # add H
            k = int(np.random.choice(np.arange(nmode - 1)))
            angle_1 = np.random.uniform(0, 2 * np.pi)
            test_gate.add([k, k+1], BS.H(angle_1))
            dq_gate.bs_h([k, k+1], angle_1)
        if 2 < j < 3: # add Rx
            k = int(np.random.choice(np.arange(nmode - 1)))
            angle_1 = np.random.uniform(0, 2 * np.pi)
            test_gate.add([k, k+1], BS.Rx(angle_1))
            dq_gate.bs_rx([k, k+1], angle_1)
        if 1 < j < 2: # add Ry
            k = int(np.random.choice(np.arange(nmode - 1)))
            angle_1 = np.random.uniform(0, 2 * np.pi)
            test_gate.add([k, k+1], BS.Ry(angle_1))
            dq_gate.bs_ry([k, k+1], angle_1)
        if 0 < j < 1:
            temp_1 = int(np.random.choice(np.arange(nmode)))
            angle_1 = np.random.uniform(0, 2 * np.pi)
            test_gate.add((temp_1), comp.PS(angle_1))
            dq_gate.ps([temp_1], angle_1, encode=encode)
        else:
            k = int(np.random.choice(np.arange(nmode - 1)))
            angle_2 = np.random.uniform(0, 2 * np.pi)
            test_gate.add((k, k+1), BS.Rx(angle_2))
            dq_gate.bs_theta([k, k+1], angle_2 / 2, encode=encode)
    backend = pcvl.BackendFactory().get_backend('Naive')
    backend.set_circuit(test_gate)
    input_state = pcvl.BasicState(ini_state)
    backend.set_input_state(input_state)
    re1 = backend.evolve()
    re2 = dq_gate()
    # calculating the difference for two simu approach
    max_error = -1.0
    for key in re1.keys():
        key2 = list(key)
        key3 = dq.FockState(key2)
        tmp_error = abs(re2[(key3)] - re1[(key)])
        # tmp_error = abs(re2[tuple(key2)] - re1[(key)])
        if tmp_error > max_error:
            max_error = tmp_error
    assert max_error < 1e-4
