import deepquantum as dq
import numpy as np
import perceval as pcvl
import perceval.components as comp
import pytest
from perceval.components import BS
import torch


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
    re2 = dq_gate(is_prob=False)
    # calculating the difference for two simu approach
    max_error = -1.0
    for key in re1.keys():
        key2 = list(key)
        key3 = dq.FockState(key2)
        tmp_error = abs(re2[key3] - re1[key])
        # tmp_error = abs(re2[tuple(key2)] - re1[(key)])
        if tmp_error > max_error:
            max_error = tmp_error
    assert max_error < 1e-4

def test_loss_fock_basis_True():
    n = 3
    angles = np.random.rand(6) * np.pi
    transmittance = np.random.rand(6)

    cir = pcvl.Processor("SLOS",n)
    cir.add(0, pcvl.LC(loss=1 - transmittance[0]))
    cir.add(0, pcvl.PS(phi=angles[0]))
    cir.add(1, pcvl.PS(phi=angles[1]))
    cir.add((0,1), pcvl.BS(theta=angles[2]))
    cir.add(0, pcvl.LC(loss=1 - transmittance[1]))
    cir.add(2, pcvl.LC(loss=1 - transmittance[2]))
    cir.add((1,2), pcvl.BS(theta=angles[3]))
    cir.add(0, pcvl.LC(loss=1 - transmittance[3]))
    cir.add(1, pcvl.LC(loss=1 - transmittance[4]))
    cir.add(2, pcvl.LC(loss=1 - transmittance[5]))

    cir.with_input(pcvl.BasicState([1, 1, 1]))
    cir.min_detected_photons_filter(0)

    imperfect_sampler = pcvl.algorithm.Sampler(cir)
    output = imperfect_sampler.probs()["results"]

    nmode = n
    cir = dq.QumodeCircuit(nmode=nmode, init_state=[1,1,1], backend='fock', basis=True)
    cir.loss_t(0, transmittance[0])
    cir.ps(0, angles[0])
    cir.ps(1, angles[1])
    cir.bs_rx([0,1], [angles[2]])
    cir.loss_t(0, transmittance[1])
    cir.loss_t(2, transmittance[2])
    cir.bs_rx([1,2], [angles[3]])
    cir.loss_t(0, transmittance[3])
    cir.loss_t(1, transmittance[4])
    cir.loss_t(2, transmittance[5])
    cir.to(torch.float64)
    state = cir(is_prob=True)
    for key in state.keys():
        dq_prob = state[key]
        fock_lst = key.state.tolist()
        pcvl_prob = output[pcvl.BasicState(fock_lst)]
        err = abs(dq_prob - pcvl_prob)
        assert err < 1e-4,f'key={key},dq_prob={dq_prob},pcvl_prob={pcvl_prob}'