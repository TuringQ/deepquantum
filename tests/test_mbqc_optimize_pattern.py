import deepquantum as dq
import numpy as np
import torch

def test_standardize():
    alpha = np.random.rand(100)
    results_1 = 0
    results_2 = 1

    while results_1 !=results_2:
        pattern = Pattern([0])
        for l in range(5):
            pattern.add(command.N(1+2*l))
            pattern.add(command.N(2+2*l))
            pattern.add(command.E(nodes=(0+2*l, 1+2*l)))
            pattern.add(command.E(nodes=(1+2*l, 2+2*l)))
            if l == 0:
                pattern.add(command.M(node=0+2*l, angle = alpha[50]))
                pattern.add(command.M(node=1+2*l, angle = alpha[51], s_domain= {0}))
            else:
                pattern.add(command.M(node=0+2*l, angle = alpha[2*l], s_domain = {2*l-1}))
                pattern.add(command.M(node=1+2*l, angle = alpha[2*l+1], s_domain = {(2*l-1),2*l}))
            pattern.add(command.X(node=2 +2*l, domain={0+2*l, 1+2*l}))
            pattern.add(command.Z(node=2 +2*l, domain={1+2*l}))
        pattern.standardize()
        state = pattern.simulate_pattern(backend="statevector")
        results_1 = pattern.results

        circ_mbqc = dq.MBQC(nqubit=1)
        for l in range(5):
            circ_mbqc.node(1+2*l)
            circ_mbqc.node(2+2*l)
            circ_mbqc.entanglement([0+2*l, 1+2*l])
            circ_mbqc.entanglement([1+2*l, 2+2*l])
            if l == 0:
                circ_mbqc.measurement(wires=0+2*l, angle = alpha[50]*torch.pi)
                circ_mbqc.measurement(wires=1+2*l, angle = alpha[51]*torch.pi, s_domain= [0])
            else:
                circ_mbqc.measurement(wires=0+2*l, angle = alpha[2*l]*torch.pi, s_domain = [2*l-1])
                circ_mbqc.measurement(wires=1+2*l, angle = alpha[2*l+1]*torch.pi, s_domain = [(2*l-1),2*l])
            circ_mbqc.X(wires=2 +2*l, signal_domain=[0+2*l, 1+2*l])
            circ_mbqc.Z(wires=2 +2*l, signal_domain=[1+2*l] )

        circ_mbqc.standardize()
        state2 = circ_mbqc()
        results_2 = circ_mbqc.measured_dic

    assert torch.allclose(torch.abs(torch.tensor(state.flatten(), dtype=torch.complex64)), torch.abs(state2))

from graphix import Pattern, command
from graphix.pauli import Pauli, Plane

def compare_with_graphix():
    alpha = np.random.rand(100)

    pattern = Pattern([0])
    for l in range(2):
        pattern.add(command.N(1+2*l))
        pattern.add(command.N(2+2*l))
        pattern.add(command.E(nodes=(0+2*l, 1+2*l)))
        pattern.add(command.E(nodes=(1+2*l, 2+2*l)))
        if l == 0:
            pattern.add(command.M(node=0+2*l, angle = alpha[50]))
            pattern.add(command.M(node=1+2*l, angle = alpha[51], s_domain= {0}))
        else:
            pattern.add(command.M(node=0+2*l, angle = alpha[2*l], s_domain = {2*l-1}))
            pattern.add(command.M(node=1+2*l, angle = alpha[2*l+1], s_domain = {(2*l-1),2*l}))
        pattern.add(command.X(node=2 +2*l, domain={0+2*l, 1+2*l}))
        pattern.add(command.Z(node=2 +2*l, domain={1+2*l}))
    pattern.standardize()
    pattern.print_pattern()

    circ_mbqc = dq.MBQC(nqubit=1)
    for l in range(2):
        circ_mbqc.node(1+2*l)
        circ_mbqc.node(2+2*l)
        circ_mbqc.entanglement([0+2*l, 1+2*l])
        circ_mbqc.entanglement([1+2*l, 2+2*l])
        if l == 0:
            circ_mbqc.measurement(wires=0+2*l, angle = alpha[50])
            circ_mbqc.measurement(wires=1+2*l, angle = alpha[51], s_domain= [0])
        else:
            circ_mbqc.measurement(wires=0+2*l, angle = alpha[2*l], s_domain = [2*l-1])
            circ_mbqc.measurement(wires=1+2*l, angle = alpha[2*l+1], s_domain = [(2*l-1),2*l])
        circ_mbqc.X(wires=2 +2*l, signal_domain=[0+2*l, 1+2*l])
        circ_mbqc.Z(wires=2 +2*l, signal_domain=[1+2*l] )
    circ_mbqc.standardize()
    print(circ_mbqc)
