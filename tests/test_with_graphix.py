import deepquantum as dq
import numpy as np
import torch
from graphix import Pattern, command
from graphix.pauli import Pauli, Plane

def test_random_with_graphix():
    n = np.random.randint(4, 10)
    results_1 = 0
    results_2 = 1
    while results_1 !=results_2:
        pattern = Pattern([0,1]) # initial state
        for i in range(2, n+1):
            pattern.add(command.N(i))
        for i in range(2, n+1):
            pattern.add(command.E(nodes=(0, i)))
        pattern.add(command.E(nodes=(n-1, n)))
        pattern.add(command.M(node=0, angle=1, plane=Plane.XY))
        pattern.add(command.M(node=n-1, angle=1, plane=Plane.XY))
        pattern.add(command.X(node=n, domain={0, n-1}))

        out_state = pattern.simulate_pattern(backend="statevector")
        results_1 = pattern.results

        circ_mbqc = dq.MBQC(nqubit=2)
        for i in range(2, n+1):
            circ_mbqc.node(i)
        for i in range(2, n+1):
            circ_mbqc.entanglement([0,i])
        circ_mbqc.entanglement([n-1, n])
        circ_mbqc.measurement(wires=0, angle=np.pi)
        circ_mbqc.measurement(wires=n-1, angle=np.pi)
        circ_mbqc.X(wires=n, signal_domain=[0, n-1])
        state = circ_mbqc()
        results_2  = circ_mbqc.measured_dic
    err = abs(state.flatten() - torch.tensor(out_state.flatten())).sum()
    assert err < 1e-5