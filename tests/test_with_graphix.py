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
        pattern_ = Pattern([0,1]) # initial state
        for i in range(2, n+1):
            pattern_.add(command.N(i))
        for i in range(2, n+1):
            pattern_.add(command.E(nodes=(0, i)))
        pattern_.add(command.E(nodes=(n-1, n)))
        pattern_.add(command.M(node=0, angle=1, plane=Plane.XY))
        pattern_.add(command.M(node=1, angle=1, plane=Plane.XY, s_domain={0}))
        pattern_.add(command.M(node=n-1, angle=1, plane=Plane.XY, s_domain={0}, t_domain={1}))
        pattern_.add(command.X(node=n, domain={0,1, n-1}))
        out_state = pattern_.simulate_pattern(backend="statevector")
        results_1 = pattern_.results

        circ_mbqc = dq.Pattern(n_input_nodes=2)
        for i in range(2, n+1):
            circ_mbqc.n(i)
        for i in range(2, n+1):
            circ_mbqc.e([0,i])
        circ_mbqc.e([n-1, n])
        circ_mbqc.m(node=0, angle=np.pi)
        circ_mbqc.m(node=1, angle=np.pi, s_domain=[0])
        circ_mbqc.m(node=n-1, angle=np.pi, s_domain=[0], t_domain=[1])
        circ_mbqc.x(node=n, signal_domain=[0, 1, n-1])
        state = circ_mbqc()
        results_2  = circ_mbqc.measured_dic
    err = abs(state.flatten() - torch.tensor(out_state.flatten())).sum()
    assert err < 1e-5