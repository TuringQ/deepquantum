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

        circ_mbqc = dq.Pattern(nodes_state=[0,1])
        for i in range(2, n+1):
            circ_mbqc.n(i)
        for i in range(2, n+1):
            circ_mbqc.e([0,i])
        circ_mbqc.e([n-1, n])
        circ_mbqc.m(node=0, angle=np.pi)
        circ_mbqc.m(node=1, angle=np.pi, s_domain=[0])
        circ_mbqc.m(node=n-1, angle=np.pi, s_domain=[0], t_domain=[1])
        circ_mbqc.c_x(node=n, domain=[0, 1, n-1])
        state = circ_mbqc()
        results_2  = circ_mbqc.state.measure_dict
    assert torch.allclose(torch.abs(torch.tensor(out_state.flatten(), dtype=torch.complex64)), torch.abs(state.flatten()), atol=1e-6)

def test_batch_init_state():
    n = np.random.randint(4, 10)
    results_1 = 0
    results_2 = 1

    circ_mbqc = dq.Pattern(nodes_state=[0,1], state=torch.tensor([[1.,0.,0.,0.],
                                                                [0.5,0.5,0.5,0.5],
                                                                [0.,0.,1.,0.]]))
    for i in range(2, n+1):
        circ_mbqc.n(i)
    for i in range(2, n+1):
        circ_mbqc.e([0,i])
    circ_mbqc.e([n-1, n])
    circ_mbqc.m(node=0, angle=np.pi)
    circ_mbqc.m(node=1, angle=np.pi, s_domain=[0])
    circ_mbqc.m(node=n-1, angle=np.pi, s_domain=[0], t_domain=[1])
    circ_mbqc.c_x(node=n, domain=[0, 1, n-1])
    state = circ_mbqc()
    results  = circ_mbqc.state.measure_dict
    results_batch = []
    for i in range(3):
        rst = {}
        for key in results.keys():
            rst[key] = [results[key][i]]
        results_batch.append(rst)

    while results_1 !=results_batch[0]:
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
        out_state = pattern_.simulate_pattern(backend="statevector", input_state=[1,0,0,0])
        results_1 = pattern_.results
    assert torch.allclose(torch.abs(torch.tensor(out_state.flatten(), dtype=torch.complex64)), torch.abs(state[0].flatten()), atol=1e-6)

    while results_1 !=results_batch[1]:
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
        out_state = pattern_.simulate_pattern(backend="statevector", input_state=[0.5,0.5,0.5,0.5])
        results_1 = pattern_.results
    assert torch.allclose(torch.abs(torch.tensor(out_state.flatten(), dtype=torch.complex64)), torch.abs(state[1].flatten()), atol=1e-6)

    while results_1 !=results_batch[2]:
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
        out_state = pattern_.simulate_pattern(backend="statevector", input_state=[0,0,1,0])
        results_1 = pattern_.results
    assert torch.allclose(torch.abs(torch.tensor(out_state.flatten(), dtype=torch.complex64)), torch.abs(state[2].flatten()), atol=1e-6)