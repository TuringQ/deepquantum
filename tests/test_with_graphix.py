import numpy as np
import torch
from graphix import Pattern, command
from graphix.fundamentals import Plane

import deepquantum as dq


def test_random_with_graphix():
    n = np.random.randint(4, 10)
    results_1 = 0
    results_2 = 1
    while results_1 != results_2:
        pat_gx = Pattern([0, 1])  # initial state
        for i in range(2, n + 1):
            pat_gx.add(command.N(i))
        for i in range(2, n + 1):
            pat_gx.add(command.E(nodes=(0, i)))
        pat_gx.add(command.E(nodes=(n - 1, n)))
        pat_gx.add(command.M(node=0, angle=1, plane=Plane.XY))
        pat_gx.add(command.M(node=1, angle=1, plane=Plane.XY, s_domain={0}))
        pat_gx.add(command.M(node=n - 1, angle=1, plane=Plane.XY, s_domain={0}, t_domain={1}))
        pat_gx.add(command.X(node=n, domain={0, 1, n - 1}))
        out_state = pat_gx.simulate_pattern(backend='statevector')
        results_1 = pat_gx.results

        pat_dq = dq.Pattern(nodes_state=[0, 1])
        for i in range(2, n + 1):
            pat_dq.n(i)
        for i in range(2, n + 1):
            pat_dq.e(0, i)
        pat_dq.e(n - 1, n)
        pat_dq.m(node=0, angle=np.pi)
        pat_dq.m(node=1, angle=np.pi, s_domain=[0])
        pat_dq.m(node=n - 1, angle=np.pi, s_domain=[0], t_domain=[1])
        pat_dq.x(node=n, domain=[0, 1, n - 1])
        state = pat_dq().full_state
        results_2 = pat_dq.state.measure_dict
    assert torch.allclose(
        torch.abs(torch.tensor(out_state.flatten(), dtype=torch.complex64)), torch.abs(state.flatten()), atol=1e-6
    )


def test_batch_init_state():
    n = np.random.randint(4, 10)
    init_state = [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 1.0, 0.0]]
    pat_dq = dq.Pattern(nodes_state=[0, 1], state=torch.tensor(init_state))
    for i in range(2, n + 1):
        pat_dq.n(i)
    for i in range(2, n + 1):
        pat_dq.e(0, i)
    pat_dq.e(n - 1, n)
    pat_dq.m(node=0, angle=np.pi)
    pat_dq.m(node=1, angle=np.pi, s_domain=[0])
    pat_dq.m(node=n - 1, angle=np.pi, s_domain=[0], t_domain=[1])
    pat_dq.x(node=n, domain=[0, 1, n - 1])
    state = pat_dq().full_state
    results = pat_dq.state.measure_dict
    for i in range(3):
        rst = {}
        for key in results:
            rst[key] = [results[key][i]]

        results_1 = 0
        while results_1 != rst:
            pat_gx = Pattern([0, 1])  # initial state
            for j in range(2, n + 1):
                pat_gx.add(command.N(j))
            for j in range(2, n + 1):
                pat_gx.add(command.E(nodes=(0, j)))
            pat_gx.add(command.E(nodes=(n - 1, n)))
            pat_gx.add(command.M(node=0, angle=1, plane=Plane.XY))
            pat_gx.add(command.M(node=1, angle=1, plane=Plane.XY, s_domain={0}))
            pat_gx.add(command.M(node=n - 1, angle=1, plane=Plane.XY, s_domain={0}, t_domain={1}))
            pat_gx.add(command.X(node=n, domain={0, 1, n - 1}))
            out_state = pat_gx.simulate_pattern(backend='statevector', input_state=init_state[i])
            results_1 = pat_gx.results
            print(out_state.flatten())
            print(state[0].flatten())
        assert torch.allclose(
            torch.abs(torch.tensor(out_state.flatten(), dtype=torch.complex64)),
            torch.abs(state[i].flatten()),
            atol=1e-6,
        )


def test_standardize():
    alpha = np.random.rand(100)
    results_1 = 0
    results_2 = 1

    while results_1 != results_2:
        pat_gx = Pattern([0])
        for i in range(3):
            pat_gx.add(command.N(1 + 2 * i))
            pat_gx.add(command.N(2 + 2 * i))
            pat_gx.add(command.E(nodes=(0 + 2 * i, 1 + 2 * i)))
            pat_gx.add(command.E(nodes=(1 + 2 * i, 2 + 2 * i)))
            if i == 0:
                pat_gx.add(command.M(node=0 + 2 * i, angle=alpha[50]))
                pat_gx.add(command.M(node=1 + 2 * i, angle=alpha[51], s_domain={0}))
            else:
                pat_gx.add(command.M(node=0 + 2 * i, angle=alpha[2 * i], s_domain={2 * i - 1}))
                pat_gx.add(command.M(node=1 + 2 * i, angle=alpha[2 * i + 1], s_domain={(2 * i - 1), 2 * i}))
            pat_gx.add(command.X(node=2 + 2 * i, domain={0 + 2 * i, 1 + 2 * i}))
            pat_gx.add(command.Z(node=2 + 2 * i, domain={1 + 2 * i}))
        pat_gx.standardize()
        state = pat_gx.simulate_pattern(backend='statevector')
        results_1 = pat_gx.results

        pat_dq = dq.Pattern(nodes_state=[0])
        for i in range(3):
            pat_dq.n(1 + 2 * i)
            pat_dq.n(2 + 2 * i)
            pat_dq.e(0 + 2 * i, 1 + 2 * i)
            pat_dq.e(1 + 2 * i, 2 + 2 * i)
            if i == 0:
                pat_dq.m(node=0 + 2 * i, angle=alpha[50] * torch.pi)
                pat_dq.m(node=1 + 2 * i, angle=alpha[51] * torch.pi, s_domain=[0])
            else:
                pat_dq.m(node=0 + 2 * i, angle=alpha[2 * i] * torch.pi, s_domain=[2 * i - 1])
                pat_dq.m(node=1 + 2 * i, angle=alpha[2 * i + 1] * torch.pi, s_domain=[2 * i - 1, 2 * i])
            pat_dq.x(node=2 + 2 * i, domain=[0 + 2 * i, 1 + 2 * i])
            pat_dq.z(node=2 + 2 * i, domain=[1 + 2 * i])
        pat_dq.standardize()
        state2 = pat_dq().full_state
        results_2 = pat_dq.state.measure_dict
    assert torch.allclose(torch.abs(torch.tensor(state.flatten(), dtype=torch.complex64)), torch.abs(state2.flatten()))


def test_signal_shifting():
    alpha = np.random.rand(100)
    results_1 = 0
    results_2 = 1

    while results_1 != results_2:
        pat_gx = Pattern([0])
        for i in range(3):
            pat_gx.add(command.N(1 + 2 * i))
            pat_gx.add(command.N(2 + 2 * i))
            pat_gx.add(command.E(nodes=(0 + 2 * i, 1 + 2 * i)))
            pat_gx.add(command.E(nodes=(1 + 2 * i, 2 + 2 * i)))
            if i == 0:
                pat_gx.add(command.M(node=0 + 2 * i, angle=alpha[50]))
                pat_gx.add(command.M(node=1 + 2 * i, angle=alpha[51], s_domain={0}, t_domain={0}))
            else:
                pat_gx.add(command.M(node=0 + 2 * i, angle=alpha[2 * i], s_domain={2 * i - 1}, t_domain={2 * i - 1}))
                pat_gx.add(command.M(node=1 + 2 * i, angle=alpha[2 * i + 1], s_domain={(2 * i - 1), 2 * i}))
            pat_gx.add(command.X(node=2 + 2 * i, domain={0 + 2 * i, 1 + 2 * i}))
            pat_gx.add(command.Z(node=2 + 2 * i, domain={1 + 2 * i}))
        pat_gx.standardize()
        pat_gx.shift_signals()
        state = pat_gx.simulate_pattern(backend='statevector')
        results_1 = pat_gx.results

        pat_dq = dq.Pattern(nodes_state=[0])
        for i in range(3):
            pat_dq.n(1 + 2 * i)
            pat_dq.n(2 + 2 * i)
            pat_dq.e(0 + 2 * i, 1 + 2 * i)
            pat_dq.e(1 + 2 * i, 2 + 2 * i)
            if i == 0:
                pat_dq.m(node=0 + 2 * i, angle=alpha[50] * torch.pi)
                pat_dq.m(node=1 + 2 * i, angle=alpha[51] * torch.pi, s_domain=[0], t_domain=[0])
            else:
                pat_dq.m(node=0 + 2 * i, angle=alpha[2 * i] * torch.pi, s_domain=[2 * i - 1], t_domain=[2 * i - 1])
                pat_dq.m(node=1 + 2 * i, angle=alpha[2 * i + 1] * torch.pi, s_domain=[2 * i - 1, 2 * i])
            pat_dq.x(node=2 + 2 * i, domain=[0 + 2 * i, 1 + 2 * i])
            pat_dq.z(node=2 + 2 * i, domain=[1 + 2 * i])
        pat_dq.standardize()
        pat_dq.shift_signals()
        state2 = pat_dq().full_state
        results_2 = pat_dq.state.measure_dict
    assert torch.allclose(torch.abs(torch.tensor(state.flatten(), dtype=torch.complex64)), torch.abs(state2.flatten()))


def test_signal_shifting_plane_yz():
    alpha = np.random.rand(100)
    results_1 = 0
    results_2 = 1

    while results_1 != results_2:
        pat_gx = Pattern([0])
        for i in range(3):
            pat_gx.add(command.N(1 + 2 * i))
            pat_gx.add(command.N(2 + 2 * i))
            pat_gx.add(command.E(nodes=(0 + 2 * i, 1 + 2 * i)))
            pat_gx.add(command.E(nodes=(1 + 2 * i, 2 + 2 * i)))
            if i == 0:
                pat_gx.add(command.M(node=0 + 2 * i, angle=alpha[50]))
                pat_gx.add(command.M(node=1 + 2 * i, angle=alpha[51], s_domain={0}, t_domain={0}))
            else:
                pat_gx.add(
                    command.M(
                        node=0 + 2 * i, angle=alpha[2 * i], s_domain={2 * i - 1}, t_domain={2 * i - 1}, plane=Plane.YZ
                    )
                )
                pat_gx.add(command.M(node=1 + 2 * i, angle=alpha[2 * i + 1], s_domain={(2 * i - 1), 2 * i}))
            pat_gx.add(command.X(node=2 + 2 * i, domain={0 + 2 * i, 1 + 2 * i}))
            pat_gx.add(command.Z(node=2 + 2 * i, domain={1 + 2 * i}))
        pat_gx.standardize()
        pat_gx.shift_signals()
        state = pat_gx.simulate_pattern(backend='statevector')
        results_1 = pat_gx.results

        pat_dq = dq.Pattern(nodes_state=[0])
        for i in range(3):
            pat_dq.n(1 + 2 * i)
            pat_dq.n(2 + 2 * i)
            pat_dq.e(0 + 2 * i, 1 + 2 * i)
            pat_dq.e(1 + 2 * i, 2 + 2 * i)
            if i == 0:
                pat_dq.m(node=0 + 2 * i, angle=alpha[50] * torch.pi)
                pat_dq.m(node=1 + 2 * i, angle=alpha[51] * torch.pi, s_domain=[0], t_domain=[0])
            else:
                pat_dq.m(
                    node=0 + 2 * i,
                    angle=torch.pi / 2 - alpha[2 * i] * torch.pi,
                    plane='yz',
                    s_domain=[2 * i - 1],
                    t_domain=[2 * i - 1],
                )
                pat_dq.m(node=1 + 2 * i, angle=alpha[2 * i + 1] * torch.pi, s_domain=[2 * i - 1, 2 * i])
            pat_dq.x(node=2 + 2 * i, domain=[0 + 2 * i, 1 + 2 * i])
            pat_dq.z(node=2 + 2 * i, domain=[1 + 2 * i])
        pat_dq.standardize()
        pat_dq.shift_signals()
        state2 = pat_dq().full_state
        results_2 = pat_dq.state.measure_dict
    assert torch.allclose(torch.abs(torch.tensor(state.flatten(), dtype=torch.complex64)), torch.abs(state2.flatten()))


def test_signal_shifting_plane_xz():
    alpha = np.random.rand(100)
    results_1 = 0
    results_2 = 1

    while results_1 != results_2:
        pat_gx = Pattern([0])
        for i in range(3):
            pat_gx.add(command.N(1 + 2 * i))
            pat_gx.add(command.N(2 + 2 * i))
            pat_gx.add(command.E(nodes=(0 + 2 * i, 1 + 2 * i)))
            pat_gx.add(command.E(nodes=(1 + 2 * i, 2 + 2 * i)))
            if i == 0:
                pat_gx.add(command.M(node=0 + 2 * i, angle=alpha[50]))
                pat_gx.add(command.M(node=1 + 2 * i, angle=alpha[51], s_domain={0}, t_domain={0}))
            else:
                pat_gx.add(
                    command.M(
                        node=0 + 2 * i, angle=alpha[2 * i], s_domain={2 * i - 1}, t_domain={2 * i - 1}, plane=Plane.XZ
                    )
                )
                pat_gx.add(command.M(node=1 + 2 * i, angle=alpha[2 * i + 1], s_domain={(2 * i - 1), 2 * i}))
            pat_gx.add(command.X(node=2 + 2 * i, domain={0 + 2 * i, 1 + 2 * i}))
            pat_gx.add(command.Z(node=2 + 2 * i, domain={1 + 2 * i}))
        pat_gx.standardize()
        pat_gx.shift_signals()
        state = pat_gx.simulate_pattern(backend='statevector')
        results_1 = pat_gx.results

        pat_dq = dq.Pattern(nodes_state=[0])
        for i in range(3):
            pat_dq.n(1 + 2 * i)
            pat_dq.n(2 + 2 * i)
            pat_dq.e(0 + 2 * i, 1 + 2 * i)
            pat_dq.e(1 + 2 * i, 2 + 2 * i)
            if i == 0:
                pat_dq.m(node=0 + 2 * i, angle=alpha[50] * torch.pi)
                pat_dq.m(node=1 + 2 * i, angle=alpha[51] * torch.pi, s_domain=[0], t_domain=[0])
            else:
                pat_dq.m(
                    node=0 + 2 * i,
                    angle=alpha[2 * i] * torch.pi,
                    plane='xz',
                    s_domain=[2 * i - 1],
                    t_domain=[2 * i - 1],
                )
                pat_dq.m(node=1 + 2 * i, angle=alpha[2 * i + 1] * torch.pi, s_domain=[2 * i - 1, 2 * i])
            pat_dq.x(node=2 + 2 * i, domain=[0 + 2 * i, 1 + 2 * i])
            pat_dq.z(node=2 + 2 * i, domain=[1 + 2 * i])
        pat_dq.standardize()
        pat_dq.shift_signals()
        state2 = pat_dq().full_state
        results_2 = pat_dq.state.measure_dict
    assert torch.allclose(torch.abs(torch.tensor(state.flatten(), dtype=torch.complex64)), torch.abs(state2.flatten()))
