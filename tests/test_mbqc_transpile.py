import random

import pytest
import torch

import deepquantum as dq


def test_random_circuit_transpilation():
    # Create a circuit with random number of qubits (2-5)
    n_qubits = random.randint(2, 5)
    batch_size = random.randint(1, 3)
    init_state = torch.rand(batch_size, 2**n_qubits)

    cir = dq.QubitCircuit(n_qubits, init_state=init_state)

    # Available single-qubit gates
    single_gates = [
        lambda q: cir.h(q),
        lambda q: cir.x(q),
        lambda q: cir.rx(q, inputs=torch.rand(1) * 2 * torch.pi),
        lambda q: cir.ry(q, inputs=torch.rand(1) * 2 * torch.pi),
        lambda q: cir.rz(q, inputs=torch.rand(1) * 2 * torch.pi),
    ]

    # Add random number of gates (3-10)
    n_gates = random.randint(3, 10)

    for _ in range(n_gates):
        # 70% chance for single-qubit gate, 30% for CNOT
        if random.random() < 0.7:
            # Random single-qubit gate
            qubit = random.randint(0, n_qubits - 1)
            random.choice(single_gates)(qubit)
        else:
            # Random CNOT
            control = random.randint(0, n_qubits - 1)
            target = random.randint(0, n_qubits - 1)
            # Ensure control and target are different
            while target == control:
                target = random.randint(0, n_qubits - 1)
            cir.cnot(control=control, target=target)

    # Transpile circuit to measurement pattern
    pattern = cir.pattern()

    # Execute both circuits
    state_cir = cir()
    state_pattern = pattern()
    # Assert that both states are equal (up to global phase)
    assert torch.allclose(torch.abs(state_cir), torch.abs(state_pattern.graph.full_state), atol=1e-6)


@pytest.mark.parametrize(
    'batched_data,reupload',
    [(False, False), (True, False), (True, True)],
    ids=['batched-state', 'batched-data', 'reupload'],
)
def test_encoded_transpilation(batched_data, reupload):
    generator = torch.Generator().manual_seed(23)
    init_state = torch.rand(8 if batched_data else (2, 8), generator=generator)
    cir = dq.QubitCircuit(3, init_state=init_state, reupload=reupload)
    cir.h(0)
    cir.rx(0, encode=True)
    cir.cnot(0, 1)
    cir.ry(1, encode=True)
    cir.rz(2, encode=True)
    cir.ry(0, inputs=0.27)
    cir.cnot(2, 1)
    cir.rx(1, encode=True)
    pattern = cir.pattern()
    data = torch.randn(2, 2 if reupload else 4, generator=generator)
    if not batched_data:
        data = data[0]

    expected = cir(data).reshape(2, 8, 1)
    actual = pattern(-data).graph.full_state.reshape(2, 8, 1)
    # Compare each output state up to its global phase.
    torch.testing.assert_close(actual @ actual.mH, expected @ expected.mH, atol=1e-6, rtol=1e-5)
