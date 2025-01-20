import deepquantum as dq
import numpy as np
import torch
import random

def test_random_circuit_transpilation():
    # Create a circuit with random number of qubits (2-5)
    n_qubits = random.randint(2, 5)
    batch_size = random.randint(1,3)
    init_state = torch.rand(batch_size, 2**n_qubits)
    init_state = init_state / torch.norm(init_state, dim=-1, keepdim=True)  # normalize

    cir = dq.QubitCircuit(n_qubits, init_state=init_state)

    # Available single-qubit gates
    single_gates = [
        lambda q: cir.h(q),
        lambda q: cir.x(q),
        lambda q: cir.rx(q, inputs=torch.rand(1) * 2 * torch.pi),
        lambda q: cir.ry(q, inputs=torch.rand(1) * 2 * torch.pi),
        lambda q: cir.rz(q, inputs=torch.rand(1) * 2 * torch.pi)
    ]

    # Add random number of gates (3-10)
    n_gates = random.randint(3, 10)

    for _ in range(n_gates):
        # 70% chance for single-qubit gate, 30% for CNOT
        if random.random() < 0.7:
            # Random single-qubit gate
            qubit = random.randint(0, n_qubits-1)
            random.choice(single_gates)(qubit)
        else:
            # Random CNOT
            control = random.randint(0, n_qubits-1)
            target = random.randint(0, n_qubits-1)
            # Ensure control and target are different
            while target == control:
                target = random.randint(0, n_qubits-1)
            cir.cnot(control=control, target=target)

    # Transpile circuit to measurement pattern
    pattern = cir.pattern()

    # Execute both circuits
    state_cir = cir()
    state_pattern = pattern()
    # Assert that both states are equal (up to global phase)
    assert torch.allclose(
        torch.abs(state_cir),
        torch.abs(state_pattern.graph.full_state),
        atol=1e-6
    )