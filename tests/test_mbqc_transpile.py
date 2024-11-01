import deepquantum as dq
import numpy as np
import torch
from deepquantum.mbqc import transpile

def test_random_circuit_transpilation():
    # Create a 3-qubit circuit with random initial state
    init_state = torch.rand(8)
    init_state = init_state / torch.norm(init_state)  # normalize

    cir = dq.QubitCircuit(3, init_state=init_state)

    # Add random gates
    cir.h(0)  # Hadamard on first qubit
    cir.ry(1, inputs=torch.pi/3)  # Rotation Y on second qubit
    cir.cnot(control=0, target=2)  # CNOT between first and third qubit
    cir.rz(2, inputs=torch.pi/4)  # Rotation Z on third qubit
    cir.cnot(control=1, target=2)  # CNOT between second and third qubit
    # Transpile circuit to measurement pattern
    pattern = transpile(cir)

    # Execute both circuits
    state_cir = cir()
    state_pattern = pattern()

    # Assert that both states are equal (up to global phase)
    assert torch.allclose(
        torch.abs(state_cir.flatten()),
        torch.abs(state_pattern),
        atol=1e-6
    )

# import deepquantum as dq
# import numpy as np
# import torch
# from deepquantum.mbqc import transpile
# import random

# def test_random_circuit_transpilation():
#     # Create a circuit with random number of qubits (2-5)
#     n_qubits = random.randint(2, 5)
#     init_state = torch.rand(2**n_qubits)
#     init_state = init_state / torch.norm(init_state)  # normalize

#     cir = dq.QubitCircuit(n_qubits, init_state=init_state)

#     # Available single-qubit gates
#     single_gates = [
#         lambda q: cir.h(q),
#         lambda q: cir.x(q),
#         lambda q: cir.rx(q, inputs=torch.rand(1) * 2 * torch.pi),
#         lambda q: cir.ry(q, inputs=torch.rand(1) * 2 * torch.pi),
#         lambda q: cir.rz(q, inputs=torch.rand(1) * 2 * torch.pi)
#     ]

#     # Add random number of gates (3-10)
#     n_gates = random.randint(3, 10)

#     for _ in range(n_gates):
#         # 70% chance for single-qubit gate, 30% for CNOT
#         if random.random() < 0.7:
#             # Random single-qubit gate
#             qubit = random.randint(0, n_qubits-1)
#             random.choice(single_gates)(qubit)
#         else:
#             # Random CNOT
#             control = random.randint(0, n_qubits-1)
#             target = random.randint(0, n_qubits-1)
#             # Ensure control and target are different
#             while target == control:
#                 target = random.randint(0, n_qubits-1)
#             cir.cnot(control=control, target=target)

#     # Transpile circuit to measurement pattern
#     pattern = transpile(cir)

#     # Execute both circuits
#     state_cir = cir()
#     state_pattern = pattern()

#     # Assert that both states are equal (up to global phase)
#     assert torch.allclose(
#         torch.abs(state_cir.flatten()),
#         torch.abs(state_pattern),
#         atol=1e-4
#     )