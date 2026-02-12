import json
import random
import time

import deepquantum as dq
import torch
from tqdm import tqdm

# Print version
print(dq.__version__)


def benchmark(f, *args, trials=10):
    r = f(*args)
    time0 = time.time()
    for _ in range(trials):
        r = f(*args)
    time1 = time.time()

    ts = (time1 - time0) / trials

    return r, ts


def random_circuit_simulation(n_qubits, n_gates):
    random.seed(10)

    cir = dq.QubitCircuit(n_qubits)

    # Available single-qubit gates
    single_gates = [
        lambda q: cir.h(q),
        lambda q: cir.x(q),
        lambda q: cir.rx(q, inputs=random.random() * 2 * torch.pi),
        lambda q: cir.ry(q, inputs=random.random() * 2 * torch.pi),
        lambda q: cir.rz(q, inputs=random.random() * 2 * torch.pi),
    ]

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

    def simulation():
        return pattern().full_state

    return benchmark(simulation)


results = {}

platform = 'deepquantum'

nqubits = [2, 5, 10, 20]
ngates = [5, 10, 100]

for n in tqdm(nqubits):
    for ng in tqdm(ngates):
        _, ts = random_circuit_simulation(n, ng)
        results[str(n) + '-' + str(ng)] = ts

with open('simulation_mbqc_' + platform + '_results.data', 'w') as f:
    json.dump(results, f)

with open('simulation_mbqc_' + platform + '_results.data') as f:
    print(json.load(f))
