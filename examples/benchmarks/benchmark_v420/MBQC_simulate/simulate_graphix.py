import json
import random
import time

import graphix
from tqdm import tqdm


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
    cir = graphix.Circuit(n_qubits)

    # Available single-qubit gates
    single_gates = [
        lambda q: cir.h(q),
        lambda q: cir.x(q),
        lambda q: cir.rx(q, angle=random.random() * 2),
        lambda q: cir.ry(q, angle=random.random() * 2),
        lambda q: cir.rz(q, angle=random.random() * 2),
    ]

    # Add random number of gates (3-10)
    for _ in range(n_gates):
        # 70% chance for single-qubit gate, 20% for CNOT
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
    pattern = cir.transpile().pattern

    def simulate():
        return pattern.simulate_pattern(backend='statevector', input_state=graphix.states.BasicStates.ZERO)

    # Transpile circuit to measurement pattern
    if n_qubits == 20 and n_gates == 100:
        return benchmark(simulate, trials=1)
    else:
        return benchmark(simulate)


results = {}

platform = 'graphix'

nqubits = [2, 5, 10, 20]
ngates = [5, 10, 100]

for n in tqdm(nqubits):
    for ng in tqdm(ngates):
        _, ts = random_circuit_simulation(n, ng)
        results[str(n) + '-' + str(ng)] = ts

with open('simulate_mbqc_' + platform + '_results.data', 'w') as f:
    json.dump(results, f)

with open('simulate_mbqc_' + platform + '_results.data') as f:
    print(json.load(f))
