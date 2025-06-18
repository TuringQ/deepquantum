import json
import random
import time

import graphix
from tqdm import tqdm

def benchmark(f, *args, trials=100):
    r = f(*args)
    time0 = time.time()
    for _ in range(trials):
        r = f(*args)
    time1 = time.time()

    ts = (time1 - time0) / trials

    return r, ts

def random_circuit_transpilation(n_qubits, n_gates):

    random.seed(10)
    cir = graphix.Circuit(n_qubits)

    # Available single-qubit gates
    single_gates = [
        lambda q: cir.h(q),
        lambda q: cir.x(q),
        lambda q: cir.rx(q, angle=random.random() * 2),
        lambda q: cir.ry(q, angle=random.random() * 2),
        lambda q: cir.rz(q, angle=random.random() * 2)
    ]

    # Add random number of gates (3-10)
    n_gates = n_gates

    for _ in range(n_gates):
        # 70% chance for single-qubit gate, 20% for CNOT
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
    def transpile():
        pattern = cir.transpile().pattern
    # Transpile circuit to measurement pattern
    return benchmark(transpile)

results = {}

platform = 'graphix'
n_list = [2, 5, 10, 20]

l_list = [5, 10, 100]

for n in tqdm(n_list):
    for l in tqdm(l_list):
        _, ts = random_circuit_transpilation(n, l)
        results[str(n) + '-' + str(l)] = ts

with open('transpile_mbqc_'+platform+'_results.data', 'w') as f:
    json.dump(results, f)

with open('transpile_mbqc_'+platform+'_results.data', 'r') as f:
    print(json.load(f))
