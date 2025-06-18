import json
import time
from functools import reduce

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from tqdm import tqdm

# Print version
print(qml.__version__)

def benchmark(f, *args, trials=10):
    r = f(*args)
    time0 = time.time()
    for _ in range(trials):
        r = f(*args)
    time1 = time.time()

    ts = (time1 - time0) / trials

    return r, ts

def grad_pennylane(n, l):
    dev = qml.device("default.qubit", wires=n)

    params = pnp.array(np.ones(3 * n * l), requires_grad=True)

    @qml.qnode(dev)
    def circuit(w):
        for j in range(l):
            for i in range(n - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(n):
                qml.RX(w[3 * n * j + i], wires=i)
            for i in range(n):
                qml.RZ(w[3 * n * j + i + n], wires=i)
            for i in range(n):
                qml.RX(w[3 * n * j + i + 2 * n], wires=i)

        obs = reduce(lambda x, y: x @ y, [qml.PauliX(i) for i in range(n)])
        return qml.expval(obs)

    grad_fn = qml.grad(circuit)
    return benchmark(grad_fn, params)

results = {}
platform = 'pennylane'

n_list = [2, 6, 10, 14, 18, 22]
l_list = [1, 5, 10]

for n in tqdm(n_list):
    for l in l_list:
        _, ts = grad_pennylane(n, l)
        results[f"{n}-{l}"] = ts

with open('gradient_'+platform+'_results.data', 'w') as f:
    json.dump(results, f)

with open('gradient_'+platform+'_results.data', 'r') as f:
    print(json.load(f))
