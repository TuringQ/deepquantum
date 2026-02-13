import json
import time

import deepquantum as dq
import torch
from torch.autograd.functional import hessian
from tqdm import tqdm

# Print version
print(dq.__version__)


def benchmark(f, *args, trials=10):
    # r = f(*args)
    time0 = time.time()
    for _ in range(trials):
        r = f(*args)
    time1 = time.time()

    ts = (time1 - time0) / trials

    return r, ts


def hessian_dq(n, layer):
    def f(params):
        cir = dq.QubitCircuit(n)
        for _ in range(layer):
            for i in range(n - 1):
                cir.cnot(i, i + 1)
            cir.rxlayer(encode=True)
            cir.rzlayer(encode=True)
            cir.rxlayer(encode=True)
        cir.observable(basis='x' * n)
        cir(data=params)
        return cir.expectation()

    def get_hs_dq(x):
        return hessian(f, x)

    return benchmark(get_hs_dq, torch.ones([3 * n * layer]))


results = {}

platform = 'deepquantum'

n_list = [2, 6, 10, 14, 18]
l_list = [1, 5, 10]

for n in tqdm(n_list):
    for layer in tqdm(l_list):
        _, ts = hessian_dq(n, layer)
        results[str(n) + '-' + str(layer)] = ts

with open('hessian_' + platform + '_results.data', 'w') as f:
    json.dump(results, f)

with open('hessian_' + platform + '_results.data') as f:
    print(json.load(f))
