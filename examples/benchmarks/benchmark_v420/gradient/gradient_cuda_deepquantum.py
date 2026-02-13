import json
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


def grad_dq(n, layer):
    def get_grad_dq(params):
        if params.grad is not None:
            params.grad.zero_()
        cir = dq.QubitCircuit(n)
        for _ in range(layer):
            for i in range(n - 1):
                cir.cnot(i, i + 1)
            cir.rxlayer(encode=True)
            cir.rzlayer(encode=True)
            cir.rxlayer(encode=True)
        cir.observable(basis='x' * n)
        cir.to('cuda')
        cir(data=params)
        exp = cir.expectation()
        exp.backward()
        return params.grad

    return benchmark(get_grad_dq, torch.ones([3 * n * layer], requires_grad=True, device='cuda'))


results = {}

platform = 'deepquantum_gpu'

n_list = [2, 6, 10, 14, 18, 22]
l_list = [1, 5, 10]

for n in tqdm(n_list):
    for layer in l_list:
        _, ts = grad_dq(n, layer)
        results[str(n) + '-' + str(layer)] = ts

with open('gradient_' + platform + '_results.data', 'w') as f:
    json.dump(results, f)

with open('gradient_' + platform + '_results.data') as f:
    print(json.load(f))
