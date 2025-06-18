import time
def benchmark(f, *args, trials=10):
    r = f(*args)
    time0 = time.time()
    for _ in range(trials):
        r = f(*args)
    time1 = time.time()

    ts = (time1 - time0) / trials

    return r, ts

import deepquantum as dq
# print version
print(dq.__version__)

import torch

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def grad_dq(n, l):
    def get_grad_dq(params):
        if params.grad != None:
            params.grad.zero_()
        cir = dq.QubitCircuit(n)
        for j in range(l):
            for i in range(n - 1):
                cir.cnot(i, i + 1)
            cir.rxlayer(encode=True)
            cir.rzlayer(encode=True)
            cir.rxlayer(encode=True)
        cir.observable(basis='x'*n)
        cir.to('cuda')
        cir(data=params)
        exp = cir.expectation()
        exp.backward()
        return params.grad

    return benchmark(get_grad_dq, torch.ones([3 * n * l], requires_grad=True, device='cuda'))

import json
from tqdm import tqdm

results = {}

platform = 'deepquantum_gpu'
n_list = [2, 6, 10, 14, 18, 22]

l_list = [1, 5, 10]

# 生成一个 n 量子比特的量子线路，深度为 l
for n in tqdm(n_list):
    for l in l_list:
        _, ts = grad_dq(n, l)
        results[str(n) + '-' + str(l)] = ts

with open('gradient_'+platform+'_results.data', 'w') as f:
    json.dump(results, f)

with open('gradient_'+platform+'_results.data', 'r') as f:
    print(json.load(f))