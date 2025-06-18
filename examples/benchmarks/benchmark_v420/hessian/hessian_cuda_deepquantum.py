import time
def benchmark(f, *args, trials=1):
    # r = f(*args)
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
from torch.autograd.functional import hessian
import deepquantum as dq

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def hessian_dq(n, l):
    def f(params):
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
        return cir.expectation()

    def get_hs_dq(x):
        return hessian(f, x)

    return benchmark(get_hs_dq, torch.ones([3 * n * l], device='cuda'))

import json
from tqdm import tqdm

results = {}

platform = 'deepquantum_gpu_22'
n_list = [22]
# n_list = [2, 6, 10, 14, 18]
l_list = [1, 5, 10]

for n in tqdm(n_list):
    for l in tqdm(l_list):
        _, ts = hessian_dq(n, l)
        results[str(n) + '-' + str(l)] = ts

with open('hessian_'+platform+'_results.data', 'w') as f:
    json.dump(results, f)

with open('hessian_'+platform+'_results.data', 'r') as f:
    print(json.load(f))
