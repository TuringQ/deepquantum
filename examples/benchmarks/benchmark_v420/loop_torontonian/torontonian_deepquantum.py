import time

import deepquantum as dq
# print version
print(dq.__version__)

import torch
import deepquantum as dq
from deepquantum.photonic.torontonian_ import torontonian

device='cpu'

def torontonian_dq(n, l):
    A = torch.load(f"tor_matrix_{n}_{1000}.pt").to(device)

    def get_torontonian_dq(matrix):
        trials = 10
        if l == 100 or l == 1000:
            trials = 1
        time0 = time.time()
        for i in range(trials):
            gamma = torch.diagonal(A[i*l:(i+1)*l], dim1=1, dim2=2)
            results = torch.vmap(torontonian)(A[i*l:(i+1)*l], gamma)
            # print(results)
        time1 = time.time()
        ts = (time1 - time0) / trials
        return ts

    return get_torontonian_dq(A)


import json
from tqdm import tqdm

results = {}

platform = 'deepquantum'

n_list = [2, 6, 10, 14]
l_list = [1, 10, 100, 1000]

# 生成一个 n 量子比特的量子线路，深度为 l
for n in tqdm(n_list):
    for l in tqdm(l_list):
        print(n,l)
        ts = torontonian_dq(n, l)
        results[str(n)+'+'+str(l)] = ts

with open('loop_torontonian_'+platform+'_results.data', 'w') as f:
    json.dump(results, f)

with open('loop_torontonian_'+platform+'_results.data', 'r') as f:
    print(json.load(f))
