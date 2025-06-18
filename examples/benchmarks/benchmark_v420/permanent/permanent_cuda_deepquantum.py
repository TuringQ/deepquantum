import time

import deepquantum as dq
# print version
print(dq.__version__)

import torch
import deepquantum as dq
from deepquantum.photonic.qmath import permanent

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

device='cuda'

def permanent_dq(n, l):
    A = torch.load(f"u_matrix_{n}_{1000}.pt").to(device)

    def get_perm_dq(matrix):
        trials = 1
        if l == 10 or l == 100 or l == 1000:
            trials = 1
        time0 = time.time()
        for i in range(trials):
            if n > 21 and l >= 100:
                    results = torch.vmap(permanent, chunk_size=1)(matrix[i*l:(i+1)*l])
            if n == 18 and l == 1000:
                    results = torch.vmap(permanent, chunk_size=200)(matrix[i*l:(i+1)*l])
            else:
                results = torch.vmap(permanent)(matrix[i*l:(i+1)*l])
        time1 = time.time()
        ts = (time1 - time0) / trials
        return ts

    return get_perm_dq(A)


import json
from tqdm import tqdm

results = {}

platform = 'deepquantum_gpu'

n_list = [2, 6, 10, 14, 18, 22]
l_list = [1, 10, 100, 1000]

# 生成一个 n 量子比特的量子线路，深度为 l
for n in tqdm(n_list):
    for l in tqdm(l_list):
        print(n,l)
        ts = permanent_dq(n, l)
        results[str(n)+'+'+str(l)] = ts

with open('permanent_'+platform+'_results.data', 'w') as f:
    json.dump(results, f)

with open('permanent_'+platform+'_results.data', 'r') as f:
    print(json.load(f))
