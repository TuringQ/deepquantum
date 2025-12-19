import json
import time

import numpy as np
import piquasso
import torch
from piquasso._math.torontonian import loop_torontonian
from tqdm import tqdm

# Print version
print(piquasso.__version__)

def _tor_pq_loop(mat):
    gamma = mat.diagonal()
    return loop_torontonian(mat, gamma)

def torontonian_pq(n, l):
    A = torch.load(f"tor_matrix_{n}_{1000}.pt").numpy()

    def get_torontonian_pq(matrix):
        trials = 10
        if l == 100 or l == 1000:
            trials = 1
        np.vectorize(_tor_pq_loop, signature='(n,n)->()')(A[0:1])
        time0 = time.time()
        for i in range(trials):
            results = np.vectorize(_tor_pq_loop, signature='(n,n)->()')(A[i*l:(i+1)*l])
            # print(results)
        time1 = time.time()
        ts = (time1 - time0) / trials
        return ts

    return get_torontonian_pq(A)

results = {}

platform = 'piquasso'

n_list = [2, 6, 10, 14, 18]
l_list = [1, 10, 100, 1000]

for n in tqdm(n_list):
    for l in tqdm(l_list):
        print(n,l)
        ts = torontonian_pq(n, l)
        results[str(n)+'+'+str(l)] = ts

with open('loop_torontonian_'+platform+'_results.data', 'w') as f:
    json.dump(results, f)

with open('loop_torontonian_'+platform+'_results.data', 'r') as f:
    print(json.load(f))
