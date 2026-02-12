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


def torontonian_pq(nmode, batch_size):
    mat_a = torch.load(f'tor_matrix_{nmode}_{1000}.pt').numpy()

    def get_torontonian_pq(matrix):
        trials = 10
        if batch_size == 100 or batch_size == 1000:
            trials = 1
        np.vectorize(_tor_pq_loop, signature='(n,n)->()')(matrix[0:1])
        time0 = time.time()
        for i in range(trials):
            np.vectorize(_tor_pq_loop, signature='(n,n)->()')(matrix[i * batch_size : (i + 1) * batch_size])
        time1 = time.time()
        ts = (time1 - time0) / trials
        return ts

    return get_torontonian_pq(mat_a)


results = {}

platform = 'piquasso'

nmodes = [2, 6, 10, 14, 18]
batch_sizes = [1, 10, 100, 1000]

for n in tqdm(nmodes):
    for bs in tqdm(batch_sizes):
        print(n, bs)
        ts = torontonian_pq(n, bs)
        results[str(n) + '+' + str(bs)] = ts

with open('loop_torontonian_' + platform + '_results.data', 'w') as f:
    json.dump(results, f)

with open('loop_torontonian_' + platform + '_results.data') as f:
    print(json.load(f))
