import json
import time

import numpy as np
import piquasso
import torch
from piquasso._math.permanent import permanent
from tqdm import tqdm

# Print version
print(piquasso.__version__)


def _perm_pq(mat):
    r = np.ones(n)
    return permanent(mat, r, r)


def perm_pq(nmode, batch_size):
    mat_a = torch.load(f'u_matrix_{nmode}_{1000}.pt').numpy()

    def get_perm_pq(matrix):
        trials = 10
        if batch_size == 100 or batch_size == 1000:
            trials = 1
        np.vectorize(_perm_pq, signature='(n,n)->()')(matrix[0:1])
        time0 = time.time()
        for i in range(trials):
            np.vectorize(_perm_pq, signature='(n,n)->()')(matrix[i * batch_size : (i + 1) * batch_size])
        time1 = time.time()
        ts = (time1 - time0) / trials
        return ts

    return get_perm_pq(mat_a)


results = {}

platform = 'piquasso'

nmodes = [2, 6, 10, 14, 18, 22, 26, 30]
batch_sizes = [1, 10, 100]

for n in tqdm(nmodes):
    for bs in tqdm(batch_sizes):
        print(n, bs)
        ts = perm_pq(n, bs)
        results[str(n) + '+' + str(bs)] = ts

with open('permanent_' + platform + '_results.data', 'w') as f:
    json.dump(results, f)

with open('permanent_' + platform + '_results.data') as f:
    print(json.load(f))
