import json
import time

import numpy as np
import strawberryfields as sf
import torch
from thewalrus import ltor
from tqdm import tqdm

# Print version
print(sf.__version__)


def _tor_sf_loop(mat):
    gamma = mat.diagonal()
    return ltor(mat, gamma)


def torontonian_sf(n, l):
    A = torch.load(f'tor_matrix_{n}_{1000}.pt').numpy()

    def get_torontonian_sf(matrix):
        trials = 10
        # if l == 100 or l == 1000:
        if l == 1000:
            trials = 1
        np.vectorize(_tor_sf_loop, signature='(n,n)->()')(A[0:1])
        time0 = time.time()
        for i in range(trials):
            results = np.vectorize(_tor_sf_loop, signature='(n,n)->()')(A[i * l : (i + 1) * l])
        time1 = time.time()
        ts = (time1 - time0) / trials
        return ts

    return get_torontonian_sf(A)


results = {}

platform = 'strawberryfields'

n_list = [2, 6, 10, 14]
l_list = [1, 10, 100, 1000]

for n in tqdm(n_list):
    for l in tqdm(l_list):
        print(n, l)
        ts = torontonian_sf(n, l)
        results[str(n) + '+' + str(l)] = ts

with open('loop_torontonian_' + platform + '_results.data', 'w') as f:
    json.dump(results, f)

with open('loop_torontonian_' + platform + '_results.data') as f:
    print(json.load(f))
