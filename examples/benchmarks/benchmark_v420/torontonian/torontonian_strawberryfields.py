import json
import time

import numpy as np
import strawberryfields as sf
import torch
from thewalrus import tor
from tqdm import tqdm

# Print version
print(sf.__version__)


def torontonian_sf(nmode, batch_size):
    mat_a = torch.load(f'tor_matrix_{nmode}_{1000}.pt').numpy()

    def get_torontonian_sf(matrix):
        trials = 10
        if batch_size == 100 or batch_size == 1000:
            trials = 1
        np.vectorize(tor, signature='(n,n)->()')(matrix[0:1])
        time0 = time.time()
        for i in range(trials):
            np.vectorize(tor, signature='(n,n)->()')(matrix[i * batch_size : (i + 1) * batch_size])

        time1 = time.time()
        ts = (time1 - time0) / trials
        return ts

    return get_torontonian_sf(mat_a)


results = {}

platform = 'strawberryfields'

nmodes = [2, 6, 10, 14]
batch_sizes = [1, 10, 100, 1000]

for n in tqdm(nmodes):
    for bs in tqdm(batch_sizes):
        print(n, bs)
        ts = torontonian_sf(n, bs)
        results[str(n) + '+' + str(bs)] = ts

with open('torontonian_' + platform + '_results.data', 'w') as f:
    json.dump(results, f)

with open('torontonian_' + platform + '_results.data') as f:
    print(json.load(f))
