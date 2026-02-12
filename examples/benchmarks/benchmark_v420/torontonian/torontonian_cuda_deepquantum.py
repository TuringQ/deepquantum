import json
import time

import deepquantum as dq
import torch
from deepquantum.photonic.torontonian_ import torontonian
from tqdm import tqdm

# Print version
print(dq.__version__)

device = 'cuda'


def torontonian_dq(nmode, batch_size):
    mat_a = torch.load(f'tor_matrix_{nmode}_{1000}.pt').to(device)

    def get_torontonian_dq(matrix):
        trials = 10
        if batch_size == 100 or batch_size == 1000:
            trials = 1
        torontonian(matrix[0])
        time0 = time.time()
        for i in range(trials):
            torch.vmap(torontonian)(matrix[i * batch_size : (i + 1) * batch_size])

        time1 = time.time()
        ts = (time1 - time0) / trials
        return ts

    return get_torontonian_dq(mat_a)


results = {}

platform = 'deepquantum_gpu'

nmodes = [2, 6, 10, 14]
batch_sizes = [1, 10, 100, 1000]

for n in tqdm(nmodes):
    for bs in tqdm(batch_sizes):
        print(n, bs)
        ts = torontonian_dq(n, bs)
        results[str(n) + '+' + str(bs)] = ts

with open('torontonian_' + platform + '_results.data', 'w') as f:
    json.dump(results, f)

with open('torontonian_' + platform + '_results.data') as f:
    print(json.load(f))
