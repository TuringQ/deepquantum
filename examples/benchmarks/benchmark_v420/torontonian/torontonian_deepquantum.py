import json
import time

import deepquantum as dq
import torch
from deepquantum.photonic.torontonian_ import torontonian
from tqdm import tqdm

# Print version
print(dq.__version__)

device = 'cpu'

def torontonian_dq(n, l):
    A = torch.load(f"tor_matrix_{n}_{1000}.pt").to(device)

    def get_torontonian_dq(matrix):
        trials = 10
        if l == 100 or l == 1000:
            trials = 1
        torontonian(A[0])
        time0 = time.time()
        for i in range(trials):
            results = torch.vmap(torontonian)(A[i*l:(i+1)*l])

        time1 = time.time()
        ts = (time1 - time0) / trials
        return ts

    return get_torontonian_dq(A)

results = {}

platform = 'deepquantum'

n_list = [2, 6, 10, 14, 18]
l_list = [1, 10, 100, 1000]

for n in tqdm(n_list):
    for l in tqdm(l_list):
        print(n,l)
        ts = torontonian_dq(n, l)
        results[str(n)+'+'+str(l)] = ts

with open('torontonian_'+platform+'_results.data', 'w') as f:
    json.dump(results, f)

with open('torontonian_'+platform+'_results.data') as f:
    print(json.load(f))
