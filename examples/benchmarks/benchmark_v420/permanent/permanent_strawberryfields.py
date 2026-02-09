import json
import time

import numpy as np
import strawberryfields as sf
import torch
from thewalrus import perm
from tqdm import tqdm

# Print version
print(sf.__version__)

def perm_sf(n, l):
    A = torch.load(f"u_matrix_{n}_{1000}.pt").numpy()

    def get_perm_sf(matrix):
        trials = 10
        if l == 1000:
            trials = 1
        np.vectorize(perm,signature='(n,n)->()')(A[0:1])
        time0 = time.time()
        for i in range(trials):
            results = np.vectorize(perm,signature='(n,n)->()')(A[i*l:(i+1)*l])

        time1 = time.time()
        ts = (time1 - time0) / trials
        return ts

    return get_perm_sf(A)

results = {}

platform = 'strawberryfields'

n_list = [2, 6, 10, 14, 18, 22, 26, 30]
l_list = [1, 10, 100, 1000]

for n in tqdm(n_list):
    for l in tqdm(l_list):
        print(n,l)
        ts = perm_sf(n, l)
        results[str(n)+'+'+str(l)] = ts

with open('permanent_'+platform+'_results.data', 'w') as f:
    json.dump(results, f)

with open('permanent_'+platform+'_results.data') as f:
    print(json.load(f))
