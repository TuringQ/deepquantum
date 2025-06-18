import time

import strawberryfields as sf
# print version
print(sf.__version__)

import time
import json
import numpy as np
import torch

from thewalrus import perm
from tqdm import tqdm


def perm_sf(n, l):
    A = torch.load(f"u_matrix_{n}_{1000}.pt").numpy()

    def get_perm_sf(matrix):
        trials = 10
        if l == 1000:
            trials = 1
        time0 = time.time()
        for i in range(trials):
            results = np.vectorize(perm,signature='(n,n)->()')(A[i*l:(i+1)*l])

        time1 = time.time()
        ts = (time1 - time0) / trials
        return ts

    return get_perm_sf(A)


import json
from tqdm import tqdm

results = {}

platform = 'strawberryfields'

n_list = [2, 6, 10, 14, 18, 22, 26, 30]
l_list = [1, 10, 100, 1000]

# 生成一个 n 量子比特的量子线路，深度为 l
for n in tqdm(n_list):
    for l in tqdm(l_list):
        print(n,l)
        ts = perm_sf(n, l)
        results[str(n)+'+'+str(l)] = ts

with open('permanent_'+platform+'_results.data', 'w') as f:
    json.dump(results, f)

with open('permanent_'+platform+'_results.data', 'r') as f:
    print(json.load(f))
