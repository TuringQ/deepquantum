import time

import piquasso
# print version
print(piquasso.__version__)

import time
import json
import numpy as np
import torch

from piquasso._math.permanent import permanent
from tqdm import tqdm


def _perm_pq(mat):
    r = np.ones(n)
    return permanent(mat, r, r)

def perm_pq(n, l):
    A = torch.load(f"u_matrix_{n}_{1000}.pt").numpy()

    def get_perm_pq(matrix):
        trials = 10
        if l == 100 or l == 1000:
            trials = 1
        time0 = time.time()
        for i in range(trials):
            results = np.vectorize(_perm_pq, signature='(n,n)->()')(A[i*l:(i+1)*l])
            # print(results)
        time1 = time.time()
        ts = (time1 - time0) / trials
        return ts

    return get_perm_pq(A)


import json
from tqdm import tqdm

results = {}

platform = 'piquasso'

l_list = [1, 10, 100]
n_list = [2, 6, 10, 14, 18, 22, 26, 30]


# 生成一个 n 量子比特的量子线路，深度为 l
for n in tqdm(n_list):
    for l in tqdm(l_list):
        print(n,l)
        ts = perm_pq(n, l)
        results[str(n)+'+'+str(l)] = ts

with open('permanent_'+platform+'_results.data', 'w') as f:
    json.dump(results, f)

with open('permanent_'+platform+'_results.data', 'r') as f:
    print(json.load(f))
