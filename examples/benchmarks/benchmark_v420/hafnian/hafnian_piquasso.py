import json
import time

import numpy as np
from GlobalHafnian import *
from piquasso._math.hafnian import hafnian_with_reduction
from tqdm import tqdm


def hafnian_pq(n, l):
    """Generate a random hafnian matrix and calculate its hafnian using DeepQuantum."""
    A = test_sequence_hafnian(n, number_of_sequence=number_of_sequence)

    # 计算 hafnian
    def get_hafnian_pq(A):
        trials = 10
        if l == 100 or l == 1000:
            trials = 1
        hafnian_with_reduction(np.array(A[0], dtype=np.complex128), np.array([1]*2*n))
        time0 = time.time()
        for i in tqdm(range(trials)):
            for j in range(i*l, (i+1)*l):
                results = hafnian_with_reduction(np.array(A[j], dtype=np.complex128), np.array([1]*2*n))
        time1 = time.time()
        ts = (time1 - time0) / trials
        return ts

    return get_hafnian_pq(A)

# 进行测试
results = {}
platform = 'piquasso'

for n in tqdm(n_list):
    for l in l_list:
        print(f"n={n}, l={l}")
        ts = hafnian_pq(n, l)
        results[str(n)+'+'+str(l)] = ts

# 保存结果
with open(f'hafnian/hafnian_{platform}_results.data', 'w') as f:
    json.dump(results, f)

# 读取并打印
with open(f'hafnian/hafnian_{platform}_results.data') as f:
    print(json.load(f))
