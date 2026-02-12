import json
import time

import numpy as np
from GlobalHafnian import test_sequence_hafnian
from piquasso._math.hafnian import hafnian_with_reduction
from tqdm import tqdm

number_of_sequence = 1000


def hafnian_pq(nmode, batch_size):
    """Generate a random hafnian matrix and calculate its hafnian using DeepQuantum."""
    mat_a = test_sequence_hafnian(nmode, number_of_sequence=number_of_sequence)

    # 计算 hafnian
    def get_hafnian_pq(matrix):
        trials = 10
        if batch_size == 100 or batch_size == 1000:
            trials = 1
        hafnian_with_reduction(np.array(matrix[0], dtype=np.complex128), np.array([1] * 2 * nmode))
        time0 = time.time()
        for i in tqdm(range(trials)):
            for j in range(i * batch_size, (i + 1) * batch_size):
                hafnian_with_reduction(np.array(matrix[j], dtype=np.complex128), np.array([1] * 2 * nmode))
        time1 = time.time()
        ts = (time1 - time0) / trials
        return ts

    return get_hafnian_pq(mat_a)


# 进行测试
results = {}

platform = 'piquasso'

nmodes = [2, 6, 10, 14]
batch_sizes = [1, 10, 100]

for n in tqdm(nmodes):
    for bs in batch_sizes:
        print(f'n={n}, bs={bs}')
        ts = hafnian_pq(n, bs)
        results[str(n) + '+' + str(bs)] = ts

# 保存结果
with open(f'hafnian/hafnian_{platform}_results.data', 'w') as f:
    json.dump(results, f)

# 读取并打印
with open(f'hafnian/hafnian_{platform}_results.data') as f:
    print(json.load(f))
