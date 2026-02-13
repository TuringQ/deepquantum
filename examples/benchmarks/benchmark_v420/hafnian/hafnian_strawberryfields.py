import json
import time

import numpy as np
import strawberryfields as sf
from GlobalHafnian import test_sequence_hafnian
from thewalrus import hafnian
from tqdm import tqdm

number_of_sequence = 1000

# Print version
print(sf.__version__)


def hafnian_sf(nmode, batch_size):
    """Generate a random hafnian matrix and calculate its hafnian using DeepQuantum."""
    mat_a = test_sequence_hafnian(nmode, number_of_sequence=number_of_sequence)

    # 计算 hafnian
    def get_hafnian_sf(matrix):
        trials = 100
        if batch_size == 100 or batch_size == 1000:
            trials = 1
        hafnian(np.array(matrix[0]))
        time0 = time.time()
        for i in tqdm(range(trials)):
            for j in range(i * batch_size, (i + 1) * batch_size):
                hafnian(np.array(matrix[j]))
        time1 = time.time()
        ts = (time1 - time0) / trials
        return ts

    return get_hafnian_sf(mat_a)


# 进行测试
results = {}

platform = 'strawberryfields'

nmodes = [2, 6, 10, 14]
batch_sizes = [1, 10, 100]

for n in tqdm(nmodes):
    for bs in batch_sizes:
        print(f'n={n}, bs={bs}')
        ts = hafnian_sf(n, bs)
        results[str(n) + '+' + str(bs)] = ts

# 保存结果
with open(f'hafnian/hafnian_{platform}_results.data', 'w') as f:
    json.dump(results, f)

# 读取并打印
with open(f'hafnian/hafnian_{platform}_results.data') as f:
    print(json.load(f))
