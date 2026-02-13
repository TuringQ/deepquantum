import json
import time

import deepquantum as dq
from GlobalHafnian import test_sequence_hafnian
from deepquantum.photonic.hafnian_ import hafnian_batch
from tqdm import tqdm

number_of_sequence = 1000

# Print version
print(dq.__version__)


def hafnian_batch_dq(nmode, batch_size):
    """Generate a random hafnian matrix and calculate its hafnian using DeepQuantum."""
    mat_a = test_sequence_hafnian(nmode, number_of_sequence=number_of_sequence).to('cuda')
    print(mat_a.device)

    def get_hafnian_batch_dq(matrix):
        trials = 100
        if batch_size == 100 or batch_size == 1000:
            trials = 1
        hafnian_batch(matrix[0:1])
        time0 = time.time()
        for i in tqdm(range(trials)):
            hafnian_batch(matrix[i * batch_size : (i + 1) * batch_size])
        time1 = time.time()
        ts = (time1 - time0) / trials
        return ts

    return get_hafnian_batch_dq(mat_a)


results = {}

platform = 'deepquantum_gpu'

nmodes = [2, 6, 10, 14]
batch_sizes = [1, 10, 100]

for n in tqdm(nmodes):
    for bs in batch_sizes:
        print(f'n={n}, bs={bs}')
        ts = hafnian_batch_dq(n, bs)
        results[str(n) + '+' + str(bs)] = ts

with open('hafnian/hafnian_' + platform + '_results.data', 'w') as f:
    json.dump(results, f)

with open('hafnian/hafnian_' + platform + '_results.data') as f:
    print(json.load(f))
