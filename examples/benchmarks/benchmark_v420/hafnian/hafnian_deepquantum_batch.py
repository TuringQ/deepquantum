import time

import deepquantum as dq
# print version
print(dq.__version__)

import deepquantum as dq
from deepquantum.photonic.hafnian_ import hafnian_batch
from GlobalHafnian import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def hafnian_batch_dq(n, l): # n modes 的数量 ， l batchsize
    """Generate a random hafnian matrix and calculate its hafnian using DeepQuantum."""
    A = test_sequence_hafnian(n, number_of_sequence=number_of_sequence).to('cuda') # （number_of_sequence=l， 2n， 2n）
    print(A.device)

    def get_hafnian_batch_dq(A):
        trials = 100
        if l == 100 or l == 1000:
            trials = 1
        time0 = time.time()
        for i in tqdm(range(trials)):
            results = hafnian_batch(A[i*l:(i+1)*l])
            # print(results)
        time1 = time.time()
        ts = (time1 - time0) / trials
        return ts

    return get_hafnian_batch_dq(A)

results = {}
platform = 'deepquantum_gpu'

# 生成一个 n 量子比特的量子线路，深度为 l
for n in tqdm(n_list):
    for l in l_list:
        print('n =', n, 'l =', l)
        ts = hafnian_batch_dq(n, l)
        results[str(n)+'+'+str(l)] = ts

with open('hafnian/hafnian_'+platform+'_results.data', 'w') as f:
    json.dump(results, f)

with open('hafnian/hafnian_'+platform+'_results.data', 'r') as f:
    print(json.load(f))
