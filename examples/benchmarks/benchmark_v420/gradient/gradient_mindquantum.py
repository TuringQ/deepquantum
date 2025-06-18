import time
def benchmark(f, *args, trials=10):
    r = f(*args)
    time0 = time.time()
    for _ in range(trials):
        r = f(*args)
    time1 = time.time()

    ts = (time1 - time0) / trials

    return r, ts

import numpy as np
import json
from tqdm import tqdm

import mindquantum as mq
import numpy as np
from mindquantum.core import Circuit, RX, RZ, X
from mindquantum.simulator import Simulator

# print version
print(mq.__version__)


def grad_mindquantum(n, l):
    sim = Simulator('mqvector', n)  # 选择 MindQuantum 矢量模拟器
    params = [f'theta_{i}' for i in range(3 * n * l)]  # 参数名

    # 构造量子线路
    circ = Circuit()
    for j in range(l):
        for i in range(n - 1):
            circ += X.on(i + 1, i)  # CNOT
        for i in range(n):
            circ += RX(params[3 * n * j + i]).on(i)
        for i in range(n):
            circ += RZ(params[3 * n * j + i + n]).on(i)
        for i in range(n):
            circ += RX(params[3 * n * j + i + 2 * n]).on(i)

    obs = ' '.join(f'X{i}' for i in range(n))
    hamiltonian = mq.Hamiltonian(mq.QubitOperator(obs))  # 观察量 Pauli X

    # 计算梯度
    grad_fn = sim.get_expectation_with_grad(hamiltonian, circ)

    def compute_grad(values):
        values = np.array(values)
        _, grad = grad_fn(values)
        return np.array(grad)

    return benchmark(compute_grad, np.ones(3 * n * l, dtype=np.float32))


results = {}
platform = 'mindquantum'
# n_list = [2, 4, 6, 8, 10]
# n_list = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
# # l_list = [5, 10, 15, 20, 25, 30]
# l_list = [5]
n_list = [2, 6, 10, 14, 18, 22]

l_list = [1, 5, 10]

for n in tqdm(n_list):
    for l in l_list:
        _, ts = grad_mindquantum(n, l)
        results[f"{n}-{l}"] = ts

with open('gradient_'+platform+'_results.data', 'w') as f:
    json.dump(results, f)

with open('gradient_'+platform+'_results.data', 'r') as f:
    print(json.load(f))
