import deepquantum as dq
import deepquantum.photonic as dqp
import numpy as np
import torch
from scipy.stats import unitary_group

def test_schur_antisymmetric():
    def generate_real_antisymmetric(n, low=-10, high=10):
        if n <= 0:
            return np.array([])
        a = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                random_val = np.random.uniform(low, high)
                a[i, j] = random_val
                a[j, i] = -random_val
        return a

    n = np.random.randint(1, 6)
    a = generate_real_antisymmetric(2*n)
    t, o = dqp.qmath.schur_antisymmetric(torch.tensor(a))
    err1 = abs(o @ t @ o.mT - torch.tensor(a)).sum()
    err2 = abs(o@o.mT - torch.eye(2*n)).sum()
    assert err1 + err2 < 1e-8


def test_williamson():
    nmode = np.random.randint(2, 10)
    u = unitary_group.rvs(nmode)
    sqs = torch.rand(nmode)
    cir = dq.QumodeCircuit(nmode, init_state='vac', cutoff=4, backend='gaussian')
    for i in range(nmode):
        cir.s(i, sqs[i])
    cir.any(unitary=u, wires=list(range(nmode)))
    covs, _ = cir()
    mat = covs[0]
    t, s = dq.williamson(mat)
    err1 = abs((s @ t @ s.mT) - mat).sum() # 验证分解正确性
    omega = torch.diag_embed(torch.cat([-mat.new_ones(nmode), mat.new_ones(nmode)]))
    omega = omega.reshape(2, nmode, 2 * nmode).flip(0).reshape(2 * nmode, 2 * nmode)
    err2 = abs((s.mT @ omega @ s) - omega).sum() # 验证辛形式
    assert err1 + err2 < 5e-4

