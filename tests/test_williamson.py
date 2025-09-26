import deepquantum as dq
import numpy as np
import torch
from scipy.stats import unitary_group

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
    t, s, o = dq.williamson(mat)
    err1 = abs((s @ t @ s.mT) - mat).sum() # 验证分解正确性
    err2 = abs((s.mT @ o @ s) - o).sum() # 验证辛形式
    assert err1 + err2 < 5e-4

