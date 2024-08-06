import deepquantum as dq
import pytest
import torch


def test_gaussian_backend_auto_grad():
    def get_vac_prob(paras):
        nmode = 2
        cir = dq.QumodeCircuit(nmode=nmode, init_state='vac', cutoff=3, backend='gaussian')
        cir.s(wires=0, r=paras[0])
        cir.s(wires=1, r=paras[1])
        cir.d(wires=0, r=paras[2])
        cir.d(wires=1, r=paras[3])
        cir.bs(wires=[0,1], inputs=[paras[4], paras[5]])

        state = cir(is_prob=True)
        target_state = dq.FockState([0,0])
        vac_prob = state[target_state]
        return vac_prob

    r_s1 = torch.tensor([1.], requires_grad=True)
    r_s2 = torch.tensor([1.], requires_grad=True)
    r_d1 = torch.tensor([1.], requires_grad=True)
    r_d2 = torch.tensor([1.], requires_grad=True)
    theta = torch.tensor([0.1], requires_grad=True)
    phi = torch.tensor([0.1], requires_grad=True)
    para_ini = [r_s1, r_s2, r_d1, r_d2, theta, phi]

    target_prob = 0.5 # set vacuum state prob
    optimizer = torch.optim.Adam(para_ini, lr=0.05)
    best_para = []
    for _ in range(500):
        optimizer.zero_grad()
        vac_prob = get_vac_prob(para_ini) # forward
        loss = abs(target_prob - vac_prob)
        if loss < 1e-4:
            best_para.append([i.detach().clone() for i in para_ini])
        loss.backward() # backpropagetion
        optimizer.step() # update parameters
    best_result = get_vac_prob(best_para[0])
    assert abs(best_result - target_prob) < 1e-4
