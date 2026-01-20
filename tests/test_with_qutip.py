import deepquantum as dq
import numpy as np
import pytest
import qutip as qp
import torch

def test_with_qutip_fock_wigner():
    r, d = torch.rand(2)
    cutoff = 50
    cir1 = dq.QumodeCircuit(nmode=1, init_state='vac', backend='fock', basis=False, cutoff=cutoff)
    cir1.s(0, r=r, theta=0)
    cir1.d(0, r=d)
    cir1.to(torch.double)
    re  = cir1()
    fock_state = dq.FockState(state=re, basis=False)
    psi = re[0]
    psi = psi.reshape(cutoff, 1)
    rho = psi @ psi.mH

    npoints = 100
    xrange = 10
    prange = 10
    xvec = np.linspace(-xrange, xrange, npoints)
    pvec = np.linspace(-prange, prange, npoints)
    wigner_qp = qp.wigner(qp.Qobj(psi), xvec, pvec, g=1)
    w = fock_state.wigner(wire=0, xrange=xrange, prange=prange, npoints=npoints, plot=False)
    err = torch.sum(abs(w - torch.tensor(wigner_qp.mT)), dim=[1,2])
    assert err < 1e-6

def test_with_qutip_gaussian_wigner():
    r, d = torch.rand(2)
    cutoff = 100
    cir1 = dq.QumodeCircuit(nmode=1, init_state='vac', backend='fock', basis=False, cutoff=cutoff)
    cir1.s(0, r=r, theta=0)
    cir1.d(0, r=d)
    cir1.to(torch.double)
    re  = cir1()
    psi = re[0]
    psi = psi.reshape(cutoff, 1)
    rho = psi @ psi.mH

    cir2 = dq.QumodeCircuit(nmode=1, init_state='vac', backend='gaussian')
    cir2.s(0, r=r, theta=0)
    cir2.d(0, r=d)
    cir2.to(torch.double)
    re2 = cir2()
    gaussian_state = dq.GaussianState(re2)

    npoints = 100
    xrange = 10
    prange = 10
    xvec = np.linspace(-xrange, xrange, npoints)
    pvec = np.linspace(-prange, prange, npoints)
    wigner_qp = qp.wigner(qp.Qobj(psi), xvec, pvec, g=1)
    wigner_dq = gaussian_state.wigner(wire=0, xrange=xrange, prange=prange, npoints=npoints, plot=False)
    err = torch.sum(abs(wigner_dq - torch.tensor(wigner_qp.mT)), dim=[1,2])
    assert err < 1e-2