import deepquantum as dq
import numpy as np
import pytest
import strawberryfields as sf
import torch
from deepquantum.photonic import xxpp_to_xpxp


def test_catstate():
    r = np.random.rand(1)[0]
    theta = 2 * np.pi * np.random.rand(1)[0]
    nmodes = 1
    prog_cat_bosonic = sf.Program(nmodes)
    hbar = 2
    with prog_cat_bosonic.context as q:
        sf.ops.Catstate(a=r, phi=theta, p=1) | q[0] # superposition of 4 states
    eng = sf.Engine("bosonic", backend_options={"hbar": hbar}) # xpxp order
    state = eng.run(prog_cat_bosonic).state
    means_sf = state.means()
    covs_sf = state.covs()
    weights_sf = state.weights()

    cat = dq.CatState(r=r, theta=theta, p=1)
    err1 = abs(cat.cov - covs_sf).sum()
    err2 = abs(cat.mean[0].squeeze() - means_sf).sum()
    err3 = abs(cat.weight - weights_sf).sum() / abs(weights_sf).sum() # relative error
    assert err1 + err2 + err3 < 3e-4


def test_forward_cov_mean():
    r = np.random.rand(1)[0]
    theta = 2 * np.pi * np.random.rand(1)[0]
    cat = dq.CatState(r=r, theta=theta, p=1)
    vac = dq.BosonicState(state='vac', nmode=1)
    cir = dq.QumodeCircuit(nmode=2, init_state=[cat, vac], backend='bosonic')
    angles = 2 * np.pi * np.random.rand(2)
    cir.s(1, r=2.)
    cir.bs([0,1], angles)
    test = cir()

    nmodes = 2
    prog_cat_bosonic = sf.Program(nmodes)
    hbar = 2
    with prog_cat_bosonic.context as q:
        sf.ops.Catstate(a=r, phi=theta, p=1) | q[0] # superposition of 4 states
    #     sf.ops.Squeezed(r=1) | q[0] # catstate 不能加压缩门
        sf.ops.Squeezed(r=2) | q[1]
        sf.ops.BSgate(angles[0], angles[1]) | [q[0], q[1]]
    eng = sf.Engine("bosonic", backend_options={"hbar": hbar}) # xpxp order
    state = eng.run(prog_cat_bosonic).state
    means_sf = state.means()
    covs_sf = state.covs()
    weights_sf = state.weights()
    err1 = abs(xxpp_to_xpxp(test[0][0]) - covs_sf).sum()
    err2 = abs(xxpp_to_xpxp(test[1][0]).squeeze() - means_sf).sum()
    err3 = abs(test[2] - weights_sf).sum() / abs(weights_sf).sum() # relative error
    assert err1 + err2 + err3 < 3e-4


def test_photon_number_mean_var():
    r = np.random.rand(1)[0]
    theta = 2 * np.pi * np.random.rand(1)[0]
    nmodes = 1
    prog_cat_bosonic = sf.Program(nmodes)
    hbar = 2
    with prog_cat_bosonic.context as q:
        sf.ops.Catstate(a=r, phi=theta, p=1) | q[0] # superposition of 4 states
    eng = sf.Engine("bosonic", backend_options={"hbar": hbar}) # xpxp order
    state = eng.run(prog_cat_bosonic).state

    cir = dq.QumodeCircuit(nmode=1, init_state='vac', backend='bosonic')
    cir.cat(wires=0, r=r, theta=theta, p=1)
    cir.s(0, r=0)
    cir.to(torch.float)
    cir()
    test1 = cir.photon_number_mean_var()
    test2 = state.mean_photon(0)
    err = abs(torch.tensor(test1) - np.array(test2)).sum()
    assert err < 1e-4


def test_wigner():
    xrange = 5
    prange = 5
    npoints = 200
    theta = 2 * np.pi * np.random.rand(1)[0]
    phi = 2 * np.pi * np.random.rand(1)[0]
    gkp = dq.GKPState(theta=theta, phi=phi, amp_cutoff=0.01, epsilon=0.05)
    gkp.to(torch.double)

    nmodes = 1
    prog_cat_bosonic = sf.Program(nmodes)
    with prog_cat_bosonic.context as q:
        sf.ops.GKP(state=[theta, phi], epsilon=0.05, ampl_cutoff=0.01) | q[0] # superposition of 4 states
    eng = sf.Engine("bosonic", backend_options={"hbar": 2}) # xpxp order
    state = eng.run(prog_cat_bosonic).state
    xvec = torch.linspace(-xrange, xrange, npoints)
    pvec = torch.linspace(-prange, prange, npoints)
    w_sf = state.wigner(0, xvec, pvec)
    w_dq = gkp.wigner(0, xrange, prange, npoints, plot=False, normalize=False)
    err = abs(w_dq[0].mT - w_sf).max()
    assert err < 1e-5
