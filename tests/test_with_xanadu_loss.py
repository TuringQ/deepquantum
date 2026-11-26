import itertools

import numpy as np
import pytest
import torch

import deepquantum as dq

try:
    import strawberryfields as sf
    from strawberryfields.ops import BSgate, Dgate, Fock, LossChannel, Rgate, Sgate
except ModuleNotFoundError as exc:
    if exc.name != 'strawberryfields':
        raise
    pytest.skip(f'{exc.name} is required for this comparison', allow_module_level=True)


@pytest.mark.parametrize('basis', [True, False], ids=['basis', 'density-matrix'])
def test_loss_fock(basis):
    nmode = 3
    cutoff = 4
    rng = np.random.default_rng(51)
    angles = rng.uniform(0, np.pi, 6)
    transmittance = rng.uniform(0.1, 0.9, 6)
    prog = sf.Program(nmode)
    with prog.context as q:
        Fock(1) | q[0]
        Fock(1) | q[1]
        Fock(1) | q[2]
        LossChannel(transmittance[0]) | q[0]
        Rgate(angles[0]) | q[0]
        Rgate(angles[1]) | q[1]
        BSgate(angles[2], angles[3]) | (q[0], q[2])
        LossChannel(transmittance[1]) | q[0]
        LossChannel(transmittance[2]) | q[2]
        BSgate(angles[4], angles[5]) | (q[1], q[2])
        LossChannel(transmittance[3]) | q[0]
        LossChannel(transmittance[4]) | q[1]
        LossChannel(transmittance[5]) | q[2]
    eng = sf.Engine('fock', backend_options={'cutoff_dim': cutoff})
    reference = eng.run(prog).state

    initial = [1, 1, 1] if basis else [(1, [1, 1, 1])]
    cir = dq.QumodeCircuit(nmode, initial, cutoff=cutoff, backend='fock', basis=basis, den_mat=not basis)
    cir.loss_t(0, transmittance[0])
    cir.ps(0, angles[0])
    cir.ps(1, angles[1])
    cir.bs([0, 2], [angles[2], angles[3]])
    cir.loss_t(0, transmittance[1])
    cir.loss_t(2, transmittance[2])
    cir.bs([1, 2], [angles[4], angles[5]])
    cir.loss_t(0, transmittance[3])
    cir.loss_t(1, transmittance[4])
    cir.loss_t(2, transmittance[5])
    if not basis:
        cir.double()
    result = cir(is_prob=True)
    if basis:
        result = {tuple(key.state.tolist()): value.item() for key, value in result.items()}
    error = 0.0
    for event in itertools.product(range(cutoff), repeat=nmode):
        probability = result.get(event, 0.0) if basis else result[0][event].item()
        error += abs(probability - reference.fock_prob(event))
    assert error < 1e-6


def test_loss_gaussian():
    n = 3
    angles = np.random.rand(6) * np.pi
    transmittance = np.linspace(0.3, 0.8, 6)
    prog = sf.Program(n)
    with prog.context as q:
        Sgate(0.375) | q[0]
        Dgate(0.5) | q[1]
        LossChannel(transmittance[0]) | q[0]
        Rgate(angles[0]) | q[0]
        Rgate(angles[1]) | q[1]
        BSgate(angles[2], angles[3]) | (q[0], q[1])
        LossChannel(transmittance[1]) | q[0]
        LossChannel(transmittance[2]) | q[2]
        BSgate(angles[4], angles[5]) | (q[1], q[2])
        LossChannel(transmittance[3]) | q[0]
        LossChannel(transmittance[4]) | q[1]
        LossChannel(transmittance[5]) | q[2]
    eng = sf.Engine('gaussian')
    result = eng.run(prog)
    sf_cov = result.state.cov()
    sf_mean = result.state.means()

    nmode = n
    cir = dq.QumodeCircuit(nmode=nmode, init_state='vac', cutoff=4, backend='gaussian')
    cir.s(0, 0.375, 0.0)
    cir.d(1, 0.5, 0.0)
    cir.loss_t(0, transmittance[0])
    cir.ps(0, angles[0])
    cir.ps(1, angles[1])
    cir.bs([0, 1], [angles[2], angles[3]])
    cir.loss_t(0, transmittance[1])
    cir.loss_t(2, transmittance[2])
    cir.bs([1, 2], [angles[4], angles[5]])
    cir.loss_t(0, transmittance[3])
    cir.loss_t(1, transmittance[4])
    cir.loss_t(2, transmittance[5])
    cir.to(torch.float64)
    state = cir()
    cov, mean = state
    err = (abs(cov[0].numpy() - sf_cov).flatten()).sum() + (abs(mean[0].flatten().numpy() - sf_mean).flatten()).sum()
    assert err < 1e-6
