import numpy as np
import strawberryfields as sf
import torch
from strawberryfields.ops import CKgate, CXgate, CZgate, Kgate, Pgate, Vgate

import deepquantum as dq


def test_quadratic_phase_gate():
    nmode = 1
    cutoff = 10
    params = np.random.rand(1)

    prog = sf.Program(nmode)
    with prog.context as q:
        Pgate(params[0]) | q[0]
    eng = sf.Engine('fock', backend_options={'cutoff_dim': cutoff})
    result = eng.run(prog)

    cir = dq.QumodeCircuit(nmode=nmode, init_state='vac', cutoff=cutoff, backend='fock', basis=False)
    cir.qp(0, params[0])
    cir.to(torch.double)
    state = cir()

    assert np.allclose(result.state.data, state.numpy())


def test_cx_gate():
    nmode = 2
    cutoff = 10
    params = np.random.rand(1)

    prog = sf.Program(nmode)
    with prog.context as q:
        CXgate(params[0]) | (q[0], q[1])
    eng = sf.Engine('fock', backend_options={'cutoff_dim': cutoff})
    result = eng.run(prog)

    cir = dq.QumodeCircuit(nmode=nmode, init_state='vac', cutoff=cutoff, backend='fock', basis=False)
    cir.cx([0,1], params[0])
    cir.to(torch.double)
    state = cir()

    assert np.allclose(result.state.data, state.numpy())


def test_cz_gate():
    nmode = 2
    cutoff = 10
    params = np.random.rand(1)

    prog = sf.Program(nmode)
    with prog.context as q:
        CZgate(params[0]) | (q[0], q[1])
    eng = sf.Engine('fock', backend_options={'cutoff_dim': cutoff})
    result = eng.run(prog)

    cir = dq.QumodeCircuit(nmode=nmode, init_state='vac', cutoff=cutoff, backend='fock', basis=False)
    cir.cz([0,1], params[0])
    cir.to(torch.double)
    state = cir()

    assert np.allclose(result.state.data, state.numpy())


def test_cubic_phase_gate():
    nmode = 1
    cutoff = 10
    params = np.random.rand(1)

    prog = sf.Program(nmode)
    with prog.context as q:
        Vgate(params[0]) | q[0]
    eng = sf.Engine('fock', backend_options={'cutoff_dim': cutoff})
    result = eng.run(prog)

    cir = dq.QumodeCircuit(nmode=nmode, init_state='vac', cutoff=cutoff, backend='fock', basis=False)
    cir.cp(0, params[0])
    cir.to(torch.double)
    state = cir()

    assert np.allclose(result.state.data, state.numpy())


def test_kerr_gate():
    nmode = 1
    cutoff = 10
    params = np.random.rand(1)

    prog = sf.Program(nmode)
    with prog.context as q:
        Kgate(params[0]) | q[0]
    eng = sf.Engine('fock', backend_options={'cutoff_dim': cutoff})
    result = eng.run(prog)

    cir = dq.QumodeCircuit(nmode=nmode, init_state='vac', cutoff=cutoff, backend='fock', basis=False)
    cir.k(0, params[0])
    cir.to(torch.double)
    state = cir()

    assert np.allclose(result.state.data, state.numpy())


def test_cross_kerr_gate():
    nmode = 2
    cutoff = 10
    params = np.random.rand(1)

    prog = sf.Program(nmode)
    with prog.context as q:
        CKgate(params[0]) | (q[0], q[1])
    eng = sf.Engine('fock', backend_options={'cutoff_dim': cutoff})
    result = eng.run(prog)

    cir = dq.QumodeCircuit(nmode=nmode, init_state='vac', cutoff=cutoff, backend='fock', basis=False)
    cir.ck([0,1], params[0])
    cir.to(torch.double)
    state = cir()

    assert np.allclose(result.state.data, state.numpy())
