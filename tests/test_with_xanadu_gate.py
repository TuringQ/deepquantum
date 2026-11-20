import numpy as np
import pytest
import strawberryfields as sf
import torch
from strawberryfields.ops import CKgate, CXgate, CZgate, Ket, Kgate, Pgate, Vgate

import deepquantum as dq


@pytest.mark.parametrize(
    'gate,reference,nmode',
    [('qp', Pgate, 1), ('cx', CXgate, 2), ('cz', CZgate, 2), ('cp', Vgate, 1), ('k', Kgate, 1), ('ck', CKgate, 2)],
)
def test_fock_gate_matches_strawberry_fields(gate, reference, nmode):
    cutoff = 10
    strength = 0.25
    ket = np.zeros((cutoff,) * nmode, dtype=np.complex128)
    # The relative phase makes the action of Kerr and cross-Kerr observable.
    ket[(0,) * nmode] = 2**-0.5
    ket[(2,) if nmode == 1 else (1, 2)] = 2**-0.5
    program = sf.Program(nmode)
    with program.context as q:
        Ket(ket) | q
        reference(strength) | q
    engine = sf.Engine('fock', backend_options={'cutoff_dim': cutoff})
    expected = engine.run(program).state.data

    circuit = dq.QumodeCircuit(nmode, torch.from_numpy(ket).unsqueeze(0), cutoff=cutoff, backend='fock', basis=False)
    getattr(circuit, gate)(list(range(nmode)), strength)
    actual = circuit.double()()[0]
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12, rtol=1e-10)
