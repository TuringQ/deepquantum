import math
from fractions import Fraction

import pytest

import deepquantum as dq


def test_quantum_phase_estimation_single_qubit():
    t = 3
    phase = 1 / 8
    qpe = dq.QuantumPhaseEstimationSingleQubit(t, phase)
    qpe()
    res = qpe.measure(wires=list(range(t)))
    max_key = max(res, key=res.get)
    phase_est = int(max_key, 2) / 2**t
    assert phase_est == phase


def test_phi_adder():
    nqubit = 5
    n1 = 1
    n2 = 8
    enc = dq.NumberEncoder(nqubit, n1)
    qft = dq.QuantumFourierTransform(nqubit, reverse=True)
    phiadd = dq.PhiAdder(nqubit, n2)
    iqft = qft.inverse()
    cir = enc + qft + phiadd + iqft
    cir()
    res = cir.measure()
    max_key = max(res, key=res.get)
    assert int(max_key, 2) == n1 + n2


def test_phi_modular_adder():
    n1 = 5
    n2 = 1
    mod = 8
    assert n1 < mod and n1 + n2 < 2 * mod
    nqubit = len(bin(mod))
    minmax = [0, nqubit - 2]
    ancilla = [nqubit - 1]
    enc = dq.NumberEncoder(nqubit, n1, minmax)
    qft = dq.QuantumFourierTransform(nqubit, minmax, reverse=True)
    pma = dq.PhiModularAdder(nqubit, n2, mod, minmax, ancilla)
    iqft = qft.inverse()
    cir = enc + qft + pma + iqft
    cir()
    res = cir.measure(wires=list(range(minmax[0], minmax[1] + 1)))
    max_key = max(res, key=res.get)
    assert int(max_key, 2) == (n1 + n2) % mod


def test_controlled_multiplier():
    n1 = 1
    n2 = 2
    n3 = 14
    mod = 15
    assert n1 < mod and n1 + n2 * n3 < 2 * mod
    nx = len(bin(n3)) - 2
    nb = len(bin(mod)) - 1
    nqubit = nx + nb + 1
    minmax1 = [0, nx - 1]
    minmax2 = [nx, nqubit - 2]
    ancilla = [nqubit - 1]
    enc1 = dq.NumberEncoder(nqubit, n3, minmax1)
    enc2 = dq.NumberEncoder(nqubit, n1, minmax2)
    cmult = dq.ControlledMultiplier(nqubit, n2, mod, [0, nqubit - 2], nx, ancilla)
    cir = enc1 + enc2 + cmult
    cir()
    res = cir.measure(wires=list(range(minmax2[0], minmax2[1] + 1)))
    max_key = max(res, key=res.get)
    assert int(max_key, 2) == (n1 + n2 * n3) % mod


def test_controlled_ua():
    mod = 15
    a = 8
    x = 3
    assert a < mod and x < mod
    assert math.gcd(a, mod) == 1
    nreg = len(bin(mod)) - 2
    nqubit = 2 * nreg + 2
    minmax = [0, nreg - 1]
    ancilla = list(range(minmax[1] + 1, minmax[1] + 1 + nreg + 2))
    enc = dq.NumberEncoder(nqubit, x, minmax)
    ua = dq.ControlledUa(nqubit, a, mod, minmax, ancilla)
    cir = enc + ua
    cir()
    res = cir.measure(wires=list(range(minmax[0], minmax[1] + 1)))
    max_key = max(res, key=res.get)
    assert int(max_key, 2) == (a * x) % mod
    res = cir.measure(wires=ancilla)
    max_key = max(res, key=res.get)
    assert int(max_key, 2) == 0


@pytest.mark.parametrize('circuit_type,kwargs', [(dq.ShorCircuit, {'mod': 15}), (dq.ShorCircuitFor15, {})])
def test_shor_finds_factors_of_15(circuit_type, kwargs):
    a = 7
    ncount = 8
    circuit = circuit_type(ncount=ncount, a=a, **kwargs)
    probabilities = circuit().reshape(2**ncount, -1).abs().square().sum(-1)
    # The order of 7 modulo 15 is four, giving peaks at k/4 on the counting register.
    peaks = probabilities.topk(4).indices.sort().values.tolist()
    assert peaks == [0, 64, 128, 192]
    period = Fraction(peaks[1], 2**ncount).denominator
    factors = {math.gcd(a ** (period // 2) + sign, 15) for sign in (-1, 1)}
    assert factors == {3, 5}
