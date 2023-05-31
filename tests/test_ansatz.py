import pytest
import deepquantum as dq


def test_quantum_phase_estimation_single_qubit():
    t = 3
    phase = 1 / 8
    qpe = dq.QuantumPhaseEstimationSingleQubit(t, phase)
    qpe()
    res = qpe.measure(wires=list(range(t)))
    max_key = max(res, key=res.get)
    phase_est = int(max_key, 2) / 2 ** t
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
    assert n1 < mod and n2 < mod
    nqubit = len(bin(max(n1, n2, mod)))
    minmax = [0, nqubit - 2]
    ancilla = [nqubit - 1]
    enc = dq.NumberEncoder(nqubit, n1, minmax)
    qft = dq.QuantumFourierTransform(nqubit, minmax, reverse=True)
    pma = dq.PhiModularAdder(nqubit, n2, mod, minmax, ancilla)
    iqft = qft.inverse()
    cir = enc + qft + pma + iqft
    cir()
    print(cir.measure(wires=ancilla))
    res = cir.measure(wires=list(range(minmax[0], minmax[1] + 1)))
    max_key = max(res, key=res.get)
    assert int(max_key, 2) == (n1 + n2) % mod