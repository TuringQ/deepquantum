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