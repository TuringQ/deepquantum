import math
from fractions import Fraction

import deepquantum as dq
import pytest


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
    ancilla = [nqubit -1]
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


def test_shor_general():
    a = 7
    mod = 15
    ncount = 8
    found = False
    trial = 0
    while not found:
        trial += 1
        print(f'\ntrial {trial}:')
        cir = dq.ShorCircuit(mod, ncount, a)
        cir()
        res = cir.measure(wires=list(range(ncount)), shots=1)
        max_key = max(res, key=res.get)
        phase = int(max_key, 2) / 2 ** ncount
        frac = Fraction(phase).limit_denominator(mod)
        r = frac.denominator
        print(f'Result: r = {r}')
        if phase != 0:
            guesses = [math.gcd(a ** (r // 2) - 1, mod), math.gcd(a ** (r // 2) + 1, mod)]
            print(f'Guessed Factors: {guesses[0]} and {guesses[1]}')
            for guess in guesses:
                if guess not in [1, mod] and (mod % guess) == 0:
                    print(f'*** Non-trivial factor found: {guess} ***')
                    found = True
                    assert guess in [3, 5]


def test_shor_special():
    a = 7
    mod = 15
    ncount = 8
    found = False
    trial = 0
    while not found:
        trial += 1
        print(f'\ntrial {trial}:')
        cir = dq.ShorCircuitFor15(ncount, a)
        cir()
        res = cir.measure(wires=list(range(ncount)), shots=1)
        max_key = max(res, key=res.get)
        phase = int(max_key, 2) / 2 ** ncount
        frac = Fraction(phase).limit_denominator(mod)
        r = frac.denominator
        print(f'Result: r = {r}')
        if phase != 0:
            guesses = [math.gcd(a ** (r // 2) - 1, mod), math.gcd(a ** (r // 2) + 1, mod)]
            print(f'Guessed Factors: {guesses[0]} and {guesses[1]}')
            for guess in guesses:
                if guess not in [1, mod] and (mod % guess) == 0:
                    print(f'*** Non-trivial factor found: {guess} ***')
                    found = True
                    assert guess in [3, 5]
