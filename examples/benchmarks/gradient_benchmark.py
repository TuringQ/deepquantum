"""
Gradient evaluation comparison between qiskit, tensorcircuit and deepquantum
Modified from the implementation of tensorcircuit
"""

import time
import json
from functools import reduce
from operator import xor
import numpy as np

from qiskit.opflow import X, StateFn
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.opflow.gradients import Gradient, Hessian

import tensorcircuit as tc

import torch
from torch.autograd.functional import hessian
import deepquantum as dq


def benchmark(f, *args, trials=10):
    time0 = time.time()
    r = f(*args)
    time1 = time.time()
    for _ in range(trials):
        r = f(*args)
    time2 = time.time()
    if trials > 0:
        time21 = (time2 - time1) / trials
    else:
        time21 = 0
    ts = (time1 - time0, time21)
    print('staging time: %.6f s' % ts[0])
    if trials > 0:
        print('running time: %.6f s' % ts[1])
    return r, ts


def grad_qiskit(n, l, trials=2):
    hamiltonian = reduce(xor, [X for _ in range(n)])
    wavefunction = QuantumCircuit(n)
    params = ParameterVector('theta', length=3 * n * l)
    for j in range(l):
        for i in range(n - 1):
            wavefunction.cnot(i, i + 1)
        for i in range(n):
            wavefunction.rx(params[3 * n * j + i], i)
        for i in range(n):
            wavefunction.rz(params[3 * n * j + i + n], i)
        for i in range(n):
            wavefunction.rx(params[3 * n * j + i + 2 * n], i)

    # Define the expectation value corresponding to the energy
    op = ~StateFn(hamiltonian) @ StateFn(wavefunction)
    grad = Gradient().convert(operator=op, params=params)

    def get_grad_qiskit(values):
        value_dict = {params: values}
        grad_result = grad.assign_parameters(value_dict).eval()
        return grad_result

    return benchmark(get_grad_qiskit, np.ones([3 * n * l]), trials=trials)


def hessian_qiskit(n, l, trials=0):
    hamiltonian = reduce(xor, [X for _ in range(n)])
    wavefunction = QuantumCircuit(n)
    params = ParameterVector("theta", length=3 * n * l)
    for j in range(l):
        for i in range(n - 1):
            wavefunction.cnot(i, i + 1)
        for i in range(n):
            wavefunction.rx(params[3 * n * j + i], i)
        for i in range(n):
            wavefunction.rz(params[3 * n * j + i + n], i)
        for i in range(n):
            wavefunction.rx(params[3 * n * j + i + 2 * n], i)

    # Define the expectation value corresponding to the energy
    op = ~StateFn(hamiltonian) @ StateFn(wavefunction)
    grad = Hessian().convert(operator=op, params=params)

    def get_hs_qiskit(values):
        value_dict = {params: values}
        grad_result = grad.assign_parameters(value_dict).eval()
        return grad_result

    return benchmark(get_hs_qiskit, np.ones([3 * n * l]), trials=trials)


def grad_tc(n, l, trials=10):
    def f(params):
        c = tc.Circuit(n)
        for j in range(l):
            for i in range(n - 1):
                c.cnot(i, i + 1)
            for i in range(n):
                c.rx(i, theta=params[3 * n * j + i])
            for i in range(n):
                c.rz(i, theta=params[3 * n * j + i + n])
            for i in range(n):
                c.rx(i, theta=params[3 * n * j + i + 2 * n])
        return tc.backend.real(c.expectation(*[[tc.gates.x(), [i]] for i in range(n)]))

    get_grad_tc = tc.backend.jit(tc.backend.grad(f))
    return benchmark(get_grad_tc, tc.backend.ones([3 * n * l], dtype='float32'))


def hessian_tc(n, l, trials=10):
    def f(params):
        c = tc.Circuit(n)
        for j in range(l):
            for i in range(n - 1):
                c.cnot(i, i + 1)
            for i in range(n):
                c.rx(i, theta=params[3 * n * j + i])
            for i in range(n):
                c.rz(i, theta=params[3 * n * j + i + n])
            for i in range(n):
                c.rx(i, theta=params[3 * n * j + i + 2 * n])
        return tc.backend.real(c.expectation(*[[tc.gates.x(), [i]] for i in range(n)]))

    get_hs_tc = tc.backend.jit(tc.backend.hessian(f))
    return benchmark(get_hs_tc, tc.backend.ones([3 * n * l], dtype='float32'))


def grad_dq(n, l, trials=10):
    def get_grad_dq(params):
        if params.grad != None:
            params.grad.zero_()
        cir = dq.QubitCircuit(n)
        for j in range(l):
            for i in range(n - 1):
                cir.cnot(i, i + 1)
            cir.rxlayer(encode=True)
            cir.rzlayer(encode=True)
            cir.rxlayer(encode=True)
        cir.observable(basis='x')
        cir(data=params)
        exp = cir.expectation()
        exp.backward()
        return params.grad

    return benchmark(get_grad_dq, torch.ones([3 * n * l], requires_grad=True))


def hessian_dq(n, l, trials=10):
    def f(params):
        cir = dq.QubitCircuit(n)
        for j in range(l):
            for i in range(n - 1):
                cir.cnot(i, i + 1)
            cir.rxlayer(encode=True)
            cir.rzlayer(encode=True)
            cir.rxlayer(encode=True)
        cir.observable(basis='x')
        cir(data=params)
        return cir.expectation()
    
    def get_hs_dq(x):
        return hessian(f, x) 

    return benchmark(get_hs_dq, torch.ones([3 * n * l]))


results = {}

for n in [4, 6, 8, 10, 12]:
    for l in [2, 4, 6]:
        _, ts = grad_qiskit(n, l)
        results[str(n) + '-' + str(l) + '-' + 'grad' + '-qiskit'] = ts
        _, ts = hessian_qiskit(n, l)
        results[str(n) + '-' + str(l) + '-' + 'hs' + '-qiskit'] = ts
        with tc.runtime_backend('tensorflow'):
            _, ts = grad_tc(n, l)
            results[str(n) + '-' + str(l) + '-' + 'grad' + '-tc-tf'] = ts
            _, ts = hessian_tc(n, l)
            results[str(n) + '-' + str(l) + '-' + 'hs' + '-tc-tf'] = ts
        with tc.runtime_backend('jax'):
            _, ts = grad_tc(n, l)
            results[str(n) + '-' + str(l) + '-' + 'grad' + '-tc-jax'] = ts
            _, ts = hessian_tc(n, l)
            results[str(n) + '-' + str(l) + '-' + 'hs' + '-tc-jax'] = ts
        with tc.runtime_backend('pytorch'):
            _, ts = grad_tc(n, l)
            results[str(n) + '-' + str(l) + '-' + 'grad' + '-tc-pytorch'] = ts
            # _, ts = hessian_tc(n, l)
            # results[str(n) + '-' + str(l) + '-' + 'hs' + '-tc-pytorch'] = ts
        _, ts = grad_dq(n, l)
        results[str(n) + '-' + str(l) + '-' + 'grad' + '-dq'] = ts
        _, ts = hessian_dq(n, l)
        results[str(n) + '-' + str(l) + '-' + 'hs' + '-dq'] = ts

# print(results)

with open('gradient_results.data', 'w') as f:
    json.dump(results, f)

with open('gradient_results.data', 'r') as f:
    print(json.load(f))
