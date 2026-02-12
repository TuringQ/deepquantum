"""
Gradient evaluation comparison between qiskit, tensorcircuit, pyvqnet and deepquantum
Modified from the implementation of tensorcircuit
"""

import json
import time
from functools import reduce
from operator import xor

import deepquantum as dq
import numpy as np
import pyqpanda as pq
import tensorcircuit as tc
import torch
from pyvqnet.qnn import grad
from pyvqnet.qnn.measure import expval
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.opflow import StateFn, X
from qiskit.opflow.gradients import Gradient, Hessian
from torch.autograd.functional import hessian


def benchmark(f, *args, trials=10):
    time0 = time.time()
    r = f(*args)
    time1 = time.time()
    for _ in range(trials):
        r = f(*args)
    time2 = time.time()
    time21 = (time2 - time1) / trials if trials > 0 else 0
    ts = (time1 - time0, time21)
    print(f'staging time: {ts[0]:.6f} s')
    if trials > 0:
        print(f'running time: {ts[1]:.6f} s')
    return r, ts


def grad_qiskit(n, layer, trials=2):
    hamiltonian = reduce(xor, [X for _ in range(n)])
    wavefunction = QuantumCircuit(n)
    params = ParameterVector('theta', length=3 * n * layer)
    for j in range(layer):
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

    return benchmark(get_grad_qiskit, np.ones([3 * n * layer]), trials=trials)


def hessian_qiskit(n, layer, trials=0):
    hamiltonian = reduce(xor, [X for _ in range(n)])
    wavefunction = QuantumCircuit(n)
    params = ParameterVector('theta', length=3 * n * layer)
    for j in range(layer):
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

    return benchmark(get_hs_qiskit, np.ones([3 * n * layer]), trials=trials)


def grad_tc(n, layer, trials=10):
    def f(params):
        c = tc.Circuit(n)
        for j in range(layer):
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
    return benchmark(get_grad_tc, tc.backend.ones([3 * n * layer], dtype='float32'))


def hessian_tc(n, layer, trials=10):
    def f(params):
        c = tc.Circuit(n)
        for j in range(layer):
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
    return benchmark(get_hs_tc, tc.backend.ones([3 * n * layer], dtype='float32'))


def grad_dq(n, layer, trials=10):
    def get_grad_dq(params):
        if params.grad is not None:
            params.grad.zero_()
        cir = dq.QubitCircuit(n)
        for _ in range(layer):
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

    return benchmark(get_grad_dq, torch.ones([3 * n * layer], requires_grad=True))


def hessian_dq(n, layer, trials=10):
    def f(params):
        cir = dq.QubitCircuit(n)
        for _ in range(layer):
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

    return benchmark(get_hs_dq, torch.ones([3 * n * layer]))


def grad_pyvqnet(n, layer, trials=10):
    def pqctest(param):
        machine = pq.CPUQVM()
        machine.init_qvm()
        qubits = machine.qAlloc_many(n)
        circuit = pq.QCircuit()
        for j in range(layer):
            for i in range(n - 1):
                circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
            for i in range(n):
                circuit.insert(pq.RX(qubits[i], param[3 * n * j + i]))
                circuit.insert(pq.RZ(qubits[i], param[3 * n * j + i + n]))
                circuit.insert(pq.RX(qubits[i], param[3 * n * j + i + 2 * n]))
        prog = pq.QProg()
        prog.insert(circuit)
        xn_string = ', '.join([f'X{i}' for i in range(n)])
        pauli_dict = {xn_string: 1.0}
        exp = expval(machine, prog, pauli_dict, qubits)
        return exp

    def get_grad(values):
        return grad(pqctest, values)

    return benchmark(get_grad, np.ones([3 * n * layer]), trials=trials)


results = {}

for n in [4, 6, 8, 10, 12]:
    for layer in [2, 4, 6]:
        _, ts = grad_qiskit(n, layer)
        results[str(n) + '-' + str(layer) + '-' + 'grad' + '-qiskit'] = ts
        _, ts = hessian_qiskit(n, layer)
        results[str(n) + '-' + str(layer) + '-' + 'hs' + '-qiskit'] = ts
        with tc.runtime_backend('tensorflow'):
            _, ts = grad_tc(n, layer)
            results[str(n) + '-' + str(layer) + '-' + 'grad' + '-tc-tf'] = ts
            _, ts = hessian_tc(n, layer)
            results[str(n) + '-' + str(layer) + '-' + 'hs' + '-tc-tf'] = ts
        with tc.runtime_backend('jax'):
            _, ts = grad_tc(n, layer)
            results[str(n) + '-' + str(layer) + '-' + 'grad' + '-tc-jax'] = ts
            _, ts = hessian_tc(n, layer)
            results[str(n) + '-' + str(layer) + '-' + 'hs' + '-tc-jax'] = ts
        with tc.runtime_backend('pytorch'):
            _, ts = grad_tc(n, layer)
            results[str(n) + '-' + str(layer) + '-' + 'grad' + '-tc-pytorch'] = ts
            # _, ts = hessian_tc(n, layer)
            # results[str(n) + '-' + str(layer) + '-' + 'hs' + '-tc-pytorch'] = ts
        _, ts = grad_dq(n, layer)
        results[str(n) + '-' + str(layer) + '-' + 'grad' + '-dq'] = ts
        _, ts = hessian_dq(n, layer)
        results[str(n) + '-' + str(layer) + '-' + 'hs' + '-dq'] = ts
        _, ts = grad_pyvqnet(n, layer)
        results[str(n) + '-' + str(layer) + '-' + 'grad' + '-pyvqnet'] = ts

# print(results)

with open('gradient_results.data', 'w') as f:
    json.dump(results, f)

with open('gradient_results.data') as f:
    print(json.load(f))
