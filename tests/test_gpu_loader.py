"""SDK discovery and CPU-only compiler/import contracts."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import deepquantum as dq
from deepquantum.backends.gpu.compiler import compile_circuit
from deepquantum.backends.gpu.loader import load_gpu_executor, load_ixquantum_executor, resolve_backend, resolve_sdk


@pytest.mark.parametrize('layout', ['flat', 'lib', 'lib64'])
def test_resolve_sdk(tmp_path, monkeypatch, layout):
    include = tmp_path if layout == 'flat' else tmp_path / 'include'
    library = tmp_path if layout == 'flat' else tmp_path / layout
    include.mkdir(exist_ok=True)
    library.mkdir(exist_ok=True)
    (include / 'custatevec.h').touch()
    (library / 'libcustatevec.so.1.8.0').touch()
    monkeypatch.setenv('IXQUANTUM_ROOT', str(tmp_path))
    assert resolve_sdk('ixquantum') == (include, library / 'libcustatevec.so.1.8.0')


def test_missing_sdk(tmp_path, monkeypatch):
    monkeypatch.setenv('IXQUANTUM_ROOT', str(tmp_path))
    with pytest.raises(RuntimeError, match='IXQUANTUM_ROOT'):
        resolve_sdk('ixquantum')


@pytest.mark.parametrize('layout', ['lib', 'lib64'])
def test_nvidia_sdk_versioned_library(tmp_path, monkeypatch, layout):
    include = tmp_path / 'include'
    include.mkdir()
    library = tmp_path / layout
    library.mkdir()
    (include / 'custatevec.h').touch()
    target = library / 'libcustatevec.so.1'
    target.touch()
    monkeypatch.setenv('CUQUANTUM_ROOT', str(tmp_path))
    assert resolve_sdk('cuquantum') == (include, target)


def test_backend_selection(monkeypatch):
    monkeypatch.setattr(torch, 'corex', True, raising=False)
    assert resolve_backend() == 'ixquantum'
    with pytest.raises(RuntimeError, match='NVIDIA CUDA'):
        resolve_backend('cuquantum')
    monkeypatch.setattr(torch, 'corex', False)
    monkeypatch.setattr(torch.version, 'cuda', '12.6')
    assert resolve_backend() == 'cuquantum'
    with pytest.raises(RuntimeError, match='CoreX'):
        resolve_backend('ixquantum')
    monkeypatch.setattr(torch.version, 'cuda', None)
    with pytest.raises(RuntimeError, match='NVIDIA CUDA'):
        resolve_backend()
    with pytest.raises(ValueError, match='backend'):
        resolve_backend('unknown')


@pytest.mark.parametrize('backend', ['ixquantum', 'cuquantum'])
def test_backend_build_configuration(tmp_path, monkeypatch, backend):
    import torch.utils.cpp_extension as extension

    import deepquantum.backends.gpu.loader as loader

    monkeypatch.setattr(torch, 'corex', backend == 'ixquantum', raising=False)
    monkeypatch.setattr(torch.version, 'cuda', '12.6')
    monkeypatch.setattr(loader, 'resolve_sdk', lambda selected: (tmp_path, tmp_path / 'libcustatevec.so.1'))
    calls = []
    monkeypatch.setattr(extension, 'load', lambda **kwargs: calls.append(kwargs) or SimpleNamespace())
    load_gpu_executor(backend)
    assert calls[0]['name'] == f'deepquantum_gpu_executor_{backend}'
    assert [Path(p).name for p in calls[0]['sources']] == ['_gpu_executor.cpp', 'adjoint_kernels.cu']
    assert ('ivcore' in calls[0]['extra_cuda_cflags']) == (backend == 'ixquantum')
    assert calls[0]['extra_ldflags'][0] == str(tmp_path / 'libcustatevec.so.1')


def test_missing_nvidia_sdk_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv('CUQUANTUM_ROOT', str(tmp_path))
    with pytest.raises(RuntimeError, match='CUQUANTUM_ROOT'):
        resolve_sdk('cuquantum')


def test_corex_optional(monkeypatch):
    monkeypatch.setattr(torch, 'corex', False, raising=False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    importlib.reload(importlib.import_module('deepquantum.backends.gpu.loader'))
    circuit = dq.QubitCircuit(2)
    circuit.h(0)
    circuit.cnot(0, 1)
    state = circuit().reshape(-1)
    torch.testing.assert_close(state.abs().square(), torch.tensor([0.5, 0.0, 0.0, 0.5]))
    with pytest.raises(RuntimeError, match='available GPU|NVIDIA CUDA'):
        dq.QubitCircuitGPU(2)
    with pytest.raises(RuntimeError, match='CoreX'):
        load_ixquantum_executor()


def test_compiler_order_and_shared_parameters():
    circuit = dq.QubitCircuit(4)
    circuit.h(0)
    circuit.cnot(0, 3)
    gate = dq.Rx(nqubit=4, wires=1, requires_grad=True)
    circuit.add(gate)
    circuit.add(gate, wires=2)
    circuit.add(gate.inverse())
    circuit.observable([0, 3], 'xy')
    plan = compile_circuit(circuit)
    assert plan.targets[:2] == ((3,), (0,))
    assert plan.controls[1] == (3,)
    assert plan.observable_bits == ((3, 0),)
    assert plan.parameter_indices.tolist() == [-1, -1, 0, 0, 0]
    assert plan.parameter_scales[-3:].tolist() == [1, 1, -1]


def test_compiler_rejects_data_encoding():
    circuit = dq.QubitCircuit(2)
    circuit.rx(0, encode=True)
    with pytest.raises(ValueError, match='encoding'):
        compile_circuit(circuit)


def test_compiler_rejects_invalid_observable():
    from deepquantum.backends.gpu.compiler import compile_pauli_observables

    with pytest.raises(ValueError, match='unique circuit wires'):
        compile_pauli_observables(2, [SimpleNamespace(wires=[0, 0], basis='zz')])


def test_specialized_api_dispatch():
    from deepquantum.backends.gpu.compiler import (
        GRAD_PAULI_GENERATOR,
        GRAD_PHASE,
        OP_MATRIX,
        OP_PAULI_ROTATION,
        OP_PERMUTATION,
        OP_PHASE,
    )

    circuit = dq.QubitCircuit(5)
    circuit.h(0)
    circuit.cx(0, 4)
    circuit.cz(3, 1)
    circuit.swap([4, 2])
    circuit.toffoli(0, 1, 2)
    circuit.fredkin(0, 3, 4)
    circuit.cp(0, 1)
    circuit.rzz([1, 4])
    circuit.rxy([2, 3])
    plan = compile_circuit(circuit)
    assert plan.opcodes.tolist() == [OP_MATRIX] + [OP_PERMUTATION] * 5 + [OP_PHASE] + [OP_PAULI_ROTATION] * 3
    assert plan.controls[4] == (4, 3)
    assert plan.targets[4] == (2,)
    assert plan.controls[5] == (4,)
    assert plan.targets[5] == (0, 1)
    assert all(matrix is None for matrix in plan.matrices[1:])
    assert plan.gradient_kinds[6:].tolist() == [GRAD_PHASE] + [GRAD_PAULI_GENERATOR] * 3
    assert plan.parameter_indices[-1] == plan.parameter_indices[-2]


def test_compiler_refreshes_fixed_inverse_and_skips_barrier():
    circuit = dq.QubitCircuit(1)
    gate = dq.U3Gate(inputs=[0.3, 0.4, 0.5], nqubit=1, wires=0)
    original = gate.matrix.clone()
    circuit.add(gate.inverse())
    circuit.barrier()
    plan = compile_circuit(circuit)
    assert len(plan.opcodes) == 1
    torch.testing.assert_close(plan.matrices[0], original.mH)
