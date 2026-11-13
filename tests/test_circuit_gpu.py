"""Numerical comparisons for the optional single-GPU ixquantum executor."""

import math

import pytest
import torch

import deepquantum as dq

pytestmark = pytest.mark.skipif(
    not getattr(torch, 'corex', False) or not torch.cuda.is_available(),
    reason='requires CoreX PyTorch and an Iluvatar GPU',
)


@pytest.fixture
def gpu():
    circuits = []

    def create(nqubit, **kwargs):
        circuit = dq.QubitCircuitGPU(nqubit, **kwargs)
        circuits.append(circuit)
        return circuit

    yield create
    for circuit in circuits:
        circuit.close()


def add_gates(circuit):
    circuit.h(0)
    circuit.rx(1, inputs=0.23)
    circuit.ry(2, inputs=-0.41)
    circuit.rz(0, inputs=0.61)
    circuit.cnot(0, 3)
    circuit.cy(3, 1)
    circuit.cz(1, 2)
    circuit.cp(0, 2, inputs=0.37)
    circuit.swap([0, 3])
    circuit.iswap([1, 2])
    circuit.rzz([1, 3], inputs=0.21)
    circuit.observable([0, 2], 'xy')
    circuit.observable(3, 'z')


@pytest.mark.parametrize('random_initial', [False, True])
def test_forward_expectations(gpu, random_initial):
    generator = torch.Generator().manual_seed(123)
    initial = torch.randn(16, dtype=torch.complex64, generator=generator)
    initial /= initial.norm()
    initial = initial if random_initial else 'zeros'
    actual = gpu(4, init_state=initial)
    expected = dq.QubitCircuit(4, init_state=initial)
    add_gates(actual)
    add_gates(expected)
    expected.double()
    torch.testing.assert_close(actual().cpu().to(torch.complex128), expected(), atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(actual.expectation().cpu().double(), expected.expectation(), atol=3e-5, rtol=3e-4)
    external = actual.expectation_from_state(actual.state, list(actual.observables))
    torch.testing.assert_close(external, actual.expectation())


def add_trainable_gates(circuit):
    circuit.h(0)
    circuit.rx(1)
    circuit.ry(2, controls=0)
    circuit.rzz([0, 3])
    circuit.cp(3, 2)
    circuit.cnot(1, 3)
    circuit.swap([0, 2])
    circuit.observable([0, 1], 'yz')
    circuit.observable(2, 'x')


def test_gradients(gpu):
    actual = gpu(4, diff_method='adjoint')
    expected = dq.QubitCircuit(4)
    add_trainable_gates(actual)
    add_trainable_gates(expected)
    expected.double()
    values = torch.tensor([0.23, -0.41, 0.61, 0.37], device='cuda', requires_grad=True)
    weights = torch.tensor([0.3, 0.7], device='cuda')
    with torch.no_grad():
        for parameter, value in zip(expected.parameters(), values.cpu(), strict=True):
            parameter.fill_(value)
        for parameter, value in zip(actual.parameters(), values.detach(), strict=True):
            parameter.fill_(value)
    expected()
    reference = (expected.expectation() * weights.cpu()).sum()
    reference.backward()
    gradient = torch.cat([parameter.grad.reshape(-1) for parameter in expected.parameters()])
    value, derivative = actual.value_and_grad(values, weights)
    torch.testing.assert_close(value.cpu().double(), reference.detach(), atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(derivative.cpu().double(), gradient, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(actual.adjoint_vjp(values, weights), derivative, atol=1e-4, rtol=1e-3)
    (actual.expectation(parameters=values) * weights).sum().backward()
    torch.testing.assert_close(values.grad, derivative, atol=1e-4, rtol=1e-3)
    (actual.expectation() * weights).sum().backward()
    module_gradient = torch.cat([parameter.grad.reshape(-1) for parameter in actual.parameters()])
    torch.testing.assert_close(module_gradient.cpu().double(), gradient, atol=1e-4, rtol=1e-3)


def test_shared_inverse_parameters(gpu):
    actual = gpu(3, diff_method='adjoint')
    expected = dq.QubitCircuit(3)
    for circuit in (actual, expected):
        gate = dq.Ry(nqubit=3, wires=0, requires_grad=True)
        with torch.no_grad():
            gate.theta.fill_(0.37)
        circuit.add(gate)
        circuit.cnot(0, 1)
        circuit.add(gate, wires=2)
        circuit.add(gate.inverse(), wires=1)
        circuit.observable(2, 'z')
        circuit.observable([0, 1], 'xx')
    expected.double()
    expected()
    expected.expectation().sum().backward()
    actual.expectation().sum().backward()
    torch.testing.assert_close(
        next(actual.parameters()).grad.double(), next(expected.parameters()).grad, atol=1e-4, rtol=1e-3
    )
    assert len(actual._compiled_plan.parameter_sources) == 1


def test_multirz(gpu):
    circuit = gpu(5, diff_method='adjoint')
    circuit.h(0)
    circuit.multirz([0, 1, 3, 4])
    circuit.observable(0, 'x')
    value, gradient = circuit.value_and_grad(torch.tensor([0.37], device='cuda'), torch.ones(1, device='cuda'))
    assert abs(value.item() - math.cos(0.37)) < 3e-5
    assert abs(gradient.item() + math.sin(0.37)) < 1e-4


def test_five_target_permutation(gpu):
    generator = torch.Generator().manual_seed(91)
    initial = torch.randn(128, dtype=torch.complex64, generator=generator)
    initial /= initial.norm()
    permutation = torch.randperm(32, generator=generator)
    diagonal = torch.exp(1j * torch.randn(32, generator=generator))
    matrix = torch.diag(diagonal)[:, permutation]
    expected = dq.QubitCircuit(7, init_state=initial)
    expected.any(matrix, wires=[4, 0, 6, 1, 3], controls=[2])
    reference = expected()
    circuit = gpu(7, init_state=initial)
    circuit.any(matrix, wires=[4, 0, 6, 1, 3], controls=[2])
    torch.testing.assert_close(circuit().cpu(), reference, atol=2e-5, rtol=2e-4)


def test_sampling_order_and_reuse(gpu):
    circuit = gpu(4)
    circuit.x(0)
    circuit.x(2)
    circuit()
    assert circuit.measure(shots=7) == {'1010': 7}
    assert circuit.measure(shots=128) == {'1010': 128}
    assert circuit.measure(shots=17, wires=[3, 0]) == {'10': 17}
    circuit.close()
    circuit()
    assert circuit.measure(shots=9) == {'1010': 9}


def test_matrix_observable(gpu):
    circuit = gpu(2)
    circuit.h(0)
    circuit.observable_matrix(torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64), wires=0)
    circuit()
    assert abs(circuit.expectation().item() - 1) < 3e-5


@pytest.mark.parametrize('wires', [[1], [0, 2], [2, 0], [3, 0, 2]])
def test_matrix_observable_wire_order(gpu, wires):
    generator = torch.Generator().manual_seed(42)
    initial = torch.randn(16, dtype=torch.complex64, generator=generator)
    initial /= initial.norm()
    dimension = 2 ** len(wires)
    matrix = torch.randn(dimension, dimension, dtype=torch.complex64, generator=generator)
    matrix = (matrix + matrix.mH) / 2
    circuit = gpu(4, init_state=initial)
    circuit.observable_matrix(matrix, wires=wires)
    circuit()
    axes = wires + [wire for wire in range(4) if wire not in wires]
    state = initial.reshape((2,) * 4).permute(axes).reshape(dimension, -1)
    expected = (state.conj() * (matrix @ state)).sum().real
    torch.testing.assert_close(circuit.expectation().cpu().reshape(()), expected, atol=2e-5, rtol=2e-4)


def test_non_default_stream(gpu):
    with torch.cuda.stream(torch.cuda.Stream()):
        circuit = gpu(3, diff_method='adjoint')
        circuit.h(0)
        circuit.ry(1, controls=0)
        circuit.observable(1)
        value, gradient = circuit.value_and_grad(torch.tensor([0.37], device='cuda'), torch.ones(1, device='cuda'))
        assert abs(value.item() - (1 + math.cos(0.37)) / 2) < 3e-5
        assert abs(gradient.item() + math.sin(0.37) / 2) < 1e-4


@pytest.mark.parametrize('mutation', ['forward', 'add', 'compile', 'close', 'parameter'])
def test_stale_backward(gpu, mutation):
    circuit = gpu(2, diff_method='adjoint')
    circuit.rx(0)
    circuit.observable(0)
    parameter = torch.tensor([0.37], device='cuda', requires_grad=True)
    value = circuit.expectation(parameters=parameter)
    if mutation == 'forward':
        circuit()
    elif mutation == 'add':
        circuit.x(1)
    elif mutation == 'parameter':
        with torch.no_grad():
            parameter.add_(0.1)
    else:
        getattr(circuit, mutation)()
    with pytest.raises(RuntimeError):
        value.sum().backward()


def test_invalid_modes(gpu):
    with pytest.raises(ValueError, match='complex64'):
        gpu(2, dtype=torch.complex128)
    circuit = gpu(2)
    with pytest.raises(ValueError, match='runtime data'):
        circuit(torch.ones(1))
    circuit.rx(0, encode=True)
    with pytest.raises(ValueError, match='encoding'):
        circuit.compile()


def test_state_lifecycle(gpu):
    circuit = gpu(2)
    circuit.observable(0)
    with pytest.raises(RuntimeError, match='completed forward'):
        circuit.expectation()
    circuit()
    circuit.x(0)
    assert circuit.measure(10) is None
    with pytest.raises(RuntimeError, match='completed forward'):
        circuit.expectation()
    circuit()
    assert abs(circuit.expectation().item() + 1) < 3e-5


def test_non_default_stream_matrix_and_sampling(gpu):
    with torch.cuda.stream(torch.cuda.Stream()):
        circuit = gpu(3)
        circuit.x(0)
        circuit.swap([0, 2])
        circuit.observable_matrix(torch.diag(torch.tensor([1, -1], dtype=torch.complex64)), wires=2)
        circuit()
        assert abs(circuit.expectation().item() + 1) < 3e-5
        assert circuit.measure(31) == {'001': 31}


@pytest.mark.parametrize('gate_type', ['Rx', 'Ry', 'Rz', 'Rxx', 'Ryy', 'Rzz', 'Rxy'])
@pytest.mark.parametrize('inverse', [False, True])
def test_controlled_rotation_derivatives(gpu, gate_type, inverse):
    generator = torch.Generator().manual_seed(712)
    initial = torch.randn(16, dtype=torch.complex64, generator=generator)
    initial /= initial.norm()
    actual = gpu(4, init_state=initial, diff_method='adjoint')
    expected = dq.QubitCircuit(4, init_state=initial)
    wires = [2, 0] if len(gate_type) == 3 else [2]
    for circuit in (actual, expected):
        gate = getattr(dq, gate_type)(nqubit=4, wires=wires, controls=3, requires_grad=True)
        with torch.no_grad():
            gate.theta.fill_(0.37)
        circuit.add(gate.inverse() if inverse else gate)
        circuit.observable([0, 1], 'xy')
        circuit.observable([2, 3], 'zx')
    expected.double()
    expected()
    loss = expected.expectation().sum()
    loss.backward()
    value = actual.expectation().sum()
    value.backward()
    torch.testing.assert_close(value.cpu().double(), loss.detach(), atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(
        next(actual.parameters()).grad.double(), next(expected.parameters()).grad, atol=1e-4, rtol=1e-3
    )


def test_ghz_sampling_distribution(gpu):
    circuit = gpu(3)
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.cnot(1, 2)
    circuit()
    counts = circuit.measure(4096)
    assert set(counts) == {'000', '111'}
    assert sum(counts.values()) == 4096
    assert abs(counts['000'] / 4096 - 0.5) < 0.05


def test_cnot_with_additional_controls(gpu):
    initial = torch.randn(32, dtype=torch.complex64, generator=torch.Generator().manual_seed(91))
    initial /= initial.norm()
    actual = gpu(5, init_state=initial)
    expected = dq.QubitCircuit(5, init_state=initial)
    for circuit in (actual, expected):
        circuit.add(dq.CNOT(nqubit=5, wires=[1, 4]), wires=[1, 4], controls=[0, 3])
    torch.testing.assert_close(actual().cpu(), expected(), atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize('controls', [[], [1], [1, 3]])
@pytest.mark.parametrize('inverse', [False, True])
def test_phase_generator_gradients(gpu, controls, inverse):
    initial = torch.randn(16, dtype=torch.complex64, generator=torch.Generator().manual_seed(482))
    initial /= initial.norm()
    actual = gpu(4, init_state=initial, diff_method='adjoint')
    expected = dq.QubitCircuit(4, init_state=initial)
    for circuit in (actual, expected):
        gate = dq.PhaseShift(nqubit=4, wires=2, controls=controls, requires_grad=True)
        with torch.no_grad():
            gate.theta.fill_(0.37)
        circuit.add(gate.inverse() if inverse else gate)
        circuit.observable([0, 2], 'xy')
        circuit.observable([2, 3], 'yz')
    expected.double()
    expected()
    weights = torch.tensor([0.3, -0.7])
    reference = (expected.expectation() * weights).sum()
    reference.backward()
    value = (actual.expectation() * weights.cuda()).sum()
    value.backward()
    torch.testing.assert_close(value.cpu().double(), reference, atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(
        next(actual.parameters()).grad.double(), next(expected.parameters()).grad, atol=1e-4, rtol=1e-3
    )


@pytest.mark.parametrize('inverse', [False, True])
def test_fixed_gate_matrix_refresh(gpu, inverse):
    initial = torch.tensor([0.6, 0.8j], dtype=torch.complex64)
    actual = gpu(1, init_state=initial)
    expected = dq.QubitCircuit(1, init_state=initial)
    for circuit in (actual, expected):
        gate = dq.U3Gate(inputs=[0.3, 0.4, 0.5], nqubit=1, wires=0)
        circuit.add(gate.inverse() if inverse else gate)
    torch.testing.assert_close(actual().cpu(), expected(), atol=2e-5, rtol=2e-4)
    for circuit in (actual, expected):
        circuit.operators[0].theta.fill_(0.9)
    actual.invalidate()
    torch.testing.assert_close(actual().cpu(), expected(), atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize('assign', [False, True])
@pytest.mark.parametrize('trainable', [False, True])
def test_checkpoint_recompiles_nested_circuit(gpu, assign, trainable):
    circuit = gpu(1, diff_method='adjoint' if trainable else 'none')
    circuit.rx(0, inputs=None if trainable else 0.2)
    circuit.observable(0)
    circuit()
    parent = torch.nn.ModuleDict({'circuit': circuit})
    checkpoint = {key: value.clone() for key, value in parent.state_dict().items()}
    checkpoint['circuit.operators.0.theta'].fill_(1.0)
    parent.load_state_dict(checkpoint, assign=assign)
    circuit()
    value = circuit.expectation().sum()
    assert abs(value.item() - math.cos(1.0)) < 3e-5
    if trainable:
        value.backward()
        assert abs(circuit.operators[0].theta.grad.item() + math.sin(1.0)) < 1e-4


def test_checkpoint_invalidates_pending_backward(gpu):
    circuit = gpu(1, diff_method='adjoint')
    circuit.rx(0)
    circuit.observable(0)
    value = circuit.expectation().sum()
    circuit.load_state_dict(circuit.state_dict())
    with pytest.raises(RuntimeError, match='changed or closed'):
        value.backward()


def test_barrier_and_circuit_reset(gpu):
    circuit = gpu(2)
    circuit.h(0)
    circuit.barrier()
    circuit.cnot(0, 1)
    circuit.observable_matrix(torch.diag(torch.tensor([1.0, -1.0])), wires=0)
    state = circuit().cpu()
    torch.testing.assert_close(state.flatten(), torch.tensor([1, 0, 0, 1], dtype=torch.complex64) / math.sqrt(2))
    assert abs(circuit.expectation().item()) < 3e-5
    circuit.reset_circuit()
    circuit.x(1)
    circuit.observable(1)
    circuit()
    assert abs(circuit.expectation().item() + 1) < 3e-5


def test_unsupported_circuit_composition(gpu):
    circuit = gpu(2)
    circuit.multirz([0, 1])
    with pytest.raises(NotImplementedError, match='circuit addition'):
        circuit + gpu(2)
    with pytest.raises(NotImplementedError, match='circuit inverse'):
        circuit.inverse()


def test_fixed_execution_device(gpu):
    circuit = gpu(1)
    circuit.rx(0)
    parameter = circuit.operators[0].theta
    with pytest.raises(ValueError, match='execution device is fixed'):
        circuit.cpu()
    assert circuit.operators[0].theta is parameter
    circuit.to(circuit.gpu_device).double()
    assert circuit.operators[0].theta.dtype == torch.float64
    assert circuit().device == circuit.gpu_device
    if torch.cuda.device_count() > 1:
        other = (circuit.gpu_device.index + 1) % torch.cuda.device_count()
        with pytest.raises(ValueError, match='execution device is fixed'):
            circuit.cuda(other)
        assert circuit.operators[0].theta.device == circuit.gpu_device


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires two visible GPUs')
def test_native_calls_restore_device(gpu):
    original = torch.cuda.current_device()
    device = (original + 1) % torch.cuda.device_count()
    circuit = gpu(2, device=f'cuda:{device}')
    circuit.x(0)
    circuit.observable(0)
    circuit.compile()
    assert torch.cuda.current_device() == original
    circuit()
    assert torch.cuda.current_device() == original
    assert abs(circuit.expectation().item() + 1) < 3e-5
    assert torch.cuda.current_device() == original
    assert circuit.measure(8) == {'10': 8}
    assert torch.cuda.current_device() == original
    value = circuit.expectation_from_state(circuit.state, list(circuit.observables))
    assert abs(value.item() + 1) < 3e-5
    assert torch.cuda.current_device() == original
    with pytest.raises(ValueError, match='parameters must be'):
        circuit._executor.execute(torch.empty(0))
    assert torch.cuda.current_device() == original
    circuit.close()
    circuit.close()
    assert torch.cuda.current_device() == original
