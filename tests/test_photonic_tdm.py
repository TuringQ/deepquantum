from copy import deepcopy

import pytest
import torch

import deepquantum as dq
import deepquantum.photonic as dqp


def test_tdm_output_wires():
    circuit = dq.QumodeCircuit(2, init_state='vac', backend='gaussian')
    circuit.delay(0, ntau=2, inputs=[0.3, 0.1], convention='mzi')
    circuit.delay(1, ntau=3, inputs=[0.4, -0.2], convention='mzi')

    output_wires = circuit.tdm_output_wires(3)

    assert torch.equal(output_wires, torch.tensor([[2, 6], [7, 8], [9, 10]]))


def test_tdm_substate_matches_global_circuit():
    circuit = dq.QumodeCircuit(2, init_state='vac', backend='gaussian')
    circuit.s(0, r=0.2)
    circuit.bs([0, 1], [0.3, 0.1])
    circuit.delay(0, ntau=2, inputs=[0.4, 0.2], convention='mzi')
    circuit.delay(1, ntau=3, inputs=[0.5, -0.1], convention='mzi')
    circuit.loss_db(0, 0.2)
    circuit.to(torch.double)
    nstep = 5
    time_steps = torch.tensor([1, 3, 4])
    wires = torch.tensor([1, 0])

    actual_cov, actual_mean = circuit.tdm_substate(nstep, time_steps, wires)
    global_circuit = circuit.global_circuit(nstep)
    global_circuit.to(torch.double)
    expected_cov, expected_mean = global_circuit()
    output_wires = circuit.tdm_output_wires(nstep)[time_steps][:, wires].reshape(-1)
    indices = torch.cat([output_wires, output_wires + global_circuit.nmode])

    torch.testing.assert_close(actual_cov, expected_cov[:, indices[:, None], indices])
    torch.testing.assert_close(actual_mean, expected_mean[:, indices])


def test_encoded_tdm_substate_matches_global_circuit():
    circuit = dq.QumodeCircuit(2, init_state='vac', backend='gaussian')
    circuit.s(0, encode=True)
    circuit.bs([0, 1], encode=True)
    circuit.delay(0, ntau=2, convention='mzi', encode=True)
    circuit.delay(1, ntau=3, inputs=[0.5, -0.1], convention='mzi')
    circuit.to(torch.double)
    nstep = 4
    time_steps = torch.tensor([1, 3])
    wires = torch.tensor([1, 0])
    data = torch.linspace(-0.3, 0.6, 2 * nstep * circuit.ndata, dtype=torch.double).reshape(2, -1)
    data.requires_grad_()
    state_before = {name: value.clone() for name, value in circuit.state_dict().items()}

    actual_cov, actual_mean = circuit.tdm_substate(nstep, time_steps, wires, data)
    for name, value in state_before.items():
        torch.testing.assert_close(circuit.state_dict()[name], value)
    global_circuit = circuit.global_circuit(nstep)
    global_circuit.to(torch.double)
    expected_cov, expected_mean = global_circuit(data)
    output_wires = circuit.tdm_output_wires(nstep)[time_steps][:, wires].reshape(-1)
    indices = torch.cat([output_wires, output_wires + global_circuit.nmode])

    torch.testing.assert_close(actual_cov, expected_cov[:, indices[:, None], indices])
    torch.testing.assert_close(actual_mean, expected_mean[:, indices])
    actual_grad = torch.autograd.grad(actual_cov.square().sum(), data)[0]
    expected_grad = torch.autograd.grad(expected_cov[:, indices[:, None], indices].square().sum(), data)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    single_cov, single_mean = circuit.tdm_substate(nstep, time_steps, wires, data[0])
    torch.testing.assert_close(single_cov, actual_cov[:1])
    torch.testing.assert_close(single_mean, actual_mean[:1])


@pytest.mark.parametrize('encoded', [False, True])
@pytest.mark.parametrize('hardware', [False, True])
def test_tdm_stream_windows_and_gradients_match_global_circuit(encoded, hardware):
    cir = dq.QumodeCircuit(2, 'vac', backend='gaussian')
    strength = torch.tensor(0.24, dtype=torch.double, requires_grad=True)
    cir.s(0, r=strength)
    cir.d(1, 0.1, 0.3)
    cir.bs([0, 1], [0.4, 0.1])
    cir.delay(0, ntau=2, inputs=[0.3, -0.2], encode=encoded)
    cir.delay(1, ntau=3, inputs=[0.6, 0.2])
    if hardware:
        model = dqp.HardwareModel()
        model.input(0).loss(t=0.91)
        model.bs().theta.gaussian(mu=0.02, sigma=0.0)
        model.output(1).loss(t=0.84)
        cir.set_hardware_model(model)
    cir.double()
    nstep = 5
    data = None
    if encoded:
        data = torch.linspace(0.1, 0.6, 2 * nstep * cir.ndata, dtype=torch.double).reshape(2, nstep, cir.ndata)
        data.requires_grad_()
    initial = {name: value.clone() for name, value in cir.state_dict().items()}
    outputs = []
    for output in cir.tdm_stream(nstep, data=data, wires=[1, 0], window=2):
        outputs.append(output)
        for name, value in cir.state_dict().items():
            torch.testing.assert_close(value, initial[name])
    unrolled = cir.global_circuit(nstep)
    unrolled.double()
    cov, mean = unrolled(None if data is None else data.flatten(1))
    references = []
    for step, output in enumerate(outputs):
        wires = cir.tdm_output_wires(nstep)[max(0, step - 1) : step + 1, [1, 0]].flatten()
        indices = torch.cat([wires, wires + unrolled.nmode])
        expected = [cov[:, indices[:, None], indices], mean[:, indices]]
        torch.testing.assert_close(output, expected)
        references.append(expected)
    variables = (strength,) if data is None else (strength, data)
    actual_grad = torch.autograd.grad(sum(value.square().sum() for output in outputs for value in output), variables)
    expected_grad = torch.autograd.grad(
        sum(value.square().sum() for output in references for value in output), variables
    )
    torch.testing.assert_close(actual_grad, expected_grad)


def test_tdm_stream_pause_close_and_restart():
    cir = dq.QumodeCircuit(1, 'vac', backend='gaussian')
    cir.s(0, encode=True)
    cir.delay(0, ntau=2, inputs=[0.4, 0.2])
    cir.double()
    data = torch.tensor([[0.2, 0.1], [0.3, -0.2], [0.1, 0.4]], dtype=torch.double)
    stream = cir.tdm_stream(3, data)
    first = next(stream)
    torch.testing.assert_close(first, cir.tdm_substate(1, [0], data=data[:1].flatten()))
    second = next(stream)
    torch.testing.assert_close(second, cir.tdm_substate(2, [1], data=data[:2].flatten()))
    stream.close()
    torch.testing.assert_close(next(cir.tdm_stream(3, data)), first)


def test_tdm_stream_hardware_noise_matches_substate():
    cir = dq.QumodeCircuit(1, 'vac', backend='gaussian')
    cir.s(0, 0.25)
    cir.ps(0, encode=True)
    cir.delay(0, ntau=2, inputs=[0.4, 0.2])
    model = dqp.HardwareModel()
    model.ps().theta.gaussian(sigma=0.04)
    cir.set_hardware_model(model)
    cir.double()
    data = torch.linspace(0.1, 0.7, 8, dtype=torch.double).reshape(2, 4, 1)
    model.manual_seed(21)
    output = list(cir.tdm_stream(4, data, window=2))[-1]
    model.manual_seed(21)
    expected = cir.tdm_substate(4, [2, 3], data=data.flatten(1))
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize('query', ['get_unitary', 'get_symplectic', 'get_displacement'])
def test_tdm_queries_refresh_after_edits(query):
    circuit = dq.QumodeCircuit(2, init_state='vac', backend='gaussian')
    circuit.ps(0, 0.2)

    def check():
        circuit.to(torch.double)
        nmode = circuit.nmode + sum(op.ntau for op in circuit.operators if isinstance(op, dqp.DelayBS))
        args = (torch.arange(2 * nmode, dtype=torch.double).reshape(-1, 1),) if query == 'get_displacement' else ()
        previous = circuit._tdm_gate_map
        actual = getattr(circuit, query)(*args)
        assert circuit._tdm_gate_map is not previous
        ref = deepcopy(circuit)
        ref._tdm_structure = None
        expected = getattr(ref.global_circuit(1), query)(*args)
        torch.testing.assert_close(actual, expected)

    check()
    delay = dqp.DelayBS([0.3, 0.1], ntau=2, nmode=2, wires=0)
    circuit.operators.append(delay)
    check()
    circuit.operators.insert(0, dqp.PhaseShift(0.4, nmode=2, wires=1))
    delay.ntau, delay.wires = 3, [1]
    delay.gates.append(dqp.PhaseShift(0.5))
    check()
    circuit.operators[0] = dqp.PhaseShift(0.6, nmode=2, wires=0)
    check()
    delay.gates[1] = dqp.PhaseShift(0.7)
    check()
    circuit.operators[0], circuit.operators[1] = circuit.operators[1], circuit.operators[0]
    check()
    del circuit.operators[-1]
    check()


@pytest.mark.parametrize('hardware', [False, True])
def test_tdm_entries_reuse_mapping(hardware):
    circuit = dq.QumodeCircuit(1, init_state='vac', backend='gaussian')
    circuit.s(0, r=0.2)
    circuit.delay(0, ntau=2, inputs=[0.3, 0.1])
    circuit.homodyne(0, phi=0.2)
    circuit.to(torch.double)
    if hardware:
        circuit.set_hardware_model(dqp.HardwareModel().gaussian(sigma=0.01))
    circuit()
    mapping = circuit._tdm_gate_map
    for call in [
        circuit.get_symplectic,
        lambda: circuit.get_displacement(torch.zeros(6, 1, dtype=torch.double)),
        lambda: circuit.tdm_output_wires(3),
        lambda: circuit.tdm_substate(3, [1, 2]),
        lambda: circuit.global_circuit(3),
        lambda: circuit.measure_homodyne(shots=2),
    ]:
        call()
        assert circuit._tdm_gate_map is mapping


def test_tdm_cache_keeps_live_parameters_and_refreshes_measurements():
    circuit = dq.QumodeCircuit(2, init_state='vac', backend='gaussian')
    circuit.s(0, encode=True)
    circuit.delay(0, ntau=2, inputs=[0.3, 0.1])
    circuit.homodyne(0, phi=0.2)
    circuit._prepare_unroll_dict()
    state_keys = tuple(circuit.state_dict())
    for dtype in (torch.float, torch.double):
        mapping = circuit._tdm_gate_map
        circuit.to(dtype)
        data = torch.tensor([0.3, 0.2], dtype=dtype, requires_grad=True)
        actual = circuit(data)
        actual_grad = torch.autograd.grad(actual[0].square().sum(), data)[0]
        assert circuit._tdm_gate_map is mapping
        circuit._tdm_structure = None
        expected = circuit(data)
        expected_grad = torch.autograd.grad(expected[0].square().sum(), data)[0]
        for left, right in zip(actual + [actual_grad], expected + [expected_grad], strict=True):
            torch.testing.assert_close(left, right)
    assert tuple(circuit.state_dict()) == state_keys

    measurement = circuit.measurements[0]
    measurement.wires = [1]
    circuit._prepare_unroll_dict()
    assert circuit._unroll_measurements()[0].wires == [3]
    replacement = dqp.Homodyne(phi=torch.tensor(0.4), nmode=2, wires=1)
    circuit.measurements[0] = replacement
    circuit._prepare_unroll_dict()
    assert circuit._unroll_measurements()[0].phi is replacement.phi
    circuit.homodyne(0, phi=0.1)
    circuit._prepare_unroll_dict()
    assert [op.wires for op in circuit._unroll_measurements()] == [[3], [2]]
    circuit.operators[1].ntau, replacement.wires = 3, [2]
    with pytest.raises(IndexError):
        circuit._prepare_unroll_dict()
    circuit.operators[1].ntau, replacement.wires = 2, [1]
    circuit._prepare_unroll_dict()
    assert circuit._nmode_tdm == 4
    assert [op.wires for op in circuit._unroll_measurements()] == [[3], [2]]
