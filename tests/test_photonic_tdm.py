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
    time_steps = torch.tensor([4, 1, 3])
    wires = torch.tensor([1, 0])

    actual_cov, actual_mean = circuit.tdm_substate(time_steps, wires)
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
    time_steps = torch.tensor([3, 1])
    wires = torch.tensor([1, 0])
    ntimes = 2
    data = torch.linspace(-0.3, 0.6, 2 * ntimes * circuit.ndata, dtype=torch.double).reshape(2, ntimes, -1)
    data.requires_grad_()
    state_before = {name: value.clone() for name, value in circuit.state_dict().items()}

    actual_cov, actual_mean = circuit.tdm_substate(time_steps, wires, data)
    for name, value in state_before.items():
        torch.testing.assert_close(circuit.state_dict()[name], value)
    global_circuit = circuit.global_circuit(nstep)
    global_circuit.to(torch.double)
    expected_cov, expected_mean = global_circuit(data[:, torch.arange(nstep) % ntimes].flatten(1))
    output_wires = circuit.tdm_output_wires(nstep)[time_steps][:, wires].reshape(-1)
    indices = torch.cat([output_wires, output_wires + global_circuit.nmode])

    torch.testing.assert_close(actual_cov, expected_cov[:, indices[:, None], indices])
    torch.testing.assert_close(actual_mean, expected_mean[:, indices])
    actual_grad = torch.autograd.grad(actual_cov.square().sum(), data)[0]
    expected_grad = torch.autograd.grad(expected_cov[:, indices[:, None], indices].square().sum(), data)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    single_cov, single_mean = circuit.tdm_substate(time_steps, wires, data[0])
    torch.testing.assert_close(single_cov, actual_cov[:1])
    torch.testing.assert_close(single_mean, actual_mean[:1])


@pytest.mark.parametrize(
    'encoded,hardware,time_steps',
    [(False, False, slice(5)), (False, True, [2, 4]), (True, False, [2, 4]), (True, True, slice(5))],
)
def test_tdm_substate_windows_and_gradients_match_global_circuit(encoded, hardware, time_steps):
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
        data = torch.linspace(0.1, 0.6, 2 * 2 * cir.ndata, dtype=torch.double).reshape(2, 2, cir.ndata)
        data.requires_grad_()
    initial = {name: value.clone() for name, value in cir.state_dict().items()}
    outputs = []
    for output in cir.tdm_substate(time_steps, data=data, wires=[1, 0], window=2):
        outputs.append(output)
        for name, value in cir.state_dict().items():
            torch.testing.assert_close(value, initial[name])
    unrolled = cir.global_circuit(nstep)
    unrolled.double()
    expanded_data = None if data is None else data[:, torch.arange(nstep) % data.shape[1]].flatten(1)
    cov, mean = unrolled(expanded_data)
    references = []
    selected = list(range(nstep))[time_steps] if isinstance(time_steps, slice) else time_steps
    for step, output in zip(selected, outputs, strict=True):
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


def test_tdm_substate_windows_pause_close_and_restart():
    cir = dq.QumodeCircuit(1, 'vac', backend='gaussian')
    cir.s(0, encode=True)
    cir.delay(0, ntau=2, inputs=[0.4, 0.2])
    cir.double()
    data = torch.tensor([[0.2, 0.1], [0.3, -0.2], [0.1, 0.4]], dtype=torch.double)
    stream = cir.tdm_substate(slice(3), data=data, window=1)
    first = next(stream)
    torch.testing.assert_close(first, cir.tdm_substate(0, data=data))
    second = next(stream)
    torch.testing.assert_close(second, cir.tdm_substate([1], data=data))
    stream.close()
    torch.testing.assert_close(next(cir.tdm_substate(slice(3), data=data, window=1)), first)
    # A scalar time selects one window, rather than the number of iteration steps.
    windows = list(cir.tdm_substate(1, data=data, window=3))
    assert len(windows) == 1
    torch.testing.assert_close(windows[0], cir.tdm_substate([0, 1], data=data))


def test_tdm_substate_windows_hardware_noise_matches_joint_state():
    cir = dq.QumodeCircuit(1, 'vac', backend='gaussian')
    cir.s(0, 0.25)
    cir.ps(0, encode=True)
    cir.delay(0, ntau=2, inputs=[0.4, 0.2])
    model = dqp.HardwareModel()
    model.ps().theta.gaussian(sigma=0.04)
    cir.set_hardware_model(model)
    cir.double()
    data = torch.linspace(0.1, 0.7, 4, dtype=torch.double).reshape(2, 2, 1)
    model.manual_seed(21)
    output = next(cir.tdm_substate(4, data=data, window=2))
    model.manual_seed(21)
    expected = cir.tdm_substate(slice(3, 5), data=data)
    torch.testing.assert_close(output, expected)


def test_tdm_selected_times_and_periodic_data_validation():
    cir = dq.QumodeCircuit(1, 'vac', backend='gaussian')
    cir.s(0, encode=True)
    cir.delay(0, ntau=2, inputs=[0.4, 0.2])
    data = torch.tensor([[0.2, 0.1], [0.3, -0.2]])
    for time_steps in ([], [-1], [0, 0], [0.5], True, slice(None), slice(0, 3, 0)):
        with pytest.raises(ValueError, match='time_steps'):
            cir.tdm_substate(time_steps, data=data)
    for invalid in (data.flatten(), data[:0], data[:, :1]):
        with pytest.raises(ValueError, match='data'):
            cir.tdm_substate([0, 3], data=invalid)
    for wires in ([], -1, [0, 0], [0.5], True, [[0]]):
        with pytest.raises(ValueError, match='wires'):
            cir.tdm_substate(0, wires=wires, data=data)
    with pytest.raises(ValueError, match='ascending'):
        cir.tdm_substate([2, 1], data=data, window=2)
    for window in (0, True, 1.5):
        with pytest.raises(ValueError, match='window'):
            cir.tdm_substate(slice(4), data=data, window=window)


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
        lambda: circuit.tdm_substate([1, 2]),
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


@pytest.mark.parametrize('backend', ['gaussian', 'bargmann'])
def test_loop_forward_keeps_encoded_controls_when_repeated_or_unrolled(backend):
    cir = dq.QumodeCircuit(1, 'vac', backend=backend)
    cir.s(0, r=0.0, theta=0.0, encode=True)
    cir.delay(0, ntau=1, inputs=[0.35, 0.1])
    cir.double()
    data = torch.tensor([0.4, 0.3], dtype=torch.double, requires_grad=True)

    def covariance(circuit, controls=None):
        output = circuit(controls)
        return output.quadrature_moments()[0] if backend == 'bargmann' else output[0]

    encoded = covariance(cir, data)
    torch.testing.assert_close(covariance(cir), encoded)
    unrolled = covariance(cir.global_circuit(1))
    torch.testing.assert_close(unrolled, encoded)
    torch.testing.assert_close(
        torch.autograd.grad(unrolled.square().sum(), data, retain_graph=True)[0],
        torch.autograd.grad(encoded.square().sum(), data)[0],
    )


def test_gaussian_loop_accepts_homodyne_memory_in_next_step_order():
    cir = dq.QumodeCircuit(2, 'vac', backend='gaussian')
    cir.thermal(1, nbar=0.2)
    cir.s(0, r=0.45, theta=0.12)
    cir.bs([0, 1], [0.35, 0.24])
    cir.delay(0, ntau=2, inputs=[0.57, 0.2])
    cir.delay(1, ntau=1, inputs=[0.31, -0.18])
    cir.double()
    full = cir.global_circuit(3).double()
    output = cir.tdm_output_wires(3)
    records = torch.tensor([[0.3, -0.2], [-0.4, 0.1], [0.2, 0.5]], dtype=torch.double, requires_grad=True)
    measure = dqp.Generaldyne(nmode=5, wires=output[0].tolist())
    loop = torch.tensor([0, 1, 3, 5, 6, 8])  # three delay modes in xxpp order
    memory = None
    for i, record in enumerate(records):
        cov, mean = measure(cir(memory=memory), samples=record)
        memory = [cov[:, loop[:, None], loop], mean[:, loop]]
        if i == 0:
            memory = dq.GaussianState(memory)  # both native input forms are accepted
    # Condition the global state in one measurement on all six outcomes; it never rotates its fixed delay wires.
    joint_measure = dqp.Generaldyne(nmode=full.nmode, wires=output.flatten().tolist())
    cov, mean = joint_measure(full(), samples=records.flatten())
    loop_wires = torch.tensor([1, 0, 3])  # the two-bin loop swaps mode order after three steps
    idx = torch.cat([loop_wires, loop_wires + full.nmode])
    expected = [cov[:, idx[:, None], idx], mean[:, idx]]
    torch.testing.assert_close(memory, expected)
    torch.testing.assert_close(
        torch.autograd.grad(memory[1].sum(), records)[0], torch.autograd.grad(expected[1].sum(), records)[0]
    )


def test_bargmann_single_photon_windows_match_global_density():
    cir = dq.QumodeCircuit(1, 'vac', backend='bargmann', cutoff=1)
    cir.single_photon(0)
    cir.delay(0, ntau=1, inputs=[0.7, 0.3])
    cir.double()
    full = cir.global_circuit(3)
    evolved = full()
    outputs = cir.tdm_output_wires(3)[:, 0].tolist()

    def expected(times):
        selected = [outputs[t] for t in times]
        marginal = evolved.partial_trace([w for w in range(full.nmode) if w not in selected])
        order = sorted(selected)
        return marginal.reorder([order.index(w) for w in selected]).to_fock(3)

    # Even a ket circuit returns a reduced density matrix. Cutoff=1 must not remove source photons.
    for window, times in zip(cir.tdm_substate([0, 2], window=2), [[0], [1, 2]], strict=True):
        assert isinstance(window, dq.BargmannState) and window.den_mat and window.cutoff == 1
        torch.testing.assert_close(window.to_fock(3), expected(times))
    torch.testing.assert_close(cir.tdm_substate([2, 0]).to_fock(3), expected([2, 0]))


@pytest.mark.parametrize('backend', ['gaussian', 'bargmann'])
def test_tdm_batched_windows_share_controls_and_preserve_output_order(backend):
    cir = dq.QumodeCircuit(2, 'vac', backend=backend, den_mat=True)
    cir.thermal(1, nbar=0.1, requires_grad=True)
    cir.s(0, encode=True)
    cir.bs([0, 1], [0.61, 0.42])
    cir.delay(0, ntau=2, inputs=[0.51, 0.14])
    cir.delay(1, ntau=1, inputs=[0.64, -0.3])
    model = dqp.HardwareModel().output(1).loss(t=0.85)
    model.s().r.gaussian(sigma=0.025)
    cir.set_hardware_model(model)
    cir.double()
    data = torch.tensor([[[0.23, 0.14], [0.34, -0.18]]], dtype=torch.double).expand(2, -1, -1).clone().requires_grad_()

    def moments(state):
        return state.quadrature_moments() if isinstance(state, dq.BargmannState) else state

    params = {k: v.clone() for k, v in cir.state_dict().items()}
    model.manual_seed(83)
    stream = cir.tdm_substate([0, 3], wires=[1, 0], data=data, window=2)
    next(stream)
    for k, v in cir.state_dict().items():
        torch.testing.assert_close(v, params[k])
    actual = moments(next(stream))
    for tensor in actual:
        torch.testing.assert_close(tensor[0], tensor[1])  # hardware sample is shared across the encoded batch
    model.manual_seed(83)
    expected = moments(cir.tdm_substate([3, 2], wires=[0, 1], data=data))
    idx = torch.tensor([3, 2, 1, 0, 7, 6, 5, 4])
    expected = [expected[0][:, idx[:, None], idx], expected[1][:, idx]]
    torch.testing.assert_close(tuple(actual), tuple(expected))
    full = cir.global_circuit(4).double()
    full.hardware_model.manual_seed(83)
    global_state = full(data.repeat(1, 2, 1).flatten(1))
    outputs = cir.tdm_output_wires(4)[2:, [1, 0]].flatten()
    if backend == 'bargmann':
        reference = global_state.quadrature_moments(outputs.tolist())
    else:
        idx = torch.cat([outputs, outputs + full.nmode])
        reference = [global_state[0][:, idx[:, None], idx], global_state[1][:, idx]]
    torch.testing.assert_close(tuple(actual), tuple(reference))
    variables = (data, cir.sources['1'].amplitude)
    torch.testing.assert_close(
        torch.autograd.grad(actual[0].square().sum(), variables),
        torch.autograd.grad(expected[0].square().sum(), variables),
    )


@pytest.mark.parametrize('backend', ['gaussian', 'bargmann'])
def test_loop_memory_validation(backend):
    cir = dq.QumodeCircuit(1, 'vac', backend=backend)
    vac = cir.init_state
    with pytest.raises(ValueError, match='delay loops'):
        cir(memory=vac)
    cir.delay(0, ntau=2, inputs=[0.5, 0.1])
    with pytest.raises(ValueError, match='either state or memory'):
        cir(state=vac, memory=vac)
    with pytest.raises(ValueError, match='2 delay modes'):
        cir(memory=vac)
    with pytest.raises(ValueError, match='2 delay modes'):
        cir(memory=cir())  # a joint state still contains the spatial output


@pytest.mark.parametrize('backend', ['fock', 'bosonic'])
def test_other_loop_backends_reject_explicit_memory(backend):
    cir = dq.QumodeCircuit(1, 'vac', backend=backend)
    cir.delay(0, ntau=1, inputs=[0.5, 0.1])
    with pytest.raises(ValueError, match='Gaussian or Bargmann'):
        cir(memory=cir.init_state)
    with pytest.raises(ValueError, match='Gaussian or Bargmann'):
        cir.tdm_substate(0)


@pytest.mark.parametrize('backend', ['gaussian', 'bargmann'])
def test_loop_readout_starts_from_vacuum_independently_of_forward(backend):
    spatial = [torch.eye(2, dtype=torch.double), torch.tensor([[0.6], [0.2]], dtype=torch.double)]
    cir = dq.QumodeCircuit(1, spatial, backend=backend)
    cir.delay(0, ntau=2, inputs=[0.5, 0.1])
    cir.double()
    if backend == 'gaussian':
        vacuum = dq.GaussianState(nmode=2).double()
        joint_mean = cir()[1]
        torch.testing.assert_close(cir(memory=vacuum)[1], joint_mean)
        output_mean = cir.tdm_substate(0)[1]
    else:
        vacuum = dq.BargmannState('vac', nmode=2, cutoff=2).double()
        joint_mean = cir().quadrature_moments()[1]
        joint = cir(memory=vacuum)
        assert joint.cutoff == cir.cutoff != vacuum.cutoff
        torch.testing.assert_close(joint.quadrature_moments()[1], joint_mean)
        output_mean = cir.tdm_substate(0).quadrature_moments()[1]
    assert joint_mean.abs().max() > 0.1
    torch.testing.assert_close(output_mean, torch.zeros_like(output_mean))


def test_automatic_homodyne_loop_reprepares_gaussian_source():
    auto = dq.QumodeCircuitTDM(1, 'vac', backend='gaussian')
    manual = dq.QumodeCircuit(1, 'vac', backend='gaussian')
    for cir in (auto, manual):
        cir.thermal(0, nbar=0.6)
        cir.delay(0, ntau=2, inputs=[0.65, 0.21])
        cir.homodyne_x(0)
        cir.double()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        expected = auto(nstep=3)
        torch.manual_seed(91)
        memory, records = None, []
        for _ in range(3):
            manual(memory=memory)
            records.append(manual.measure_homodyne(shots=1))
            cov, mean = manual.state_measured
            idx = [0, 1, 3, 4]
            memory = [cov[:, idx, :][:, :, idx], mean[:, idx]]
    torch.testing.assert_close(manual.state_measured, expected)
    torch.testing.assert_close(torch.stack(records, -1), auto.samples)
