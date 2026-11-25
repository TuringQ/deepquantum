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
    unrolled_data = None if data is None else data[:, torch.arange(nstep) % data.shape[1]].flatten(1)
    cov, mean = unrolled(unrolled_data)
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


@pytest.mark.parametrize('backend', ['gaussian', 'bargmann'])
def test_submode_loop_memory_order_matches_six_explicit_modes(backend):
    def prepare(nmode):
        cir = dq.QumodeCircuit(nmode, 'vac', backend='gaussian')
        if nmode >= 4:
            cir.s2([0, 1], 0.31, 0.2)  # correlated basis modes in the first loop position
            cir.d(2, 0.21, -0.4)  # distinguishes the second loop position
        if nmode != 4:
            cir.d(nmode - 2 if nmode == 6 else 0, 0.17, 0.3)  # fresh spatial input in basis mode 0
        return cir.double()()

    spatial, memory, concurrent = (prepare(n) for n in (1, 4, 6))
    cir = dq.QumodeCircuit(1, spatial, backend=backend, den_mat=True)
    cir.delay(0, ntau=2, inputs=[0.53, 0.16])
    cir.set_hardware_model(dqp.HardwareModel().input(0).submodes([1, 0]))
    cir.double()
    reference = dq.QumodeCircuit(6, concurrent, backend='gaussian')
    for k in (0, 1):
        reference.bs_theta([k, 4 + k], 0.53)
        reference.ps(k, 0.16)
    cov, mean = reference.double()()
    order = torch.tensor([2, 3, 0, 1, 4, 5])  # next-step order swaps the two loop positions
    idx = torch.cat([order, order + 6])
    expected = [cov[:, idx[:, None], idx], mean[:, idx]]
    if backend == 'bargmann':
        memory, concurrent = (dq.BargmannState(x, den_mat=True) for x in (memory, concurrent))
    for result in (cir(memory=memory), cir(state=concurrent)):
        assert (result.nmode, result.nsubmode, result.factorized) == (3, 2, False)
        joint = result.to_joint()
        moments = list(joint.quadrature_moments()) if backend == 'bargmann' else joint
        torch.testing.assert_close(moments, expected, atol=1e-12, rtol=0)
    # A terminal readout on spatial wire 0 observes the chosen basis mode at the output.
    torch.testing.assert_close(cir.quadrature_mean(0, phi=0, lo=[0, 1]), mean[:, [5]])
    with pytest.raises(ValueError, match='4 delay modes'):
        cir(memory=concurrent)


@pytest.mark.parametrize('detector', ['pnrd', 'threshold'])
def test_bargmann_submode_loop_keeps_joint_event_weight_and_correlations(detector):
    cir = dq.QumodeCircuit(1, 'vac', backend='bargmann', den_mat=True, detector=detector, cutoff=2)
    cir.s(0, encode=True)
    cir.delay(0, ntau=1, inputs=[0.5, 0.1])
    cir.set_hardware_model(dqp.HardwareModel().s(0).submodes(relative_r=[1, 0.4]))
    cir.double()
    full = cir.global_circuit(2)
    data = torch.tensor([[0.20, 0.12], [0.31, -0.17]], dtype=torch.double, requires_grad=True)

    result = cir(data)
    assert result.partition == ((0,), (1,))
    output_probability = cir.get_prob([1])
    memory, p1 = cir.post_select(0, 1, normalize=False)
    torch.testing.assert_close(p1, output_probability)
    assert memory.nmode == 2 and memory.den_mat
    if detector == 'pnrd':
        # The observed photon leaves odd parity in its partner loop mode. Both loops cannot be empty.
        torch.testing.assert_close(memory.to_fock(1).flatten(), torch.zeros_like(p1).to(torch.cdouble))
        for k in (0, 1):
            assert (memory.partial_trace([k]).to_fock(1).flatten().real > 0.1 * p1).all()

    assert cir(data, memory=memory).partition == ((0, 1),)
    final, p10, info = cir.post_select(0, 0, normalize=False, return_info=True)
    assert info['remaining_wires'] == (0,)
    torch.testing.assert_close(info['input_weight'], p1)
    torch.testing.assert_close(info['conditional_probability'], p10 / p1)
    full(data.repeat(1, 2))
    oracle, weight = full.post_select(cir.tdm_output_wires(2).flatten(), [1, 0], normalize=False)
    torch.testing.assert_close(p10, weight, atol=1e-12, rtol=0)
    torch.testing.assert_close(final.to_fock(3), oracle.to_fock(3), atol=1e-12, rtol=0)
    args = dict(wires=[0, 1], phi=[0, 0.3], powers=[(2, 2)], normalize=False)
    actual, expected = (s.quadrature_moments(**args) for s in (final, oracle))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), data), torch.autograd.grad(expected.sum(), data))


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

    # A partial trace of the ket returns a density operator. Source photons survive the default readout cutoff of 1.
    for window, times in zip(cir.tdm_substate([0, 2], window=2), [[0], [1, 2]], strict=True):
        assert isinstance(window, dq.BargmannState) and window.den_mat and window.cutoff == 1
        torch.testing.assert_close(window.to_fock(3), expected(times))
    torch.testing.assert_close(cir.tdm_substate([2, 0]).to_fock(3), expected([2, 0]))


def test_submode_loop_reprepares_photon_after_post_selection():
    coeff = torch.tensor([0.8, 0.6j], dtype=torch.cdouble)
    theta = torch.tensor(0.67, dtype=torch.double)
    cir = dq.QumodeCircuit(1, 'vac', backend='bargmann', den_mat=True, cutoff=1)
    cir.single_photon(0, coefficients=coeff)
    cir.delay(0, ntau=1, inputs=[theta, 0.22])
    cir.double()
    cir()
    memory, p1 = cir.post_select(0, 1, normalize=False)
    # Detect the first photon, then leave the entire second photon in the loop.
    cir(memory=memory)
    memory, p10 = cir.post_select(0, 0, normalize=False)
    torch.testing.assert_close(p1, theta.cos().square().reshape(1))
    torch.testing.assert_close(p10, p1 * theta.sin().square())
    ket = coeff.new_zeros(2, 2)
    ket[1, 0], ket[0, 1] = coeff
    torch.testing.assert_close(memory.to_fock(2)[0], p10 * torch.einsum('ij,kl->ijkl', ket, ket.conj()))


def test_tdm_submode_single_photon_coherences_and_source_gradient():
    cir = dq.QumodeCircuit(1, 'vac', backend='bargmann', cutoff=1)
    cir.single_photon(0, coefficients=torch.tensor([0.8, 0.6j], dtype=torch.cdouble), requires_grad=True)
    cir.delay(0, ntau=1, inputs=[0.67, 0.22])
    cir.double()
    result = cir.tdm_substate([1, 0])
    assert (result.nmode, result.nsubmode, result.den_mat, result.partition) == (2, 2, True, ((0, 1),))
    streamed = result.to_joint()

    # Two fresh photons in total. Cutoff 3 retains every Fock occupation, including bunching.
    full = cir.global_circuit(2)().to_joint()
    spatial = cir.tdm_output_wires(2).flip(0)
    keep = (2 * spatial + torch.arange(2)).flatten().tolist()
    reduced = full.partial_trace([w for w in range(full.nmode) if w not in keep])
    reference = reduced.reorder([sorted(keep).index(w) for w in keep])
    assert streamed.cutoff == 1
    torch.testing.assert_close(streamed.to_fock(3), reference.to_fock(3), atol=1e-12, rtol=0)
    source = cir.sources['0'].components
    gradients = [
        torch.autograd.grad(s.quadrature_moments([0], 0.2, powers=[(4,)]).sum(), source)[0]
        for s in (streamed, reference)
    ]
    assert gradients[0].abs().max() > 0.01
    torch.testing.assert_close(*gradients, atol=1e-10, rtol=0)


@pytest.mark.parametrize('backend', ['gaussian', 'bargmann'])
@pytest.mark.parametrize('feedback', [False, True], ids=['output-marginal', 'loop-memory'])
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS is not available')
def test_tdm_submode_mps_control_gradients_match_cpu(backend, feedback):
    def readout(device):
        cir = dq.QumodeCircuit(2, 'vac', backend=backend, den_mat=True)
        cir.s(0, encode=True)
        cir.s2([0, 1], encode=True)
        cir.d(1, encode=True)
        cir.d_channel(1, convention='loss_theta', encode=True)
        cir.bs([1, 0], encode=True)
        cir.ps(0, encode=True)
        cir.delay(0, ntau=1, convention='mzi', encode=True)
        cir.set_hardware_model(dqp.HardwareModel().s(0).submodes(relative_r=[1, 0.6]))
        cir.to(device)
        data = torch.linspace(0.07, 0.32, cir.ndata, device=device).reshape(1, -1).requires_grad_()
        if feedback:
            joint = cir(data).to_joint()
            if backend == 'bargmann':
                memory, _ = cir.post_select([0, 1], [1, 0], normalize=False)
            else:
                idx = [0, 1, 6, 7]  # first loop position, both basis modes in xxpp order
                memory = [joint[0][:, idx][:, :, idx], joint[1][:, idx]]
            joint = cir(data, memory=memory).to_joint()
        else:
            joint = cir.tdm_substate([1, 0], data=data).to_joint()
        cov, mean = joint.quadrature_moments() if backend == 'bargmann' else joint
        (grad,) = torch.autograd.grad(cov.square().sum() + mean.sum(), data)
        return tuple(x.detach().cpu() for x in (cov, mean, grad))

    torch.testing.assert_close(readout('mps'), readout('cpu'), atol=4e-5, rtol=3e-4)


@pytest.mark.parametrize('backend', ['gaussian', 'bargmann'])
@pytest.mark.parametrize('submodes', [False, True], ids=['ordinary', 'partial-basis-coupling'])
def test_tdm_batched_windows_share_controls_and_preserve_output_order(backend, submodes):
    cir = dq.QumodeCircuit(2, 'vac', backend=backend, den_mat=True)
    cir.thermal(1, nbar=0.1, requires_grad=True)
    cir.s(0, encode=True)
    cir.bs([0, 1], [0.61, 0.42])
    cir.delay(0, ntau=2, inputs=[0.51, 0.14])
    cir.delay(1, ntau=1, inputs=[0.64, -0.3], loop_gates=[dqp.Displacement([0.14, 0.2])])
    model = dqp.HardwareModel().output(1).loss(t=0.85)
    model.s().r.gaussian(sigma=0.025)
    d = 3 if submodes else 1
    if submodes:
        model.s(0).submodes(relative_r=[1, 0.4, 0.2])
        model.input(1).submodes(torch.tensor([0.8, 0, 0.6j], dtype=torch.cdouble))
        model.delay(1).d().r.gaussian(sigma=0.01)
    cir.set_hardware_model(model)
    cir.double()
    data = torch.tensor([[[0.23, 0.14], [0.34, -0.18]]], dtype=torch.double).expand(2, -1, -1).clone().requires_grad_()

    def moments(state):
        if submodes:
            assert isinstance(state, dqp.SubmodeResult)
            assert state.partition == ((0, 2), (1,))
            state = state.to_joint()
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
    order = (torch.tensor([3, 2, 1, 0])[:, None] * d + torch.arange(d)).flatten()
    idx = torch.cat([order, order + 4 * d])
    expected = [expected[0][:, idx[:, None], idx], expected[1][:, idx]]
    torch.testing.assert_close(tuple(actual), tuple(expected))
    full = cir.global_circuit(4).double()
    full.hardware_model.manual_seed(83)
    cov, mean = moments(full(data.repeat(1, 2, 1).flatten(1)))
    outputs = cir.tdm_output_wires(4)[2:, [1, 0]].reshape(-1, 1)
    modes = (outputs * d + torch.arange(d)).flatten()
    idx = torch.cat([modes, modes + full.nmode * d])
    reference = [cov[:, idx[:, None], idx], mean[:, idx]]
    torch.testing.assert_close(tuple(actual), tuple(reference))
    variables = (data, cir.sources['1'].amplitude)
    torch.testing.assert_close(
        torch.autograd.grad(actual[0].square().sum(), variables),
        torch.autograd.grad(reference[0].square().sum(), variables),
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


def test_equal_submode_windows_share_hardware_draws_and_preserve_gradients():
    ratios = [1.0, 0.6, 0.25]
    r = torch.tensor(0.28, dtype=torch.double, requires_grad=True)
    data = torch.tensor([[[-0.2], [0.3]], [[0.4], [0.1]]], dtype=torch.double, requires_grad=True)

    def build(ratio=None):
        cir = dq.QumodeCircuit(1, 'vac', backend='gaussian')
        cir.s(0, r if ratio is None else r * ratio, 0.23)
        cir.ps(0, encode=True)
        cir.delay(0, ntau=3, inputs=[0.62, -0.24])
        model = dqp.HardwareModel().ps(0).theta.gaussian(sigma=0.13)
        if ratio is None:
            model.s(0).submodes(relative_r=ratios)
        cir.set_hardware_model(model)
        model.manual_seed(174)
        cir.double()
        return next(cir.tdm_substate(5, window=3, data=data))

    result = build()
    assert result.partition == ((0,), (1,), (2,))
    total, reference = 0.0, 0.0
    for ((k,), state), ratio in zip(result._factors(), ratios, strict=True):
        expected = build(ratio)
        torch.testing.assert_close(state, expected, atol=2e-8, rtol=2e-7)
        total += (k + 1) * state[0].square().sum()
        reference += (k + 1) * expected[0].square().sum()
    torch.testing.assert_close(
        torch.autograd.grad(total, (r, data)), torch.autograd.grad(reference, (r, data)), atol=2e-7, rtol=2e-7
    )
