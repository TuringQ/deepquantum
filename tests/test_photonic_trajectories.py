"""Photon-count records paired with their native conditional states."""

import math

import pytest
import torch

import deepquantum as dq
import deepquantum.photonic as dqp


@pytest.mark.parametrize('backend', ['fock', 'bargmann'])
def test_unresolved_herald_records_and_weighted_inputs(backend):
    # Half a pair, half vacuum. After 50:50 taps the four probabilities are 5/8, 1/8, 1/8, 1/8.
    rho = torch.zeros(2, 16, 16, dtype=torch.cdouble)
    rho[0, 0, 0], rho[0, 3, 3], rho[1, 0, 0] = 0.2, 0.2, 0.7
    source = dq.FockState(rho.reshape(2, *([2] * 8)), basis=False, den_mat=True)
    cir = dq.QumodeCircuit(4, source, cutoff=2, backend=backend, basis=False, den_mat=True)
    cir.bs([2, 0], [math.pi / 4, 0.0])
    cir.bs([3, 1], [math.pi / 4, 0.0])
    cir.double()
    cir()
    torch.manual_seed(18)
    rec, retained, p = cir.measure(shots=32, wires=[1, 0], detector='threshold', with_state=True, with_prob=True)
    assert rec.shape == (2, 32, 2) and rec.dtype == torch.long and p.shape == (2, 32)
    assert not rec[1].any()
    assert len(rec[0].unique(dim=0)) == 4
    dm = retained.to_fock(2) if backend == 'bargmann' else retained.state
    for event in rec.flatten(0, 1).unique(dim=0):
        b, s = (rec == event).all(-1).nonzero(as_tuple=True)
        target, weight = cir.post_select([1, 0], event.tolist(), detector='threshold', normalize=False)
        target = target.to_fock(2) if backend == 'bargmann' else target.state
        torch.testing.assert_close(dm[b * 32 + s], target[b] / weight[b, None, None, None, None])
        prob = torch.where(b == 1, 1.0, 0.125 if event.any() else 0.625).to(p)
        torch.testing.assert_close(p[b, s], prob, atol=2e-7, rtol=0)
    if backend == 'bargmann':
        torch.manual_seed(18)
        shots = 8192
        counts = cir.measure(shots, wires=[0, 1], detector='threshold', with_prob=True)
        for event, (count, probability) in counts[0].items():
            target = 0.125 if event.state.any() else 0.625
            assert probability.item() == pytest.approx(target, abs=2e-7)
            assert abs(count - shots * target) < 5 * math.sqrt(shots * target * (1 - target))
        assert sum(v[0] for v in counts[0].values()) == shots
        assert counts[1][dq.FockState([0, 0])][0] == shots
    _, nothing, p_all = cir.measure(shots=3, detector='threshold', with_state=True, with_prob=True)
    assert nothing is None and p_all.shape == (2, 3)


@pytest.mark.parametrize('submodes', [False, True])
def test_sampled_threshold_memory_feeds_next_step(submodes):
    loop = dq.QumodeCircuit(1, 'vac', cutoff=2, backend='bargmann', den_mat=True, detector='threshold')
    loop.s(0, torch.tensor(0.55, dtype=torch.double), 0.0)
    loop.ps(0, encode=True)
    loop.delay(0, ntau=1, inputs=[0.65, 0.12])
    if submodes:
        loop.set_hardware_model(dqp.HardwareModel().s(0).submodes(relative_r=[1.0, 0.4]))
    loop.double()
    data = torch.tensor([[0.0], [0.32]], dtype=torch.double, requires_grad=True)
    loop(data)
    torch.manual_seed(9)
    shots = 12
    rec, mem, p = loop.measure(shots, with_state=True, with_prob=True)
    assert set(rec.flatten().tolist()) == {0, 1}
    torch.testing.assert_close(mem.norm(), torch.ones_like(p.flatten()))
    feedback = rec.reshape(-1, 1).to(data) * (math.pi / 2)
    loop(feedback, memory=mem)
    _, next_vac = loop.post_select(0, 0)
    full = loop.global_circuit(2)
    full(torch.cat((data.repeat_interleave(shots, dim=0), feedback), dim=-1))
    ports = loop.tdm_output_wires(2).flatten()
    joint = torch.stack([full.post_select(ports, [i, 0], normalize=False)[1] for i in (0, 1)])
    expected = joint[rec.flatten(), torch.arange(p.numel())]
    torch.testing.assert_close(next_vac * p.flatten(), expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad((next_vac * p.flatten()).sum(), data)[0],
        torch.autograd.grad(expected.sum(), data)[0],
        atol=2e-11,
        rtol=0,
    )
    if not submodes:
        shot = rec[0, :, 0].nonzero()[0, 0]
        phases = data.new_tensor([[0.0], [math.pi / 2]])
        loop(phases, memory=mem._select_batch(shot))
        _, alternative = loop.post_select(0, 0)
        full(torch.cat((torch.zeros_like(phases), phases), dim=1))
        torch.testing.assert_close(alternative * p[0, shot], full.post_select(ports, [1, 0], normalize=False)[1])
        torch.testing.assert_close(alternative, data.new_tensor([0.719704, 0.622951]), atol=1e-6, rtol=0)


@pytest.mark.parametrize('backend', ['bargmann', 'gaussian'])
def test_coherent_pnr_tail_matches_poisson(backend):
    cir = dq.QumodeCircuit(2, 'vac', backend=backend, cutoff=2)
    cir.d(1, torch.tensor(2.0, dtype=torch.double), 0.0)
    cir.double()
    cir(detector='threshold')
    torch.manual_seed(12)
    rec, state, p = cir.measure(40, wires=[1], detector='pnrd', with_state=True, with_prob=True)
    n = rec.flatten().to(p)
    assert n.max() > cir.cutoff
    torch.testing.assert_close(p.flatten(), (n * math.log(4.0) - 4.0 - torch.lgamma(n + 1)).exp())
    torch.testing.assert_close(state.norm(), torch.ones_like(n))


@pytest.mark.parametrize('detector', ['pnrd', 'threshold'])
def test_gaussian_count_instruments_return_the_same_native_branch(detector):
    r = torch.tensor(0.67, dtype=torch.double, requires_grad=True)
    cir = dq.QumodeCircuit(2, 'vac', backend='gaussian', cutoff=2)
    cir.s2([0, 1], r, 0.28)
    cir.double()
    cir(is_prob=True, detector='threshold')
    torch.manual_seed(8)
    records, sampled, probabilities = cir.measure(48, wires=[1], detector=detector, with_prob=True, with_state=True)
    assert isinstance(sampled, dq.BargmannState) and sampled.den_mat == (detector == 'threshold')
    q = r.tanh().square()
    assert records.max() >= (2 if detector == 'pnrd' else 1)
    for n in records.unique().tolist():
        rows = (records[0, :, 0] == n).nonzero().flatten()
        selected, p = cir.post_select(1, n, detector=detector, cutoff=1)
        assert isinstance(selected, dq.BargmannState) and selected.den_mat == sampled.den_mat
        draws = sampled._select_batch(rows)
        cutoff = max(n + 1, 4)
        tensor = draws.to_fock(cutoff)
        torch.testing.assert_close(tensor, selected.to_fock(cutoff).expand_as(tensor))
        expected = (1 - q) * q**n if detector == 'pnrd' or n == 0 else q
        torch.testing.assert_close(probabilities[0, rows], expected.expand(len(rows)))
        torch.testing.assert_close(p[0], expected)
        torch.testing.assert_close(selected.norm(), torch.ones_like(p))
        photons = p.new_tensor(n) if detector == 'pnrd' or not n else 1 / (1 - q)
        torch.testing.assert_close(selected.photon_number_mean_var()[0].squeeze(), photons)
        torch.testing.assert_close(
            torch.autograd.grad(probabilities[0, rows[0]], r, retain_graph=True),
            torch.autograd.grad(p, r, retain_graph=True),
        )


def test_gaussian_counting_preserves_thermal_derivative_at_vacuum():
    occupation = torch.tensor(0.0, dtype=torch.double, requires_grad=True)
    cov = torch.stack([1 + 2 * occupation, occupation.new_ones(())] * 2).diag()[None]
    cir = dq.QumodeCircuit(2, 'vac', backend='gaussian').double()
    cir(state=[cov, cov.new_zeros(1, 4, 1)])
    record, sampled, p = cir.measure(1, wires=0, with_state=True, with_prob=True)
    selected, weight = cir.post_select(0, 0)
    assert record.item() == 0 and selected.den_mat and sampled.den_mat
    for probability in [p, weight]:
        torch.testing.assert_close(
            torch.autograd.grad(probability, occupation, retain_graph=True)[0], -torch.ones_like(occupation)
        )


def test_gaussian_delay_count_memory_order_after_probability_table():
    cir = dq.QumodeCircuit(1, 'vac', backend='gaussian', cutoff=2, detector='threshold')
    cir.d(0, 0.5, 0.1)
    cir.delay(0, ntau=2, inputs=[0.6, 0.27])
    cir.double()
    mean = torch.tensor([0.4, 2.1, 0.3, -0.8], dtype=torch.double).reshape(1, 4, 1)
    memory = [torch.eye(4, dtype=torch.double)[None], mean]
    full = cir(memory=memory)
    # Coherent inputs stay a product. Conditioning either port outcome leaves these loop amplitudes.
    expected = cir.tdm_memory(full)
    table = cir(is_prob=True, memory=memory)
    position, momentum = expected[1][0, :, 0].chunk(2)
    occupations = (position.square() + momentum.square()) * dqp.kappa**2 / dqp.hbar
    for wire, photons in enumerate(occupations):
        vacuum = sum(p for event, p in table.items() if event.state[wire] == 0)
        torch.testing.assert_close(vacuum, (-photons).exp().reshape(1))
    torch.manual_seed(18)
    records, sampled, probs = cir.measure(12, with_prob=True, with_state=True)
    for n in records.unique().tolist():
        native, p = cir.post_select(0, n)
        weight = sum(mass for event, mass in table.items() if event.state[-1] == n)
        rows = (records[0, :, 0] == n).nonzero().flatten()
        torch.testing.assert_close(p, weight)
        torch.testing.assert_close(probs[0, rows], p.expand(len(rows)))
        torch.testing.assert_close(native.quadrature_moments(), tuple(expected))
        for actual, target in zip(sampled._select_batch(rows).quadrature_moments(), expected, strict=True):
            torch.testing.assert_close(actual, target.expand(len(rows), -1, -1))


def test_actual_herald_record_selects_homodyne_angle():
    cir = dq.QumodeCircuit(4, 'vac', backend='bargmann', den_mat=True)
    for signal, tap, r in [(2, 0, 0.55), (3, 1, 0.40)]:
        cir.s(signal, torch.tensor(r, dtype=torch.double), 0.0)
        cir.bs([signal, tap], [torch.tensor(0.45, dtype=torch.double), 0.0])
    cir.double()
    cir()
    torch.manual_seed(27)
    records, state, p = cir.measure(64, wires=[1, 0], detector='threshold', with_state=True, with_prob=True)
    adaptive, fixed = 0.0, 0.0
    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        event = records[0].new_tensor([i, j])
        branch, prob = cir.post_select([1, 0], event, detector='threshold')
        phi = [j * math.pi / 2, i * math.pi / 2]
        target = branch.quadrature_moments([0, 1], phi, powers=[(2, 2)])
        adaptive += (prob * target).item()
        fixed += (prob * branch.quadrature_moments([0, 1], 0.0, powers=[(2, 2)])).item()
        rows = (records[0] == event).all(-1).nonzero().flatten()
        assert len(rows) > 0
        observed = state._select_batch(rows).quadrature_moments([0, 1], phi, powers=[(2, 2)])
        torch.testing.assert_close(observed, target.expand_as(observed))
        torch.testing.assert_close(p[0, rows], prob.expand(len(rows)))
    assert adaptive == pytest.approx(0.525343, abs=1e-6)
    assert fixed == pytest.approx(0.254112, abs=1e-6)


def test_threshold_histogram_uses_sorted_labels_and_skips_zero_mass():
    cir = dq.QumodeCircuit(2, [0, 1], backend='bargmann', den_mat=True)
    cir()
    torch.manual_seed(71)
    records, state = cir.measure(7, wires=[1, 0], detector='threshold', with_state=True)
    assert state is None and records.tolist() == [[[1, 0]] * 7]
    assert cir.measure(0, detector='threshold') == {}
    histogram = cir.measure(7, wires=[1, 0], detector='threshold', with_prob=True)
    count, prob = histogram[dq.FockState([0, 1])]
    assert count == 7 and prob.item() == 1.0
    cir = dq.QumodeCircuit(1, torch.zeros(1, 2, 2, dtype=torch.cdouble), backend='bargmann', den_mat=True)
    cir()
    with pytest.raises(ValueError, match='zero-weight'):
        cir.measure(1, detector='threshold', with_state=True)


def test_two_port_records_pair_with_partial_outputs_and_next_control():
    loop = dq.QumodeCircuit(2, 'vac', backend='bargmann', den_mat=True, detector='threshold')
    loop.s(0, 0.52, 0.1)
    loop.d(1, 0.8, -0.2)
    loop.ps(0, encode=True)
    loop.bs([0, 1], [0.42, -0.11])
    loop.delay(0, ntau=2, inputs=[0.7, 0.23])
    loop.double()
    data = torch.tensor([[0.1], [0.57]], dtype=torch.double)
    loop(data)
    torch.manual_seed(25)
    partial, retained, prob = loop.measure(8, wires=[1], with_state=True, with_prob=True)
    assert retained.nmode == 3
    for event in [0, 1]:
        condition, p = loop.post_select(1, event)
        b, s = (partial[:, :, 0] == event).nonzero(as_tuple=True)
        if len(b):
            torch.testing.assert_close(prob[b, s], p[b])
            torch.testing.assert_close(
                retained._select_batch(b * 8 + s).to_fock(1), condition._select_batch(b).to_fock(1)
            )
    rec, mem, p = loop.measure(8, wires=[1, 0], with_state=True, with_prob=True)
    controls = (rec[:, :, 0] != rec[:, :, 1]).reshape(-1, 1).to(data) * (math.pi / 2)
    oracles = {}
    for key in rec.flatten(0, 1).unique(dim=0).tolist():
        oracles[tuple(key)] = loop.post_select([1, 0], key)
    loop(controls, memory=mem)
    q = loop.post_select([0, 1], [0, 0])[1]
    for b in range(2):
        for s in range(8):
            i = 8 * b + s
            branch, weight = oracles[tuple(rec[b, s].tolist())]
            torch.testing.assert_close(p[b, s], weight[b])
            loop(controls[i], memory=branch._select_batch(b))
            torch.testing.assert_close(q[i], loop.post_select([0, 1], [0, 0])[1][0])


@pytest.mark.parametrize('backend', ['bargmann', 'gaussian'])
def test_unresolved_submode_pnr_above_cutoff_matches_weighted_posterior(backend):
    cir = dq.QumodeCircuit(2, 'vac', backend=backend, cutoff=1, den_mat=True)
    cir.s2([0, 1], torch.tensor(0.55, dtype=torch.double), 0.0)
    cir.set_hardware_model(dqp.HardwareModel().s2([0, 1]).submodes(relative_r=[1.0, 0.75]))
    cir.double()
    cir()
    torch.manual_seed(6)
    rec, state, p = cir.measure(12, wires=[1], with_state=True, with_prob=True)
    assert rec.max() > 0 and state.nmode == 2
    for n in rec.unique().tolist():
        ref, weight = cir.post_select(1, n)
        rows = (rec[0, :, 0] == n).nonzero().flatten()
        selected = state._select_batch(rows)
        torch.testing.assert_close(p[0, rows], weight.expand(len(rows)))
        torch.testing.assert_close(selected.norm(), torch.ones_like(p[0, rows]))
        torch.testing.assert_close(selected.to_fock(2), ref.to_fock(2).expand(len(rows), -1, -1, -1, -1))


def test_bargmann_loop_promotes_fresh_vacuum_to_sampled_memory_precision():
    # The built-in vacuum has complex64 buffers. A double gate can produce complex128 conditional memory.
    loop = dq.QumodeCircuit(1, 'vac', backend='bargmann', den_mat=True)
    loop.s(0, torch.tensor(0.4, dtype=torch.double), 0.0)
    loop.delay(0, ntau=1, inputs=[0.6, 0.2])
    loop()
    memory, _ = loop.post_select(0, 1, detector='threshold')
    assert memory.mat_a.dtype == torch.cdouble
    next_state = loop(memory=memory)
    torch.testing.assert_close(next_state.norm(), torch.ones(1, dtype=torch.double))


def test_sampled_unresolved_single_photon_feeds_joint_loop_memory():
    loop = dq.QumodeCircuit(1, 'vac', backend='bargmann', cutoff=2, den_mat=True)
    loop.s(0, torch.tensor(0.2, dtype=torch.double), 0.12)
    loop.delay(0, ntau=1, inputs=[0.5, 0.1])
    loop.set_hardware_model(dqp.HardwareModel().s(0).submodes(relative_r=[1.0, 0.4]))
    loop.double()
    loop()
    torch.manual_seed(173)
    rec, memory, probability = loop.measure(512, with_state=True, with_prob=True)
    row = (rec[0, :, 0] == 1).nonzero()[0, 0]
    branch = memory._select_batch(row)
    # The unresolved single count correlates the two basis modes left in the loop.
    torch.testing.assert_close(branch.to_fock(1).real.reshape(()), probability.new_zeros(()), atol=1e-12, rtol=0)
    loop(memory=branch)
    _, p0 = loop.post_select(0, 0)
    full = loop.global_circuit(2)
    full()
    _, joint = full.post_select(loop.tdm_output_wires(2).flatten(), [1, 0], normalize=False)
    torch.testing.assert_close(probability[0, row] * p0, joint, atol=2e-13, rtol=0)
    torch.testing.assert_close(p0, p0.new_tensor([0.75739548]), atol=1e-7, rtol=0)
