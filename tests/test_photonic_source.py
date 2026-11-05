import math

import pytest
import torch

import deepquantum.photonic as dqp


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _run_joint(circuit, data=None):
    from deepquantum.photonic.submode import _SubmodeCircuit

    source = circuit._prepare_sources() if circuit.sources else None
    expanded = _SubmodeCircuit(circuit, source, joint=True)
    circuit.__dict__['_submode_circuit'] = expanded
    expanded(data)
    circuit.state = expanded.result()
    return circuit.state


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('basis,den_mat', [(True, False), (False, False), (False, True)])
def test_spatial_hom_and_loss(dtype, basis, den_mat):
    prepared = dtype == torch.float64
    cir = dqp.QumodeCircuit(2, 'vac' if prepared else [1, 1], cutoff=3, basis=basis, den_mat=den_mat)
    cir.bs([0, 1], [torch.pi / 4, 0.0])
    model = dqp.HardwareModel()
    if prepared:
        cir.single_photon(0)
        cir.single_photon(1, [0.8j, 0.6])
    else:
        model.input(1).submodes([0.8j, 0.6])
    transmission = 0.8 if den_mat or basis else 1
    if transmission < 1:
        model.output().loss(t=transmission)
    cir.set_hardware_model(model)
    cir.to(dtype=dtype)
    probabilities = cir(is_prob=True)
    assert cir.nmode == 2 and cir._submode_circuit.nsubmode == 2
    expected = torch.tensor([transmission**2 * (1 - 0.8**2) / 2], dtype=dtype)
    torch.testing.assert_close(probabilities[dqp.FockState([1, 1])], expected)
    torch.testing.assert_close(sum(probabilities.values()), torch.ones(1, dtype=dtype))
    torch.testing.assert_close(cir.get_prob([1, 1]), expected)
    assert sum(cir.measure(50).values()) == 50
    if basis:
        cir()
        torch.testing.assert_close(cir.get_prob([1, 1]), expected)
        assert sum(count for count, _ in cir.measure(10, with_prob=True).values()) == 10
    elif prepared:
        rho, probability = cir.post_select(1, 1, keep=0, lo=[0.8j, 0.6])
        assert rho.nmode == 1 and rho.den_mat
        expected = sum(value for key, value in probabilities.items() if key.state[1] == 1)
        torch.testing.assert_close(probability, expected)
        readout = dqp.QumodeCircuit(1, rho)
        readout()
        assert readout.measure_homodyne(8, wires=0, phi=0.3).shape == (8,)


@pytest.mark.parametrize('den_mat', [False, True])
def test_input_isometry_preserves_number_coherences(den_mat):
    initial = torch.tensor([[2**-0.5, 0, 2**-0.5]])
    if den_mat:
        initial = initial.unsqueeze(-1) * initial.unsqueeze(-2)
    cir = dqp.QumodeCircuit(1, initial, cutoff=3, basis=False, den_mat=den_mat)
    cir.set_hardware_model(dqp.HardwareModel().input(0).submodes([0.6j, 0.8]))
    cir.to(torch.double)
    state = cir().to_joint()
    ket = torch.zeros(3, 3, dtype=torch.cdouble)
    ket[0, 0] = 2**-0.5
    ket[2, 0] = 2**-0.5 * (0.6j) ** 2
    ket[1, 1] = 0.6j * 0.8
    ket[0, 2] = 2**-0.5 * 0.8**2
    expected = torch.outer(ket.flatten(), ket.flatten().conj()) if den_mat else ket
    torch.testing.assert_close(state.reshape(expected.shape), expected)


@pytest.mark.parametrize('backend', ['gaussian', 'bosonic', 'fock'])
def test_squeezed_modes_matches_explicit_circuit(backend, monkeypatch):
    monkeypatch.setattr(dqp, 'hbar', 1.0)
    monkeypatch.setattr(dqp, 'kappa', 0.6)
    cir = dqp.QumodeCircuit(2, 'vac', cutoff=4, backend=backend, basis=False, den_mat=backend == 'fock')
    cir.s(0, r=0.3, theta=0.2)
    cir.bs([0, 1], [0.4, 0.1])
    model = dqp.HardwareModel().s(0).submodes(relative_r=[1, 1 / 3])
    model.s(0).after.loss(t=0.8)
    cir.set_hardware_model(model)
    cir.to(torch.double)
    actual = cir().to_joint()
    explicit = dqp.QumodeCircuit(4, 'vac', cutoff=4, backend=backend, basis=False, den_mat=backend == 'fock')
    for k, ratio in enumerate([1, 1 / 3]):
        explicit.s(k, r=ratio * cir.operators[0].r, theta=cir.operators[0].theta)
        explicit.loss_t(k, torch.tensor(0.8))
        explicit.bs([k, k + 2], [cir.operators[1].theta, cir.operators[1].phi])
    expected = explicit.to(torch.double)()
    torch.testing.assert_close(actual, expected.reshape_as(actual) if backend == 'fock' else expected)


@pytest.mark.parametrize('backend', ['gaussian', 'bosonic'])
def test_complex_input_covariance_and_bosonic_components(backend):
    source = dqp.QumodeCircuit(1, 'vac', backend=backend)
    if backend == 'bosonic':
        source.cat(0, r=0.3, theta=0.4)
    source.s(0, r=0.3, theta=0.2)
    state = source.to(torch.double)()
    cir = dqp.QumodeCircuit(1, state, backend=backend)
    cir.set_hardware_model(dqp.HardwareModel().input(0).submodes([0.6j, 0.8]))
    actual = cir.to(torch.double)().to_joint()
    c = torch.tensor([0.6j, 0.8])
    embed = torch.stack([torch.cat([c.real, c.imag]), torch.cat([-c.imag, c.real])], -1)
    vacuum = dqp.hbar / (4 * dqp.kappa**2)
    torch.testing.assert_close(actual[0], embed @ (state[0] - vacuum * torch.eye(2)) @ embed.mT + vacuum * torch.eye(4))
    torch.testing.assert_close(actual[1], embed.to(state[1].dtype) @ state[1])
    if backend == 'bosonic':
        torch.testing.assert_close(actual[2], state[2])


def test_mixed_single_photon_hom():
    a = torch.diag(torch.tensor([0.8, 0.2]))
    factor = torch.tensor([[0.7, 0.2j], [-0.1j, 0.5]], requires_grad=True)
    b = factor @ factor.mH
    b = b / b.trace().real
    initial = [(math.sqrt(0.7), [0, 0]), (math.sqrt(0.3), [1, 1])]
    cir = dqp.QumodeCircuit(2, initial, cutoff=3, basis=False, den_mat=True)
    cir.bs([0, 1], [torch.pi / 4, 0.0])
    cir.single_photon(0, density_matrix=a)
    cir.single_photon(1, factor=factor)
    cir.to(torch.double)
    cir()
    probability = cir.get_prob([1, 1])
    expected = ((1 - (a.to(b) @ b).trace().real) / 2).reshape(1)
    torch.testing.assert_close(probability, expected)
    torch.testing.assert_close(
        torch.autograd.grad(probability.sum(), factor)[0], torch.autograd.grad(expected, factor)[0]
    )


@pytest.mark.parametrize('backend', ['gaussian', 'fock'])
def test_encoding_shared_control_noise_and_gradients(backend):
    cir = dqp.QumodeCircuit(2, 'vac', cutoff=3, backend=backend, basis=False)
    cir.s(0, encode=True)
    cir.bs([0, 1], [0.4, 0.1])
    model = dqp.HardwareModel(seed=2).s(0).submodes(relative_r=[1, 0.5])
    model.bs().theta.gaussian(sigma=0.1)
    cir.set_hardware_model(model)
    cir.to(torch.double)
    data = torch.tensor([[0.2, 0.0], [0.3, 0.1]], requires_grad=True)
    result = cir(data).data
    torch.testing.assert_close(cir.get_prob([99, 0]), torch.zeros(2))
    loss = result[0].square().sum() if backend == 'gaussian' else result.abs().square().sum()
    grad = torch.autograd.grad(loss, data)[0]
    assert torch.isfinite(grad).all()
    model.manual_seed(2)
    single = cir(data[0]).data
    torch.testing.assert_close(result[0][0], single[0][0]) if backend == 'gaussian' else torch.testing.assert_close(
        result[0], single[0]
    )


@pytest.mark.parametrize('den_mat', [False, True])
def test_unresolved_herald_traces_auxiliary_without_losing_its_weight(den_mat):
    # Main family: sqrt(.7)|0,0> + sqrt(.3)|1,1>.
    # Auxiliary family: sqrt(.8)|0,0> + sqrt(.2)|1,1>.
    # Ordering is [retained main, retained auxiliary, detected main, detected auxiliary].
    components = [(math.sqrt(a * b), [i, j, i, j]) for i, a in enumerate([0.7, 0.3]) for j, b in enumerate([0.8, 0.2])]
    explicit = dqp.QumodeCircuit(4, components, cutoff=2, basis=False, den_mat=den_mat)
    state = explicit.to(torch.double)()
    from deepquantum.photonic.qmath import _pnr_condition

    density, probability = _pnr_condition(state, 4, 2, den_mat, [[2, 3]], [1], [0])
    torch.testing.assert_close(probability, torch.tensor([0.3 * 0.8 + 0.7 * 0.2]))
    expected = torch.diag(torch.tensor([0.7 * 0.2, 0.3 * 0.8]) / 0.38).to(density)
    torch.testing.assert_close(density[0], expected)


@pytest.mark.parametrize('backend', ['gaussian', 'bosonic', 'fock'])
def test_thermal_preparation(backend):
    cir = dqp.QumodeCircuit(1, 'vac', backend=backend, cutoff=4, basis=False, den_mat=backend == 'fock')
    cir.thermal(0, purity=0.8)
    actual = cir.to(torch.double)()
    if backend == 'fock':
        q = (1 / 0.8 - 1) / (1 / 0.8 + 1)
        expected = torch.diag((1 - q) * q ** torch.arange(4)).to(actual)
        torch.testing.assert_close(actual.reshape(4, 4), expected)
    else:
        torch.testing.assert_close(actual[0].reshape(2, 2), torch.eye(2) * dqp.hbar / (4 * dqp.kappa**2) / 0.8)


def test_validation():
    with pytest.raises(ValueError, match='positive reference'):
        dqp.Submodes(relative_r=[0, 0.3])
    with pytest.raises(ValueError, match='orthogonal'):
        dqp.Submodes([[1, 0], [1, 0]], relative_r=[1, 1])
    with pytest.raises(ValueError, match='positive semidefinite'):
        dqp.SinglePhotonSource([[1.2, 0], [0, -0.2]])
    with pytest.raises(NotImplementedError, match='Fock density'):
        dqp.QumodeCircuit(1, [1]).thermal(0, nbar=0.1)
    with pytest.raises(NotImplementedError, match='Fock backend'):
        dqp.QumodeCircuit(1, 'vac', backend='gaussian').single_photon(0)
    density_circuit = dqp.QumodeCircuit(1, 'vac', cutoff=2, basis=False, den_mat=True)
    density_circuit.single_photon(0, density_matrix=[[1, 0], [0, 0]])
    with pytest.raises(ValueError, match='already defines'):
        density_circuit.set_hardware_model(dqp.HardwareModel().input(0).submodes([1, 0]))
    density_circuit.thermal(0, nbar=0.1)
    assert isinstance(density_circuit.sources['0'], dqp.ThermalSource)
    cir = dqp.QumodeCircuit(2, 'vac', cutoff=2, basis=False)
    cir()
    with pytest.raises(ValueError, match='zero-probability'):
        cir.post_select(1, 1)
    with pytest.raises(ValueError, match='at least one'):
        cir.post_select([], [])


def test_internal_mode_calibration_preserves_reference_squeezing():
    modes = dqp.Submodes([[3, 4j], [4j, 3]], relative_r=[3, 2])
    torch.testing.assert_close(modes.coefficients, torch.tensor([[0.6, 0.8j], [0.8j, 0.6]]))
    large = torch.tensor([3e38 + 3e38j, 3e38 - 3e38j], dtype=torch.cfloat)
    expected = torch.tensor([0.5 + 0.5j, 0.5 - 0.5j], dtype=torch.cfloat)
    torch.testing.assert_close(dqp.Submodes(large).coefficients[0], expected)
    torch.testing.assert_close(
        dqp.SinglePhotonSource(coefficients=large).density_matrix, torch.outer(expected, expected.conj())
    )
    torch.testing.assert_close(dqp.Submodes(relative_r=[3, 2]).relative_r, torch.tensor([1, 2 / 3]))
    nbar = torch.tensor([0.6, 0.08, 0.01])
    modes = dqp.Submodes.from_nbar(nbar)
    r = nbar[0].sqrt().asinh()
    torch.testing.assert_close(modes.mean_photon_weights(r), nbar / nbar.sum())
    torch.testing.assert_close(modes.relative_r[0], torch.tensor(1.0))
    torch.testing.assert_close(modes.effective_mode_number(r), nbar.sum().square() / nbar.square().sum())
    zero = torch.tensor(0.0, requires_grad=True)
    low_gain = modes.relative_r.square()
    torch.testing.assert_close(modes.mean_photon_weights(zero), low_gain / low_gain.sum())
    assert torch.isfinite(torch.autograd.grad(modes.effective_mode_number(zero), zero)[0])
    cir = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
    cir.s(0, r=r, theta=0.0)
    cir.set_hardware_model(dqp.HardwareModel().s(0).submodes(nbar=nbar))
    cir.to(torch.double)()
    torch.testing.assert_close(cir.photon_number_mean_var()[0], nbar.sum())
    torch.testing.assert_close(cir.operators[0].r, r)
    with pytest.raises(ValueError, match='not both'):
        dqp.HardwareModel().s(0).submodes(relative_r=[3, 2], nbar=[0.6, 0.1])
    with pytest.raises(ValueError, match='already specifies'):
        dqp.HardwareModel().s(0).submodes(modes, nbar=nbar)


def test_fock_isometry_high_occupation_and_zero_coefficient_gradient():
    from deepquantum.photonic.qmath import _fock_isometry

    coefficients = torch.tensor([2**-0.5, 1j * 2**-0.5], dtype=torch.cfloat)
    matrix = _fock_isometry(coefficients, 301)
    column = matrix[..., 300]
    assert torch.isfinite(column).all()
    torch.testing.assert_close(column.abs().square().sum(), torch.ones((), dtype=torch.float32), atol=2e-5, rtol=2e-5)
    expected = -math.exp((math.lgamma(301) - 2 * math.lgamma(151)) / 2 - 150 * math.log(2))
    torch.testing.assert_close(column[150, 150].real, torch.tensor(expected, dtype=torch.float32), atol=1e-5, rtol=1e-5)
    coefficients = torch.tensor([1.0, 0.0], dtype=torch.cdouble, requires_grad=True)
    matrix = _fock_isometry(coefficients, 5)
    gradient = torch.autograd.grad(matrix[3, 1, 4].real, coefficients)[0]
    torch.testing.assert_close(gradient, torch.tensor([0, 2], dtype=torch.cdouble))


@pytest.mark.parametrize('backend,den_mat', [('gaussian', False), ('fock', False), ('fock', True)])
def test_factorized_spatial_initial_batch_matches_joint(backend, den_mat):
    if backend == 'gaussian':
        preparation = dqp.QumodeCircuit(2, 'vac', backend='gaussian')
        preparation.s2([0, 1], r=0.2, theta=0.3)
        preparation.d(1, r=0.1, theta=0.2)
        cov, mean = preparation.to(torch.double)()
        initial = [cov.expand(2, -1, -1), torch.cat([mean, -mean])]
    else:
        initial = torch.zeros(2, 3, 3, dtype=torch.cdouble)
        initial[0, 0, 0], initial[0, 1, 1] = 0.7**0.5, 0.3**0.5
        initial[1, 0, 1], initial[1, 1, 0] = 0.4**0.5, 1j * 0.6**0.5
        if den_mat:
            initial = torch.einsum('bij,bkl->bijkl', initial, initial.conj())
    results, probabilities = [], []
    for coefficients in [None, torch.eye(2)]:
        cir = dqp.QumodeCircuit(2, 'vac', cutoff=3, backend=backend, basis=False, den_mat=den_mat)
        cir.s(1, r=0.12, theta=0.3)
        cir.bs([0, 1], [0.4, 0.1])
        cir.set_hardware_model(dqp.HardwareModel().s(1).submodes(coefficients, relative_r=[1, 0.4]))
        results.append(cir.to(torch.double)(state=initial).to_joint())
        probabilities.append(cir.get_prob([1, 1]))
    torch.testing.assert_close(results[0], results[1])
    torch.testing.assert_close(probabilities[0], probabilities[1])
    assert probabilities[0].shape == (2,)


@pytest.mark.parametrize(
    'dtype,backend,den_mat',
    [(torch.float32, 'gaussian', False), (torch.float64, 'fock', False), (torch.float64, 'fock', True)],
)
def test_factorized_evolution_readout_and_gradients_match_joint(dtype, backend, den_mat):
    r = torch.tensor(0.32, dtype=dtype, requires_grad=True)
    circuits = []
    for joint in [False, True]:
        cir = dqp.QumodeCircuit(3, 'vac', cutoff=3, backend=backend, basis=False, den_mat=den_mat)
        cir.s(0, r=r, theta=0.2)
        cir.s2([1, 2], r=0.15, theta=0.6)
        cir.bs([0, 1], [0.35, 0.2])
        cir.bs([1, 2], [0.42, 0.1])
        cir.d(2, r=0.03, theta=0.2)
        model = dqp.HardwareModel(seed=17)
        basis = torch.eye(2, dtype=dtype) if joint else None
        model.s(0).submodes(basis, relative_r=[1, 0.4])
        model.s2([1, 2]).submodes(basis, relative_r=[1, 0.2])
        model.bs().theta.gaussian(sigma=torch.tensor(0.01, dtype=dtype))
        if den_mat or backend == 'gaussian':
            cir.thermal(0, nbar=0.02)
            model.output().loss(t=0.85)
        cir.set_hardware_model(model)
        circuits.append(cir.to(dtype))
    product, joint = circuits
    actual, reference = product(), _run_joint(joint)
    assert actual.factorized and not reference.factorized
    torch.testing.assert_close(actual.to_joint(), reference.data)
    rng = product.hardware_model._generator.get_state().clone()
    torch.testing.assert_close(rng, joint.hardware_model._generator.get_state())
    for pattern in ([0, 1, 1], [2, 0, 0], [4, 0, 0], [99, 0, 0]):
        torch.testing.assert_close(product.get_prob(pattern), joint.get_prob(pattern))
    torch.testing.assert_close(product.photon_number_mean_var(), joint.photon_number_mean_var())
    torch.testing.assert_close(product.quadrature_mean([2, 0], 0.3), joint.quadrature_mean([2, 0], 0.3))
    if backend == 'fock':
        for internal in [0, 1, None]:
            rho, probability = product.post_select([1, 2], [1, 0], keep=0, submode=internal)
            ref, p_ref = joint.post_select([1, 2], [1, 0], keep=0, submode=internal)
            torch.testing.assert_close(rho.state, ref.state)
            torch.testing.assert_close(probability, p_ref)
        reordered, p_reordered = product.post_select(1, 1, keep=[2, 0])
        reference_order, p_order = joint.post_select(1, 1, keep=[2, 0])
        torch.testing.assert_close(reordered.state, reference_order.state)
        torch.testing.assert_close(p_reordered, p_order)
        value = probability + 0.1 * rho.state.flatten()[0].real
        ref_value = p_ref + 0.1 * ref.state.flatten()[0].real
    else:
        value, ref_value = product.get_prob([0, 1, 1]), joint.get_prob([0, 1, 1])
    grad = torch.autograd.grad(value.sum(), r)[0]
    ref_grad = torch.autograd.grad(ref_value.sum(), r)[0]
    torch.testing.assert_close(grad, ref_grad)
    assert torch.equal(rng, product.hardware_model._generator.get_state())
    samples = product.measure(12, wires=[0, 2], with_prob=True)
    assert sum(n for n, _ in samples.values()) == 12
    for key, (_, probability) in samples.items():
        if backend == 'gaussian':
            # Gaussian marginalization traces the full unmeasured oscillator;
            # summing only its counts below cutoff would omit a physical tail.
            modes = [0, 1, 4, 5]
            indices = modes + [w + 6 for w in modes]
            cov, mean = reference.data
            reduced = dqp.QumodeCircuit(
                4, [cov[:, indices, :][:, :, indices], mean[:, indices]], cutoff=3, backend='gaussian'
            )
            reduced(stepwise=True)
            from deepquantum.photonic.qmath import fock_combinations

            expected = sum(
                reduced.get_prob(a + b)
                for a in fock_combinations(2, int(key.state[0]), 3)
                for b in fock_combinations(2, int(key.state[1]), 3)
            )
        else:
            expected = sum(joint.get_prob([key.state[0], n, key.state[1]]) for n in range(5))
        torch.testing.assert_close(probability, expected.squeeze())


def test_many_internal_modes_condition_without_a_joint_state():
    count = 16
    cir = dqp.QumodeCircuit(2, 'vac', cutoff=3, basis=False)
    cir.s(0, r=0.3, theta=0.1)
    cir.bs([0, 1], [0.4, 0.2])
    cir.set_hardware_model(dqp.HardwareModel().s(0).submodes(relative_r=[1] + [0.2] * (count - 1)))
    result = cir.to(torch.double)()
    assert result.factorized and result.data.numel() == count * 3**2
    rho, probability = cir.post_select(1, 1, keep=0, submode=0)
    selected, selected_probability = cir.post_select(1, 1, keep=0, lo=[1] + [0] * (count - 1))
    torch.testing.assert_close(selected.state, rho.state)
    torch.testing.assert_close(selected_probability, probability)
    block_probabilities = []
    for ratio in [1, 0.2]:
        reference = dqp.QumodeCircuit(2, 'vac', cutoff=3, basis=False)
        reference.s(0, r=cir.operators[0].r * ratio, theta=cir.operators[0].theta)
        reference.bs([0, 1], [cir.operators[1].theta, cir.operators[1].phi])
        reference.to(torch.double)()
        block_probabilities.append([reference.post_select(1, n, keep=0)[1] for n in [0, 1]])
    (a0, a1), (b0, b1) = block_probabilities
    expected = a1 * b0 ** (count - 1) + a0 * (count - 1) * b1 * b0 ** (count - 2)
    torch.testing.assert_close(probability, expected)
    torch.testing.assert_close(rho.state.reshape(3, 3).trace().real, torch.tensor(1.0))
    assert sum(cir.measure(10).values()) == 10


@pytest.mark.parametrize('active', ['s', 'opa'])
def test_active_complex_profile_and_spatial_moments(active):
    r = torch.tensor(0.4, requires_grad=True)
    one = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
    cir = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
    for circuit in [one, cir]:
        if active == 's':
            circuit.s(0, r=r, theta=0.3)
        else:
            circuit.opa(0, r=r, theta=0.3, loss=0.9, convention='t')
    c = torch.tensor([0.5j, math.sqrt(0.5), -0.5])
    model = dqp.HardwareModel()
    getattr(model, active)(0).submodes(c)
    cir.set_hardware_model(model)
    expected = one.to(torch.double)()
    actual = cir.to(torch.double)().to_joint()
    embedding = torch.stack([torch.cat([c.real, c.imag]), torch.cat([-c.imag, c.real])], -1)
    v0 = dqp.hbar / (4 * dqp.kappa**2)
    cov = embedding @ (expected[0] - v0 * torch.eye(2)) @ embedding.mT + v0 * torch.eye(6)
    torch.testing.assert_close(actual[0], cov)
    torch.testing.assert_close(cir.photon_number_mean_var(), one.photon_number_mean_var())
    grad = torch.autograd.grad(actual[0].square().sum(), r, retain_graph=True)[0]
    ref = torch.autograd.grad(cov.square().sum(), r)[0]
    torch.testing.assert_close(grad, ref)


def test_global_tdm_expansion_and_source_placement():
    cir = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
    cir.thermal(0, nbar=0.02)
    cir.s(0, r=0.3, theta=0.2)
    cir.delay(0, ntau=1, inputs=[0.4, 0.1])
    model = dqp.HardwareModel().s(0).submodes(relative_r=[1, 0.5])
    cir.set_hardware_model(model)
    global_circuit = cir.global_circuit(3).to(torch.double)
    actual = global_circuit().to_joint()
    # Identically acting internal families must remain independent.
    n = global_circuit.nmode
    main = torch.tensor(list(range(0, 2 * n, 2)) + list(range(2 * n, 4 * n, 2)))
    auxiliary = main + 1
    torch.testing.assert_close(actual[0][:, main[:, None], auxiliary], torch.zeros(1, 2 * n, 2 * n))
    for k, ratio in enumerate([1, 0.5]):
        reference = dqp.QumodeCircuit(1, 'vac', backend='gaussian')
        if k == 0:
            reference.thermal(0, nbar=0.02)
        reference.s(0, r=ratio * cir.operators[0].r, theta=cir.operators[0].theta)
        reference.delay(0, ntau=1, inputs=[cir.operators[1].gates[0].theta, cir.operators[1].gates[1].theta])
        expected = reference.global_circuit(3).to(torch.double)()
        indices = main + k
        torch.testing.assert_close(actual[0][:, indices[:, None], indices], expected[0])
    with pytest.raises(NotImplementedError, match='Streaming'):
        cir.tdm_substate(3, [0, 2])


@pytest.mark.parametrize(
    'backend,rotated_input',
    [('fock', False), ('fock', True), ('gaussian', False), ('gaussian', True), ('bosonic', True)],
)
def test_readout_uses_saved_state_and_module_vmap(backend, rotated_input):
    from torch.func import functional_call, stack_module_state, vmap

    circuits = []
    for r in [0.2, 0.4]:
        cir = dqp.QumodeCircuit(1, 'vac', backend=backend, cutoff=3, basis=False)
        if backend == 'bosonic':
            cir.cat(0, r=0.2, theta=0.1)
        cir.s(0, r=r, theta=0.1)
        model = dqp.HardwareModel(seed=7)
        if rotated_input:
            model.input(0).submodes([0.8, 0.6j])
        model.s(0).submodes(relative_r=[1, 0.5])
        cir.set_hardware_model(model)
        circuits.append(cir.to(torch.double))
    parameters, buffers = stack_module_state(circuits)
    actual = vmap(lambda p, b: functional_call(circuits[0], (p, b), ()))(parameters, buffers).to_joint()
    for i, cir in enumerate(circuits):
        expected = cir().to_joint()
        if backend == 'fock':
            torch.testing.assert_close(actual[i], expected)
        else:
            for result, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(result[i], reference)
    if backend == 'bosonic':
        assert torch.isfinite(cir.photon_number_mean_var()[0]).all()
        return
    before = cir.hardware_model._generator.get_state().clone()
    if backend == 'gaussian':
        expected = cir._submode_circuit.spatial_probabilities()[dqp.FockState([0])]
        torch.testing.assert_close(cir.get_prob([0]), expected)
    assert sum(count for count, _ in cir.measure(10, with_prob=True).values()) == 10
    assert torch.equal(before, cir.hardware_model._generator.get_state())


@pytest.mark.parametrize('backend,den_mat', [('gaussian', False), ('fock', False), ('fock', True)])
def test_partial_blocks_match_joint_state_conditioning_and_gradients(backend, den_mat):
    r = torch.tensor(0.27, requires_grad=True)
    circuits = []
    coefficients = [[0.8, 0, 0.6j], [0, 1, 0], [0.6j, 0, 0.8]]
    for _ in range(2):
        cir = dqp.QumodeCircuit(2, 'vac', cutoff=3, backend=backend, basis=False, den_mat=den_mat)
        cir.s(0, r=r, theta=0.2)
        cir.s2([0, 1], r=0.1, theta=0.3)
        cir.bs([0, 1], [0.4, 0.2])
        cir.d(1, r=0.02, theta=0.1)
        model = dqp.HardwareModel(seed=19).s(0).submodes(coefficients, relative_r=[3, 1, 2])
        model.s2().submodes(relative_r=[1, 0.2, 0.3])
        model.bs().theta.gaussian(sigma=0.01)
        if den_mat or backend == 'gaussian':
            cir.thermal(0, nbar=0.01)
            model.output().loss(t=0.9)
        cir.set_hardware_model(model)
        circuits.append(cir.to(torch.double))
    partial, joint = circuits
    actual, expected = partial(), _run_joint(joint)
    assert actual.partition == ((0, 2), (1,))
    torch.testing.assert_close(actual.to_joint(), expected.data)
    torch.testing.assert_close(
        partial.hardware_model._generator.get_state(), joint.hardware_model._generator.get_state()
    )
    torch.testing.assert_close(partial.get_prob([1, 1]), joint.get_prob([1, 1]))
    torch.testing.assert_close(partial.photon_number_mean_var(), joint.photon_number_mean_var())
    lo = torch.tensor([0.6, 0.6j, 0.2])
    torch.testing.assert_close(partial.quadrature_mean(0, phi=0.4, lo=lo), joint.quadrature_mean(0, phi=0.4, lo=lo))
    if backend == 'fock':
        for selection in ({'submode': 0}, {}, {'lo': lo}):
            rho, probability = partial.post_select(1, 1, keep=0, **selection)
            reference, p_ref = joint.post_select(1, 1, keep=0, **selection)
            torch.testing.assert_close(rho.state, reference.state)
            torch.testing.assert_close(probability, p_ref)
        value, ref = probability + rho.state.flatten()[0].real, p_ref + reference.state.flatten()[0].real
    else:
        value, ref = partial.get_prob([1, 1]), joint.get_prob([1, 1])
    torch.testing.assert_close(torch.autograd.grad(value.sum(), r)[0], torch.autograd.grad(ref.sum(), r)[0])
    assert sum(partial.measure(5, wires=0).values()) == 5


@pytest.mark.parametrize('backend', ['gaussian', 'bosonic', 'fock'])
def test_lo_projection_matches_displaced_wavepacket(backend):
    cir = dqp.QumodeCircuit(1, 'vac', cutoff=8, backend=backend, basis=False)
    cir.d(0, r=0.2, theta=0.3)
    profile = torch.tensor([0.8, 0.6j])
    cir.set_hardware_model(dqp.HardwareModel().d(0).submodes(profile))
    cir.to(torch.double)()
    for lo in (profile, torch.tensor([0.6j, 0.8]), torch.tensor([1.0, 0.0])):
        alpha = torch.vdot(lo.to(profile), profile) * 0.2 * torch.exp(torch.tensor(0.3j))
        expected = math.sqrt(dqp.hbar) / dqp.kappa * (alpha * torch.exp(torch.tensor(-0.2j))).real
        torch.testing.assert_close(
            cir.quadrature_mean(0, 0.2, lo=3 * lo).squeeze().real, expected, atol=1e-8, rtol=1e-6
        )
    samples = cir.measure_homodyne(8, wires=0, lo=profile, phi=0.2)
    assert samples.shape == (8,) and torch.isfinite(samples).all()


@pytest.mark.parametrize('backend', ['gaussian', 'fock'])
def test_partial_blocks_encoded_batch_and_module_vmap(backend):
    from torch.func import functional_call, stack_module_state, vmap

    circuits = []
    for displacement in [0.05, 0.1]:
        cir = dqp.QumodeCircuit(1, 'vac', cutoff=3, backend=backend, basis=False)
        cir.s(0, encode=True)
        cir.d(0, r=displacement, theta=0.4)
        model = dqp.HardwareModel(seed=17).s(0).submodes([0.8, 0.6j, 0])
        model.d(0).theta.gaussian(sigma=0.01)
        cir.set_hardware_model(model)
        circuits.append(cir.to(torch.double))
    data = torch.tensor([[0.2, 0.1], [0.3, 0.4]], requires_grad=True)
    actual = circuits[0](data).to_joint()
    circuits[0].hardware_model.manual_seed(17)
    expected = _run_joint(circuits[0], data).data
    torch.testing.assert_close(actual, expected)
    value = actual.abs().square().sum() if backend == 'fock' else actual[0].square().sum()
    assert torch.isfinite(torch.autograd.grad(value, data)[0]).all()
    for cir in circuits:
        cir.encode(data[0].detach())
        cir.hardware_model.manual_seed(17)
    parameters, buffers = stack_module_state(circuits)
    outputs = vmap(lambda p, b: functional_call(circuits[0], (p, b), (data[0].detach(),)), randomness='same')(
        parameters, buffers
    ).to_joint()
    for i, cir in enumerate(circuits):
        cir.hardware_model.manual_seed(17)
        expected = cir(data[0].detach()).to_joint()
        if backend == 'fock':
            torch.testing.assert_close(outputs[i], expected)
        else:
            torch.testing.assert_close([x[i] for x in outputs], expected)
