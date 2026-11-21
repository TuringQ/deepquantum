"""Sparse products of simultaneous, angle-resolved Homodyne outcomes."""

import math

import pytest
import torch

import deepquantum as dq
import deepquantum.photonic as dqp


@pytest.mark.parametrize('backend', ['gaussian', 'bargmann'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_displaced_correlated_readout_and_gradients(backend, dtype, monkeypatch):
    monkeypatch.setattr(dqp, 'hbar', 1.0)
    monkeypatch.setattr(dqp, 'kappa', 1.0)
    preparation = dq.QumodeCircuit(2, 'vac', backend='gaussian')
    preparation.s2([0, 1], encode=True)
    preparation.d(0, 0.25, -0.5)
    preparation.d(1, 0.18, 0.7)
    preparation.to(dtype)
    data = torch.tensor([[0.21, -0.23], [0.32, 0.54]], dtype=dtype, requires_grad=True)
    cov, mean = preparation(data)
    reader = preparation if backend == 'gaussian' else dq.BargmannState([cov, mean], den_mat=True)

    # Reversed wires, two different angles and a repeated request.
    phi = torch.tensor(0.43, dtype=dtype, requires_grad=True)
    powers = [(0, 4), (2, 2), (4, 0), (0, 0), (2, 2)]
    readout = {'wires': [1, 0], 'phi': [0.0, phi]}
    actual = reader.quadrature_moments(**readout, powers=powers)
    central = reader.quadrature_moments(**readout, powers=powers, centered=True)
    c, s = phi.cos(), phi.sin()
    x, y = mean[:, 1, 0], c * mean[:, 0, 0] + s * mean[:, 2, 0]
    vx = cov[:, 1, 1]
    vy = c**2 * cov[:, 0, 0] + 2 * c * s * cov[:, 0, 2] + s**2 * cov[:, 2, 2]
    vxy = c * cov[:, 1, 0] + s * cov[:, 1, 2]
    mixed = (vx + x**2) * (vy + y**2) + 2 * vxy**2 + 4 * x * y * vxy
    expected = torch.stack(
        [y**4 + 6 * y**2 * vy + 3 * vy**2, mixed, x**4 + 6 * x**2 * vx + 3 * vx**2, torch.ones_like(x), mixed], -1
    )
    central_mixed = vx * vy + 2 * vxy**2
    expected_central = torch.stack([3 * vy**2, central_mixed, 3 * vx**2, torch.ones_like(x), central_mixed], -1)
    for moments, target in ((actual, expected), (central, expected_central)):
        torch.testing.assert_close(moments, target)
        torch.testing.assert_close(
            torch.autograd.grad(moments.sum(), (data, phi), retain_graph=True),
            torch.autograd.grad(target.sum(), (data, phi), retain_graph=True),
        )


def _fock_products(density, phi, powers, *, centered=False):
    """Trace the unnormalized two-mode Fock matrix against padded local operator powers."""
    cutoff = density.shape[-1]
    size = cutoff + max(max(row) for row in powers)
    a = torch.diag(torch.arange(1, size, dtype=torch.double).sqrt(), diagonal=1)
    phase = torch.exp(-1j * torch.as_tensor(phi, dtype=torch.double))
    lowering = phase[:, None, None] * a
    x, y = (lowering + lowering.mH) * (dqp.hbar**0.5 / (2 * dqp.kappa))
    if centered:
        weight = torch.einsum('...ijij->...', density).real
        mean_x = torch.einsum('...ijkj,ki->...', density, x[:cutoff, :cutoff]).real / weight
        mean_y = torch.einsum('...ijil,lj->...', density, y[:cutoff, :cutoff]).real / weight
        eye = torch.eye(size, dtype=x.dtype)
        x, y = x - mean_x[..., None, None] * eye, y - mean_y[..., None, None] * eye
    values = []
    for i, j in powers:
        op_x, op_y = x.matrix_power(i)[..., :cutoff, :cutoff], y.matrix_power(j)[..., :cutoff, :cutoff]
        values.append(torch.einsum('...ijkl,...ki,...lj->...', density, op_x, op_y).real)
    return torch.stack(values, -1)


def _split_pair_fock(cutoff, r, eta, herald_angle, signal_angles, clicks):
    """Prepare a split Fock signal independently from the pair-number law and the two on/off outcomes."""
    n = torch.arange(cutoff, dtype=torch.double)
    pair_ratio = r.tanh().square()
    pairs = (1 - pair_ratio) * pair_ratio**n
    # Each herald photon is lost or routed to exactly one counter.
    silent_2 = (1 - eta * herald_angle.cos().square()) ** n
    silent_3 = (1 - eta * herald_angle.sin().square()) ** n
    neither = (1 - eta) ** n
    likelihood = silent_3 - neither if clicks == (1, 0) else 1 - silent_2 - silent_3 + neither
    rho = torch.zeros(1, cutoff, cutoff, cutoff, cutoff, dtype=torch.complex128)
    rho[0, :, 0, :, 0] = (pairs * likelihood).diag().to(rho)
    splitter = dq.QumodeCircuit(2, 'vac', cutoff=cutoff, backend='fock', basis=False, den_mat=True)
    splitter.bs([0, 1], signal_angles)
    splitter.double()
    return splitter(state=rho)


@pytest.mark.parametrize('clicks', [(1, 0), (1, 1)])
def test_lossy_heralds_against_converged_fock_operators(clicks):
    r, eta, herald_angle, mixing, phase = torch.tensor([0.26, 0.71, 0.6, 0.53, -0.37], dtype=torch.double)
    cir = dq.QumodeCircuit(4, 'vac', cutoff=1, backend='bargmann', den_mat=True)
    cir.s2([0, 2], r)
    cir.loss_t(2, eta)
    cir.bs([2, 3], [herald_angle, 0.0])
    cir.bs([0, 1], [mixing, phase])
    cir.double()
    cir()
    state, probability = cir.post_select([2, 3], clicks, detector='threshold', normalize=False)
    phi = [0.0, 0.63]
    powers = [(4, 0), (2, 2), (0, 4), (0, 0)]
    weighted = state.quadrature_moments([0, 1], phi, powers=powers, normalize=False)
    conditional = state.quadrature_moments([0, 1], phi, powers=powers)
    torch.testing.assert_close(weighted[:, -1], probability)
    normalized, _ = cir.post_select([2, 3], clicks, detector='threshold')
    torch.testing.assert_close(normalized.quadrature_moments(phi=phi, powers=powers, normalize=False), conditional)

    def reference(cutoff):
        density = _split_pair_fock(cutoff, r, eta, herald_angle, [mixing, phase], clicks)
        return _fock_products(density, phi, powers)

    previous, final = reference(14), reference(18)
    torch.testing.assert_close(previous, final, rtol=0, atol=1e-9)
    torch.testing.assert_close(weighted, final, rtol=0, atol=1e-8)
    final_conditional = final / final[:, -1:]
    torch.testing.assert_close(previous / previous[:, -1:], final_conditional, rtol=0, atol=1e-9)
    torch.testing.assert_close(conditional, final_conditional, rtol=0, atol=1e-8)


def test_rare_and_impossible_clicks_keep_batch_and_event_weight():
    def click(r):
        cir = dq.QumodeCircuit(2, 'vac', backend='bargmann')
        cir.s2([0, 1], encode=True)
        cir.double()
        cir(torch.stack([r, torch.zeros_like(r)], -1))
        return cir.post_select(0, 1, detector='threshold', normalize=False)

    r = torch.tensor([0, 0.0015, 0.2], dtype=torch.double, requires_grad=True)
    state, probability = click(r)
    powers = [(0,), (4,)]
    weighted = state.quadrature_moments(0, powers=powers, normalize=False)
    ratio = r.tanh().square()
    # On a click the remaining photon number has a geometric law on 1, 2, ...
    fourth = 3 + 12 / (1 - ratio).square()
    torch.testing.assert_close(probability, ratio)
    torch.testing.assert_close(weighted, torch.stack([ratio, ratio * fourth], -1), atol=1e-12, rtol=0)
    assert torch.count_nonzero(weighted[0]) == 0
    assert torch.count_nonzero(state.quadrature_moments(0, powers=powers, centered=True, normalize=False)[0]) == 0
    with pytest.raises(ValueError, match='positive state weight'):
        state.quadrature_moments(0, powers=powers)
    positive, _ = click(r[1:])
    torch.testing.assert_close(positive.quadrature_moments(0, powers=[(4,)])[:, 0], fourth[1:], atol=1e-8, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad(weighted[:, 1].sum(), r), torch.autograd.grad((ratio * fourth).sum(), r)
    )


@pytest.mark.parametrize('centered', [False, True])
def test_polynomial_state_high_orders_and_angle_gradient(centered):
    ket = torch.tensor([[[0.1, 0.3j], [-0.2j, 0.4]]], dtype=torch.complex128)
    state = dq.BargmannState(ket, nmode=2)
    density = torch.einsum('bij,bkl->bijkl', ket, ket.conj())
    phi = torch.tensor([0.32, 0.8], dtype=torch.double, requires_grad=True)
    powers = [(0, 0), (1, 0), (1, 3), (0, 8), (3, 5)]
    actual = state.quadrature_moments(phi=phi, powers=powers, centered=centered, normalize=False)
    expected = _fock_products(density, phi, powers, centered=centered)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), phi), torch.autograd.grad(expected.sum(), phi))


def test_displaced_click_products_center_on_the_whole_branch():
    cir = dq.QumodeCircuit(3, 'vac', backend='bargmann')
    cir.s2([0, 2], 0.22, 0.1)
    cir.bs([0, 1], [0.44, 0.3])
    cir.d(2, 0.42, 0.5)
    cir.d(1, 0.17, -0.2)
    cir.double()
    cir()
    branch, _ = cir.post_select(2, 1, detector='threshold', normalize=False)
    angles, powers = [0.1, 0.63], [(2, 2), (0, 4)]
    final = _fock_products(branch.to_fock(14), angles, powers, centered=True)
    previous = _fock_products(branch.to_fock(10), angles, powers, centered=True)
    torch.testing.assert_close(previous, final, rtol=0, atol=1e-9)
    weighted = branch.quadrature_moments(phi=angles, powers=powers, centered=True, normalize=False)
    torch.testing.assert_close(weighted, final, rtol=0, atol=1e-9)


@pytest.mark.parametrize('backend', ['gaussian', 'bargmann'])
def test_high_order_with_finite_physical_moment(backend, monkeypatch):
    monkeypatch.setattr(dqp, 'hbar', 1 / 64)
    monkeypatch.setattr(dqp, 'kappa', 1.0)
    degree = 1000
    # In these units the 1000th vacuum moment is finite, though 1000! exceeds float64.
    variance = dqp.hbar / (4 * dqp.kappa**2)
    expected = math.exp(math.lgamma(degree + 1) - math.lgamma(degree // 2 + 1) + degree / 2 * math.log(variance / 2))
    if backend == 'gaussian':
        reader = dq.QumodeCircuit(1, 'vac', backend=backend).double()
        reader()
    else:
        reader = dq.BargmannState([1]).double()
        # The single-photon quadrature density is x² / variance times the vacuum density.
        expected *= degree + 1
    actual = reader.quadrature_moments(powers=[(degree,)])
    torch.testing.assert_close(actual, torch.full_like(actual, expected), rtol=1e-10, atol=0)


@pytest.mark.parametrize('backend', ['gaussian', 'bargmann'])
def test_vacuum_quadrature_products(backend):
    cir = dq.QumodeCircuit(2, 'vac', backend=backend).double()
    cir()
    powers = torch.tensor([(4, 0), (2, 2), (0, 4), (0, 0), (1, 0), (0, 8)])
    actual = cir.quadrature_moments(phi=math.pi / 3, powers=powers, normalize=False)
    torch.testing.assert_close(actual, torch.tensor([[3, 1, 3, 1, 0, 105]], dtype=torch.double))
