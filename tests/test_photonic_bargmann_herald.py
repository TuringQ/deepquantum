"""Native PNR projection, compatibility and differentiable continuation."""

import pytest
import torch

import deepquantum as dq
from deepquantum.photonic.bargmann import PolyExpAnsatz
from deepquantum.photonic.state import BargmannState


@pytest.mark.parametrize('density', [False, True])
def test_gaussian_reference_and_cutoff(density):
    circuits = []
    for backend in ['gaussian', 'bargmann']:
        cir = dq.QumodeCircuit(2, 'vac', backend=backend, den_mat=density, cutoff=5)
        cir.s2([0, 1], 0.31, 0.27)
        cir.to(torch.float64)
        cir()
        circuits.append(cir)
    reference, expected_p = circuits[0].heralded_state(0, 2, normalize=True)
    cir = circuits[1]
    original = cir.state
    state, p, info = cir.heralded_state(0, 2, cutoff=1, normalize=True, return_info=True)
    assert isinstance(state, BargmannState)
    assert cir.state is original
    assert state.cutoff == 1
    assert state.wires == (0,)
    assert info['remaining_wires'] == (1,)
    assert info['retained_fraction'] is None
    assert info['probability_in_cutoff'] is None
    torch.testing.assert_close(p, expected_p)
    torch.testing.assert_close(state.to_fock(5), reference)
    torch.testing.assert_close(state.norm(), torch.ones_like(p))
    assert state.to_fock().abs().max() < 1e-12


@pytest.mark.parametrize('density', [False, True])
def test_existing_derived_unsorted_batch_and_joint_weight(density):
    torch.manual_seed(7)
    psi = torch.randn(2, 3, 3, 3, dtype=torch.complex128)
    psi = psi / psi.flatten(1).norm(dim=1).reshape(2, 1, 1, 1)
    psi = psi * 0.4  # intentional prior event weight
    cir = dq.QumodeCircuit(3, BargmannState(psi, nmode=3, den_mat=density), backend='bargmann')
    cir()
    state, p, info = cir.heralded_state([2, 0], [1, 2], return_info=True)
    selected = psi[:, 2, :, 1]
    expected = selected[:, :, None] * selected.conj()[:, None, :] if density else selected
    torch.testing.assert_close(state.to_fock(3), expected)
    torch.testing.assert_close(p, selected.abs().square().sum(-1))
    torch.testing.assert_close(info['conditional_probability'], p / 0.16)
    assert info['remaining_wires'] == (1,)


def test_superposition_axis():
    # Two coherent exponentials, preserving complex interference between L terms.
    alpha = torch.tensor([0.2 + 0.1j, -0.2 - 0.1j], dtype=torch.complex128)
    a = torch.zeros(1, 2, 2, 2, dtype=torch.complex128)
    b = torch.stack([alpha, alpha * 0.5], -1).unsqueeze(0)
    c = torch.tensor([[0.7, 0.3j]], dtype=torch.complex128)
    state = BargmannState(PolyExpAnsatz(a, b, c, 2))
    cir = dq.QumodeCircuit(2, state, backend='bargmann')
    full = cir().to_fock(8)
    result, p = cir.heralded_state(0, 1)
    torch.testing.assert_close(result.to_fock(8), full[:, 1])
    torch.testing.assert_close(p, full[:, 1].abs().square().sum(-1), atol=1e-12, rtol=1e-9)


@pytest.mark.parametrize('density', [False, True])
def test_two_rounds_and_gradient(density):
    def run(r, theta):
        first = dq.QumodeCircuit(3, 'vac', backend='bargmann', den_mat=density)
        first.s2([0, 1], r, 0.0)
        first.to(torch.float64)
        first()
        state, p1 = first.heralded_state(0, 1)
        second = dq.QumodeCircuit(2, state, backend='bargmann')
        second.bs([0, 1], [theta, 0.0])
        second.to(torch.float64)
        second()
        result, p12, info = second.heralded_state(1, 0, return_info=True)
        torch.testing.assert_close(info['input_weight'], p1)
        torch.testing.assert_close(result.norm(), p12)
        expected = (1 - r.tanh() ** 2) * r.tanh() ** 2 * theta.cos() ** 2
        torch.testing.assert_close(p12, expected.reshape(1))
        return p12

    r = torch.tensor(0.29, dtype=torch.float64, requires_grad=True)
    theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(run, (r, theta), atol=1e-5)


def test_validation_zero_and_cache():
    cir = dq.QumodeCircuit(2, 'vac', backend='bargmann')
    with pytest.raises(RuntimeError):
        cir.heralded_state(0, 0)
    cir()
    for wires, photons in [([], []), ([0, 1], [0, 0]), ([0, 0], [0, 0]), ([0], [-1])]:
        with pytest.raises(ValueError):
            cir.heralded_state(wires, photons)
    zero, p = cir.heralded_state(0, 1)
    torch.testing.assert_close(p, torch.zeros_like(p))
    torch.testing.assert_close(zero.to_fock(2), torch.zeros(1, 2, dtype=zero.a.dtype))
    with pytest.raises(ValueError, match='zero-probability'):
        cir.heralded_state(0, 1, normalize=True)
    cir.to(torch.float64)
    with pytest.raises(RuntimeError):
        cir.heralded_state(0, 0)


def test_mixed_loss_against_gaussian_and_no_fock_expansion(monkeypatch):
    results = []
    for backend in ['gaussian', 'bargmann']:
        cir = dq.QumodeCircuit(2, 'vac', backend=backend, den_mat=True, cutoff=5)
        cir.s2([0, 1], 0.3, 0.1)
        cir.loss_t(0, 0.7)
        cir.to(torch.float64)
        cir()
        if backend == 'bargmann':

            def reject(*args, **kwargs):
                raise AssertionError('No physical Fock readout during native heralding')

            with monkeypatch.context() as patch:
                patch.setattr(BargmannState, 'to_fock', reject)
                state, p = cir.heralded_state(0, 1, normalize=True)
            results.append((state.to_fock(5), p))
        else:
            results.append(cir.heralded_state(0, 1, normalize=True))
    torch.testing.assert_close(results[0][0], results[1][0])
    torch.testing.assert_close(results[0][1], results[1][1])
