import pytest
import torch

import deepquantum as dq
import deepquantum.photonic as dqp


def test_local_gaussian_gate_preserves_correlations_and_complex_means():
    rng = torch.Generator().manual_seed(37)
    params = torch.tensor([0.4, -0.3], dtype=torch.double, requires_grad=True)
    gate = dqp.BeamSplitter(params, nmode=4, wires=[3, 1])
    # Bosonic terms can have different complex means and share a single covariance.
    factor = torch.randn(2, 1, 8, 8, dtype=torch.double, generator=rng)
    cov = (factor @ factor.mT).requires_grad_()
    mean = torch.randn(2, 3, 8, 1, dtype=torch.cfloat, generator=rng, requires_grad=True)
    weight = torch.ones(2, 3, dtype=torch.cdouble) / 3
    actual_cov, actual_mean, actual_weight = gate([cov, mean, weight])
    matrix = gate.get_symplectic()
    expected_cov = matrix @ cov @ matrix.mT
    expected_mean = matrix.to(mean.dtype) @ mean + gate.get_displacement()
    torch.testing.assert_close(actual_cov, expected_cov)
    torch.testing.assert_close(actual_mean, expected_mean)
    assert actual_weight is weight
    inputs = (params, cov, mean)
    # Include a correlation between an addressed quadrature and a spectator.
    actual_grad = torch.autograd.grad(actual_cov[..., 7, 4].sum() + actual_mean[..., 1, 0].imag.sum(), inputs)
    expected_grad = torch.autograd.grad(expected_cov[..., 7, 4].sum() + expected_mean[..., 1, 0].imag.sum(), inputs)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_gaussian_gate_uses_custom_global_transform():
    class GlobalShift(dqp.PhaseShift):
        def get_displacement(self):
            return self.theta.expand(2 * self.nmode, 1)

    gate = GlobalShift(0.3, nmode=3, wires=1)
    cov, mean = gate([torch.eye(6), torch.zeros(6, 1)])
    torch.testing.assert_close(cov, torch.eye(6))
    torch.testing.assert_close(mean, torch.full((6, 1), 0.3))


def test_local_gaussian_gate_preserves_autocast_dtypes():
    gate = dqp.BeamSplitter([0.35, 0.27], nmode=3, wires=[2, 0])
    cov = torch.eye(6) + torch.ones(6, 6) / 117
    mean = torch.arange(6, dtype=torch.float32).reshape(6, 1) / 7
    with torch.autocast('cpu', dtype=torch.bfloat16):
        matrix = gate.get_symplectic()
        expected = [matrix @ cov @ matrix.mT, matrix @ mean + gate.get_displacement()]
        actual = gate([cov, mean])
    torch.testing.assert_close(actual, expected)


def test_2_mode_squeezing_gate():
    r = torch.rand(1)[0]
    theta = torch.rand(1)[0] * 2 * torch.pi
    cutoff = 5
    cir1 = dq.QumodeCircuit(nmode=2, init_state='vac', cutoff=cutoff, backend='gaussian')
    cir1.s2([0, 1], r, theta)
    cov1, mean1 = cir1()
    sym1 = cir1.get_symplectic()

    cir2 = dq.QumodeCircuit(nmode=2, init_state='vac', cutoff=cutoff, backend='gaussian')
    cir2.bs([0, 1], [torch.pi / 4, 0])
    cir2.s(0, r, theta)
    cir2.s(1, -r, theta)
    cir2.bs([0, 1], [-torch.pi / 4, 0])
    cov2, mean2 = cir2()
    sym2 = cir2.get_symplectic()
    assert torch.allclose(cov1, cov2, atol=1e-6)
    assert torch.allclose(mean1, mean2, atol=1e-6)
    assert torch.allclose(sym1, sym2, atol=1e-6)


@pytest.mark.parametrize('gate,wires,cutoff', [('s', 0, 128), ('s2', [0, 1], 64)])
def test_squeezing_gate_numerical_stability(gate, wires, cutoff):
    circuit = dq.QumodeCircuit(2, 'vac', cutoff=cutoff, backend='fock', basis=False)
    getattr(circuit, gate)(wires, r=1)
    getattr(circuit, gate)(wires, r=1)
    reference = dq.QumodeCircuit(2, 'vac', cutoff=cutoff, backend='fock', basis=False)
    getattr(reference, gate)(wires, r=2)
    assert torch.allclose(circuit.double()(), reference.double()())
