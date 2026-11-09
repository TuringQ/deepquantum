"""Invalid-input checks and numerical tolerances for native Bargmann states."""

import pytest
import torch

from deepquantum.photonic.bargmann import PolyExpAnsatz, loss_kernel
from deepquantum.photonic.channel import PhotonLoss
from deepquantum.photonic.state import BargmannState


def make_ansatz(dtype=torch.complex128, coefficient=1):
    a = torch.zeros(1, 1, 2, 2, dtype=dtype)
    b = torch.zeros(1, 1, 2, dtype=dtype)
    c = torch.full((1, 1), coefficient, dtype=dtype)
    return a, b, c


@pytest.mark.parametrize('field', [0, 1, 2])
@pytest.mark.parametrize('invalid', [float('nan'), float('inf')])
def test_nonfinite_parameters_rejected(field, invalid):
    values = make_ansatz()
    values[field].reshape(-1)[0] = invalid
    with pytest.raises(ValueError, match='finite'):
        PolyExpAnsatz(*values, 2)


def test_asymmetric_matrix_rejected():
    a, b, c = make_ansatz()
    a[0, 0, 0, 1] = 1
    with pytest.raises(ValueError, match='symmetric'):
        PolyExpAnsatz(a, b, c, 2)


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_complex_symmetric_matrix_and_roundoff_accepted(dtype):
    a, b, c = make_ansatz(dtype)
    a[..., 0, 1] = 0.1j
    a[..., 1, 0] = 0.1j + torch.finfo(a.real.dtype).eps
    state = PolyExpAnsatz(a, b, c, 2)
    assert torch.isfinite(state.to_fock((3, 3))).all()


@pytest.mark.parametrize('invalid', [-0.1, 1.1, float('nan'), float('inf'), 0.2j])
def test_invalid_loss_transmittance_rejected(invalid):
    with pytest.raises(ValueError, match='transmittance'):
        loss_kernel(torch.tensor(invalid))
    with pytest.raises(ValueError, match='loss|Loss'):
        PhotonLoss(inputs=invalid, convention='t')


@pytest.mark.parametrize('convention, invalid', [('db', -1.0), ('db', float('inf')), ('theta', float('nan'))])
def test_invalid_loss_other_conventions(convention, invalid):
    with pytest.raises(ValueError):
        PhotonLoss(inputs=invalid, convention=convention)


@pytest.mark.parametrize('transmittance', [0.0, 1.0])
def test_loss_endpoints_accepted(transmittance):
    assert torch.isfinite(loss_kernel(torch.tensor(transmittance)).a).all()
    assert torch.isfinite(PhotonLoss(inputs=transmittance, convention='t').t).all()


def test_loss_interior_gradient():
    t = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    coefficient = loss_kernel(t).a[0, 0, 0, 2].real
    derivative = torch.autograd.grad(coefficient, t)[0]
    torch.testing.assert_close(derivative, 0.5 / t.sqrt())
    torch.autograd.gradcheck(lambda x: PhotonLoss(inputs=x, convention='t').t, (t,))


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_norm_rejects_significant_imaginary_trace(dtype):
    state = BargmannState(PolyExpAnsatz(*make_ansatz(dtype, 1 + 0.01j), 2), den_mat=True)
    with pytest.raises(ValueError, match='imaginary'):
        state.norm()


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_norm_allows_roundoff_and_preserves_gradient(dtype):
    eps = torch.finfo(torch.empty((), dtype=dtype).real.dtype).eps
    a, b, c = make_ansatz(dtype, 1 + 1j * eps)
    c.requires_grad_()
    state = BargmannState(PolyExpAnsatz(a, b, c, 2), den_mat=True)
    value = state.norm()
    torch.testing.assert_close(value, torch.ones_like(value))
    gradient = torch.autograd.grad(value.sum(), c)[0]
    torch.testing.assert_close(gradient, torch.ones_like(c))
