from unittest.mock import patch

import pytest
import torch

import deepquantum as dq
import deepquantum.photonic as dqp


def test_forward_unitary():
    batch = 4
    cir = dq.QumodeCircuit(2, init_state=[1, 0])
    cir.bs([0, 1])
    cir.ps(0, encode=True)
    x = torch.randn(batch, 1)
    u = cir(x)
    assert u.shape == (batch, 2, 2)


def test_gaussian_shape():
    cir = dq.QumodeCircuit(nmode=1, init_state='vac', cutoff=3, backend='gaussian')
    cir.s(0, 0.0, encode=True)

    data2 = torch.tensor([[0, 0], [0, 1]])
    state = cir()
    assert tuple(state[0].shape) == (1, 2, 2) and tuple(state[1].shape) == (1, 2, 1)
    state = cir(data=data2)
    assert tuple(state[0].shape) == (2, 2, 2) and tuple(state[1].shape) == (2, 2, 1)


def test_gaussian_batch_shape():
    batch = torch.randint(1, 10, size=[1])[0]
    covs = torch.stack([torch.eye(2)] * batch)
    means = torch.tensor([[0, 0]] * batch)
    cir = dq.QumodeCircuit(nmode=1, init_state=[covs, means], cutoff=3, backend='gaussian')
    cir.s(0, 0.0, encode=True)

    data2 = torch.tensor([[0, 0]] * batch)
    state = cir()
    assert tuple(state[0].shape) == (batch, 2, 2) and tuple(state[1].shape) == (batch, 2, 1)
    state = cir(data=data2)
    assert tuple(state[0].shape) == (batch, 2, 2) and tuple(state[1].shape) == (batch, 2, 1)


@pytest.mark.parametrize(
    ('device', 'dtype'),
    [('cpu', torch.float32), ('cpu', torch.float64)]
    + ([('cuda', torch.float32), ('cuda', torch.float64)] if torch.cuda.is_available() else [])
    + ([('mps', torch.float32)] if torch.backends.mps.is_available() else []),
)
def test_empty_gaussian_receiver_uses_input_dtype_and_device(device, dtype):
    eye = torch.eye(4, dtype=dtype, device=device)
    cov = torch.stack([1.1 * eye, 1.3 * eye])
    mean = eye.new_tensor([[0, 0, 0, 0], [0.2, -0.1, 0.3, 0.1]]).unsqueeze(-1)
    state = [cov, mean]
    receiver = dq.QumodeCircuit(2, 'vac', backend='gaussian', detector='click', cutoff=2).double()
    torch.testing.assert_close(receiver(state=state), state)
    actual = receiver(state=state, is_prob=True)

    reference = dq.QumodeCircuit(2, state, backend='gaussian', detector='click', cutoff=2)
    torch.testing.assert_close(reference.get_symplectic(), eye)
    reference.ps(0, 0.0)
    reference.to(device=device, dtype=dtype)
    torch.testing.assert_close(actual, reference(is_prob=True))
    for event, probability in actual.items():
        torch.testing.assert_close(probability, receiver.get_prob(event.state, refer_state=state))


def test_gaussian_get_prob_batch_shapes():
    cir = dq.QumodeCircuit(2, 'vac', cutoff=3, backend='gaussian', detector='click')
    cir.s(0, r=0.3)
    cir.bs([0, 1], [0.4, 0.2])
    cir.to(torch.double)
    state = cir()
    state_batch = [state[0].expand(2, -1, -1), state[1].expand(2, -1, -1)]
    patterns = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]])

    probs = [
        cir.get_prob(patterns[0], state),
        cir.get_prob(patterns[0], state_batch),
        cir.get_prob(patterns[:1], state),
        cir.get_prob(patterns[:1], state_batch),
        cir.get_prob(patterns, state),
        cir.get_prob(patterns, state_batch),
    ]
    expected = torch.stack([cir.get_prob(pattern, state) for pattern in patterns])

    assert [prob.shape for prob in probs] == [(), (2,), (1,), (2, 1), (4,), (2, 4)]
    torch.testing.assert_close(probs[-2], expected)
    torch.testing.assert_close(probs[-1], expected.expand(2, -1))

    multiplexed = torch.tensor([[2, 0], [0, 2], [2, 1], [1, 2], [0, 0]])
    expected = torch.stack([cir.get_prob(pattern, state) for pattern in multiplexed])
    torch.testing.assert_close(cir.get_prob(multiplexed, state), expected)


@pytest.mark.parametrize('detector', ['pnrd', 'threshold', 'click'])
def test_gaussian_chunk_budget_preserves_values_and_gradients(detector, monkeypatch):
    monkeypatch.setattr(dqp.utils, '_chunk_budgets', {})
    cir = dq.QumodeCircuit(1, 'vac', cutoff=3, backend='gaussian', detector=detector)
    cov = (torch.eye(2, dtype=torch.double) * 1.4).expand(5, -1, -1)
    mean = torch.linspace(0.1, 1, 10, dtype=torch.double).reshape(5, 2, 1).requires_grad_()
    expected = cir.get_prob([[1], [0]], [cov, mean])
    dqp.set_chunk_budget('cpu', 256)
    with patch.object(cir, '_get_probs_gaussian_helper', wraps=cir._get_probs_gaussian_helper) as read:
        actual = cir.get_prob([[1], [0]], [cov, mean])
    assert read.call_count > 1
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual[:, 0].sum(), mean), torch.autograd.grad(expected[:, 0].sum(), mean)
    )


def test_gaussian_pnrd_get_prob_mixed_totals_in_requested_order():
    cir = dq.QumodeCircuit(2, 'vac', cutoff=5, backend='gaussian', detector='pnrd')
    amplitude = torch.tensor([0.4, 0.0, 0.65, 0.0, 0.22, 0.55, 0.28], dtype=torch.float64, requires_grad=True)
    nbar = amplitude.new_tensor([0.0, 0.3, 0.5, 0.0, 0.6, 0.0, 0.2])
    ones = torch.ones_like(nbar)
    zeros = torch.zeros_like(nbar)
    cov = torch.stack([ones, 2 * nbar + 1, ones, 2 * nbar + 1], dim=-1).diag_embed()
    mean = torch.stack([2 * amplitude, zeros, zeros, zeros], dim=-1).unsqueeze(-1)
    cir(state=[cov, mean])
    patterns = torch.tensor([[2, 0], [0, 0], [3, 4], [0, 1], [1, 1], [1, 0], [2, 0], [0, 2]])
    actual = cir.get_prob(patterns)

    # Independent coherent and thermal modes have Poisson and geometric photon counts.
    n, m = patterns.T
    rate = amplitude.square().unsqueeze(-1)
    factorial = (n.to(rate) + 1).lgamma().exp()
    thermal = nbar.unsqueeze(-1)
    expected = (-rate).exp() * rate**n / factorial * (thermal / (1 + thermal)) ** m / (1 + thermal)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(cir.get_prob(patterns[0]), expected[:, 0], atol=1e-12, rtol=0)
    torch.testing.assert_close(cir.get_prob(patterns, [cov[0], mean[0]]), expected[0], atol=1e-12, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), amplitude), torch.autograd.grad(expected.sum(), amplitude), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    'dtype,lossy,angles',
    [
        (torch.float64, False, [[0.1]]),
        (torch.float64, True, [[0.1], [0.4], [0.9]]),
        (torch.float32, False, [[0.1], [0.4], [0.9]]),
        (torch.float32, True, [[0.1]]),
    ],
    ids=['double-pure-single', 'double-loss-batch', 'float-pure-batch', 'float-loss-single'],
)
def test_gaussian_threshold_probabilities_in_requested_order(dtype, lossy, angles, request):
    previous_dtype = torch.get_default_dtype()
    request.addfinalizer(lambda: torch.set_default_dtype(previous_dtype))
    torch.set_default_dtype(torch.float64)
    cir = dq.QumodeCircuit(2, 'vac', cutoff=1, backend='gaussian', detector='threshold')
    cir.s(0, r=0.55)
    cir.d(1, r=0.23, theta=0.4)
    cir.ps(0, encode=True)
    cir.bs([0, 1], inputs=[0.78, 0.0])
    if lossy:
        loss_angle = torch.tensor(2.6, dtype=dtype, requires_grad=True)
        cir.loss(0, inputs=loss_angle)
        cir.loss_t(1, inputs=0.1)
    cir.to(dtype)
    phases = torch.tensor(angles, dtype=dtype, requires_grad=True)
    patterns = [[1, 1], [0, 0], [1, 0], [0, 1], [1, 0]]
    cov, mean = cir(phases)
    double_click = cir.get_prob([1, 1])
    one_bin = dq.QumodeCircuit(2, 'vac', cutoff=2, backend='gaussian', detector='click')
    expected = torch.stack(
        [one_bin.get_prob(patterns, [cov_b.double(), mean_b.double()]) for cov_b, mean_b in zip(cov, mean, strict=True)]
    ).to(dtype)
    table = cir(phases, is_prob=True)
    actual = cir.get_prob(patterns)
    tol = 1e-11 if dtype == torch.float64 else 1e-6
    torch.testing.assert_close(actual, expected.squeeze(0), atol=tol, rtol=0)
    torch.testing.assert_close(double_click, expected[:, 0].squeeze(0), atol=tol, rtol=0)
    torch.testing.assert_close(
        torch.stack([table[dq.FockState(p)] for p in patterns], dim=-1), expected, atol=tol, rtol=0
    )
    torch.testing.assert_close(sum(table.values()), torch.ones_like(phases[:, 0]), atol=tol, rtol=0)
    variables = (phases, loss_angle) if lossy else (phases,)
    torch.testing.assert_close(
        torch.autograd.grad(actual[..., 0].sum(), variables),
        torch.autograd.grad(expected[:, 0].sum(), variables),
        atol=tol,
        rtol=0,
    )


@pytest.mark.parametrize('count', [0.5, 2])
def test_gaussian_threshold_rejects_nonbinary_counts(count):
    cir = dq.QumodeCircuit(1, 'vac', cutoff=3, backend='gaussian', detector='threshold')
    cir()
    with pytest.raises(AssertionError, match='0 or 1'):
        cir.get_prob([count])


def test_bosonic_shape():
    cir = dq.QumodeCircuit(nmode=2, init_state='vac', cutoff=3, backend='bosonic')
    cir.cat(0, r=1, theta=0.0)
    cir.gkp(1, theta=0.0, phi=0.0)
    cir.s(0, 0.0, encode=True)

    data2 = torch.tensor([[0, 0], [0, 1]])
    state = cir()
    assert (
        tuple(state[0].shape) == (1, 1, 4, 4)
        and tuple(state[1].shape) == (1, 356, 4, 1)
        and tuple(state[2].shape) == (1, 356)
    )
    state = cir(data=data2)
    assert (
        tuple(state[0].shape) == (2, 1, 4, 4)
        and tuple(state[1].shape) == (2, 356, 4, 1)
        and tuple(state[2].shape) == (1, 356)
    )


def test_bosonic_batch_shape():
    batch = torch.randint(1, 10, size=[1])[0]
    cat = dq.CatState(r=1.0, theta=0.0, p=1)
    cov_in = cat.cov.expand(batch, 1, 2, 2)
    mean_in = cat.mean.expand(batch, 4, 2, 1)
    weight_in = cat.weight.expand(batch, 4)
    cir = dq.QumodeCircuit(nmode=1, init_state=[cov_in, mean_in, weight_in], cutoff=3, backend='bosonic')
    cir.s(0, 0.0, encode=True)

    data2 = torch.tensor([[0, 0]] * batch)
    state = cir()
    assert (
        tuple(state[0].shape) == (batch, 1, 2, 2)
        and tuple(state[1].shape) == (batch, 4, 2, 1)
        and tuple(state[2].shape) == (batch, 4)
    )
    state = cir(data=data2)
    assert (
        tuple(state[0].shape) == (batch, 1, 2, 2)
        and tuple(state[1].shape) == (batch, 4, 2, 1)
        and tuple(state[2].shape) == (batch, 4)
    )
