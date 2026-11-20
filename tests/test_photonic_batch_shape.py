import pytest
import torch

import deepquantum as dq


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


@pytest.mark.parametrize('detector', ['pnrd', 'threshold'])
def test_gaussian_get_prob_fixed_total_pattern_batch(detector):
    cir = dq.QumodeCircuit(2, 'vac', cutoff=3, backend='gaussian', detector=detector)
    cir.s(0, r=0.2)
    cir.to(torch.double)
    state = cir()
    patterns = torch.tensor([[1, 0], [0, 1]])

    batched = cir.get_prob(patterns, state)
    expected = torch.stack([cir.get_prob(pattern, state) for pattern in patterns])

    torch.testing.assert_close(batched, expected)
    with pytest.raises(AssertionError, match='same total occupation'):
        cir.get_prob(torch.tensor([[0, 0], [1, 0]]), state)


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
