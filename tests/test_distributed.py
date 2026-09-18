from datetime import timedelta
from itertools import combinations

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import deepquantum as dq
from deepquantum.photonic.distributed import dist_swap_gate as photonic_swap

pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason='CPU distributed tests require Gloo'
)


def _assert_state_matches(state, expected):
    expected_local = expected.reshape(-1).chunk(state.world_size)[state.rank]
    torch.testing.assert_close(state.amps.reshape(-1), expected_local, atol=1e-6, rtol=1e-5)
    norm = state.amps.abs().square().sum()
    dist.all_reduce(norm)
    torch.testing.assert_close(norm, torch.ones_like(norm), atol=1e-6, rtol=1e-5)


def _check_qubit_swaps():
    circuit = dq.DistributedQubitCircuit(4)
    reference = dq.QubitCircuit(4)
    for cir in (circuit, reference):
        cir.h(0)
        cir.swap([0, 1])  # Both targets are rank bits with four processes.
        cir.h(0)  # Another exchange exposes amps/buffer aliasing after SWAP.
    _assert_state_matches(circuit(), reference())


def _check_qubit_gates():
    circuit = dq.DistributedQubitCircuit(4)
    reference = dq.QubitCircuit(4)
    for cir in (circuit, reference):
        cir.hlayer()
        for target in range(4):
            others = [wire for wire in range(4) if wire != target]
            for ncontrol in range(4):
                for controls in combinations(others, ncontrol):
                    cir.rx(target, 0.37, controls=list(controls))
        for wires in combinations(range(4), 2):
            cir.rxx(list(wires), 0.23)
        cir.rx(0, controls=[3], encode=True)
        cir.ry(1, controls=[0, 3], encode=True)
        cir.observable([0, 1], 'zy')
        cir.to(torch.double)
    data = torch.tensor([0.12, 0.29], dtype=torch.double, requires_grad=True)
    data_ref = data.detach().clone().requires_grad_()
    _assert_state_matches(circuit(data), reference(data_ref))
    value = circuit.expectation().sum()
    value_ref = reference.expectation().sum()
    value.backward()
    value_ref.backward()
    torch.testing.assert_close(value, value_ref)
    torch.testing.assert_close(data.grad, data_ref.grad)


def _check_photonic_gates():
    circuit = dq.DistributedQumodeCircuit(4, [1, 0, 0, 0], cutoff=2)
    reference = dq.QumodeCircuit(4, [1, 0, 0, 0], cutoff=2, basis=False)
    for cir in (circuit, reference):
        for wires in ([0, 1], [1, 2], [2, 3], [0, 3]):
            cir.bs(wires, [0.27, -0.19])
    _assert_state_matches(circuit(), reference())


def _check_photonic_swaps():
    # Measurement also uses this helper directly, including swaps between two rank digits.
    generator = torch.Generator().manual_seed(917)
    expected = torch.randn([2] * 4, dtype=torch.cdouble, generator=generator)
    expected /= expected.norm()
    state = dq.DistributedFockState([1, 0, 0, 0], nmode=4, cutoff=2).to(torch.double)
    state.amps.copy_(expected.reshape(-1).chunk(state.world_size)[state.rank].reshape(state.amps.shape))
    for target1, target2 in ((0, 1), (2, 3), (2, 3), (0, 3), (1, 2), (0, 1)):
        photonic_swap(state, target1, target2)
        expected = expected.transpose(3 - target1, 3 - target2).contiguous()
        _assert_state_matches(state, expected)


def _check_measurements():
    bits = [1, 0, 0, 1]
    qubit = dq.DistributedQubitCircuit(4)
    qubit.x(0)
    qubit.x(3)
    photonic = dq.DistributedQumodeCircuit(4, bits, cutoff=2)
    for circuit in (qubit, photonic):
        state = circuit()
        original = state.amps.clone()
        for nwires in range(1, 5):
            for wires in combinations(range(4), nwires):
                # Match dense measurement's ascending wire order, even for an unsorted request.
                result = circuit.measure(shots=16, with_prob=True, wires=list(reversed(wires)))
                torch.testing.assert_close(state.amps, original)
                if state.rank == 0:
                    actual = {}
                    for key, (count, prob) in result.items():
                        if not isinstance(key, str):
                            key = key.state.tolist()
                        actual[tuple(map(int, key))] = (count, float(prob))
                    assert actual == {tuple(bits[wire] for wire in wires): (16, 1.0)}
                else:
                    assert result == {}


def _distributed_worker(rank, world_size, init_method, check):
    torch.set_num_threads(1)
    dist.init_process_group(
        'gloo', init_method=init_method, rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    exchange = dist.all_to_all_single

    def exchange_without_overlap(output, input, *args, **kwargs):  # noqa: A002
        if input.numel() and output.numel():
            assert input.is_contiguous() and output.is_contiguous()
            assert input.untyped_storage().data_ptr() != output.untyped_storage().data_ptr(), (
                'Receiving must not overwrite amplitudes that are still being sent'
            )
        return exchange(output, input, *args, **kwargs)

    # Aliased communication can succeed by chance; enforce the buffer invariant as well as numerical accuracy.
    dist.all_to_all_single = exchange_without_overlap
    try:
        check()
    finally:
        dist.all_to_all_single = exchange
        dist.destroy_process_group()


def _run_distributed(tmp_path, world_size, check):
    # A fresh file store avoids fixed-port collisions and also works under pytest-xdist.
    mp.spawn(
        _distributed_worker,
        args=(world_size, (tmp_path / 'rendezvous').as_uri(), check),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize('world_size', [2, 4])
@pytest.mark.parametrize(
    'check',
    [_check_qubit_swaps, _check_qubit_gates, _check_photonic_gates, _check_photonic_swaps],
    ids=['qubit_swaps', 'qubit_gates', 'photonic_gates', 'photonic_swaps'],
)
def test_distributed(tmp_path, world_size, check):
    _run_distributed(tmp_path, world_size, check)


@pytest.mark.parametrize('world_size', [2, 4, 8])
def test_distributed_measurements(tmp_path, world_size):
    _run_distributed(tmp_path, world_size, _check_measurements)
