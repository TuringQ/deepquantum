"""
Distributed operations
"""

from collections import Counter

import torch
import torch.distributed as dist

from ..communication import comm_exchange_arrays
from ..distributed import get_local_targets
from ..qmath import block_sample, decimal_to_list, evolve_state, inverse_permutation, list_to_decimal
from .state import DistributedFockState, FockState


# The 0-th mode is the rightmost for the `target`
def get_pair_rank(rank: int, target_rank: int, new_digit: int, cutoff: int, ndigit: int) -> int:
    """Get the pair rank for communication."""
    digits = decimal_to_list(rank, cutoff, ndigit)
    digits[-(target_rank + 1)] = new_digit
    return list_to_decimal(digits, cutoff)


def get_digit(decimal: int, target: int, cutoff: int) -> int:
    """Get the digit of a decimal number at a specific position."""
    digits = decimal_to_list(decimal, cutoff)
    if target >= len(digits):
        return 0
    else:
        return digits[-(target + 1)]


def local_gate(state: torch.Tensor, targets: list[int], matrix: torch.Tensor) -> torch.Tensor:
    """Apply a gate to a Fock state tensor locally."""
    shape = state.shape
    nmode = len(shape)
    cutoff = shape[0]
    wires = [nmode - target - 1 for target in targets]
    state[:] = evolve_state(state.unsqueeze(0), matrix, nmode, wires, cutoff).squeeze(0)
    return state


def local_swap_gate(state: torch.Tensor, target1: int, target2: int) -> torch.Tensor:
    """Apply a SWAP operation to a Fock state tensor locally."""
    nmode = len(state.shape)
    wire1 = nmode - target1 - 1
    wire2 = nmode - target2 - 1
    return state.transpose(wire1, wire2)


def dist_swap_gate(state: DistributedFockState, target1: int, target2: int):
    """Apply a SWAP operation to a distributed Fock state tensor."""
    if target1 > target2:
        target1, target2 = target2, target1
    if target2 < state.nmode_local:
        state.amps = local_swap_gate(state.amps, target1, target2)
    elif target1 >= state.nmode_local:
        target1_rank = target1 - state.nmode_local
        target2_rank = target2 - state.nmode_local
        digit1 = get_digit(state.rank, target1_rank, state.cutoff)
        digit2 = get_digit(state.rank, target2_rank, state.cutoff)
        if digit1 != digit2:
            new_rank = get_pair_rank(state.rank, target1_rank, digit2, state.cutoff, state.nmode_global)
            pair_rank = get_pair_rank(new_rank, target2_rank, digit1, state.cutoff, state.nmode_global)
            comm_exchange_arrays(state.amps, state.buffer, pair_rank)
            state.amps = state.buffer
    else:
        target2_rank = target2 - state.nmode_local
        wire1 = state.nmode_local - target1 - 1
        pm_shape = list(range(state.nmode_local))
        pm_shape.remove(wire1)
        pm_shape = [wire1] + pm_shape
        state.amps = state.amps.permute(pm_shape).contiguous()
        qudits = list(range(state.cutoff))
        io_sizes = [0] * state.world_size
        for i in qudits:
            pair_rank = get_pair_rank(state.rank, target2_rank, i, state.cutoff, state.nmode_global)
            io_sizes[pair_rank] = 1
        dist.all_to_all_single(state.buffer, state.amps, io_sizes, io_sizes)
        state.amps = state.buffer.permute(inverse_permutation(pm_shape)).contiguous()
    return state


def dist_gate(state: DistributedFockState, targets: list[int], matrix: torch.Tensor) -> DistributedFockState:
    """Apply a gate to a distributed Fock state tensor."""
    nt = len(targets)
    assert nt <= state.nmode_local
    if max(targets) < state.nmode_local:
        state.amps = local_gate(state.amps, targets, matrix)
    else:
        targets_new = get_local_targets(targets, state.nmode_local)
        for i in range(nt):
            if targets_new[i] != targets[i]:
                dist_swap_gate(state, targets_new[i], targets[i])
        state.amps = local_gate(state.amps, targets_new, matrix)
        for i in range(nt):
            if targets_new[i] != targets[i]:
                dist_swap_gate(state, targets_new[i], targets[i])
    return state


def measure_dist(
    state: DistributedFockState,
    shots: int = 1024,
    with_prob: bool = False,
    wires: int | list[int] | None = None,
    block_size: int = 2**24,
) -> dict:
    """Measure a distributed Fock state tensor."""
    if isinstance(wires, int):
        wires = [wires]
    nwires = len(wires) if wires else state.nmode
    if wires is not None:
        targets = [state.nmode - wire - 1 for wire in wires]
        pm_shape = list(range(state.nmode_local))
        # Assume nmode_global < nmode_local
        if nwires <= state.nmode_local:  # All targets move to local modes
            if max(targets) >= state.nmode_local:
                targets_new = get_local_targets(targets, state.nmode_local)
                for i in range(nwires):
                    if targets_new[i] != targets[i]:
                        dist_swap_gate(state, targets_new[i], targets[i])
                wires_local = sorted([state.nmode_local - target - 1 for target in targets_new])
            else:
                wires_local = sorted([state.nmode_local - target - 1 for target in targets])
            for w in wires_local:
                pm_shape.remove(w)
            pm_shape = wires_local + pm_shape
            probs = torch.abs(state.amps) ** 2
            probs = probs.permute(pm_shape).reshape([state.cutoff] * nwires + [-1]).sum(-1).reshape(-1)
            dist.all_reduce(probs, dist.ReduceOp.SUM)
            if state.rank == 0:
                samples = Counter(block_sample(probs, shots, block_size))
                results = {FockState(decimal_to_list(k, state.cutoff, nwires)): v for k, v in samples.items()}
                if with_prob:
                    for k in results:
                        index = list_to_decimal(k.state, state.cutoff)
                        results[k] = results[k], probs[index]
                return results
            return {}
        else:  # All targets are sorted, then move to global modes
            targets_sort = sorted(targets, reverse=True)
            wires_local = []
            for i, target in enumerate(targets_sort):
                if i < state.nmode_global:
                    target_new = state.nmode - i - 1
                    if target_new != target:
                        dist_swap_gate(state, target, target_new)
                else:
                    wires_local.append(state.nmode_local - target - 1)
            for w in wires_local:
                pm_shape.remove(w)
            pm_shape = wires_local + pm_shape
            probs = torch.abs(state.amps) ** 2
            probs = probs.permute(pm_shape).reshape([state.cutoff] * len(wires_local) + [-1]).sum(-1).reshape(-1)
    else:
        probs = (torch.abs(state.amps) ** 2).reshape(-1)
    probs_rank = probs.new_empty(state.world_size)
    dist.all_gather_into_tensor(probs_rank, probs.sum().unsqueeze(0))
    blocks = torch.multinomial(probs_rank, shots, replacement=True)
    dist.broadcast(blocks, src=0)
    block_dict = Counter(blocks.cpu().numpy())
    key_offset = state.rank * state.cutoff ** (nwires - state.nmode_global)
    if state.rank in block_dict:
        samples = Counter(block_sample(probs, block_dict[state.rank], block_size))
        results = {FockState(decimal_to_list(k + key_offset, state.cutoff, nwires)): v for k, v in samples.items()}
    else:
        results = {}
    if with_prob:
        for k in results:
            index = list_to_decimal(k.state, state.cutoff) - key_offset
            results[k] = results[k], probs[index]
    results_lst = [None] * state.world_size
    dist.all_gather_object(results_lst, results)
    if state.rank == 0:
        results = {}
        for r in results_lst:
            results.update(r)
        return results
    else:
        return {}
