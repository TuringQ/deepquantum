"""
Distributed opearations
"""

from collections import Counter
from typing import Dict, List, Union

import torch
import torch.distributed as dist

from .bitmath import log_base2, get_bit, flip_bit, flip_bits, all_bits_are_one, get_bit_mask
from .communication import comm_exchange_arrays
from .qmath import evolve_state, block_sample, measure
from .state import DistributedQubitState


# The 0-th qubit is the rightmost in a ket for the `target`
def local_one_targ_gate(state: torch.Tensor, target: int, matrix: torch.Tensor) -> torch.Tensor:
    """Apply a single-qubit gate to a state vector locally.

    See https://arxiv.org/abs/2311.01512 Alg.2
    """
    indices = torch.arange(len(state))
    indices_0 = indices[get_bit(indices, target) == 0]
    indices_1 = flip_bit(indices_0, target)
    amps_0 = state[indices_0]
    amps_1 = state[indices_1]
    state[indices_0] = matrix[0, 0] * amps_0 + matrix[0, 1] * amps_1
    state[indices_1] = matrix[1, 0] * amps_0 + matrix[1, 1] * amps_1
    return state


def local_many_ctrl_one_targ_gate(
    state: torch.Tensor,
    controls: List[int],
    target: int,
    matrix: torch.Tensor
) -> torch.Tensor:
    """Apply a multi-control single-qubit gate to a state vector locally.

    See https://arxiv.org/abs/2311.01512 Alg.3
    """
    indices = torch.arange(len(state))
    control_mask = torch.ones_like(indices, dtype=torch.bool)
    for control in controls:
        control_mask &= (get_bit(indices, control) == 1)
    mask = control_mask & (get_bit(indices, target) == 0)
    # Indices where controls are 1 AND target is 0
    indices_0 = indices[mask]
    # Indices where controls are 1 AND target is 1
    indices_1 = flip_bit(indices_0, target)
    amps_0 = state[indices_0]
    amps_1 = state[indices_1]
    state[indices_0] = matrix[0, 0] * amps_0 + matrix[0, 1] * amps_1
    state[indices_1] = matrix[1, 0] * amps_0 + matrix[1, 1] * amps_1
    return state


def local_swap_gate(state: torch.Tensor, qb1: int, qb2: int) -> torch.Tensor:
    """Apply a SWAP gate to a state vector locally."""
    indices = torch.arange(len(state))
    mask01 = (get_bit(indices, qb1) == 0) & (get_bit(indices, qb2) == 1)
    indices_01 = indices[mask01]
    indices_10 = flip_bits(indices_01, [qb1, qb2])
    state[indices_01], state[indices_10] = state[indices_10], state[indices_01]
    return state


def local_many_targ_gate(state: torch.Tensor, targets: List[int], matrix: torch.Tensor) -> torch.Tensor:
    """Apply a multi-qubit gate to a state vector locally.

    See https://arxiv.org/abs/2311.01512 Alg.4
    """
    nqubit = log_base2(len(state))
    wires = [nqubit - target - 1 for target in targets]
    state[:] = evolve_state(state.reshape([1] + [2] * nqubit), matrix, nqubit, wires, 2).reshape(-1)
    return state


def dist_one_targ_gate(state: DistributedQubitState, target: int, matrix: torch.Tensor) -> DistributedQubitState:
    """Apply a single-qubit gate to a distributed state vector.

    See https://arxiv.org/abs/2311.01512 Alg.6
    """
    nqubit_local = state.log_num_amps_per_node
    if target < nqubit_local:
        state.amps = local_one_targ_gate(state.amps, target, matrix)
    else:
        rank_target = target - nqubit_local
        pair_rank = flip_bit(state.rank, rank_target)
        comm_exchange_arrays(state.amps, state.buffer, pair_rank)
        bit = get_bit(state.rank, rank_target)
        state.amps = matrix[bit, bit] * state.amps + matrix[bit, 1 - bit] * state.buffer
    return state


def dist_many_ctrl_one_targ_gate(
    state: DistributedQubitState,
    controls: List[int],
    target: int,
    matrix: torch.Tensor
) -> DistributedQubitState:
    """Apply a multi-control single-qubit gate to a distributed state vector.

    See https://arxiv.org/abs/2311.01512 Alg.7
    """
    prefix_ctrls = []
    suffix_ctrls = []
    nqubit_local = state.log_num_amps_per_node
    for q in controls:
        if q >= nqubit_local:
            prefix_ctrls.append(q - nqubit_local)
        else:
            suffix_ctrls.append(q)
    if not all_bits_are_one(state.rank, prefix_ctrls):
        comm_exchange_arrays(state.amps, state.buffer, None)
        return state
    if target < nqubit_local:
        state.amps = local_many_ctrl_one_targ_gate(state.amps, suffix_ctrls, target, matrix)
        comm_exchange_arrays(state.amps, state.buffer, None)
    else:
        if not suffix_ctrls:
            state = dist_one_targ_gate(state, target, matrix)
        else:
            state = dist_ctrl_sub(state, suffix_ctrls, target, matrix)
    return state


def dist_ctrl_sub(
    state: DistributedQubitState,
    controls: List[int],
    target: int,
    matrix: torch.Tensor
) -> DistributedQubitState:
    """"A subroutine of `dist_many_ctrl_one_targ_gate`.

    See https://arxiv.org/abs/2311.01512 Alg.8
    """
    rank_target = target - state.log_num_amps_per_node
    pair_rank = flip_bit(state.rank, rank_target)
    indices = torch.arange(state.num_amps_per_node)
    control_mask = torch.ones_like(indices, dtype=torch.bool)
    for control in controls:
        control_mask &= (get_bit(indices, control) == 1)
    # Indices where controls are 1
    indices = indices[control_mask]
    send = state.amps[indices].contiguous()
    recv = state.buffer[:len(send)]
    comm_exchange_arrays(send, recv, pair_rank)
    bit = get_bit(state.rank, rank_target)
    state.amps[indices] = matrix[bit, bit] * send + matrix[bit, 1 - bit] * recv
    return state


def dist_swap_gate(state: DistributedQubitState, qb1: int, qb2: int):
    """Apply a SWAP gate to a distributed state vector.

    See https://arxiv.org/abs/2311.01512 Alg.9
    """
    if qb1 > qb2:
        qb1, qb2 = qb2, qb1
    nqubit_local = state.log_num_amps_per_node
    if qb2 < nqubit_local:
        state.amps = local_swap_gate(state.amps, qb1, qb2)
        comm_exchange_arrays(state.amps, state.buffer, None)
    elif qb1 >= nqubit_local:
        qb1_rank = qb1 - nqubit_local
        qb2_rank = qb2 - nqubit_local
        if get_bit(state.rank, qb1_rank) != get_bit(state.rank, qb2_rank):
            pair_rank = flip_bits(state.rank, [qb1_rank, qb2_rank])
            comm_exchange_arrays(state.amps, state.buffer, pair_rank)
            state.amps = state.buffer
    else:
        qb2_rank = qb2 - nqubit_local
        bit = 1 - get_bit(state.rank, qb2_rank)
        pair_rank = flip_bit(state.rank, qb2_rank)
        indices = torch.arange(state.num_amps_per_node)
        mask = (get_bit(indices, qb1) == bit)
        indices = indices[mask]
        send = state.amps[indices].contiguous()
        recv = state.buffer[:len(send)]
        comm_exchange_arrays(send, recv, pair_rank)
        state.amps[indices] = recv
    return state


def _get_local_targets(targets: List[int], nqubit_local: int) -> List[int]:
    mask = get_bit_mask(targets)
    min_non_targ = 0
    while get_bit(mask, min_non_targ):
        min_non_targ += 1
    targets_new = []
    for target in targets:
        if target < nqubit_local:
            targets_new.append(target)
        else:
            targets_new.append(min_non_targ)
            min_non_targ += 1
            while get_bit(mask, min_non_targ):
                min_non_targ += 1
    return targets_new


def dist_many_targ_gate(
    state: DistributedQubitState,
    targets: List[int],
    matrix: torch.Tensor
) -> DistributedQubitState:
    """Apply a multi-qubit gate to a distributed state vector.

    See https://arxiv.org/abs/2311.01512 Alg.10
    """
    nqubit_local = state.log_num_amps_per_node
    nt = len(targets)
    assert nt <= nqubit_local
    if max(targets) < nqubit_local:
        state.amps = local_many_targ_gate(state.amps, targets, matrix)
        comm_exchange_arrays(state.amps, state.buffer, None)
    else:
        targets_new = _get_local_targets(targets, nqubit_local)
        for i in range(nt):
            if targets_new[i] != targets[i]:
                dist_swap_gate(state, targets_new[i], targets[i])
        state.amps = local_many_targ_gate(state.amps, targets_new, matrix)
        for i in range(nt):
            if targets_new[i] != targets[i]:
                dist_swap_gate(state, targets_new[i], targets[i])
    return state


def measure_dist(
    state: DistributedQubitState,
    shots: int = 1024,
    with_prob: bool = False,
    wires: Union[int, List[int], None] = None,
    block_size: int = 2 ** 24
) -> Dict:
    """Measure a distributed state vector."""
    if state.world_size == 1:
        return measure(state.amps, shots, with_prob, wires, False, block_size)
    else:
        nqubit_local = state.log_num_amps_per_node
        nqubit_global = state.log_num_nodes
        if isinstance(wires, int):
            wires = [wires]
        num_bits = len(wires) if wires else state.nqubit
        if wires is not None:
            targets = [state.nqubit - wire - 1 for wire in wires]
            # Assume nqubit_global < nqubit_local
            if num_bits <= nqubit_local: # All targets move to local qubits
                if max(targets) >= nqubit_local:
                    targets_new = _get_local_targets(targets, nqubit_local)
                    for i in range(num_bits):
                        if targets_new[i] != targets[i]:
                            dist_swap_gate(state, targets_new[i], targets[i])
                    wires_local = sorted([nqubit_local - target - 1 for target in targets_new])
                else:
                    wires_local = sorted([nqubit_local - target - 1 for target in targets])
                pm_shape = list(range(nqubit_local))
                for w in wires_local:
                    pm_shape.remove(w)
                pm_shape = wires_local + pm_shape
                probs = (torch.abs(state.amps) ** 2).reshape([2] * nqubit_local)
                probs = probs.permute(pm_shape).reshape([2] * num_bits + [-1]).sum(-1).reshape(-1)
                dist.all_reduce(probs, dist.ReduceOp.SUM)
                if state.rank == 0:
                    samples = Counter(block_sample(probs, shots, block_size))
                    results = {bin(key)[2:].zfill(num_bits): value for key, value in samples.items()}
                    if with_prob:
                        for k in results:
                            index = int(k, 2)
                            results[k] = results[k], probs[index]
                    return results
                return {}
            else: # All targets are sorted, then move to global qubits
                targets_sort = sorted(targets, reverse=True)
                wires_local = []
                for i, target in enumerate(targets_sort):
                    if i < nqubit_global:
                        target_new = state.nqubit - i - 1
                        if target_new != target:
                            dist_swap_gate(state, target, target_new)
                    else:
                        wires_local.append(nqubit_local - target - 1)
                pm_shape = list(range(nqubit_local))
                for w in wires_local:
                    pm_shape.remove(w)
                pm_shape = wires_local + pm_shape
                probs = (torch.abs(state.amps) ** 2).reshape([2] * nqubit_local)
                probs = probs.permute(pm_shape).reshape([2] * len(wires_local) + [-1]).sum(-1).reshape(-1)
        else:
            probs = torch.abs(state.amps) ** 2
        probs_rank = probs.new_empty(state.world_size)
        dist.all_gather_into_tensor(probs_rank, probs.sum())
        blocks = torch.multinomial(probs_rank, shots, replacement=True)
        dist.broadcast(blocks, src=0)
        block_dict = Counter(blocks.cpu().numpy())
        key_offset = state.rank << (num_bits - nqubit_global)
        if state.rank in block_dict:
            samples = Counter(block_sample(probs, block_dict[state.rank], block_size))
            results = {bin(key + key_offset)[2:].zfill(num_bits): value for key, value in samples.items()}
        else:
            results = {}
        if with_prob:
            for k in results:
                index = int(k, 2) - key_offset
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


def inner_product_dist(bra: DistributedQubitState, ket: DistributedQubitState) -> torch.Tensor:
    """Get the inner product of two distributed state vectors."""
    value = bra.amps.conj() @ ket.amps
    dist.all_reduce(value, dist.ReduceOp.SUM)
    return value
