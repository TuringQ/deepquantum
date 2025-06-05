"""
Communication utilities
"""

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist


def setup_distributed(port = '29500', backend = 'nccl') -> Tuple[int, int, int]:
    """Initialize torch.distributed."""
    try:
        # These should be set by the launch script (e.g., torchrun)
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK']) # GPU id on the current node
    except KeyError:
        print('RANK, WORLD_SIZE, and LOCAL_RANK env vars must be set.')
        # Fallback for single-process testing (optional)
        rank = 0
        world_size = 1
        local_rank = 0
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = port

    print(f'Initializing distributed setup: Rank {rank}/{world_size}, Local Rank (GPU): {local_rank}')

    # Initialize the process group
    dist.init_process_group(backend, world_size=world_size, rank=rank)

    # Pin the current process to a specific GPU
    torch.cuda.set_device(local_rank)

    print(f'Rank {rank} initialized, using GPU {local_rank}.')
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    """Clean up the distributed environment."""
    dist.destroy_process_group()
    print('Distributed environment cleaned up.')


def comm_get_rank() -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def comm_get_world_size() -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def comm_exchange_arrays(send_data: torch.Tensor, recv_data: torch.Tensor, pair_rank: Optional[int]) -> None:
    """Simulate a point-to-point exchange using dist.all_to_all_single
    with output_split_sizes and input_split_sizes to minimize memory.
    If pair_rank is None, this rank participates in the collective call
    but sends/receives no actual data to/from other specific ranks in this logical P2P.

    Args:
        send_data (torch.Tensor): The data this rank wants to send to pair_rank.
            If pair_rank is None, this can be an empty tensor with correct dtype and device.
        recv_data (torch.Tensor): The Tensor where data received from pair_rank will be stored.
            It MUST already be allocated with the correct size if pair_rank is not None.
            If pair_rank is None, this can be an empty tensor.
        pair_rank (int or None): The rank of the process to exchange data with, or None.
    """
    world_size = comm_get_world_size()
    rank = comm_get_rank()

    if not dist.is_initialized() or world_size <= 1:
        return
    if world_size == 1 and pair_rank is not None and rank == pair_rank:
        if send_data.numel() > 0 and recv_data.numel() > 0:
            recv_data.copy_(send_data)
        return

    is_valid = (pair_rank is not None) and (0 <= pair_rank < world_size)
    io_sizes = [0] * world_size
    if is_valid:
        assert send_data.shape == recv_data.shape, 'Send/Recv shape must match for active P2P'
        assert send_data.dtype == recv_data.dtype, 'Send/Recv dtype must match for active P2P'
        io_sizes[pair_rank] = send_data.numel()
    else:
        send_data = send_data.new_empty(0)
        recv_data = recv_data.new_empty(0)

    dist.all_to_all_single(output=recv_data, input=send_data, output_split_sizes=io_sizes, input_split_sizes=io_sizes)
