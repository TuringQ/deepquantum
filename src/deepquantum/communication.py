"""Communication utilities"""

import os

import torch
import torch.distributed as dist


def setup_distributed(backend: str = 'nccl', port: str = '29500') -> tuple[int, int, int]:
    """Initialize ``torch.distributed``."""
    try:
        # These should be set by the launch script (e.g., torchrun)
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])  # GPU id on the current node
    except KeyError:
        print('RANK, WORLD_SIZE, and LOCAL_RANK env vars must be set.')
        # Fallback for single-process testing (optional)
        rank = 0
        world_size = 1
        local_rank = 0
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = port
    if backend == 'nccl':
        print(f'Initializing distributed setup: Rank {rank}/{world_size}, Local Rank (GPU): {local_rank}')
    elif backend == 'gloo':
        print(f'Initializing distributed setup: Rank {rank}/{world_size}, Local Rank (CPU): {local_rank}')
    # Initialize the process group
    dist.init_process_group(backend, world_size=world_size, rank=rank)
    if backend == 'nccl':
        # Pin the current process to a specific GPU
        torch.cuda.set_device(local_rank)
        print(f'Rank {rank} initialized, using GPU {local_rank}.')
    elif backend == 'gloo':
        print(f'Rank {rank} initialized.')
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def comm_get_rank() -> int:
    """Get the rank of the current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def comm_get_world_size() -> int:
    """Get the total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def comm_exchange_arrays(send_data: torch.Tensor, recv_data: torch.Tensor, pair_rank: int | None) -> None:
    """Exchange tensor data with a peer rank using collective communication.

    This performs a point-to-point communication via ``dist.all_to_all_single`` and allows specific ranks
    to participate in the collective call without active data transfer by setting ``pair_rank`` to ``None``.

    Args:
        send_data (torch.Tensor): Data to be sent to the ``pair_rank``.
            If ``pair_rank`` is ``None``, this can be an empty tensor with correct dtype and device.
        recv_data (torch.Tensor): Pre-allocated buffer to store received data. Must match
            ``send_data`` in shape and dtype if ``pair_rank`` is active.
            If ``pair_rank`` is ``None``, this can be an empty tensor.
        pair_rank (int or None): The target rank for exchange, or ``None`` to remain
            quiescent during the collective call.
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
        io_sizes[pair_rank] = len(send_data)
    else:
        send_data = send_data.new_empty(0)
        recv_data = recv_data.new_empty(0)

    dist.all_to_all_single(output=recv_data, input=send_data, output_split_sizes=io_sizes, input_split_sizes=io_sizes)
