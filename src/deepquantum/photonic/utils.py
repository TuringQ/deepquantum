"""Utilities"""

import gzip
import pickle
from bisect import bisect_left
from functools import cache
from operator import index

import numpy as np
import psutil
import torch

import deepquantum.photonic as dqp

_DEFAULT_CHUNK_BYTES = 8 * 1024**2
_chunk_budgets: dict[torch.device, int] = {}


def set_hbar(hbar: float) -> None:
    """Set the global reduced Planck constant."""
    dqp.hbar = hbar


def set_kappa(kappa: float) -> None:
    """Set the global kappa."""
    dqp.kappa = kappa


def load_sample(filename):
    """Load the sample data with the given filename"""
    with gzip.open('./data/' + filename + '.pkl.gz', 'rb') as f:
        sample = pickle.load(f)
    return sample


def save_sample(filename, data):
    """Save the sample data with the given filename"""
    with gzip.open('./data/' + filename + '.pkl.gz', 'wb') as f:
        pickle.dump(data, f)
    return


def load_adj(filename):
    """Load the adjacent matrix with the given filename"""
    mat = np.load('./data/' + filename + '.npy')
    return mat


def save_adj(filename, data):
    """Save the adjacent matrix with the given filename"""
    np.save('./data/' + filename + '.npy', data)
    return


def set_chunk_budget(device: torch.device | str, nbytes: int | None) -> None:
    """Set the working-memory budget for automatic photonic readout chunks on a device.

    Ryser permanents size their subsets and Gaussian ``get_prob`` queries size their input
    batches from the estimated bytes in an intermediate tensor. Other tensors, enclosing
    ``vmap`` calls and autograd can use additional memory. An explicit ``set_perm_chunksize``
    takes precedence for permanents of that device and dtype.

    Args:
        device: Target device. An unindexed accelerator sets the default for that device type.
        nbytes: Positive number of bytes. Pass ``None`` to restore the inherited or 8 MiB default.
    """
    device = torch.device(device)
    if device.type == 'cpu':
        device = torch.device('cpu')
    if nbytes is None:
        _chunk_budgets.pop(device, None)
        return
    if isinstance(nbytes, bool) or nbytes <= 0:
        raise ValueError('nbytes must be a positive integer')
    _chunk_budgets[device] = index(nbytes)


def _chunk_capacity(device: torch.device, bytes_per_item: int) -> int:
    """Return the number of work items that fit the device budget, with a minimum of one."""
    budget = _chunk_budgets.get(device)
    if budget is None and device.index is not None:
        budget = _chunk_budgets.get(torch.device(device.type))
    return max(1, (budget or _DEFAULT_CHUNK_BYTES) // bytes_per_item)


@cache
def _default_perm_chunksize(device: torch.device, dtype: torch.dtype) -> int | None:
    """Return the calibrated Ryser subset limit for the device and precision."""
    if device.type == 'cpu':
        mem_free_gb = psutil.virtual_memory().free / 1024**3
    elif device.type == 'cuda':
        mem_free_gb = torch.cuda.mem_get_info(device=device)[0] / 1024**3
    elif device.type == 'mps':
        mem_free_gb = torch.mps.recommended_max_memory() / 1024**3
    else:
        raise ValueError(f'Unknown device type: {device.type}')
    tier = bisect_left((5, 8, 20, 60, 80), mem_free_gb)
    if dtype == torch.cfloat:
        return (20_000, 125_000, 200_000, 600_000, 2_000_000, 5_000_000)[tier]
    if dtype == torch.cdouble:
        return (10_000, 60_000, 100_000, 300_000, 1_000_000, 2_000_000)[tier]
    return None


def mem_to_chunksize(device: torch.device, dtype: torch.dtype, *, elements_per_subset: int | None = None) -> int | None:
    """Return the Ryser subset limit for a device and dtype.

    Supplying the number of gathered matrix elements per subset also applies the byte budget
    to automatic chunks. A manual subset limit, including ``None``, takes precedence.
    """
    if elements_per_subset is not None:
        elements_per_subset = index(elements_per_subset)
        if elements_per_subset <= 0:
            raise ValueError('elements_per_subset must be positive')
    if (device, dtype) in dqp.perm_chunksize_dict:
        return dqp.perm_chunksize_dict[device, dtype]
    limit = _default_perm_chunksize(device, dtype)
    if elements_per_subset is None:
        return limit
    capacity = _chunk_capacity(device, dtype.itemsize * elements_per_subset)
    return capacity if limit is None else min(capacity, limit)


def set_perm_chunksize(device: torch.device, dtype: torch.dtype, chunksize: int | None) -> None:
    """Override automatic chunk sizing for Ryser permanents. ``None`` evaluates all subsets of each size together."""
    dqp.perm_chunksize_dict[device, dtype] = chunksize
