"""
Utilities
"""

import gzip
import pickle

import numpy as np
import psutil
import torch

import deepquantum.photonic as dqp


def set_hbar(hbar: float) -> None:
    """Set the global reduced Planck constant."""
    dqp.hbar = hbar

def set_kappa(kappa: float) -> None:
    """Set the global kappa."""
    dqp.kappa = kappa

def load_sample(filename):
    """load the sample data with the given filename"""
    with gzip.open('./data/' + filename + '.pkl.gz','rb') as f:
        sample = pickle.load(f)
    return sample

def save_sample(filename, data):
    """save the sample data with the given filename"""
    with gzip.open('./data/' + filename + '.pkl.gz','wb') as f:
        pickle.dump(data, f)
    return

def load_adj(filename):
    """load the adjacent matrix with the given filename"""
    mat = np.load('./data/' + filename + '.npy')
    return mat

def save_adj(filename, data):
    """save the adjacent matrix with the given filename"""
    np.save('./data/' + filename + '.npy', data)
    return

def mem_to_chunksize(device: torch.device, dtype: torch.dtype) -> int | None:
    """Return the chunk size of vmap according to device free memory and dtype.
    
    Note: Currently only optimized for permanent and complex dtype.
    """
    if (device, dtype) in dqp.perm_chunksize_dict:
        return dqp.perm_chunksize_dict[device, dtype]
    if device == torch.device('cpu'):
        mem_free_gb = psutil.virtual_memory().free / 1024**3
    else:
        mem_free_gb = torch.cuda.mem_get_info(device=device)[0] / 1024**3
    if dtype == torch.cfloat:
        if mem_free_gb > 80:
            # requires checking when we have such GPUs:)
            chunksize = int(5e6)
        elif mem_free_gb > 60:
            chunksize = int(2e6)
        elif mem_free_gb > 20:
            chunksize = int(6e5)
        elif mem_free_gb > 8:
            chunksize = int(2e5)
        elif mem_free_gb > 5:
            chunksize = int(1.25e5)
        else:
            chunksize = int(2e4)
    elif dtype == torch.cdouble:
        if mem_free_gb > 80:
            chunksize = int(2e6)
        elif mem_free_gb > 60:
            chunksize = int(1e6)
        elif mem_free_gb > 20:
            chunksize = int(3e5)
        elif mem_free_gb > 8:
            chunksize = int(1e5)
        elif mem_free_gb > 5:
            chunksize = int(6e4)
        else:
            chunksize = int(1e4)
    else:
        chunksize = None
    dqp.perm_chunksize_dict[device, dtype] = chunksize
    return chunksize

def set_perm_chunksize(device: torch.device, dtype: torch.dtype, chunksize: int | None) -> None:
    """Set the global chunk size for permanent calculations."""
    dqp.perm_chunksize_dict[device, dtype] = chunksize
