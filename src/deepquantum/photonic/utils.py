"""
Utilities
"""

import gzip
import pickle

import deepquantum.photonic as dqp
import numpy as np
import torch
import psutil


def set_hbar(hbar: float) -> None:
    """Set the global reduced Planck constant."""
    dqp.hbar = hbar

def set_kappa(kappa: float) -> None:
    """Set the global kappa."""
    dqp.kappa = kappa

def load_sample(filename):
    """"load the sample data with the given filename"""
    with gzip.open('./data/' + filename + '.pkl.gz','rb') as f:
        sample = pickle.load(f)
    return sample

def save_sample(filename, data):
    """"save the sample data with the given filename"""
    with gzip.open('./data/' + filename + '.pkl.gz','wb') as f:
        pickle.dump(data, f)
    return

def load_adj(filename):
    """"load the adjacent matrix with the given filename"""
    mat = np.load('./data/' + filename + '.npy')
    return mat

def save_adj(filename, data):
    """"save the adjacent matrix with the given filename"""
    np.save('./data/' + filename + '.npy', data)
    return

def mem_to_batchsize(device, dtype):
    if device == torch.device('cpu'):
        mem_free_gb = psutil.virtual_memory().free/1024**3
    else:
        mem_free_gb = torch.cuda.mem_get_info(device=device)[0]/1024**3
    if dtype == torch.complex64:
        if mem_free_gb > 80:
            # requires checking when we have such GPUs:)
            batch_size = int(1e7)
        elif mem_free_gb > 50:
            batch_size = int(8e6)
        elif mem_free_gb > 20:
            batch_size = int(6e6)
        elif mem_free_gb > 8:
            batch_size = int(5e6)
        elif mem_free_gb > 5:
            # set for PC gpu whose free memory shows only dedicated GPU memory,
            # while the total memory(including shared mem) is around 20 GB
            batch_size = int(5e6)
        else:
            batch_size = int(2e5)
    else:
        if mem_free_gb > 80:
            batch_size = int(1e7)
        elif mem_free_gb > 50:
            batch_size = int(5e6)
        elif mem_free_gb > 20:
            batch_size = int(4e6)
        elif mem_free_gb > 8:
            batch_size = int(1e6)
        elif mem_free_gb > 5:
            batch_size = int(5e5)
        else:
            batch_size = int(1e5)
    return batch_size
