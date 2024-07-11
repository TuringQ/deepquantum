"""
Utilities
"""

import numpy as np
import os
import pickle
import gzip
import deepquantum.photonic as dqp


def set_hbar(hbar: float) -> None:
    """Set the global reduced Planck constant."""
    dqp.hbar = hbar

def set_kappa(kappa: float) -> None:
    """Set the global kappa."""
    dqp.kappa = kappa

def load_sample(filename):
    """"load the sample data with the given filename"""
    # dir_ = os.path.dirname(__file__)
    # file_path = os.path.join(dir_, filename+'.pkl.gz')
    with gzip.open(filename+'.pkl.gz','rb') as f:
        sample = pickle.load(f)
    return sample

def save_sample(filename, data):
    """"save the sample data with the given filename"""
    with gzip.open(filename+'.pkl.gz','wb') as f:
        pickle.dump(data, f)
    return

def load_adj(filename):
    mat = np.load(filename+'.npy')
    return mat

def save_adj(filename, data):
    """"save the adjacent matrix with the given filename"""
    np.save(filename+'.npy', data)
    return
