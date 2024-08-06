"""
Utilities
"""

import gzip
import pickle

import deepquantum.photonic as dqp
import numpy as np


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
