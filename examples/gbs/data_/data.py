"""
functions for loading and saving data
"""

import os
import pickle
import gzip
import numpy as np

dir_ = os.path.dirname(__file__)

def load_sample(filename):
    """"load the sample data with the given filename"""
    file_path = os.path.join(dir_, filename+'.pkl.gz')
    with gzip.open(file_path,'rb') as f:
        sample = pickle.load(f)
    return sample

def save_sample(filename, data):
    """"save the sample data with the given filename"""
    with gzip.open(filename+'.pkl.gz','wb') as f:
        pickle.dump(data, f)
    return

def load_adj(filename):
    """"load the adjacent matrix with the given filename"""
    file_path = os.path.join(dir_, filename+'.npy')
    mat = np.load(file_path)
    return mat

def save_adj(filename, data):
    """"save the adjacent matrix with the given filename"""
    np.save(filename+'.npy', data)
    return
