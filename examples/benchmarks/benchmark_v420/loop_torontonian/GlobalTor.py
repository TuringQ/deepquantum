import numpy as np
import torch
from thewalrus.quantum.conversions import Amat, Xmat
from thewalrus.random import random_covariance
from tqdm import tqdm

n_list = [2, 6, 10, 14, 18]

number_of_sequence = 1000

device = 'cpu'

np.random.seed(42)


def generate_psd_matrix(n):
    cov = random_covariance(n)
    o = Xmat(n) @ Amat(cov)
    return o


for nmode in tqdm(n_list):
    u = torch.zeros((number_of_sequence, nmode * 2, nmode * 2), dtype=torch.complex128, device=device)
    for j in tqdm(range(number_of_sequence)):
        # Generate a random covariance matrix
        u[j] = torch.tensor(generate_psd_matrix(nmode), device=device)
    # Save the matrix U to a file
    torch.save(u, f'tor_matrix_{nmode}_{number_of_sequence}.pt')
print('done')
