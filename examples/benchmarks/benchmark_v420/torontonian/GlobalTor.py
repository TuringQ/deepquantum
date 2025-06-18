import torch

from tqdm import tqdm

n_list = [2, 6, 10, 14, 18]

number_of_sequence = 1000

device = 'cpu'

import numpy as np
from thewalrus.random import random_covariance
from thewalrus.quantum.conversions import Amat, Xmat

np.random.seed(42)

def generate_psd_matrix(n):

    cov = random_covariance(n)
    O = Xmat(n) @ Amat(cov)
    return O

for nmode in tqdm(n_list):
    U = torch.zeros((number_of_sequence, nmode * 2, nmode * 2), dtype=torch.complex128, device=device)
    for j in tqdm(range(number_of_sequence)):
        # Generate a random covariance matrix
        U[j] = torch.tensor(generate_psd_matrix(nmode), device=device)
    # Save the matrix U to a file
    torch.save(U, f"tor_matrix_{nmode}_{number_of_sequence}.pt")
print('done')
