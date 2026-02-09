import torch
from tqdm import tqdm

n_list = [2, 6, 10, 14, 18, 22, 26, 30]
number_of_sequence = 1000

device = 'cpu'

import numpy as np
from scipy.stats import unitary_group

np.random.seed(42)

for nmode in tqdm(n_list):
    U = torch.zeros((number_of_sequence, nmode, nmode), dtype=torch.complex128, device=device)
    for j in range(number_of_sequence):
        # Generate a random covariance matrix
        U[j] = torch.tensor(unitary_group.rvs(nmode), device=device)
    # Save the matrix U to a file
    torch.save(U, f'u_matrix_{nmode}_{number_of_sequence}.pt')
print('done')
