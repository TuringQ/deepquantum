import numpy as np
import torch
from scipy.stats import unitary_group
from tqdm import tqdm

n_list = [2, 6, 10, 14, 18, 22, 26, 30]
number_of_sequence = 1000

device = 'cpu'

np.random.seed(42)

for nmode in tqdm(n_list):
    u = torch.zeros((number_of_sequence, nmode, nmode), dtype=torch.complex128, device=device)
    for j in range(number_of_sequence):
        # Generate a random covariance matrix
        u[j] = torch.tensor(unitary_group.rvs(nmode), device=device)
    # Save the matrix U to a file
    torch.save(u, f'u_matrix_{nmode}_{number_of_sequence}.pt')
print('done')
