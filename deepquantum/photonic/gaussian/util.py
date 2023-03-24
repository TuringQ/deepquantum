"""Basic operations needed for Gaussian"""
import numpy as np
import torch
from scipy.stats import unitary_group


def split_covariance(covariance, modes_id):
    """
    Split the covariance matrix into three parts according to the IDs of modes.
    
    covariance: the covariance matrix with batch, torch tensor.
    modes_id: the IDs of modes, torch tensor.
    
    Output:
    A, the left covariance matrix after deleting the rows and columns corresponding to modes_id.
    B, delete the rows of the matrix by modes_id and select the columns.
    C, select the rows and columns of the matrix by modes_id.
    """
    modes_id = modes_id.numpy()
    A = torch.clone(covariance).numpy()
    A = np.delete(np.delete(A, modes_id, axis=1), modes_id, axis=2)
    
    B = torch.clone(covariance).numpy()
    B = np.delete(B[:, :, modes_id], modes_id, axis=1)
    
    C = torch.clone(covariance).numpy()
    C = C[:, :, modes_id][:, modes_id, :]
    
    return (torch.from_numpy(A), torch.from_numpy(B), torch.from_numpy(C))
    
    
def split_mean(mean, modes_id):
    """
    Split the mean vector into two vectors according to the IDs of modes.
    
    Input:
    mean, the mean vector, torch tensor.
    modes_id, the IDs of modes, torch, tensor.
    
    Output:
    v_c, the mean vector corresponding to the modes_id.
    v_a, the left vector after deleting the elements according to modes_id.
    """
    v = torch.clone(mean).numpy()
    v_c = v[:, modes_id]
    v_a = np.delete(v, modes_id, axis=1)
    
    return (torch.from_numpy(v_a), torch.from_numpy(v_c))


def embed_to_covariance(matrix, modes_id, dtype):
    """
    Embed a matrix into its original covariance matrix with dimension being len(modes_id)+dim(matrix).
    
    Input:
    matrix, the matrix to be embeded, torch tensor.
    modes_id, the IDs of added modes, torch tensor.
    
    Output:
    cov, a larger matrix.
    """
    n = matrix.shape[1] + modes_id.shape[0]
    indices = set(np.arange(n)) - set(modes_id.numpy())
    cov = torch.stack([torch.eye(n, dtype=dtype)] * matrix.shape[0])
    
    for i, i1 in enumerate(indices):
        for j, j1 in enumerate(indices):
            cov[:, i1, j1] = matrix[:, i, j]
    
    return cov


def embed_to_mean(mean, quad_id, dtype):
    """
    Embed a vector into its original mean vector according to the IDs of quadrature.
    mean: torch tensor.
    quad_id: torch tensor.
    """
    n = mean.shape[1] + quad_id.shape[0]
    indices = set(np.arange(n)) - set(quad_id.numpy())
    new_mean = torch.zeros(mean.shape[0], n, dtype=dtype)
    
    for i, i1 in enumerate(indices):
        new_mean[:, i1] = mean[:, i]
        
    return new_mean



def xxpp_to_xpxp(mean, cov, n_mode, dtype):
    """
    Transform the representation in xxpp ordering to the representation in xpxp ordering.
    """
    # transformation matrix
    t = torch.zeros((2*n_mode, 2*n_mode), dtype=dtype)
    for i in range(2*n_mode):
        if i % 2 == 0:
            t[i][i//2] = 1
        else:
            t[i][i//2+n_mode] = 1
    # new mean vector in xpxp ordering
    new_mean = torch.transpose(t @ mean.T, 0, 1)
    # new covariance matrix in xpxp ordering
    new_cov = t @ cov @ t.T
    
    return new_mean, new_cov


def xpxp_to_xxpp(mean, cov, n_mode, dtype):
    """
    Transform the representation in xpxp ordering to the representation in xxpp ordering.
    """
    # transformation matrix
    t = torch.zeros((2*n_mode, 2*n_mode), dtype=dtype)
    for i in range(2*n_mode):
        if i < n_mode:
            t[i][2*i] = 1
        else:
            t[i][2*(i-n_mode)+1] = 1
    # new mean in xxpp ordering
    new_mean = torch.transpose(t @ mean.T, 0, 1)
    # new covariance matrix in xxpp ordering
    new_cov = t @ cov @ t.T

    return new_mean, new_cov



def lambda_xpxp(n_mode, dtype):
    """
    Generate the matrix appearing in the cummutation relations in xpxp ordering.
    """
    t = torch.zeros((2*n_mode, 2*n_mode), dtype=dtype)
    for i in range(2*n_mode):
        if i % 2 == 0:
            t[i][i + 1] = 1
        else:
            t[i][i - 1] = -1
    return t



def double_partial(disp, cov, i):
    """
    Calculate the double partial of gauss function with respect to ith variable.
    """
    return - cov[:, i, i] - disp[:, i] * 2



def two_double_partial(disp, cov, i, j):
    """
    Calculate the double partial of gauss function with respect to i, and then w.r.t j.
    """
    t1 = (cov[:, i, i] + disp[:, i] * 2) * (cov[:, j, j] + disp[:, j] * 2)
    t2 = cov[:, i, j] * 2
    t3 = 2 * disp[:, i] * disp[:, j] * cov[:, i, j]
    
    return t1 + t2 + t3
    

def random_sympectic(mode_number):
    """
    Generate a random haar unitary matrix and construct a corresponding sympectic matrix.
    """
    # generate a random unitary matrix [mode_number, mode_number], representing a random linear optical circuit acting on creation operators
    u = torch.tensor(unitary_group.rvs(mode_number))
    # construct sympectic matrix
    return torch.cat([torch.cat([u, torch.zeros(mode_number, mode_number)], dim=1), torch.cat([torch.zeros(mode_number, mode_number), torch.conj(u)], dim=1)], dim=0)

