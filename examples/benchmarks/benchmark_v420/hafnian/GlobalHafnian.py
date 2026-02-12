import torch

n_list = [2, 6, 10, 14]
l_list = [1, 10, 100]
number_of_sequence = 1000

device = 'cpu'


def random_unitary(nmodes, dtype=torch.complex64, device=device):
    n = nmodes
    """Generate a random unitary matrix of size n x n using only PyTorch."""
    # 随机复数矩阵 A = Re + i Im
    real = torch.rand((n, n), dtype=torch.float32, device=device)
    imag = torch.rand((n, n), dtype=torch.float32, device=device)
    a = torch.complex(real, imag)

    # QR分解
    q, r = torch.linalg.qr(a)

    # 修正 Q 的相位：让 R 的对角线变成正实数
    diag = torch.diagonal(r, 0)
    phase = diag / torch.abs(diag)
    u = q * phase.unsqueeze(0)  # 广播乘法

    return u.to(dtype)


def random_covariance(nmodes, dtype=torch.complex64, device=device):
    n = nmodes * 2
    """Generate a random covariance matrix of size n x n using only PyTorch."""
    u = random_unitary(n, dtype=dtype, device=device)
    d = torch.rand(n, dtype=torch.float32, device=device)
    mat_d = torch.diag(d).to(u.dtype)
    cov = u @ mat_d @ u.conj().T
    return cov.to(dtype)


def random_hafnian_matrix(cov):
    """Generate a random matrix for hafnian calculation."""
    nmodes = cov.shape[0] // 2
    n = nmodes
    q = cov + torch.eye(n * 2, dtype=cov.dtype, device=cov.device) / 2

    identity = torch.eye(n, dtype=cov.dtype, device=cov.device)
    zeros = torch.zeros((n, n), dtype=cov.dtype, device=cov.device)

    x_top = torch.cat([zeros, identity], dim=1)
    x_bottom = torch.cat([identity, zeros], dim=1)
    x = torch.cat([x_top, x_bottom], dim=0)

    q_inv = torch.inverse(q)
    a = x @ (torch.eye(n * 2, dtype=cov.dtype, device=cov.device) - q_inv)
    return a


def test_sequence_hafnian(nmode, number_of_sequence=64, device=device, start=None, end=None):
    """Generate a sequence of hafnian matrices."""

    # 判断文件是否存在，如果不存在，则保存矩阵U(复数)
    try:
        u = torch.load(f'hafnian/hafnian_matrix_{nmode}_{number_of_sequence}.pt')
        u = u[start, end].to(device) if start is not None else u.to(device)
    except FileNotFoundError:
        print(f'File hafnian_matrix_{nmode}_{number_of_sequence}.pt not found. Saving matrix U.')

        cov = random_covariance(nmode, device=device)
        u = torch.zeros((number_of_sequence, nmode * 2, nmode * 2), dtype=cov.dtype, device=device)
        for i in range(number_of_sequence):
            # Generate a random covariance matrix
            cov = random_covariance(nmode, device=device)
            a = random_hafnian_matrix(cov)
            a = (a + a.mT) / 2
            # 把矩阵A添加到U中
            u[i] = a
        # Save the matrix U to a file
        torch.save(u, f'hafnian/hafnian_matrix_{nmode}_{number_of_sequence}.pt')
        print('done')

    return u
