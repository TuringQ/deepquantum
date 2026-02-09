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
    A = torch.complex(real, imag)

    # QR分解
    Q, R = torch.linalg.qr(A)

    # 修正 Q 的相位：让 R 的对角线变成正实数
    diag = torch.diagonal(R, 0)
    phase = diag / torch.abs(diag)
    U = Q * phase.unsqueeze(0)  # 广播乘法

    return U.to(dtype)


def random_covariance(nmodes, dtype=torch.complex64, device=device):
    n = nmodes * 2
    """Generate a random covariance matrix of size n x n using only PyTorch."""
    U = random_unitary(n, dtype=dtype, device=device)
    d = torch.rand(n, dtype=torch.float32, device=device)
    D = torch.diag(d).to(U.dtype)
    cov = U @ D @ U.conj().T
    return cov.to(dtype)


def random_hafnian_matrix(cov):
    """Generate a random matrix for hafnian calculation."""
    nmodes = cov.shape[0] // 2
    n = nmodes
    Q = cov + torch.eye(n * 2, dtype=cov.dtype, device=cov.device) / 2

    I = torch.eye(n, dtype=cov.dtype, device=cov.device)
    Z = torch.zeros((n, n), dtype=cov.dtype, device=cov.device)

    X_top = torch.cat([Z, I], dim=1)
    X_bottom = torch.cat([I, Z], dim=1)
    X = torch.cat([X_top, X_bottom], dim=0)

    Q_inv = torch.inverse(Q)
    A = X @ (torch.eye(n * 2, dtype=cov.dtype, device=cov.device) - Q_inv)
    return A


def test_sequence_hafnian(nmode, number_of_sequence=64, device=device, start=None, end=None):
    """Generate a sequence of hafnian matrices."""

    # 判断文件是否存在，如果不存在，则保存矩阵U(复数)
    try:
        U = torch.load(f'hafnian/hafnian_matrix_{nmode}_{number_of_sequence}.pt')
        if start != None:
            U = U[start, end].to(device)
        else:
            U = U.to(device)
    except FileNotFoundError:
        print(f'File hafnian_matrix_{nmode}_{number_of_sequence}.pt not found. Saving matrix U.')

        cov = random_covariance(nmode, device=device)
        U = torch.zeros((number_of_sequence, nmode * 2, nmode * 2), dtype=cov.dtype, device=device)
        for i in range(number_of_sequence):
            # Generate a random covariance matrix
            cov = random_covariance(nmode, device=device)
            A = random_hafnian_matrix(cov)
            A = (A + A.mT) / 2
            # 把矩阵A添加到U中
            U[i] = A
        # Save the matrix U to a file
        torch.save(U, f'hafnian/hafnian_matrix_{nmode}_{number_of_sequence}.pt')
        print('done')

    return U
