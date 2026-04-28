"""Decompose the unitary matrix"""

from collections import defaultdict

import numpy as np
import torch


class UnitaryDecomposer:
    """This class is to decompose a unitary matrix into the Clements/Reck architecture.

    Args:
        unitary (np.ndarray or torch.Tensor): The unitary matrix to be decomposed.
        method (str, optional): The decomposition method, only 16 values (``'rssr'``, ``'rsdr'``, ``'rdsr'``,
            ``'rddr'``, ``'rssl'``, ``'rsdl'``, ``'rdsl'``, ``'rddl'``, ``'cssr'``, ``'csdr'``, ``'cdsr'``, ``'cddr'``,
            ``'cssl'``, ``'csdl'``, ``'cdsl'``, ``'cddl'``) are valid.
            The first char denotes the Clements or Reck architecture.
            The second char denotes single or double arms of outer phase shifters.
            The third char denotes single or double arms of inner phase shifters.
            The last char denotes the position of a column of phase shifters, i.e., ``'l'`` for left
            and ``'r'`` for right. Default: ``'cssr'``
    """

    def __init__(self, unitary: np.ndarray | torch.Tensor, method: str = 'cssr') -> None:
        if isinstance(unitary, np.ndarray):
            self.unitary = unitary.copy()
            if np.abs(unitary @ unitary.conj().T - np.eye(len(unitary))).sum() / len(unitary) ** 2 > 1e-6:
                print('Make sure the input matrix is unitary, in case of an abnormal computation result.')
            self.unitary[np.abs(self.unitary) < 1e-32] = 1e-32
        elif isinstance(unitary, torch.Tensor):
            if unitary.dim() == 2:
                unitary = unitary.unsqueeze(0)
            self.unitary = unitary + 0j
            if torch.abs(unitary @ unitary.mH - torch.eye(unitary.shape[-1])).sum() / unitary.shape[-1] ** 2 > 1e-6:
                print('Make sure the input matrix is unitary, in case of an abnormal computation result.')
            self.unitary[torch.abs(self.unitary) < 1e-32] = 1e-32
        else:
            raise TypeError('The matrix to be decomposed must be in the type of numpy array or torch tensor.')
        if self.unitary.shape[-1] != self.unitary.shape[-2]:
            raise TypeError('The matrix to be decomposed must be a square matrix.')
        self.method = method

    def decomp(self) -> tuple[dict, dict, dict]:
        """Decompose the unitary matrix.

        The third dictionary is the representation of the positions and the angles of all phase shifters.
        """
        method = self.method
        if method not in [
            'rssr',
            'rsdr',
            'rdsr',
            'rddr',
            'rssl',
            'rsdl',
            'rdsl',
            'rddl',
            'cssr',
            'csdr',
            'cdsr',
            'cddr',
            'cssl',
            'csdl',
            'cdsl',
            'cddl',
        ]:
            raise LookupError('请检查分解方式！')
        elif method[0] == 'c':  # clements
            if isinstance(self.unitary, torch.Tensor):
                temp_0 = self.decomp_c_like_torch(self.unitary, method, method[-1])[0]
                temp_1 = self.sort_mzi_torch(temp_0)
                temp_2 = self.ps_pos_torch(temp_1, temp_0['phase_angle'])
                return temp_0, temp_1, temp_2
            else:
                temp_0 = self.decomp_c_like(self.unitary, method, method[-1])[0]
        elif method[0] == 'r':  # reck
            if isinstance(self.unitary, torch.Tensor):
                raise NotImplementedError('Reck decomposition not implemented for torch.tensor')
            else:
                temp_0 = self.decomp_r_like(self.unitary, method, method[-1])[0]
        temp_1 = self.sort_mzi(temp_0)
        temp_2 = self.ps_pos(temp_1, temp_0['phase_angle'])
        return temp_0, temp_1, temp_2

    def decomp_r_like(self, unitary: np.ndarray, method: str, mode: str) -> dict:
        """Decompose a unitary using the Reck structure.

        This helper unifies the behavior of the original `decomp_rr` and `decomp_rl`
        variants, controlled by `mode`, `mode=r` or `mode=l`.
        """
        n_dim = len(unitary)
        info = {}
        info['N'] = n_dim
        info['method'] = method
        info['MZI_list'] = []  # jj,ii,phi,theta
        period_theta, period_phi = self.get_periods(method)
        for i in range(n_dim):
            ii = n_dim - 1 - i  # 基准列 ii
            for jj in range(ii)[::-1]:
                if mode == 'r':
                    ratio = unitary[ii, ii] / (unitary[ii, jj] + 1e-32)
                    theta = 2 * np.arctan(np.abs(ratio))
                    phi = -np.angle(-ratio)
                    multiple = self.get_matrix_inverse_r([jj, ii, phi, theta], n_dim, method)
                    unitary = unitary @ multiple
                if mode == 'l':
                    ratio = unitary[ii, ii] / (unitary[jj, ii] + 1e-32)
                    theta = 2 * np.arctan(np.abs(ratio))
                    phi = -np.angle(-ratio)
                    multiple = self.get_matrix_inverse_l([jj, ii, phi, theta], n_dim, method)
                    unitary = multiple @ unitary
                phi = self.period_cut(phi, period_phi)
                theta = self.period_cut(theta, period_theta)
                info['MZI_list'].append([jj, ii, phi, theta])
        diagonal = np.diag(unitary)
        info['phase_angle'] = np.angle(diagonal)
        mask = np.logical_or(info['phase_angle'] >= 2 * np.pi, info['phase_angle'] < 0)
        info['phase_angle'][mask] -= np.floor(info['phase_angle'][mask] / np.pi / 2) * np.pi * 2
        return info, unitary

    def decomp_c_like(self, unitary: np.ndarray, method: str, mode: str) -> dict:
        """Decompose a unitary using the Clements structure.

        This helper unifies the behavior of the original `decomp_cr` and `decomp_cl`
        variants, controlled by `mode`, `mode=r` or `mode=l`.
        """
        assert mode == 'r', 'cssl method is not supported'
        n_dim = len(unitary)
        info = {}
        info['N'] = n_dim
        info['method'] = method
        info['MZI_list'] = []  # jj,ii,phi,theta
        info['right'] = []
        info['left'] = []
        period_theta, period_phi = self.get_periods(method)
        for i in range(n_dim - 1):  # 从下往上第i个反对角线
            if i % 2:  # 左乘, 利用TU消元；
                for j in range(i + 1):  # 反对角线的元素计数
                    # 消元顺序：从左上到右下
                    jj = j  # 当前待消元元素列号
                    ii = n_dim - 1 - i + j  # 当前待消元元素行号
                    # print(ii,jj)
                    # if unitary[ii,jj] == 0:
                    #     continue
                    ratio = unitary[ii - 1, jj] / (unitary[ii, jj] + 1e-32)
                    theta = 2 * np.arctan(np.abs(ratio))
                    if mode == 'r':
                        phi = -np.angle(ratio)
                        multiple = self.get_matrix_constr_r([ii - 1, ii, phi, theta], n_dim, method)
                    if mode == 'l':
                        phi = np.angle(ratio)
                        multiple = self.get_matrix_inverse_l([ii - 1, ii, phi, theta], n_dim, method)
                    unitary = multiple @ unitary
                    info['left'].append([ii - 1, ii, phi, theta])
            else:  # 利用UT^{-1}消元，即利用 unitary[ii,jj+1] 消去 unitary[ii,jj]
                for j in range(i + 1)[::-1]:  # 反对角线的元素计数
                    # 消元顺序：从右下到左上
                    jj = j  # 当前待消元元素列号
                    ii = n_dim - 1 - i + j  # 当前待消元元素行号
                    # print(ii,jj)
                    # if unitary[ii,jj] == 0:
                    #     continue
                    ratio = unitary[ii, jj + 1] / (unitary[ii, jj] + 1e-32)
                    theta = 2 * np.arctan(np.abs(ratio))
                    if mode == 'r':
                        phi = -np.angle(-ratio)
                        multiple = self.get_matrix_inverse_r([jj, jj + 1, phi, theta], n_dim, method)
                    if mode == 'l':
                        phi = np.angle(-ratio)
                        multiple = self.get_matrix_constr_l([jj, jj + 1, phi, theta], n_dim, method)
                    unitary = unitary @ multiple
                    info['right'].append([jj, jj + 1, phi, theta])
        phase_angle = np.angle(np.diag(unitary))
        info['phase_angle_ori'] = phase_angle.copy()  # unitary=LLLDRRR，本行保存D
        if mode == 'r':
            for idx in range(len(info['right'])):
                info['right'][idx][2] = self.period_cut(info['right'][idx][2], period_phi)
                info['right'][idx][3] = self.period_cut(info['right'][idx][3], period_theta)
                info['MZI_list'].append(info['right'][idx])
            left_list = info['left'][::-1]
        if mode == 'l':
            for idx in range(len(info['left'])):
                info['left'][idx][2] = self.period_cut(info['left'][idx][2], period_phi)
                info['left'][idx][3] = self.period_cut(info['left'][idx][3], period_theta)
                info['MZI_list'].append(info['left'][idx])
            left_list = info['right'][::-1]

        for idx in range(len(left_list)):
            jj, ii, phi, theta = left_list[idx]
            phi_, theta_, phase_angle[jj], phase_angle[ii] = self.clements_diagonal_transform(
                phi, theta, phase_angle[jj], phase_angle[ii], method
            )
            phi_ = self.period_cut(phi_, period_phi)
            theta_ = self.period_cut(theta_, period_theta)
            info['MZI_list'].append([jj, ii, phi_, theta_])
        info['phase_angle'] = phase_angle.copy()  # unitary=D'L'L'L'RRR,本行保存新的D
        mask = np.logical_or(info['phase_angle'] >= 2 * np.pi, info['phase_angle'] < 0)
        info['phase_angle'][mask] -= np.floor(info['phase_angle'][mask] / np.pi / 2) * np.pi * 2
        return info, unitary

    def decomp_c_like_torch(self, unitary: torch.Tensor, method: str, mode: str) -> dict:
        """Decompose a unitary using the Clements structure.

        This helper unifies the behavior of the original `decomp_cr` and `decomp_cl`
        variants, controlled by `mode`, `mode=r` or `mode=l`.
        """
        assert mode == 'r', 'cssl method is not supported'
        n_dim = unitary.shape[-1]
        batch = unitary.shape[0]
        dtype = unitary.dtype

        info = {}
        info['N'] = n_dim
        info['method'] = method
        info['MZI_list'] = []  # jj,ii,phi,theta
        info['right'] = []
        info['left'] = []
        period_theta, period_phi = self.get_periods(method)
        for i in range(n_dim - 1):  # 从下往上第i个反对角线
            if i % 2:  # 左乘, 利用TU消元；
                for j in range(i + 1):  # 反对角线的元素计数
                    # 消元顺序：从左上到右下
                    jj = j  # 当前待消元元素列号
                    ii = n_dim - 1 - i + j  # 当前待消元元素行号
                    # print(ii,jj)
                    # if unitary[ii,jj] == 0:
                    #     continue
                    ratio = unitary[:, ii - 1, jj] / (unitary[:, ii, jj] + 1e-32)
                    theta = 2 * torch.arctan(abs(ratio))
                    if mode == 'r':
                        phi = -torch.angle(ratio)
                        multiple = self.get_matrix_constr_r_torch([ii - 1, ii, phi, theta], n_dim, batch, method, dtype)
                    if mode == 'l':
                        phi = torch.angle(ratio)
                        multiple = self.get_matrix_inverse_l_torch(
                            [ii - 1, ii, phi, theta], n_dim, batch, method, dtype
                        )
                    unitary = multiple @ unitary
                    info['left'].append([ii - 1, ii, phi, theta])
            else:  # 利用UT^{-1}消元，即利用 unitary[ii,jj+1] 消去 unitary[ii,jj]
                for j in range(i + 1)[::-1]:  # 反对角线的元素计数
                    # 消元顺序：从右下到左上
                    jj = j  # 当前待消元元素列号
                    ii = n_dim - 1 - i + j  # 当前待消元元素行号
                    # print(ii,jj)
                    # if unitary[ii,jj] == 0:
                    #     continue
                    ratio = unitary[:, ii, jj + 1] / (unitary[:, ii, jj] + 1e-32)
                    theta = 2 * torch.arctan(abs(ratio))
                    if mode == 'r':
                        phi = -torch.angle(-ratio)
                        multiple = self.get_matrix_inverse_r_torch(
                            [jj, jj + 1, phi, theta], n_dim, batch, method, dtype
                        )
                    if mode == 'l':
                        phi = torch.angle(-ratio)
                        multiple = self.get_matrix_constr_l_torch([jj, jj + 1, phi, theta], n_dim, batch, method, dtype)
                    unitary = unitary @ multiple
                    info['right'].append([jj, jj + 1, phi, theta])
        phase_angle = torch.angle(torch.diagonal(unitary, dim1=1, dim2=2))
        info['phase_angle_ori'] = phase_angle.clone()  # unitary=LLLDRRR，本行保存D
        if mode == 'r':
            for idx in range(len(info['right'])):
                info['right'][idx][2] = self.period_cut(info['right'][idx][2], period_phi)
                info['right'][idx][3] = self.period_cut(info['right'][idx][3], period_theta)
                info['MZI_list'].append(info['right'][idx])
            left_list = info['left'][::-1]
        if mode == 'l':
            for idx in range(len(info['left'])):
                info['left'][idx][2] = self.period_cut(info['left'][idx][2], period_phi)
                info['left'][idx][3] = self.period_cut(info['left'][idx][3], period_theta)
                info['MZI_list'].append(info['left'][idx])
            left_list = info['right'][::-1]

        for idx in range(len(left_list)):
            jj, ii, phi, theta = left_list[idx]
            phi_, theta_, phase_angle[:, jj], phase_angle[:, ii] = self.clements_diagonal_transform(
                phi, theta, phase_angle[:, jj], phase_angle[:, ii], method
            )
            phi_ = self.period_cut(phi_, period_phi)
            theta_ = self.period_cut(theta_, period_theta)
            info['MZI_list'].append([jj, ii, phi_, theta_])
        info['phase_angle'] = phase_angle.clone()  # unitary=D'L'L'L'RRR,本行保存新的D
        mask = torch.logical_or(info['phase_angle'] >= 2 * torch.pi, info['phase_angle'] < 0)
        info['phase_angle'][mask] -= torch.floor(info['phase_angle'][mask] / torch.pi / 2) * torch.pi * 2
        return info, unitary

    @staticmethod
    def get_periods(method: str):
        if 'dd' in method:
            period_theta = 2 * np.pi
            period_phi = 4 * np.pi
        elif 'ds' in method:
            period_theta = 4 * np.pi
            period_phi = 4 * np.pi
        else:
            period_theta = 2 * np.pi
            period_phi = 2 * np.pi
        return period_theta, period_phi

    @staticmethod
    def period_cut(input_angle: float, period: float = np.pi * 2) -> float:
        return input_angle - np.floor(input_angle / period) * period

    @staticmethod
    def calc_factor_inverse(method, phi, theta):
        # 计算MZI矩阵T^{-1}的系数（相当于全局相位）
        if 'sd' in method:
            return -1j
        elif 'ss' in method:
            return -1j * np.exp(-1j * theta / 2)
        elif 'dd' in method:
            return -1j * np.exp(-1j * (theta - phi) / 2)
        elif 'ds' in method:
            return -1j * np.exp(1j * phi / 2)

    @staticmethod
    def calc_factor_inverse_torch(method, phi, theta):
        # 计算MZI矩阵T^{-1}的系数（相当于全局相位）
        if 'sd' in method:
            return -1j
        elif 'ss' in method:
            return -1j * torch.exp(-1j * theta / 2)
        elif 'dd' in method:
            return -1j * torch.exp(-1j * (theta - phi) / 2)
        elif 'ds' in method:
            return -1j * torch.exp(1j * phi / 2)

    def calc_factor_constr(self, method, phi, theta):
        # 计算MZI矩阵T的系数（相当于全局相位）
        return self.calc_factor_inverse(method, phi, theta).conjugate()

    def calc_factor_constr_torch(self, method, phi, theta):
        # 计算MZI矩阵T的系数（相当于全局相位）
        return self.calc_factor_inverse_torch(method, phi, theta).conj()

    def get_matrix_constr_l(self, info, n_dim, method):
        jj, ii, phi, theta = info
        factor = self.calc_factor_constr(method, phi, theta)
        multiple = np.eye(n_dim, dtype=complex)
        multiple[jj, jj] = factor * np.exp(1j * phi) * np.sin(theta / 2)
        multiple[jj, ii] = factor * np.exp(1j * phi) * np.cos(theta / 2)
        multiple[ii, jj] = factor * np.cos(theta / 2)
        multiple[ii, ii] = factor * -np.sin(theta / 2)
        return multiple

    def get_matrix_constr_l_torch(self, info, n_dim, batch, method, dtype):
        jj, ii, phi, theta = info
        factor = self.calc_factor_constr_torch(method, phi, theta)
        multiple = torch.eye(n_dim).repeat(batch, 1, 1)
        multiple = multiple.to(dtype)
        multiple[:, jj, jj] = factor * torch.exp(1j * phi) * torch.sin(theta / 2)
        multiple[:, jj, ii] = factor * torch.exp(1j * phi) * torch.cos(theta / 2)
        multiple[:, ii, jj] = factor * torch.cos(theta / 2)
        multiple[:, ii, ii] = factor * -torch.sin(theta / 2)
        return multiple

    def get_matrix_inverse_l(self, info, n_dim, method):
        jj, ii, phi, theta = info
        factor = self.calc_factor_inverse(method, phi, theta)
        multiple = np.eye(n_dim, dtype=complex)
        multiple[jj, jj] = factor * np.exp(-1j * phi) * np.sin(theta / 2)
        multiple[jj, ii] = factor * np.cos(theta / 2)
        multiple[ii, jj] = factor * np.exp(-1j * phi) * np.cos(theta / 2)
        multiple[ii, ii] = factor * -np.sin(theta / 2)
        return multiple

    def get_matrix_inverse_l_torch(self, info, n_dim, batch, method, dtype):
        jj, ii, phi, theta = info
        factor = self.calc_factor_inverse_torch(method, phi, theta)
        multiple = torch.eye(n_dim).repeat(batch, 1, 1)
        multiple = multiple.to(dtype)
        multiple[:, jj, jj] = factor * torch.exp(-1j * phi) * torch.sin(theta / 2)
        multiple[:, jj, ii] = factor * torch.cos(theta / 2)
        multiple[:, ii, jj] = factor * torch.exp(-1j * phi) * torch.cos(theta / 2)
        multiple[:, ii, ii] = factor * -torch.sin(theta / 2)
        return multiple

    def get_matrix_constr_r(self, info, n_dim, method):
        jj, ii, phi, theta = info
        factor = self.calc_factor_constr(method, phi, theta)
        multiple = np.eye(n_dim, dtype=complex)
        multiple[jj, jj] = factor * np.exp(1j * phi) * np.sin(theta / 2)
        multiple[jj, ii] = factor * np.cos(theta / 2)
        multiple[ii, jj] = factor * np.exp(1j * phi) * np.cos(theta / 2)
        multiple[ii, ii] = factor * -np.sin(theta / 2)
        return multiple

    def get_matrix_constr_r_torch(self, info, n_dim, batch, method, dtype):
        jj, ii, phi, theta = info
        factor = self.calc_factor_constr_torch(method, phi, theta)
        multiple = torch.eye(n_dim).repeat(batch, 1, 1)
        multiple = multiple.to(dtype)
        multiple[:, jj, jj] = factor * torch.exp(1j * phi) * torch.sin(theta / 2)
        multiple[:, jj, ii] = factor * torch.cos(theta / 2)
        multiple[:, ii, jj] = factor * torch.exp(1j * phi) * torch.cos(theta / 2)
        multiple[:, ii, ii] = factor * -torch.sin(theta / 2)
        return multiple

    def get_matrix_inverse_r(self, info, n_dim, method):
        jj, ii, phi, theta = info
        factor = self.calc_factor_inverse(method, phi, theta)
        multiple = np.eye(n_dim, dtype=complex)
        multiple[jj, jj] = factor * np.exp(-1j * phi) * np.sin(theta / 2)
        multiple[jj, ii] = factor * np.exp(-1j * phi) * np.cos(theta / 2)
        multiple[ii, jj] = factor * np.cos(theta / 2)
        multiple[ii, ii] = factor * -np.sin(theta / 2)
        return multiple

    def get_matrix_inverse_r_torch(self, info, n_dim, batch, method, dtype):
        jj, ii, phi, theta = info
        factor = self.calc_factor_inverse_torch(method, phi, theta)
        multiple = torch.eye(n_dim).repeat(batch, 1, 1)
        multiple = multiple.to(dtype)
        multiple[:, jj, jj] = factor * torch.exp(-1j * phi) * torch.sin(theta / 2)
        multiple[:, jj, ii] = factor * torch.exp(-1j * phi) * torch.cos(theta / 2)
        multiple[:, ii, jj] = factor * torch.cos(theta / 2)
        multiple[:, ii, ii] = factor * -torch.sin(theta / 2)
        return multiple

    @staticmethod
    def clements_diagonal_transform(phi, theta, a1, a2, method):
        if 'sd' in method:
            theta_ = theta
            phi_ = a1 - a2
            b1 = a2 - phi + np.pi
            b2 = a2 + np.pi
            return phi_, theta_, b1, b2
        elif 'ss' in method:
            # T^-1 * D = D' * T
            theta_ = theta
            phi_ = a1 - a2
            b1 = a2 - phi + np.pi - theta
            b2 = a2 + np.pi - theta
            return phi_, theta_, b1, b2
        elif 'dd' in method:
            theta_ = theta
            phi_ = a1 - a2
            b1 = a2 - phi + np.pi - theta + (phi + phi_) / 2
            b2 = a2 + np.pi - theta + (phi + phi_) / 2
            return phi_, theta_, b1, b2
        elif 'ds' in method:
            theta_ = theta
            phi_ = a1 - a2
            b1 = a2 - phi + np.pi + (phi + phi_) / 2
            b2 = a2 + np.pi + (phi + phi_) / 2
            return phi_, theta_, b1, b2

    def sort_mzi(self, mzi_info):
        """Sort mzi parameters in the same array for plotting."""
        dic_mzi = defaultdict(list)  # 当key不存在时对应的value是[]
        mzi_list = mzi_info['MZI_list']
        for i in mzi_list:
            dic_mzi[tuple(i[0:2])].append(i[2:])
        return dic_mzi

    def sort_mzi_torch(self, mzi_info):
        """Sort mzi parameters in the same array for plotting."""
        dic_mzi = defaultdict(list)  # 当key不存在时对应的value是[]
        mzi_list = mzi_info['MZI_list']
        for i in mzi_list:
            dic_mzi[tuple(i[0:2])].append(torch.stack(i[2:]).mT)
        return dic_mzi

    def ps_pos(self, dic_mzi, phase_angle):
        """Label the position of each phaseshifter for ``'cssr'`` case."""
        if self.method in ['cssr', 'cssl']:
            dic_pos = {}
            nmode = self.unitary.shape[0]
            dic_ = dic_mzi
            for mode in range(nmode):
                pair = (mode, mode + 1)
                value = dic_[pair]
                value = np.array(value).flatten()
                if self.method == 'cssr':
                    for k in range(len(value)):
                        dic_pos[(mode, k)] = np.round((value[k]), 4)
                    if mode == nmode - 1:
                        dic_pos[(mode, 0)] = np.round((phase_angle[mode]), 4)
                    else:
                        dic_pos[(mode, k + 1)] = np.round((phase_angle[mode]), 4)
                if self.method == 'cssl':
                    dic_pos[(mode, 0)] = np.round((phase_angle[mode]), 4)
                    for k in range(len(value)):
                        dic_pos[(mode, k + 1)] = np.round((value[k]), 4)
            return dic_pos
        else:
            raise NotImplementedError(f"ps_pos only supports method='cssr','cssl', but got method={self.method!r}.")

    def ps_pos_torch(self, dic_mzi, phase_angle):
        """Label the position of each phaseshifter for ``'cssr'`` case."""
        if self.method == 'cssr':
            dic_pos = {}
            dic_ = dic_mzi
            nmode = self.unitary.shape[-1]
            for mode in range(nmode - 1):
                pair = (mode, mode + 1)
                value = torch.cat(dic_[pair], dim=1)
                for k in range(value.shape[-1]):
                    dic_pos[(mode, k)] = value[:, k]
                    dic_pos[(mode, k + 1)] = phase_angle[:, mode]
            dic_pos[(nmode - 1, 0)] = phase_angle[:, nmode - 1]
            return dic_pos
        else:
            raise NotImplementedError(f"ps_pos only supports method='cssr', but got method={self.method!r}.")
