"""
Decompose the unitary matrix
"""

from collections import defaultdict
from typing import Dict, Tuple, Union

import numpy as np
import torch


class UnitaryDecomposer():
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
    def __init__(self, unitary: Union[np.ndarray, torch.Tensor], method: str = 'cssr') -> None:
        if isinstance(unitary, np.ndarray):
            self.unitary = unitary.copy()
        elif isinstance(unitary, torch.Tensor):
            self.unitary = unitary.cpu().clone().detach().numpy()
        else:
            raise TypeError('The matrix to be decomposed must be in the type of numpy array or torch tensor.')
        if len(self.unitary.shape) != 2 or self.unitary.shape[0] != self.unitary.shape[1]:
            raise TypeError('The matrix to be decomposed must be a square matrix.')
        if np.abs(unitary @ unitary.conj().T - np.eye(len(unitary))).sum() / len(unitary) ** 2 > 1e-6:
            print('Make sure the input matrix is unitary, in case of an abnormal computation result.')
        self.unitary[np.abs(self.unitary) < 1e-32] = 1e-32
        self.method = method

    def decomp(self) -> Tuple[Dict, Dict, Dict]:
        """Decompose the unitary matrix.The third dictionary is the representation
          of the positions and the angles of all phase shifters.
        """
        def period_cut(input_angle: float, period: float = np.pi * 2) -> float:
            return input_angle - np.floor(input_angle/period)*period

        def decomp_rr(unitary: np.ndarray, method: str) -> Dict:
            n_dim = len(unitary)
            info = {}
            info['N'] = n_dim
            info['method'] = method
            info['MZI_list'] = [] # jj,ii,phi,theta
            if 'dd' in method:
                period_theta = 2*np.pi
                period_phi = 4*np.pi
            elif 'ds' in method:
                period_theta = 4*np.pi
                period_phi = 4*np.pi
            else:
                period_theta = 2*np.pi
                period_phi = 2*np.pi
            for i in range(n_dim):
                ii = n_dim - 1 - i # 基准列 ii
                for jj in range(ii)[::-1]:
                    # print(ii,jj)
                    # 要用 uniatry[:,ii] 把 unitary[:,jj]的第ii号元素变成0
                    # if uniatry[ii,jj] == 0:
                    #     continue
                    ratio = unitary[ii,ii] / (unitary[ii,jj]+1e-32)
                    theta = 2 * np.arctan(np.abs(ratio))
                    phi = - np.angle(-ratio)
                    multiple = get_matrix_inverse_r([jj,ii,phi,theta], n_dim, method)
                    unitary = unitary @ multiple
                    phi = period_cut(phi, period_phi)
                    theta = period_cut(theta, period_theta)
                    info['MZI_list'].append([jj,ii,phi,theta])
            diagonal = np.diag(unitary)
            info['phase_angle'] = np.angle(diagonal)
            mask = np.logical_or(info['phase_angle']>=2*np.pi, info['phase_angle']<0)
            info['phase_angle'][mask] -= np.floor(info['phase_angle'][mask]/np.pi/2)*np.pi*2
            return info, unitary

        def decomp_cr(unitary: np.ndarray, method: str) -> Dict:
            n_dim = len(unitary)
            info = {}
            info['N']= n_dim
            info['method'] = method
            info['MZI_list'] = [] # jj,ii,phi,theta
            info['right'] = []
            info['left'] = []
            if 'dd' in method:
                period_theta = 2*np.pi
                period_phi = 4*np.pi
            elif 'ds' in method:
                period_theta = 4*np.pi
                period_phi = 4*np.pi
            else:
                period_theta = 2*np.pi
                period_phi = 2*np.pi
            for i in range(n_dim-1): # 从下往上第i个反对角线
                if i % 2: # 左乘, 利用TU消元；
                    for j in range(i+1): # 反对角线的元素计数
                        # 消元顺序：从左上到右下
                        jj = j # 当前待消元元素列号
                        ii = n_dim - 1 - i + j # 当前待消元元素行号
                        # print(ii,jj)
                        # if unitary[ii,jj] == 0:
                        #     continue
                        ratio = unitary[ii-1,jj]/(unitary[ii,jj]+1e-32)
                        theta = 2 * np.arctan( np.abs(ratio) )
                        phi = - np.angle(ratio)
                        multiple = get_matrix_constr_r([ii-1,ii,phi,theta], n_dim, method)
                        unitary = multiple @ unitary
                        info['left'].append([ii-1,ii,phi,theta])
                else: # 利用UT^{-1}消元，即利用 unitary[ii,jj+1] 消去 unitary[ii,jj]
                    for j in range(i+1)[::-1]: # 反对角线的元素计数
                        # 消元顺序：从右下到左上
                        jj = j # 当前待消元元素列号
                        ii = n_dim - 1 - i + j # 当前待消元元素行号
                        # print(ii,jj)
                        # if unitary[ii,jj] == 0:
                        #     continue
                        ratio = unitary[ii,jj+1]/(unitary[ii,jj]+1e-32)
                        theta = 2 * np.arctan( np.abs( ratio ) )
                        phi = - np.angle(- ratio)
                        multiple = get_matrix_inverse_r([jj,jj+1,phi,theta], n_dim, method)
                        unitary = unitary @ multiple
                        info['right'].append([jj,jj+1,phi,theta])
            phase_angle = np.angle(np.diag(unitary))
            info['phase_angle_ori'] = phase_angle.copy() # unitary=LLLDRRR，本行保存D
            for idx in range(len(info['right'])):
                info['right'][idx][2] = period_cut(info['right'][idx][2],period_phi)
                info['right'][idx][3] = period_cut(info['right'][idx][3],period_theta)
                info['MZI_list'].append(info['right'][idx])
            left_list = info['left'][::-1]
            for idx in range(len(left_list)):
                jj,ii,phi,theta = left_list[idx]
                phi_, theta_, phase_angle[jj], phase_angle[ii] = \
                clements_diagonal_transform(phi, theta, phase_angle[jj], phase_angle[ii],method)
                phi_ = period_cut(phi_,period_phi)
                theta_ = period_cut(theta_,period_theta)
                info['MZI_list'].append([jj,ii,phi_,theta_])
            info['phase_angle'] = phase_angle.copy() # unitary=D'L'L'L'RRR,本行保存新的D
            mask = np.logical_or(info['phase_angle']>=2*np.pi, info['phase_angle']<0)
            info['phase_angle'][mask] -= np.floor(info['phase_angle'][mask]/np.pi/2)*np.pi*2
            return info, unitary

        def decomp_rl(unitary: np.ndarray, method: str) -> Dict:
            n_dim = len(unitary)
            info = {}
            info['N'] = n_dim
            info['method'] = method
            info['MZI_list'] = [] # jj,ii,phi,theta
            if 'dd' in method:
                period_theta = 2*np.pi
                period_phi = 4*np.pi
            elif 'ds' in method:
                period_theta = 4*np.pi
                period_phi = 4*np.pi
            else:
                period_theta = 2*np.pi
                period_phi = 2*np.pi
            for i in range(n_dim):
                ii = n_dim - 1 - i # 基准行 ii
                for jj in range(ii)[::-1]:
                    # print(ii,jj)
                    # 要用 unitary[ii] 把 unitary[jj]的第ii号元素变成0
                    # if unitary[jj,ii] == 0:
                    #     continue
                    ratio = unitary[ii,ii] / (unitary[jj,ii]+1e-32)
                    theta = 2 * np.arctan( np.abs( ratio ) )
                    phi = - np.angle(- ratio)
                    multiple = get_matrix_inverse_l([jj,ii,phi,theta], n_dim, method)
                    unitary = multiple @ unitary
                    phi = period_cut(phi,period_phi)
                    theta = period_cut(theta,period_theta)
                    info['MZI_list'].append([jj,ii,phi,theta])
            diagonal = np.diag(unitary)
            info['phase_angle'] = np.angle(diagonal)
            mask = np.logical_or(info['phase_angle']>=2*np.pi, info['phase_angle']<0)
            info['phase_angle'][mask] -= np.floor(info['phase_angle'][mask]/np.pi/2)*np.pi*2
            return info, unitary

        def decomp_cl(unitary: np.ndarray, method: str) -> Dict:
            n_dim = len(unitary)
            info = {}
            info['N'] = n_dim
            info['method'] = method
            info['MZI_list'] = [] # jj,ii,phi,theta
            info['right'] = []
            info['left'] = []
            if 'dd' in method:
                period_theta = 2*np.pi
                period_phi = 4*np.pi
            elif 'ds' in method:
                period_theta = 4*np.pi
                period_phi = 4*np.pi
            else:
                period_theta = 2*np.pi
                period_phi = 2*np.pi
            for i in range(n_dim-1): # 从下往上第i个反对角线
                if i % 2: # 左乘, 利用T^{-1}U消元；
                    for j in range(i+1): # 反对角线的元素计数
                        # 消元顺序：从左上到右下
                        jj = j # 当前待消元元素列号
                        ii = n_dim - 1 - i + j # 当前待消元元素行号
                        # print(ii,jj)
                        # if unitary[ii,jj] == 0:
                        #     continue
                        ratio = unitary[ii-1,jj]/(unitary[ii,jj]+1e-32)
                        theta = 2 * np.arctan( np.abs(ratio) )
                        phi = np.angle(ratio)
                        multiple = get_matrix_inverse_l([ii-1,ii,phi,theta], n_dim, method)
                        unitary = multiple @ unitary
                        info['left'].append([ii-1,ii,phi,theta])
                else: # 利用UT消元，即利用 unitary[ii,jj+1] 消去 unitary[ii,jj]
                    for j in range(i+1)[::-1]: # 反对角线的元素计数
                        # 消元顺序：从右下到左上
                        jj = j # 当前待消元元素列号
                        ii = n_dim - 1 - i + j # 当前待消元元素行号
                        # print(ii,jj)
                        # if unitary[ii,jj] == 0:
                        #     continue
                        ratio = unitary[ii,jj+1]/(unitary[ii,jj]+1e-32)
                        theta = 2 * np.arctan( np.abs( ratio ) )
                        phi = np.angle(- ratio)
                        multiple = get_matrix_constr_l([jj,jj+1,phi,theta], n_dim, method)
                        unitary = unitary @ multiple
                        info['right'].append([jj,jj+1,phi,theta])
            phase_angle = np.angle(np.diag(unitary))
            info['phase_angle_ori'] = phase_angle.copy() # U=LLLDRRR，本行保存D
            for idx in range(len(info['left'])):
                info['left'][idx][2] = period_cut(info['left'][idx][2],period_phi)
                info['left'][idx][3] = period_cut(info['left'][idx][3],period_theta)
                info['MZI_list'].append(info['left'][idx])
            left_list = info['right'][::-1]
            for idx in range(len(left_list)):
                jj,ii,phi,theta = left_list[idx]
                phi_, theta_, phase_angle[jj], phase_angle[ii] = \
                clements_diagonal_transform(phi, theta, phase_angle[jj], phase_angle[ii], method)
                phi_ = period_cut(phi_,period_phi)
                theta_ = period_cut(theta_,period_theta)
                info['MZI_list'].append([jj,ii,phi_,theta_])
            info['phase_angle'] = phase_angle.copy() # unitary =D'L'L'L'RRR,本行保存新的D
            mask = np.logical_or(info['phase_angle']>=2*np.pi, info['phase_angle']<0)
            info['phase_angle'][mask] -= np.floor(info['phase_angle'][mask]/np.pi/2)*np.pi*2
            return info, unitary

        def calc_factor_inverse(method, phi, theta):
            # 计算MZI矩阵T^{-1}的系数（相当于全局相位）
            if 'sd' in method:
                return -1j
            elif 'ss' in method:
                return -1j*np.exp(-1j*theta/2)
            elif 'dd' in method:
                return -1j*np.exp(-1j*(theta-phi)/2)
            elif 'ds' in method:
                return -1j*np.exp(1j*phi/2)

        def calc_factor_constr(method, phi, theta):
            # 计算MZI矩阵T的系数（相当于全局相位）
            return calc_factor_inverse(method,phi,theta).conjugate()

        def get_matrix_constr_l(info, n_dim, method):
            jj,ii,phi,theta = info
            factor = calc_factor_constr(method,phi,theta)
            multiple = np.eye(n_dim, dtype=complex)
            multiple[jj,jj] = factor*np.exp(1j*phi)*np.sin(theta/2)
            multiple[jj,ii] = factor*np.exp(1j*phi)*np.cos(theta/2)
            multiple[ii,jj] = factor*np.cos(theta/2)
            multiple[ii,ii] = factor*-np.sin(theta/2)
            return multiple

        def get_matrix_inverse_l(info, n_dim, method):
            jj,ii,phi,theta = info
            factor = calc_factor_inverse(method,phi,theta)
            multiple = np.eye(n_dim, dtype=complex)
            multiple[jj,jj] = factor*np.exp(-1j*phi)*np.sin(theta/2)
            multiple[jj,ii] = factor*np.cos(theta/2)
            multiple[ii,jj] = factor*np.exp(-1j*phi)*np.cos(theta/2)
            multiple[ii,ii] = factor*-np.sin(theta/2)
            return multiple

        def get_matrix_constr_r(info, n_dim, method):
            jj,ii,phi,theta = info
            factor = calc_factor_constr(method,phi,theta)
            multiple = np.eye(n_dim, dtype=complex)
            multiple[jj,jj] = factor*np.exp(1j*phi)*np.sin(theta/2)
            multiple[jj,ii] = factor*np.cos(theta/2)
            multiple[ii,jj] = factor*np.exp(1j*phi)*np.cos(theta/2)
            multiple[ii,ii] = factor*-np.sin(theta/2)
            return multiple

        def get_matrix_inverse_r(info, n_dim, method):
            jj,ii,phi,theta = info
            factor = calc_factor_inverse(method,phi,theta)
            multiple = np.eye(n_dim, dtype=complex)
            multiple[jj,jj] = factor*np.exp(-1j*phi)*np.sin(theta/2)
            multiple[jj,ii] = factor*np.exp(-1j*phi)*np.cos(theta/2)
            multiple[ii,jj] = factor*np.cos(theta/2)
            multiple[ii,ii] = factor*-np.sin(theta/2)
            return multiple

        def clements_diagonal_transform(phi, theta, a1, a2, method):
            if 'sd' in method:
                theta_ = theta
                phi_ = a1 - a2
                b1 =  a2 - phi + np.pi
                b2 =  a2 + np.pi
                return phi_, theta_, b1, b2
            elif 'ss' in method:
                theta_ = theta
                phi_ = a1 - a2
                b1 =  a2 - phi + np.pi - theta
                b2 =  a2 + np.pi - theta
                return phi_, theta_, b1, b2
            elif 'dd' in method:
                theta_ = theta
                phi_ = a1 - a2
                b1 =  a2 - phi + np.pi - theta + (phi+phi_)/2
                b2 =  a2 + np.pi - theta + (phi+phi_)/2
                return phi_, theta_, b1, b2
            elif 'ds' in method:
                theta_ = theta
                phi_ = a1 - a2
                b1 =  a2 - phi + np.pi + (phi+phi_)/2
                b2 =  a2 + np.pi + (phi+phi_)/2
                return phi_, theta_, b1, b2

        method = self.method
        if method not in ['rssr','rsdr', 'rdsr', 'rddr', 'rssl', 'rsdl', 'rdsl', 'rddl',\
        'cssr', 'csdr', 'cdsr', 'cddr', 'cssl', 'csdl', 'cdsl', 'cddl']:
            raise LookupError('请检查分解方式！')
        elif method[0]+method[-1] == 'cr':
            temp_0 = decomp_cr(self.unitary, method)[0]
        elif method[0]+method[-1] == 'cl':
            temp_0 = decomp_cl(self.unitary, method)[0]
        elif method[0]+method[-1] == 'rr':
            temp_0 = decomp_rr(self.unitary, method)[0]
        elif method[0]+method[-1] == 'rl':
            temp_0 = decomp_rl(self.unitary, method)[0]
        temp_1 = self.sort_mzi(temp_0)
        temp_2 = self.ps_pos(temp_1, temp_0['phase_angle'])
        return temp_0, temp_1, temp_2

    def sort_mzi(self, mzi_info):
        """
        Sort mzi parameters in the same array for plotting.
        """
        dic_mzi = defaultdict(list) #当key不存在时对应的value是[]
        mzi_list = mzi_info['MZI_list']
        for i in mzi_list:
            dic_mzi[tuple(i[0:2])].append(i[2:])
        return dic_mzi

    def ps_pos(self, dic_mzi, phase_angle):
        """
        Label the position of each phaseshifter for ``'cssr'`` case.
        """
        if self.method == 'cssr':
            dic_pos = {}
            nmode = self.unitary.shape[0]
            dic_ =dic_mzi
            for mode in range(nmode):
                pair = (mode, mode+1)
                value = dic_[pair]
                value = np.array(value).flatten()
                for k in range(len(value)):
                    dic_pos[(mode, k)] = np.round((value[k]), 4)
                if mode == nmode -1:
                    dic_pos[(mode, 0)] = np.round((phase_angle[mode]), 4)
                else:
                    dic_pos[(mode, k+1)] = np.round((phase_angle[mode]), 4)
            return dic_pos
        else:
            return None
