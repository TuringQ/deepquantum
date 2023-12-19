from collections import defaultdict

import numpy as np
import torch
# 2023-12-18 更新

class UnitaryDecomposer():
    def __init__(self, U: np.array, method="cssr"):
        '''
        初始化酉矩阵。
        检查输入类型,如果是torch.Tensor转为numpy.ndarray;
        如果不是方阵/酉矩阵，则报错。
        method 默认为"cssr", clements-single-single-right, single 表示单臂,right 表示最后一列移相器位置在最右边
        method 列表["rssr","rsdr","rdsr","rddr","rssl","rsdl","rdsl","rddl",
            "cssr","csdr","cdsr","cddr","cssl","csdl","cdsl","cddl"]
        '''
        if type(U) == np.ndarray:
            self.U = U.copy()
        elif type(U) == torch.Tensor:
            self.U == U.clone.detach().numpy()
        else:
            raise TypeError("The matrix to be decomposed must be in the type of numpy array or pytorch tensor.")
        if (len(self.U.shape)!=2)or(self.U.shape[0]!=self.U.shape[1]):
            raise TypeError("The matrix to be decomposed must be a square matrix.")
        if np.abs(U@U.conj().T - np.eye(len(U))).sum()/len(U)**2>1e-12:
            print("Make sure the input matrix is unitary, in case of an abnormal computation result.")
        self.U[np.abs(self.U)<1e-32] = 1e-32
        self.method = method

    def decomp(self) -> dict:
        """
        method 前三个字母的含义如下
        <第0个字母>
        r: Reck架构
        c: Clements架构
        <第1,2个字母>
        注意,无论是列移相器在左还是在右,第一个字母都对应phi。
        sd: 单臂phi+双臂theta
        ss: 单臂+单臂
        dd: 双臂+双臂
        ds: 双臂+单臂
        """
        def period_cut(input_angle: float, period: float=np.pi*2) -> float:
            return input_angle - np.floor(input_angle/period)*period

        def decomp_rr (U : np.array, method : str) -> dict:
            N = len(U)
            I = dict()
            I["N"]=N
            I["method"] = method
            I["MZI_list"] = [] # jj,ii,phi,theta
            if "dd" in method:
                period_theta = 2*np.pi
                period_phi = 4*np.pi
            elif "ds" in method:
                period_theta = 4*np.pi
                period_phi = 4*np.pi
            else:
                period_theta = 2*np.pi
                period_phi = 2*np.pi
            for i in range(N):
                ii = N - 1 - i # 基准列 ii
                for jj in range(ii)[::-1]:
                    # print(ii,jj)
                    # 要用 U[:,ii] 把 U[:,jj]的第ii号元素变成0
                    # if U[ii,jj] == 0:
                    #     continue
                    ratio = U[ii,ii] / (U[ii,jj]+1e-32)
                    theta = 2 * np.arctan( np.abs( ratio ) )
                    phi = - np.angle(- ratio)
                    multiple = get_matrix_inverse_r([jj,ii,phi,theta],N,method)
                    U = U @ multiple
                    phi = period_cut(phi,period_phi)
                    theta = period_cut(theta,period_theta)
                    I["MZI_list"].append([jj,ii,phi,theta])
            diagonal = np.diag(U)
            I["phase_angle"] = np.angle(diagonal)
            mask = np.logical_or(I["phase_angle"]>=2*np.pi,I["phase_angle"]<0)
            I["phase_angle"][mask] -= np.floor(I["phase_angle"][mask]/np.pi/2)*np.pi*2
            return I, U

        def decomp_cr (U : np.array, method : str) -> dict:
            N = len(U)
            I = dict()
            I["N"]=N
            I["method"] = method
            I["MZI_list"] = [] # jj,ii,phi,theta
            I["right"] = []
            I["left"] = []
            if "dd" in method:
                period_theta = 2*np.pi
                period_phi = 4*np.pi
            elif "ds" in method:
                period_theta = 4*np.pi
                period_phi = 4*np.pi
            else:
                period_theta = 2*np.pi
                period_phi = 2*np.pi
            for i in range(N-1): # 从下往上第i个反对角线
                if (i % 2): # 左乘, 利用TU消元；
                    for j in range(i+1): # 反对角线的元素计数
                        # 消元顺序：从左上到右下
                        jj = j # 当前待消元元素列号
                        ii = N - 1 - i + j # 当前待消元元素行号
                        # print(ii,jj)
                        # if U[ii,jj] == 0:
                        #     continue
                        ratio = U[ii-1,jj]/(U[ii,jj]+1e-32)
                        theta = 2 * np.arctan( np.abs(ratio) )
                        phi = - np.angle(ratio)
                        multiple = get_matrix_constr_r([ii-1,ii,phi,theta],N,method)
                        U = multiple @ U
                        I["left"].append([ii-1,ii,phi,theta])
                else: # 利用UT^{-1}消元，即利用 U[ii,jj+1] 消去 U[ii,jj]
                    for j in range(i+1)[::-1]: # 反对角线的元素计数
                        # 消元顺序：从右下到左上
                        jj = j # 当前待消元元素列号
                        ii = N - 1 - i + j # 当前待消元元素行号
                        # print(ii,jj)
                        # if U[ii,jj] == 0:
                        #     continue
                        ratio = U[ii,jj+1]/(U[ii,jj]+1e-32)
                        theta = 2 * np.arctan( np.abs( ratio ) )
                        phi = - np.angle(- ratio)
                        multiple = get_matrix_inverse_r([jj,jj+1,phi,theta],N,method)
                        U = U @ multiple
                        I["right"].append([jj,jj+1,phi,theta])
            phase_angle = np.angle(np.diag(U))
            I["phase_angle_ori"] = phase_angle.copy() # U=LLLDRRR，本行保存D
            for idx in range(len(I["right"])):
                I["right"][idx][2] = period_cut(I["right"][idx][2],period_phi)
                I["right"][idx][3] = period_cut(I["right"][idx][3],period_theta)
                I["MZI_list"].append(I["right"][idx])
            left_list = I["left"][::-1]
            for idx in range(len(left_list)):
                jj,ii,phi,theta = left_list[idx]
                phi_, theta_, phase_angle[jj], phase_angle[ii] = clements_diagonal_transform(phi, theta, phase_angle[jj], phase_angle[ii],method)
                phi_ = period_cut(phi_,period_phi)
                theta_ = period_cut(theta_,period_theta)
                I["MZI_list"].append([jj,ii,phi_,theta_])
            I["phase_angle"] = phase_angle.copy() # U=D'L'L'L'RRR,本行保存新的D
            mask = np.logical_or(I["phase_angle"]>=2*np.pi,I["phase_angle"]<0)
            I["phase_angle"][mask] -= np.floor(I["phase_angle"][mask]/np.pi/2)*np.pi*2
            return I,U

        def decomp_rl (U : np.array, method : str) -> dict:
            N = len(U)
            I = dict()
            I["N"]=N
            I["method"] = method
            I["MZI_list"] = [] # jj,ii,phi,theta
            if "dd" in method:
                period_theta = 2*np.pi
                period_phi = 4*np.pi
            elif "ds" in method:
                period_theta = 4*np.pi
                period_phi = 4*np.pi
            else:
                period_theta = 2*np.pi
                period_phi = 2*np.pi
            for i in range(N):
                ii = N - 1 - i # 基准行 ii
                for jj in range(ii)[::-1]:
                    # print(ii,jj)
                    # 要用 U[ii] 把 U[jj]的第ii号元素变成0
                    # if U[jj,ii] == 0:
                    #     continue
                    ratio = U[ii,ii] / (U[jj,ii]+1e-32)
                    theta = 2 * np.arctan( np.abs( ratio ) )
                    phi = - np.angle(- ratio)
                    multiple = get_matrix_inverse_l([jj,ii,phi,theta],N,method)
                    U = multiple @ U
                    phi = period_cut(phi,period_phi)
                    theta = period_cut(theta,period_theta)
                    I["MZI_list"].append([jj,ii,phi,theta])
            diagonal = np.diag(U)
            I["phase_angle"] = np.angle(diagonal)
            mask = np.logical_or(I["phase_angle"]>=2*np.pi,I["phase_angle"]<0)
            I["phase_angle"][mask] -= np.floor(I["phase_angle"][mask]/np.pi/2)*np.pi*2
            return I, U

        def decomp_cl (U : np.array, method : str) -> dict:
            N = len(U)
            I = dict()
            I["N"]=N
            I["method"] = method
            I["MZI_list"] = [] # jj,ii,phi,theta
            I["right"] = []
            I["left"] = []
            if "dd" in method:
                period_theta = 2*np.pi
                period_phi = 4*np.pi
            elif "ds" in method:
                period_theta = 4*np.pi
                period_phi = 4*np.pi
            else:
                period_theta = 2*np.pi
                period_phi = 2*np.pi
            for i in range(N-1): # 从下往上第i个反对角线
                if (i % 2): # 左乘, 利用T^{-1}U消元；
                    for j in range(i+1): # 反对角线的元素计数
                        # 消元顺序：从左上到右下
                        jj = j # 当前待消元元素列号
                        ii = N - 1 - i + j # 当前待消元元素行号
                        # print(ii,jj)
                        # if U[ii,jj] == 0:
                        #     continue
                        ratio = U[ii-1,jj]/(U[ii,jj]+1e-32)
                        theta = 2 * np.arctan( np.abs(ratio) )
                        phi = np.angle(ratio)
                        multiple = get_matrix_inverse_l([ii-1,ii,phi,theta],N,method)
                        U = multiple @ U
                        I["left"].append([ii-1,ii,phi,theta])
                else: # 利用UT消元，即利用 U[ii,jj+1] 消去 U[ii,jj]
                    for j in range(i+1)[::-1]: # 反对角线的元素计数
                        # 消元顺序：从右下到左上
                        jj = j # 当前待消元元素列号
                        ii = N - 1 - i + j # 当前待消元元素行号
                        # print(ii,jj)
                        # if U[ii,jj] == 0:
                        #     continue
                        ratio = U[ii,jj+1]/(U[ii,jj]+1e-32)
                        theta = 2 * np.arctan( np.abs( ratio ) )
                        phi = np.angle(- ratio)
                        multiple = get_matrix_constr_l([jj,jj+1,phi,theta],N,method)
                        U = U @ multiple
                        I["right"].append([jj,jj+1,phi,theta])
            phase_angle = np.angle(np.diag(U))
            I["phase_angle_ori"] = phase_angle.copy() # U=LLLDRRR，本行保存D
            for idx in range(len(I["left"])):
                I["left"][idx][2] = period_cut(I["left"][idx][2],period_phi)
                I["left"][idx][3] = period_cut(I["left"][idx][3],period_theta)
                I["MZI_list"].append(I["left"][idx])
            left_list = I["right"][::-1]
            for idx in range(len(left_list)):
                jj,ii,phi,theta = left_list[idx]
                phi_, theta_, phase_angle[jj], phase_angle[ii] = clements_diagonal_transform(phi, theta, phase_angle[jj], phase_angle[ii], method)
                phi_ = period_cut(phi_,period_phi)
                theta_ = period_cut(theta_,period_theta)
                I["MZI_list"].append([jj,ii,phi_,theta_])
            I["phase_angle"] = phase_angle.copy() # U=D'L'L'L'RRR,本行保存新的D
            mask = np.logical_or(I["phase_angle"]>=2*np.pi,I["phase_angle"]<0)
            I["phase_angle"][mask] -= np.floor(I["phase_angle"][mask]/np.pi/2)*np.pi*2
            return I,U

        def calc_factor_inverse(method,phi,theta):
            # 计算MZI矩阵T^{-1}的系数（相当于全局相位）
            if "sd" in method:
                return -1j
            elif "ss" in method:
                return -1j*np.exp(-1j*theta/2)
            elif "dd" in method:
                return -1j*np.exp(-1j*(theta-phi)/2)
            elif "ds" in method:
                return -1j*np.exp(1j*phi/2)

        def calc_factor_constr(method,phi,theta):
            # 计算MZI矩阵T的系数（相当于全局相位）
            return calc_factor_inverse(method,phi,theta).conjugate()

        def get_matrix_constr_l(info,N,method):
            jj,ii,phi,theta = info
            factor = calc_factor_constr(method,phi,theta)
            multiple = np.eye(N,dtype=complex)
            multiple[jj,jj] = factor*np.exp(1j*phi)*np.sin(theta/2)
            multiple[jj,ii] = factor*np.exp(1j*phi)*np.cos(theta/2)
            multiple[ii,jj] = factor*np.cos(theta/2)
            multiple[ii,ii] = factor*-np.sin(theta/2)
            return multiple

        def get_matrix_inverse_l(info,N,method):
            jj,ii,phi,theta = info
            factor = calc_factor_inverse(method,phi,theta)
            multiple = np.eye(N,dtype=complex)
            multiple[jj,jj] = factor*np.exp(-1j*phi)*np.sin(theta/2)
            multiple[jj,ii] = factor*np.cos(theta/2)
            multiple[ii,jj] = factor*np.exp(-1j*phi)*np.cos(theta/2)
            multiple[ii,ii] = factor*-np.sin(theta/2)
            return multiple

        def get_matrix_constr_r(info,N,method):
            jj,ii,phi,theta = info
            factor = calc_factor_constr(method,phi,theta)
            multiple = np.eye(N,dtype=complex)
            multiple[jj,jj] = factor*np.exp(1j*phi)*np.sin(theta/2)
            multiple[jj,ii] = factor*np.cos(theta/2)
            multiple[ii,jj] = factor*np.exp(1j*phi)*np.cos(theta/2)
            multiple[ii,ii] = factor*-np.sin(theta/2)
            return multiple

        def get_matrix_inverse_r(info,N,method):
            jj,ii,phi,theta = info
            factor = calc_factor_inverse(method,phi,theta)
            multiple = np.eye(N,dtype=complex)
            multiple[jj,jj] = factor*np.exp(-1j*phi)*np.sin(theta/2)
            multiple[jj,ii] = factor*np.exp(-1j*phi)*np.cos(theta/2)
            multiple[ii,jj] = factor*np.cos(theta/2)
            multiple[ii,ii] = factor*-np.sin(theta/2)
            return multiple

        def clements_diagonal_transform(phi, theta, a1, a2, method):
            if "sd" in method:
                theta_ = theta
                phi_ = a1 - a2
                b1 =  a2 - phi + np.pi
                b2 =  a2 + np.pi
                return phi_, theta_, b1, b2
            elif "ss" in method:
                theta_ = theta
                phi_ = a1 - a2
                b1 =  a2 - phi + np.pi - theta
                b2 =  a2 + np.pi - theta
                return phi_, theta_, b1, b2
            elif "dd" in method:
                theta_ = theta
                phi_ = a1 - a2
                b1 =  a2 - phi + np.pi - theta + (phi+phi_)/2
                b2 =  a2 + np.pi - theta + (phi+phi_)/2
                return phi_, theta_, b1, b2
            elif "ds" in method:
                theta_ = theta
                phi_ = a1 - a2
                b1 =  a2 - phi + np.pi + (phi+phi_)/2
                b2 =  a2 + np.pi + (phi+phi_)/2
                return phi_, theta_, b1, b2

        def constr_r(I):
            N = I["N"]
            method = I["method"]
            U = np.eye(N,dtype=complex)

            for idx in range(len(I["MZI_list"])):
                multiple = get_matrix_constr_r(I["MZI_list"][idx],N,method)
                U = multiple @ U
            return U

        def constr_l(I):
            N = I["N"]
            method = I["method"]
            U = np.eye(N,dtype=complex)
            ordered_list = I["MZI_list"]  #  注意顺序。对于Reck，T2- T1- U = D => U = T1 T2 D
            for idx in range(len(ordered_list)):
                multiple = get_matrix_constr_l(ordered_list[idx],N,method)
                U = U @ multiple
            return U

        U = self.U.copy()
        method = self.method
        if method not in ["rssr","rsdr","rdsr","rddr","rssl","rsdl","rdsl","rddl",\
        "cssr","csdr","cdsr","cddr","cssl","csdl","cdsl","cddl"]:
            raise Exception("请检查分解方式！")
        elif method[0]+method[-1] == "cr":
            temp_0 = decomp_cr(self.U, method)[0]
        elif method[0]+method[-1] == "cl":
            temp_0 = decomp_cl(self.U, method)[0]
        elif method[0]+method[-1] == "rr":
            temp_0 = decomp_rr(self.U, method)[0]
        elif method[0]+method[-1] == "rl":
            temp_0 = decomp_rl(self.U, method)[0]
        temp_1 = self.sort_mzi(temp_0)
        temp_2 = self.ps_pos(temp_1, temp_0["phase_angle"])
        return temp_0, temp_1, temp_2

    def sort_mzi(self, mzi_info):
        """
        sort mzi parameters in the same array for plotting
        """
        dic_mzi = defaultdict( list) #当key不存在时对应的value是[]
        mzi_list = mzi_info["MZI_list"]
        for i in mzi_list:
            dic_mzi[tuple(i[0:2])].append(i[2:])
        return dic_mzi

    def ps_pos(self, dic_mzi, phase_angle):
        """
        label the position of each phaseshifter for cssr case
        """
        if self.method == "cssr":
            dic_pos = { }
            nmode = self.U.shape[0]
            phase_angle = phase_angle
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
