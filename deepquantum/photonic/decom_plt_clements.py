import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

import scipy
import torch
import numpy as np
from scipy import special
import cmath
############################
#first decomposing the unitary matrix
############################
############################
def construct_u(paras):
    """
    given nxn+nxn real values return U_nxn
    """
    n = 6
    n_para = n * n
    u_temp_real = paras[0 : n_para]
    u_temp_real = u_temp_real.reshape(n, n)
    u_temp_imag = paras[n_para : 2 * n_para]
    u_temp_imag = u_temp_imag.reshape(n, n)
    u_temp = torch.tensor(u_temp_real + u_temp_imag * (1j))

    return u_temp

def get_para(u_rnd):
    """
    from unitary U get real+image as inputs
    """
    para_real = []
    para_imag = []
    for i in range(6):
        for j in range(6):
            temp_ele = u_rnd[i][j]
            para_real.append(temp_ele.real)
            para_imag.append(temp_ele.imag)
    para_ini = np.array(para_real + para_imag)

    return para_ini


def unitary_cons(paras):
    """
    set unitary constraints for 6x6 matrix
    """
    u_temp = construct_u(paras)
    u_temp_conj = u_temp.conj()
    u_temp_dag = u_temp_conj.transpose(0, 1)
    u_product = u_temp_dag @ u_temp
    u_identity = torch.eye(6)
    all_sum = float(abs(u_product - u_identity).sum())
    
    return all_sum


#############################################
########## function for mzi decompose########
#############################################
def decomp(U : np.array, method : str) -> dict:
    
    # U[abs(U)<1e-32] = 1e-32 #TODO 避免之后除数为0的报错
    """
    method 前三个字母的含义如下
    <第0个字母>
    r: Reck架构
    c: Clements架构
    <第1,2个字母>
    注意，无论是列移相器在左还是在右，第一个字母都对应phi。
    sd: 单臂phi+双臂theta
    ss: 单臂+单臂
    dd: 双臂+双臂
    ds: 双臂+单臂
    
    """
    if method not in ["rssr","rsdr","rdsr","rddr","rssl","rsdl","rdsl","rddl",\
        "cssr","csdr","cdsr","cddr","cssl","csdl","cdsl","cddl"]:
        raise Exception("请检查分解方式（method的首尾字母）！")
    
    elif method[0] == 'c':
        if method[-1] == 'r':
            return decomp_cr(U,method)
        elif method[-1] == 'l':
            return decomp_cl(U,method)
    
    elif method[0] == 'r':
        if method[-1] == 'r':
            return decomp_rr(U,method)
        elif method[-1] == 'l':
            return decomp_rl(U,method)

def decomp_rr (U : np.array, method : str) -> dict:
    N = len(U)
    I = dict()
    I["N"]=N
    I["method"] = method
    I["MZI_list"] = [] # jj,ii,phi,theta
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
        I["MZI_list"].append(I["right"][idx])    


    left_list = I["left"][::-1]
    for idx in range(len(left_list)):
        jj,ii,phi,theta = left_list[idx]
        phi_, theta_, phase_angle[jj], phase_angle[ii] = clements_diagonal_transform(phi, theta, phase_angle[jj], phase_angle[ii],method)
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
        I["MZI_list"].append(I["left"][idx])    


    left_list = I["right"][::-1]
    for idx in range(len(left_list)):
        jj,ii,phi,theta = left_list[idx]
        phi_, theta_, phase_angle[jj], phase_angle[ii] = clements_diagonal_transform(phi, theta, phase_angle[jj], phase_angle[ii],method)
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

def clements_diagonal_transform(phi, theta, a1, a2,method):
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


###########################
#next for plotting the mzi
###########################
###########################
class Graph_Mzi():
    """
    for plotting mzi clements structure
    n_mode: int 
    mzi_info: dictionary for mzi parameters
    type: cssr or cssl, the way for clements decomposition
    """
    def __init__(self, n_mode, unitary, cl="dodgerblue", fs=30, type="cssr"):
        self.n_mode = n_mode 
        self.type = type
        self.unitary = unitary
        mzi_info = decomp(unitary, type)[0]
        self.mzi_info = mzi_info
        self.color = cl 
        self.fontsize =fs
        self.wid = 0.1
        self.height = 0.08
        self.axis_off = "off"
        self.phase_angle = self.mzi_info["phase_angle"] # for phase shifter
        self.dic_mzi = self.sort_mzi() # for mzi parameters in the same array
        self.ps_position = self.ps_pos()
    def plotting_clements(self):
        if self.type == "cssr":
            assert(self.n_mode%2 == 0), "plotting only valid for even modes"
            self.plotting_clements_1()
        if self.type == "cssl":
            self.plotting_clements_2()

    
    def plotting_clements_1(self):
        """
        plotting CSSR, order: left to right

        """

        fig, ax = plt.subplots(1, 1)
       
        plt.rcParams["figure.figsize"] = (8*3,5.0*3)
        coords1 = []
        coords2 = []
        n_mode = self.n_mode
        phase_angle = self.phase_angle
        fs = self.fontsize
        cl = self.color
        wid = self.wid 
        height = self.height
        for i in range(n_mode):
            plt.annotate('',xy=(-0.1,1-0.25*i), xytext=(-0.5,1-0.25*i), arrowprops={'arrowstyle': '-|>', "lw":5}, va="center", )
            plt.text(-0.8, 1-0.25*i, "%d"%i, fontsize = fs )
            
            plt.plot([0, 1.2], [1-0.25*i,1-0.25*i], color = cl)
            plt.text( 3.2*(n_mode/2-1)+2.2+2.1, 1-0.25*i+0.05, "%.2f"%phase_angle[i], fontsize=fs-8 )  # phase angle
            ax.add_patch(
            patches.Rectangle(
                (3.2*(n_mode/2-1)+2.2+2.1, 1-0.25*i-0.05),
                wid,
                height,
                edgecolor = 'green',
                facecolor = 'green',
                fill=True
                             ) )  ## for PS
            if n_mode%2==1:
                plt.plot([2.2+3.2*(int((n_mode+1)/2)-1), 3.2*int((n_mode+1)/2-1)+2.2+2.2], [1-0.25*i,1-0.25*i], color = cl)
        
        if n_mode%2==0:   # for even mode      
            for i in range(int(n_mode/2)):
                plt.plot([2.2+3.2*i, 3.2*i+2.2+2.2], [1,1], color = cl)
                plt.plot([2.2+3.2*i, 3.2*i+2.2+2.2], [1-0.25*(n_mode-1), 1-0.25*(n_mode-1) ], color = cl)

                
                for j in range(n_mode):
                    plt.plot([1.5+3.2*i, 3.2*i+1.9], [1-0.25*j,1-0.25*j], color = cl)
                    coords1.append( [1.5+3.2*i, 3.2*i+1.9, 1-0.25*j,1-0.25*j]) 
                    
                    if 0<j<n_mode-1: 
                        plt.plot([3.1+3.2*i, 3.2*i+3.5], [1-0.25*j,1-0.25*j], color = cl)
                        coords2.append([3.1+3.2*i, 3.2*i+3.5, 1-0.25*j,1-0.25*j])

                        plt.plot([2.2+3.2*i, 3.2*i+2.8], [1-0.25*j,1-0.25*j], color = cl)
                        plt.plot([3.8+3.2*i, 3.2*i+4.4], [1-0.25*j,1-0.25*j], color = cl)

        if n_mode%2==1:  # for odd mode
            for i in range(int((n_mode+1)/2)):
                plt.plot([2.2+3.2*i, 3.2*i+2.2+2.2], [1,1], color = cl)
            #     plt.plot([1.2+3.2*i, 3.2*i+2.2+2.2], [1-0.25*(n_mode-1), 1-0.25*(n_mode-1) ], color = cl)
                
                for j in range(n_mode):
                    if j< n_mode-1: # remove last line
                        plt.plot([1.5+3.2*i, 3.2*i+1.9], [1-0.25*j,1-0.25*j], color = cl)
                        coords1.append( [1.5+3.2*i, 3.2*i+1.9, 1-0.25*j,1-0.25*j]) 
                    if j >= n_mode-1: 
                        plt.plot([1.2+3.2*i, 3.2*i+2.2], [1-0.25*j,1-0.25*j], color = cl)
                    
                    if  i< int((n_mode+1)/2)-1 and 0<j<n_mode: # remove the last column
                        plt.plot([3.1+3.2*i, 3.2*i+3.5], [1-0.25*j,1-0.25*j], color = cl)
                        coords2.append([3.1+3.2*i, 3.2*i+3.5, 1-0.25*j,1-0.25*j])

                        plt.plot([2.2+3.2*i, 3.2*i+2.8], [1-0.25*j,1-0.25*j], color = cl)
                        plt.plot([3.8+3.2*i, 3.2*i+4.4], [1-0.25*j,1-0.25*j], color = cl)


        # connecting lines i, i+1
        for i  in range(len(coords1)):
            if i%2==0:
                self.connect1(coords1[i], ax, a=-0.5-0.4, c=0.7-0.7)
            if i%2==1:
                self.connect2(coords1[i])

        for i  in range(len(coords2)):
            if i%2==0:
                self.connect1(coords2[i], ax, a=-0.5-0.4, c=0.7-0.7)
            if i%2==1:
                self.connect2(coords2[i])
        
        # plotting paras
        self.plot_paras_1(self.dic_mzi, self.n_mode, fs=self.fontsize-8)


        plt.axis(self.axis_off)
        # if self.axis_off:
        #     plt.axis('off')

        plt.show()



    def plotting_clements_2(self):
        """
        plotting CSSL, order: right to left
        """

        fig, ax = plt.subplots(1, 1)
       
        plt.rcParams["figure.figsize"] = (8*3,5.0*3)
        coords1 = []
        coords2 = []
        n_mode = self.n_mode
        phase_angle = self.phase_angle
        fs = self.fontsize
        cl = self.color
        wid = self.wid 
        height = self.height
        for i in range(n_mode):
            plt.annotate('',xy=(-0.1,1-0.25*i), xytext=(-0.5,1-0.25*i), arrowprops={'arrowstyle': '-|>', "lw":5}, va="center", )
            plt.text(-0.8, 1-0.25*i, "%d"%i, fontsize = fs )
            
            plt.plot([0, 1.2], [1-0.25*i,1-0.25*i], color = cl)
            plt.text( 0.4,1-0.25*i+0.05, "%.2f"%phase_angle[i], fontsize=fs-8 )  # phase angle
            ax.add_patch(
            patches.Rectangle(
                (0.5,1-0.25*i-0.05),
                wid,
                height,
                edgecolor = 'blue',
                facecolor = 'blue',
                fill=True
                             ) )
            if n_mode%2==1:
                plt.plot([2.2+3.2*(int((n_mode+1)/2)-1), 3.2*int((n_mode+1)/2-1)+2.2+2.2], [1-0.25*i,1-0.25*i], color = cl)
        
        if n_mode%2==0:   # for even mode      
            for i in range(int(n_mode/2)):
                plt.plot([2.2+3.2*i, 3.2*i+2.2+2.2], [1,1], color = cl)
                plt.plot([2.2+3.2*i, 3.2*i+2.2+2.2], [1-0.25*(n_mode-1), 1-0.25*(n_mode-1) ], color = cl)

                
                for j in range(n_mode):
                    plt.plot([1.5+3.2*i, 3.2*i+1.9], [1-0.25*j,1-0.25*j], color = cl)
                    coords1.append( [1.5+3.2*i, 3.2*i+1.9, 1-0.25*j,1-0.25*j]) 
                    
                    if 0<j<n_mode-1: 
                        plt.plot([3.1+3.2*i, 3.2*i+3.5], [1-0.25*j,1-0.25*j], color = cl)
                        coords2.append([3.1+3.2*i, 3.2*i+3.5, 1-0.25*j,1-0.25*j])

                        plt.plot([2.2+3.2*i, 3.2*i+2.8], [1-0.25*j,1-0.25*j], color = cl)
                        plt.plot([3.8+3.2*i, 3.2*i+4.4], [1-0.25*j,1-0.25*j], color = cl)

        if n_mode%2==1:  # for odd mode
            for i in range(int((n_mode+1)/2)):
                plt.plot([2.2+3.2*i, 3.2*i+2.2+2.2], [1,1], color = cl)
            #     plt.plot([1.2+3.2*i, 3.2*i+2.2+2.2], [1-0.25*(n_mode-1), 1-0.25*(n_mode-1) ], color = cl)
                
                for j in range(n_mode):
                    if j< n_mode-1: # remove last line
                        plt.plot([1.5+3.2*i, 3.2*i+1.9], [1-0.25*j,1-0.25*j], color = cl)
                        coords1.append( [1.5+3.2*i, 3.2*i+1.9, 1-0.25*j,1-0.25*j]) 
                    if j >= n_mode-1: 
                        plt.plot([1.2+3.2*i, 3.2*i+2.2], [1-0.25*j,1-0.25*j], color = cl)
                    
                    if  i< int((n_mode+1)/2)-1 and 0<j<n_mode: # remove the last column
                        plt.plot([3.1+3.2*i, 3.2*i+3.5], [1-0.25*j,1-0.25*j], color = cl)
                        coords2.append([3.1+3.2*i, 3.2*i+3.5, 1-0.25*j,1-0.25*j])

                        plt.plot([2.2+3.2*i, 3.2*i+2.8], [1-0.25*j,1-0.25*j], color = cl)
                        plt.plot([3.8+3.2*i, 3.2*i+4.4], [1-0.25*j,1-0.25*j], color = cl)


        # connecting lines i, i+1
        for i  in range(len(coords1)):
            if i%2==0:
                self.connect1(coords1[i], ax)
            if i%2==1:
                self.connect2(coords1[i])

        for i  in range(len(coords2)):
            if i%2==0:
                self.connect1(coords2[i], ax)
            if i%2==1:
                self.connect2(coords2[i])
        
        # plotting paras
        self.plot_paras(self.dic_mzi, self.n_mode, fs=self.fontsize-8)


        plt.axis(self.axis_off)
        # if self.axis_off:
        #     plt.axis('off')

        plt.show()



    def sort_mzi(self):
        """
        sort mzi parameters in the same array for plotting
        """
        dic_mzi = defaultdict( list) #当key不存在时对应的value是[]
        mzi_list = self.mzi_info["MZI_list"]
        for i in mzi_list:
            dic_mzi[tuple(i[0:2])].append(i[2:])
        return dic_mzi
    
    def ps_pos(self):
        """
        label the position of each phaseshifter for cssr case
        """
        if self.type == "cssr":
            dic_pos = { }
            nmode = self.n_mode
            phase_angle = self.phase_angle
            dic_ =self.dic_mzi
            for mode in range(nmode):
                pair = (mode, mode+1)
                value = dic_[pair]
                value = np.array(value).flatten()
                for k in range(len(value)):
                    dic_pos[(mode, k)] = np.round(value[k],4)
                if mode == nmode -1: 
                    dic_pos[(mode, 0)] = np.round(phase_angle[mode],4)
                else:
                    dic_pos[(mode, k+1)] = np.round(phase_angle[mode],4)
            return dic_pos
        else:
            return None
        

    
    @staticmethod
    def connect1(coordinate, ax, cl="dodgerblue", wid=0.1, height=0.08, a=-0.05, b=-0.05, c=0.7, d=-0.05):
        """
        connect odd column
        """
        x0, x1, y0, y1 = coordinate
    #     print(x0,x1,y0,y1)
        plt.plot([x0, x0-0.3],[y0, y0-0.25], color = cl)
        plt.plot([x1, x1+0.3],[y1, y1-0.25], color = cl) 
        ax.add_patch(patches.Rectangle(
            ((x0+x1)/2 + a, y0 + b),
            wid,
            height,
            edgecolor = 'blue',
            facecolor = 'blue',
            fill=True
        ) )
        
        ax.add_patch(patches.Rectangle(
            ((x0+x1)/2 + c, y0 + d),
            wid,
            height,
            edgecolor = 'blue',
            facecolor = 'blue',
            fill=True
        ) )
    @staticmethod
    def connect2(coordinate, cl='dodgerblue'): 
        """
        connect even column
        """
        x0, x1, y0, y1 = coordinate

        plt.plot([x0, x0-0.3],[y0, y0+0.25], color = cl)
        plt.plot([x1, x1+0.3],[y1, y1+0.25], color = cl) 



    @staticmethod
    def plot_paras(sort_mzi_dic, n_mode, fs=20):
        """
        plotting mzi_paras, for CSSL
        """
        # def positive_angles(angle):
        #     while angle < 0:
        #         angle = angle + 2*np.pi
        #     while angle > 2*np.pi:
        #         angle = angle - 2*np.pi
        #     return angle
        
        def positive_angles(angle):
    
            return angle

        
        for i in sort_mzi_dic.keys():
            if i[0]%2 == 0: # 0, 2, 4, 6..
                temp_values = sort_mzi_dic[i]
                L = len(temp_values)
                for j in range(L):
                    plt.text(8.6-3.2*j+3.2*((n_mode-6)//2+n_mode%2), 1-0.25*i[0]+0.05, "%.2f"%positive_angles(temp_values[j][0]), fontsize = fs )
                    plt.text(7.8-3.2*j+3.2*((n_mode-6)//2+n_mode%2), 1-0.25*i[0]+0.05, "%.2f"%positive_angles(temp_values[j][1]), fontsize = fs )
                    
                    
            if i[0]%2 ==1: # 1, 3..
                temp_values = sort_mzi_dic[i]
                L = len(temp_values)
                for j in range(L):
                    plt.text(8.6-3.2*j +1.6+3.2*((n_mode-6)//2), 1-0.25*i[0]+0.05, "%.2f"%positive_angles(temp_values[j][0]), fontsize = fs )
                    plt.text(7.8-3.2*j +1.6+3.2*((n_mode-6)//2), 1-0.25*i[0]+0.05, "%.2f"%positive_angles(temp_values[j][1]), fontsize = fs )

    
    @staticmethod
    def plot_paras_1(sort_mzi_dic, n_mode, fs=20):
        """
        plotting mzi_paras, CSSR
        """
        # def positive_angles(angle):
        #     while angle < 0:
        #         angle = angle + 2*np.pi
        #     while angle > 2*np.pi:
        #         angle = angle - 2*np.pi
        #     return angle  ##here comes a theta/2 problem 4pi period

        def positive_angles(angle):
    
            return angle

        
        for i in sort_mzi_dic.keys():
            if i[0]%2 == 0: # 0, 2, 4, 6..
                temp_values = sort_mzi_dic[i]
                L = len(temp_values)
                for j in range(L):
                    3.2*(n_mode/2-1)+2.2+2.1
                    plt.text(3.2*j+0.6, 1-0.25*i[0]+0.05, "%.2f"%positive_angles(temp_values[j][0]), fontsize = fs )
                    # plt.text(8.6-3.2*j+3.2*((n_mode-6)//2+n_mode%2), 1-0.25*i[0]+0.05, "%.2f"%positive_angles(temp_values[j][0]), fontsize = fs )
                    plt.text(3.2*j+0.6+0.9, 1-0.25*i[0]+0.05, "%.2f"%positive_angles(temp_values[j][1]), fontsize = fs )
                    
                    
            if i[0]%2 ==1: # 1, 3..
                temp_values = sort_mzi_dic[i]
                L = len(temp_values)
                for j in range(L):
                    plt.text(3.2*j+0.6+1.6, 1-0.25*i[0]+0.05, "%.2f"%positive_angles(temp_values[j][0]), fontsize = fs )
                    plt.text(3.2*j+0.6+2.4, 1-0.25*i[0]+0.05, "%.2f"%positive_angles(temp_values[j][1]), fontsize = fs )
                        

                        

