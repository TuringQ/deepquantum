"""
Map the quantum gate to optical quantum circuit
"""
import copy
import itertools
import os
import pickle
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import special
from scipy.optimize import root
from sympy import symbols, Matrix, simplify
from torch import vmap


class UgateMap():
    """
    map the quantum U gate to unitary matrix in a optical quantum circuit.
    n_qubits: the number of qubits the quantum gates acting on.
    nmode: the number of modes for the circuits.
    ugate: the target quantum U gate.
    success: sqrt of the probability of the optimized quantum gate.
    preferred values: 1/3 for 2qubits, 1/4 for 3qubits
    aux: two auxiliary modes in quantum circuit, values could be [0,0] or [1,0].
    the positon is the last two mode in the circuit by default
    aux_pos: the position of the auxiliary modes, default=[n_mode-2, n_mode-1].
    """
    def __init__(self, n_qubits: int, n_mode: int, ugate: Any, success:Any, aux: List=None, aux_pos: List=None):
        assert 2*n_qubits<=n_mode, 'need more modes'
        self.n_mode = n_mode
        self.ugate = ugate  # the quantum gate to map
        self.success = success
        self.n_qubits = n_qubits
        self.aux = aux
        if aux_pos is None:
            aux_pos = [n_mode-2, n_mode-1]
        self.aux_position = aux_pos
        self.basis = self.create_basis(aux_pos) # these basis is changed with aux_pos

        self.u_dim =self.n_mode
        y_var = symbols('y0:%d'%((self.u_dim)**2))
        y_mat = Matrix(np.array((y_var)))
        y_mat = y_mat.reshape(self.u_dim,self.u_dim) # for symbolic computation, the matrix to optimize
        self.y_var = y_var
        self.u = y_mat

        if self.n_mode == 2*self.n_qubits: # no auxiliary mode case
            self.all_basis = self.create_basis([ ] )
        else:
            self.all_basis = self.create_basis([n_mode-2, n_mode-1] ) #with auxiliary mode case,
            #these basis is fixed  for computation

        # num_photons = int(sum(self.all_basis[0]))
        # dual_rail coding, permanent matrix dim: num_photons x num_photons
        # L = self.u_dim

        basic_dic = {}
        for k in range(len(y_var)):   # here we only consider single term
            basic_dic[y_var[k]]=k
        self.basic_dic = basic_dic

        # load indicies data
        current_path = os.path.abspath(__file__)
        cur_sep = os.path.sep # obtain seperation
        # print(cur_sep)
        path_2 = os.path.abspath(os.path.join(current_path, '../..'))
        path_3 = os.path.abspath(os.path.join(path_2, '..'))
        try:
            # load the idx tensor data
            fn = path_3+cur_sep+'cache'+cur_sep+'Idx_%dqb_%dmode_'%(n_qubits,n_mode)+'aux_%d%d'%(aux[0],aux[1])+'.pt'
            idx_ts = torch.load(fn)
        except:
            idx_ts = None

        self.idx_ts = idx_ts

        if self.idx_ts is None:
            idx_dic, idx_ts = self.indx_eqs()
            # save the idx tensor data
            fn = path_3+cur_sep+'cache'+cur_sep+'Idx_%dqb_%dmode_'%(n_qubits,n_mode)+'aux_%d%d'%(aux[0],aux[1])+'.pt'
            fn_2 = path_3+cur_sep+'cache'+cur_sep+'Idx_%dqb_%dmode_'%(n_qubits,n_mode)+'aux_%d%d'%(aux[0],aux[1])+'.pickle'
            torch.save(idx_ts, fn)
            UgateMap.save_dict(idx_dic, fn_2)

    def create_basis(self, aux_position):
        """
        creat the n qubits basis in  dual-rail coding
        """
        main_position = [i for i in range(self.n_mode) if i not in aux_position]
        # print(main_position)
        all_basis = []
        n = self.n_qubits
        temp = [[1, 0],[0, 1]] # |0> and |1>
        for state in itertools.product([0, 1], repeat=n):
            len_state = len(state)
            dual_code = []
            for m in range(len_state):
                dual_code.extend(temp[state[m]])
            temp_basis = torch.zeros(self.n_mode, dtype=int)
            if self.aux:
                temp_basis[torch.tensor(aux_position, dtype=int)] = torch.tensor(self.aux)
            temp_basis[torch.tensor(main_position, dtype=int)] = torch.tensor(dual_code)
            # dual_code.extend(self.aux)
            all_basis.append(temp_basis)
        return all_basis


    ##############################
    # using symbolic computation finding indicies to avoid repeat computing
    def get_coeff_sym(self, input_state, output_states=None):
        """
        return the transfer state coefficient in a symnolic way
        input_state: torch.tensor
        output_states: List[torch.tensor]
        """
        if output_states is None:
            output_states = self.all_basis
        u =self.u
        dic_coeff={}
        for output in output_states:
            mat = self.sub_matrix_sym(u, input_state, output)
            per = self.permanent(mat)
            temp_coeff = per/np.sqrt((self.product_factorial(input_state)*self.product_factorial(output)))
            dic_coeff[output]=simplify(temp_coeff)
        return dic_coeff

    def get_indx_coeff(self, coeff_i, all_inputs=None):
        """
        get the index of y_var for given state transfer coefficients
        """
        index_coeff = {}
        if all_inputs is None:
            all_inputs = self.all_basis
        basic_dic = self.basic_dic
        temp_ts2 = torch.empty(0, dtype=torch.int32) # set initial value 0
        for s in range(len(all_inputs)):
            input_s = all_inputs[s]
            test = simplify(coeff_i[input_s])
            dict_0 = test.as_coefficients_dict()
            dict_0_keys = list(dict_0.keys())
            temp = []
            temp_2 = []
            for key in dict_0_keys:
                map_tuple = tuple(map(lambda x: basic_dic[x], key.args))
                temp.append([map_tuple, float(dict_0[key])])
                temp_2.append(list(map_tuple))
            index_coeff[tuple(input_s.numpy())] =temp   # save as dictionary
            temp_ts = torch.IntTensor(temp_2).unsqueeze(0)   # save as torch.tensor
            temp_ts2 = torch.cat((temp_ts2, temp_ts),0)
            print('finishing find idx for state',tuple(input_s.numpy()))  ## for check the result
        print('##########')
        return index_coeff, temp_ts2


    def indx_eqs(self):
        """
        get the dic of indx of y_mat for each non_linear eqs for n_mode
        """
        all_inputs = self.all_basis
        all_outputs = list(map(self.get_coeff_sym, all_inputs))
        indices = list(map(self.get_indx_coeff, all_outputs))

        temp_ts = torch.empty(0, dtype=torch.int32)
        idx_dic = []
        for idx in indices:
            idx_dic.append(idx[0])
            temp_ts = torch.cat((temp_ts, idx[1]),0)

        self.idx_all = idx_dic
        self.idx_ts = temp_ts
        return idx_dic, temp_ts

    ##############################
    # constructing and solving nonlinear equations
    @staticmethod
    def single_prod(single_idx_i, y_test):
        temp = y_test[single_idx_i]
        return temp.prod()
    @staticmethod
    def single_output(single_idx, y_test):
        temp_ = vmap(UgateMap.single_prod, in_dims=(0, None))(single_idx, y_test)
        return temp_.sum()

    def get_transfer_mat(self, y):
        y = y.flatten()
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        idx_ts = self.idx_ts
        all_inputs = self.all_basis
        num_basis = len(all_inputs)
        temp_ = vmap(UgateMap.single_output, in_dims=(0, None))(idx_ts, y)
        temp_mat = temp_.reshape(num_basis, num_basis) # here already do transpose
        return temp_mat

    def f_real(self, y):
        """
        construct (2**n_qubit)**2 equations for nxn y, obtain real solutions for real part of u_gate
        y: array with n**2 element
        """
        transfer_mat = self.get_transfer_mat(y)
        u_gate = self.ugate*self.success # the target quantum gate with probability
        if not isinstance(u_gate, torch.Tensor):
            u_gate = torch.tensor(u_gate)
        diff_matrix = transfer_mat - u_gate.real
        eqs = diff_matrix.flatten()  # flatten 之后是按行比较的结果
        eqs_list = eqs.tolist()
        return eqs_list

    def f_real_unitary(self, y):
        """
        return quantum gate constrains + unitary constrains
        """
        eqs_1 = self.f_real(y)
        mat_y = torch.tensor(y).view(self.n_mode, self.n_mode)
        eqs_2 = self.unitary_constrains(mat_y)
        eqs = eqs_1 + eqs_2

        return eqs

    @staticmethod
    def unitary_constrains(u_temp):
        """
        input: torch.tensor
        return n**2 eqs for nxn matrix under unitary condition
        """
        # assert()
        u_size = u_temp.size()
        u_temp_conj = u_temp.conj()
        u_temp_dag = u_temp_conj.transpose(0, 1)
        u_product = u_temp_dag @ u_temp
        u_identity = torch.eye(u_size[0])
        diff_matrix = u_product - u_identity
        eqs = diff_matrix.flatten()
        eqs_list = eqs.tolist()
        return eqs_list

    def f_complex_unitary(self, paras):
        """
        return quantum gate constrains + unitary constrains
        """
        num_paras = len(paras)
        y = np.array(paras)[0 : int(num_paras/2)] + np.array(paras)[int(num_paras/2):]*1j

        eqs_1 = self.f_complex(y)
        mat_y = torch.tensor(y).view(self.n_mode, self.n_mode)
        eqs_2 = self.unitary_constrains_complex(mat_y)
        eqs = eqs_1 + eqs_2

        return eqs

    def f_complex(self, y):
        """
        construct (2**n_qubit)**2  for nxn y, obtain complex solutions
        paras: array with 2*(n**2) element
        """
        num_paras = len(y)*2
        transfer_mat = self.get_transfer_mat(y)
        u_gate = self.ugate*self.success # the target quantum gate with probability
        if not isinstance(u_gate, torch.Tensor):
            u_gate = torch.tensor(u_gate)
        diff_matrix = transfer_mat - u_gate
        eqs = diff_matrix.flatten()  # flatten 之后是按行比较的结果
        eqs_real = eqs.real
        eqs_imag = eqs.imag

        eqs_real_list = eqs_real.tolist()
        eqs_imag_list = eqs_imag.tolist()

        eqs_all = eqs_real_list + eqs_imag_list
        if num_paras > len(eqs_all): # if # parameters > # equations
            for t in range(num_paras-len(eqs_all)):
                extra_eq = y[0]*y[1]-y[0]*y[1]
                eqs_all.append(extra_eq)
        return eqs_all
    @staticmethod
    def unitary_constrains_complex(u_temp):
        """
        input: torch.tensor
        return n**2 eqs for nxn matrix under unitary condition
        """
        # assert()

        u_size = u_temp.size()
        u_temp_conj = u_temp.conj()
        u_temp_dag = u_temp_conj.transpose(0, 1)
        u_product = u_temp_dag @ u_temp
        u_identity = torch.eye(u_size[0])
        diff_matrix = u_product - u_identity
        eqs = diff_matrix.flatten()
        eqs_1 = eqs.real
        eqs_2 = eqs.imag
        eqs1_list = eqs_1.tolist()
        eqs2_list = eqs_2.tolist()
        eqs_list = eqs1_list + eqs2_list

        return eqs_list

    def solve_eqs_real(self, total_trials = 10, trials = 1000, precision = 1e-6 ):
        """
        solving the non-linear eqautions for matrix satisfying ugate with real solution
        """
        func = self.f_real_unitary
        results = []
        for t in range(total_trials):

            result =[]
            sum_=[]
            for i in range(trials):

                y0 = np.random.uniform(-1, 1, (self.u_dim)**2)
                re1 = root(func, y0, method='lm')
                temp_eqs = np.array((func(re1.x)))
                print(re1.success, sum(abs(temp_eqs)))
                if re1.success and  sum(abs(temp_eqs))<precision:
                    re2 = (re1.x).reshape(self.u_dim,self.u_dim) #the result is for aux_pos=[nmode-2,nmode-1],
                    re3 = UgateMap.exchange(re2, self.aux_position) #need row and column exchange for target aux_pos
                    result.append(re3)
                    sum_.append(sum(abs(temp_eqs)))
                print('total:',t, 'trials:', i+1, 'success:', len(result), end='\r')
            results.append(result)
        return results, sum_

    def solve_eqs_complex(self, total_trials = 10, trials = 1000, precision=1e-5 ):
        """
        solving the non-linear eqautions for matrix satisfying ugate with complex solution
        """
        func = self.f_complex_unitary
        results = []
        for t in range(total_trials):

            result = []
            sum_ = []
            for i in range(trials):

                y0 = np.random.uniform(-1, 1, 2*(self.u_dim)**2)
                re1 = root(func, y0, method='lm')
                temp_eqs = np.array((func(re1.x)))
                print(re1.success, sum(abs(temp_eqs)))
                if re1.success and  sum(abs(temp_eqs))<precision:
                    re2 = re1.x[:(self.u_dim)**2] + re1.x[(self.u_dim)**2:]*1j
                    re3 = re2.reshape(self.u_dim, self.u_dim)
                    re4 = UgateMap.exchange(re3, self.aux_position)
                    result.append(re4)
                    sum_.append(sum(abs(temp_eqs)))
                print('total:',t, 'trials:',i, 'success:', len(result), end='\r')
            results.append(result)
        return results, sum_

    @staticmethod
    def exchange(matrix, aux1_pos):
        nmode = matrix.shape[0]
        aux2_pos=[nmode-2, nmode-1]
        matrix_new = copy.deepcopy(matrix)
        for i in range(2):
            matrix_1 = np.delete(matrix_new, aux2_pos[i],0)
            matrix_2 = np.insert(matrix_1, aux1_pos[i],matrix_new[aux2_pos[i],:],0)
            matrix_3 = np.delete(matrix_2, aux2_pos[i],1)
            matrix_4 = np.insert(matrix_3, aux1_pos[i],matrix_2[:,aux2_pos[i]],1)
            matrix_new = matrix_4
        return  matrix_4


    @staticmethod
    def sub_matrix_sym(unitary, input_state, output_state):
        """
        get the sub_matrix of transfer probs for given input and output state
        """
        indx1 = UgateMap.set_copy_indx(input_state)    # indicies for copies of rows for U
        indx2 = UgateMap.set_copy_indx(output_state)   # indicies for copies of columns for
        u1 = unitary[indx1,:]  #输入取行
        u2 = u1[:,indx2] #输出取列
        return u2


    @staticmethod
    ## computing submatrix will invoke
    def set_copy_indx(state):
        """
        picking up indices from the nonezero elements of state,
        repeat times depend on the nonezero value
        """
        inds_nonzero = torch.nonzero(state, as_tuple=False) # nonezero index in state
        # len_ = int(sum(state[inds_nonzero]))
        temp_ind = []

        for i in range(len(inds_nonzero)):       # repeat times depends on the nonezero value
            temp1 = inds_nonzero[i]
            temp = state[inds_nonzero][i]
            temp_ind = temp_ind + [int(temp1)] * (int(temp))

        return  temp_ind

    @staticmethod
    def permanent(mat): # using torch tensor for calculating
        """
        Using RyserFormula for permanent, valid for square matrix
        """

        mat = np.matrix(mat)
        num_coincidence = np.shape(mat)[0]
        sets = UgateMap.create_subset(num_coincidence) # all S set
        value_perm = 0

        for subset in sets:          # sum all S set
            num_elements = len(subset)
            value_times = 1
            for i in range(num_coincidence):
                value_sum = 0
                for j in subset:
                    value_sum = value_sum + mat[i, j]  #sum(a_ij)
                value_times = value_times * value_sum
            value_perm = value_perm + value_times * (-1) ** num_elements
        value_perm = value_perm * (-1) ** num_coincidence
        return value_perm

    @staticmethod
    def product_factorial(state):
        """
        return the product of the factorial of each element
        |s_1,s_2,...s_n> -->s_1!*s_2!*...s_n!

        """
        temp = special.factorial(state)
        product_fac = 1
        for i in range(len(state)):
            product_fac = product_fac * temp[i]
        return product_fac

    @staticmethod
    ## computing permanent will invoke
    def create_subset(num_coincidence):
        """
        all subset from {1,2,...n}
        """
        num_arange = np.arange(num_coincidence)
        all_subset = []

        for index_1 in range(1, 2 ** num_coincidence):
            all_subset.append([])
            for index_2 in range(num_coincidence):
                if index_1 & (1 << index_2):
                    all_subset[-1].append(num_arange[index_2])

        return all_subset
    @staticmethod
    def save_dict(dictionary, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(dictionary, file)


########################################
######some additional function##########
#for plotting matrix and check the unitary matrix

    @staticmethod
    def plot_u(unitary, vmax=1, vmin=0, fs=20, len_ticks=5, cl='RdBu'):
        """
        plotting the matrix in graphic way
        fs: fontsize
        len_ticks: # of ticks in colorbar
        ...
        """
        plt.rcParams ['figure.figsize'] = (8, 8)
        step = (vmax -vmin)/(len_ticks-1)
        ticks_=[0]*len_ticks
        for i in range(len_ticks):
            ticks_[i] = vmin + i*step
        # print(ticks_)
        ax = plt.matshow(np.array(unitary).real, vmax = vmax, vmin =vmin, cmap=cl )
        cb = plt.colorbar(fraction =0.03, ticks=ticks_)
        plt.title('U_real', fontsize=fs)
        plt.xticks(fontsize=fs-1)
        plt.yticks(fontsize=fs-1)
        cb.ax.tick_params(labelsize = fs-6)

        ax1 = plt.matshow(np.array(unitary).imag, vmax = vmax, vmin = vmin,cmap=cl )
        cb1 = plt.colorbar( fraction =0.03, ticks=ticks_)
        plt.title('U_imag', fontsize=fs)
        plt.xticks(fontsize=fs-1)
        plt.yticks(fontsize=fs-1)
        cb1.ax.tick_params(labelsize = fs-6)

    @staticmethod
    def is_unitary(u_temp):
        """
        check the matrix is unitary or not
        """
        # U_temp = construct_u(paras)
        u_size = u_temp.size()
        u_temp_conj = u_temp.conj()
        u_temp_dag = u_temp_conj.transpose(0, 1)
        u_product = u_temp_dag @ u_temp
        u_identity = torch.eye(u_size[0])
        all_sum = float(abs(u_product - u_identity).sum())

        return all_sum

#########################################################
###########check the result using perceval #############
    def state_basis(self):
        """
        map '000' to dual_rail
        """
        state_map = {}
        map_dic = {(1, 0): '0', (0, 1): '1'}
        for i in self.all_basis:
            len_ = len(i)
            s = ''
            for j in range(int(len_/2)):
                temp = tuple(i[2*j : 2*j+2].numpy())
                s = s + map_dic[temp]
            state_map[s] = i.tolist()
        return state_map

#########################################################
#########################################################
