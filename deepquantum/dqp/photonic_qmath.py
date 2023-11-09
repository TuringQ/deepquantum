import numpy as np
import torch 
from scipy import special
import itertools
import copy 
from state import FockState
from typing import Any, List, Optional, Union




def inverse_permutation(permute_shape):
    """Calculate the inversed permutation.

    Args:
        permute_shape (List[int]): Shape of permutation.

    Returns:
        List[int]: A list of integers that is the inverse of ``permute_shape``.
    """
    # find the index of each element in the range of the list length
    return [permute_shape.index(i) for i in range(len(permute_shape))]

###################################
### this part is to obtain the submatrix for caculating the transfer amplitude and transfer probs for
# given input and output state, we take copies of rows from the output, copies of columns from 
# the input as in percelval.
###################################
class CreateSubmat():
    """
    obtain the submatrix for caculating the transfer amplitude and transfer probs for
    given input and output state.

    """
    @staticmethod
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
    def sub_matrix(u, input_state, output_state):
        """
        u: torch.tensor, the unitary matrix for the circuit or component
        
        get the sub_matrix of transfer probs for given input and output state
        here choose rows from the output and choose columns from the input
        """
        indx1 = CreateSubmat.set_copy_indx(input_state) 
        indx2 = CreateSubmat.set_copy_indx(output_state)  
        u1 = u[[indx2]] ## choose rows from the output
        u2 = u1[:, [indx1]] ## choose columns from the input
        return torch.squeeze(u2)            

    @staticmethod
    ## computing permanent 
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
    def permanent(mat):
        """
        Using RyserFormula for permanent, valid for square matrix
        """
        
#         mat = np.matrix(mat)
        if len(mat.size()) == 0 :   # for the single photon case
            return mat
        else:
            num_coincidence = mat.size()[0]
            sets = CreateSubmat.create_subset(num_coincidence) # all S set
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


###################################
### this part is to decompose the unitary matrix to clements structure
#  and plot the clements structure with parameters
###################################





###################################
### this part is to calculate all output fockstate given input state list
###################################
class FockOutput():
    """
    to calculate all output fockstate given input state
    Args: 
        ini_state(FockState): the input fock state for the circuit 
    """

    def __init__(self, ini_state) -> None:

        self.nmode = ini_state.nmode
        self.photons = ini_state.photons
        self.all_com = []

        self.decompose_num(self.photons, self.photons,[])  # decompose the integer number
        self.outfock_dic = {}
        self.calculate_fock_output()

        self.fock_outputs = []
        for key in self.outfock_dic:
            output_list = self.outfock_dic[key]
            for j in output_list:
                out_state = FockState(self.nmode, j)
                self.fock_outputs.append(out_state)                


    
    def decompose_num(self, num, m, all_com):
        """
        for decomposing an integer number, give all combination of decomposition
        num: the integer number for decomposing
        m: the maximum integer number in the decomposition
        all_com: collecting all decomposition result
        """
        if num == 0:
            self.all_com.append(all_com)
        else: 
            if m>1:
                FockOutput.decompose_num(self, num, m-1,all_com)
            if m <= num:
                FockOutput.decompose_num(self, num-m, m, all_com+[m])

    @staticmethod
    def allpermutations(list_L):
        all_permutation = list(itertools.permutations(list_L))
        return all_permutation
    
    def calculate_fock_output(self):
        for i in range (len(self.all_com)):
            list_copy = copy.deepcopy(self.all_com[i])
            temp_len = len(list_copy)
            if temp_len <= self.nmode:
                list_copy = list_copy +[0]*(self.nmode-temp_len)
                self.outfock_dic[tuple(self.all_com[i])] = (list(set(FockOutput.allpermutations(list_copy))))
            # All_com[i] = list_copy 
            # return

###################################
### this part is to calculate the matrix tensor for the optical elements which gives the transfer amplitudes of
# the quantum state given the input quantum fock state tensor 
###################################
class FockGateTensor():
    """
    to calculate the matrix tensor for the optical elements which gives the transfer amplitudes of
    the quantum state given the input quantum fock state tensor 
    see https://arxiv.org/pdf/2004.11002.pdf eqs 74, 75

    Args:
        n_mode:
        cutoff:
        parameters:

    """
    def __init__(
        self, 
        n_mode: int = None, 
        cutoff: int = None, 
        parameters:  Union[int, List[int], None] = None,
    ) -> None:
        if not isinstance(parameters, List):
            parameters = [parameters]
        self.nmode = n_mode
        self.cutoff = cutoff
        self.parameters = parameters
        self.u = None
    def bs(self, dtype=torch.complex128):
        """
        give the tensor representation of the beamsplitter
        return Z_ _, _ _ .The left part for output state, the right part for the input
        state
        """
        assert len(self.parameters)==2, "BS gate needs two parameters theata and phi"
        theta, phi = self.parameters
        cutoff = self.cutoff

        sqrt = torch.sqrt(torch.tensor(np.arange(cutoff), dtype=dtype))
        ct = torch.cos(theta)
        st = torch.sin(theta) * torch.exp(1j * phi)
        torch_zero = torch.tensor(0)
        R = torch.stack(
                [torch_zero, torch_zero, ct, -torch.conj(st),
                torch_zero, torch_zero, st, ct,
                ct, st, torch_zero, torch_zero,
                -torch.conj(st), ct, torch_zero, torch_zero]
        ) .reshape(4, 4)
        self.u = R
        Z = torch.zeros([cutoff]*4, dtype=dtype)   # 2 outputs modes + 2 inputs modes 
        Z[0, 0, 0, 0] = 1.0
        # rank 3
        for m in range(cutoff):
            for n in range(cutoff - m):
                p = m + n
                if 0 < p < cutoff:
                    Z[m, n, p, 0] = (
                        R[0, 2] * sqrt[m] / sqrt[p] * Z[m - 1, n, p - 1, 0]
                        + R[1, 2] * sqrt[n] / sqrt[p] * Z[m, n - 1, p - 1, 0]
                    )

        # rank 4
        for m in range(cutoff):
            for n in range(cutoff):
                for p in range(cutoff):
                    q = m + n - p
                    if 0 < q < cutoff:
                        Z[m, n, p, q] = (
                            R[0, 3] * sqrt[m] / sqrt[q] * Z[m - 1, n, p, q - 1]
                            + R[1, 3] * sqrt[n] / sqrt[q] * Z[m, n - 1, p, q - 1]
                        )
    #     Z = Z.transpose((0, 2, 1, 3))
        return Z
    

    def ps(self, dtype=torch.complex128):
        """
        give the tensor representation of the phase shifter
        """
        assert len(self.parameters)==1, "PS gate needs one parameter theata"
        cutoff = self.cutoff
        Z = torch.zeros([cutoff]*2, dtype=dtype)   # 1 outputs mode + 1 inputs mode1
        theta = self.parameters[0] 
        Z[0, 0] = 1.0
        for i in range(1, cutoff):
               Z[i,i] = torch.exp(1j * theta*i)
        return Z


    # def h():
    # def Rx():
    # def Ry():
    # def Rz():



    



    


