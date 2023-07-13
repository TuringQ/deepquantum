import numpy as np
import torch
from torch import nn

from torch.distributions.uniform import Uniform
from torch.distributions.multivariate_normal  import MultivariateNormal
import numbers
import gaussian.util as util

from thewalrus import hafnian
import itertools
from scipy.stats import unitary_group






########################
#### Gaussian class ####
########################

class Gaussian:
    """
    Torch version of Gaussian class.
    Class for Gaussian process: prepare a gaussian state, make gaussian unitrary transformations, and finally make guassian and non-gaussian measurments.
    """
    def __init__(self, mode_number, batch_size, kappa=1/np.sqrt(2), dtype=torch.complex128):
        """
        Initialize a gaussian system in the vaccum state with a specfic number of optical modes.
        mode_number: the number of modes.
        kappa: the constant appearing in the transformation of creation and annihilation operators with quadrature operators, the default value is 1/sqrt(2).
        """
        
        # initialize a vaccum state  
        self.reset(mode_number, batch_size, kappa, dtype)
        print(f'Initialize a gaussian system in the vaccum state with {mode_number} modes and batch size {batch_size}.')
    
    
    
    def reset(self, mode_number, batch_size, kappa=1/np.sqrt(2), dtype=torch.complex128):
        """
        Reset the system into a vaccum state with a specific number of modes.
        """
        if not isinstance(mode_number, int) or mode_number <= 0:
            raise ValueError('The number of optical modes must be a positive integer!')
        if not isinstance(kappa, numbers.Real) or kappa <= 0:
            raise ValueError('kappa must be a positive real number!')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('The batch size must be an positive integer!')
        if dtype not in (torch.complex128, torch.complex64):
            raise ValueError('The data type must be complex Pytorch data type!')
        
        self._kappa = kappa
        self._mode_number = mode_number
        self._batch_size = batch_size
        self._dtype = dtype

        # initialize a vaccum state  
        self._covariance = 1 / 2 * torch.stack([torch.eye(2*mode_number, dtype=self._dtype)] * self._batch_size)
        self._displacement = torch.zeros(batch_size, 2*mode_number, dtype=self._dtype)
    

    
    def update(self, mean, cov):
        """
        Update the covairance and displacement vector of the Gaussian state according to the 
        provided covariance and displacement vector.

        mean: torch tensor 
        cov: torch tensor
        """
        if mean.shape == self._displacement.shape:
            self._displacement = mean
        elif len(mean.shape) == 1 and mean.shape[0] == self._displacement.shape[1]:
            self._displacement[:,] = mean
        else:
            raise ValueError("The shape of the provided displacement vector or the number of modes must match the shape of the system or the number of system's modes.")

        if cov.shape == self._covariance.shape:
            self._covariance = cov
        elif len(cov) == 2 and cov.shape[0] == self._covariance[1]:
            self._covariance[:,] = cov
        else:
            raise ValueError("The shape of the provided covariance matrix must match the shape of system's covariance matrix.")
       
    
        
    def covariance(self):
        """
        Return the covariance matrix defined by creation and annihilation operators.
        """
        return self._covariance
    
    
    def displacement(self):
        """
        Return the displacement vector defined by creation and annihilation operators.
        """
        return self._displacement
    
    
    def quadrature_rep(self):
        """
        Return the displacement vector and covariance matrix of the present state, defined by the quadrature operators, 
        i.e., the position and momentum operators.
        """
        return self.annihilation_to_quadrature(self._displacement, self._covariance)
    
    
    
    def annihilation_to_quadrature(self, mean, cov):
        """
        Get the covariance matrix and mean vector of quadrature operators from the covariance matrix and mean vector of creation and annihilation operators.
        Inputs:
        mean, torch tensor, [batch_size, 2*mode_number];
        cov, torch tensor, [batch_size, 2*mode_number, 2*mode_number].
        
        Ouputs:
        
        """
        # transform matrix from creation and annihilation to quadrature
        identity = torch.eye(self._mode_number, dtype = self._dtype)
        t = 1 / (2 * self._kappa) * torch.cat([torch.cat([identity, identity], dim=1), torch.cat([-1j*identity, 1j*identity], dim=1)], dim=0)
        hermitian_t = 1 / (2 * self._kappa) * torch.cat([torch.cat([identity, 1j*identity], dim=1), torch.cat([identity, -1j*identity], dim=1)], dim=0)

        quadrature_displacement = torch.squeeze(t @ torch.unsqueeze(mean, 2),2)
        quadrature_covariance = t @ cov @ hermitian_t

        return quadrature_displacement, quadrature_covariance



    
    def quadrature_to_annihilation(self, quad_mean, quad_cov):
        """
        Get the covariance matrix and mean vector of creation and annihilation operators from the covariance matrix and mean vector of
        quadrature operators.
        """
        # transform matrix from quadrature to creation and annihilation
        identity = torch.eye(self._mode_number, dtype = self._dtype)
        t = self._kappa * torch.cat([torch.cat([identity, 1j*identity], dim=1), torch.cat([identity, -1j*identity], dim=1)], dim=0)
        hermitian_t = self._kappa * torch.cat([torch.cat([identity, identity], dim=1), torch.cat([-1j*identity, 1j*identity], dim=1)], dim=0)

        mean = torch.squeeze(t @ torch.unsqueeze(quad_mean, 2),2)
        cov = t @ quad_cov @ hermitian_t
        return mean, cov
    
    
    

    def chara_wigner(self, x, b=None):
        """
        Get the Wigner characteristic function of present gaussian state, convention from paper "Gaussian Quantum Information".
        
        Input:
        x, a real torch tensor with shape being [batch_size, 2*mode_number].
        b, a real torch tensor with shape being [batch_size, 2*mode_number].
        
        Return:
        the complex value of Wigner characteristic function at x.
        """
        if len(x.shape) == 1:
            x = torch.stack([x] * self._batch_size)
        if 2 * self._mode_number != x.shape[1]:
            raise ValueError('The dimension of input variable must equal two times the number of modes!')
        if b == None:
            b = torch.zeros(self._batch_size, 2*self._mode_number)
        
        # get the covairance matrix and displacement vector in quadrature representation
        dis, cov = self.annihilation_to_quadrature(self._displacement, self._covariance)
        # get the cov and dis in xpxp ordering
        dis, cov = util.xxpp_to_xpxp(dis, cov, self._mode_number, self._dtype)
        
        # get the commutation matrix in xpxp ordering
        lamb = util.lambda_xpxp(self._mode_number, self._dtype)
        
        # return characteristic function
        factor1 = torch.unsqueeze(x, 1) @ (lamb @ cov @ lamb.T) @ torch.unsqueeze(x, 2)
        factor2 = 1j * torch.unsqueeze((dis + b) @ lamb.T, 1) @ torch.unsqueeze(x, 2)
        chara = torch.exp(- 1/2 * factor1.reshape(self._batch_size) - factor2.reshape(self._batch_size))
        
        return chara
    
    
        
        
        
    
    #############
    #############
    # single mode operators

    def displace_one_mode(self, mode_id, distance):
        """
        Apply the displacement operator to a specific mode. Displacement operator only changes the corresponding displacement vector.
        
        mode_id: the ID of displaced mode. The modes are labelled as (0, 1, 2, ..., mode_number-1).
        distance: torch tensor with shape [batch_size] or [1], the complex distance of displacement.
        """
        if not isinstance(mode_id, int):
            raise ValueError('The ID of a mode must be an integer!')
        if mode_id >= self._mode_number:
            raise ValueError('The mode to be displaced does not exist!')
        if distance.shape[0] != self._batch_size and distance.shape[0] != 1:
            raise ValueError('The dimension of the parameter tensor should be equal to batch size or 1!')
        
        self._displacement[:, mode_id] += distance
        self._displacement[:, self._mode_number+mode_id] += torch.conj(distance)

    
    
    
    def squeeze_one_mode(self, mode_id, r, theta):
        """
        Apply the squeezing operator to a specific mode.
        
        r: real torch tensor with shape [batch_size] or [1], the absolute value of the complex squeezing parameter.
        theta: real torch tensor with shape [batch_size] or [1], the angle of the complex squeezing parameter.
        """
        if not isinstance(mode_id, int):
            raise ValueError('The ID of a mode must be an integer!')
        if mode_id >= self._mode_number:
            raise ValueError('The mode to be displaced does not exist!')
        if r.shape[0] != self._batch_size and r.shape[0] != 1:
            raise ValueError('The dimension of the r tensor should be equal to batch size or 1!')
        if theta.shape[0] != self._batch_size and theta.shape[0] != 1:
            raise ValueError('The dimension of the theta tensor should be equal to batch size or 1!')
            

        # change the dtype of r into default dtype
        r = r.type(self._dtype)
        # transformation matrix
        t = torch.zeros(self._batch_size, 2, 2, dtype=self._dtype)
        t[:,0,0] = torch.cosh(r)
        t[:,0,1] = -torch.exp(1j*theta) * torch.sinh(r)
        t[:,1,0] = -torch.exp(-1j*theta) * torch.sinh(r)
        t[:,1,1] = torch.cosh(r)

        
        # update displacement vector
        disp = self._displacement.clone()
        self._displacement[:, [mode_id, mode_id+self._mode_number]] = torch.squeeze(t @ torch.unsqueeze(disp[:,[mode_id, mode_id+self._mode_number]], 2), 2)
        
        
        # update covariance matrix
        hermitian_t = torch.transpose(torch.conj(t), 1, 2)
        rows = torch.tensor([[mode_id, mode_id], [mode_id+self._mode_number, mode_id+self._mode_number]])
        columns = torch.tensor([[mode_id, mode_id+self._mode_number], [mode_id, mode_id+self._mode_number]])
        cov = self._covariance.clone()
        self._covariance[:, rows, columns] = t @ cov[:, rows, columns] @ hermitian_t 
        
        
     
    def phase_shift_one_mode(self, mode, theta):
        """
        Apply a phase shift operator to a specific mode. 
        
        theta: real torch tensor with shape [batch_size] or [1], the parameter of phase shifter.
        """
        if not isinstance(mode, int):
            raise ValueError('The label of a mode must be an integer!')
        if mode >= self._mode_number:
            raise ValueError('The mode to be displaced does not exist!')
        if theta.shape[0] != self._batch_size and theta.shape[0] != 1:
            raise ValueError('The dimension of the theta tensor should be equal to batch size or 1!')
        
        mode2 = mode + self._mode_number
        
        # update displacement vector
        # in our convention, annihilation operator is multiplied by e^{-i theata}.
        dis = self._displacement.clone()
        self._displacement[:, mode] = torch.exp(-1j*theta) * dis[:, mode]
        self._displacement[:, mode2] = torch.exp(1j*theta) * dis[:, mode2]
        
        # update covariance matrix
        cov = self._covariance.clone()
        self._covariance[:, mode, mode2] = torch.exp(-2*1j*theta) * cov[:, mode, mode2]
        self._covariance[:, mode2, mode] = torch.exp(2*1j*theta) * cov[:, mode2, mode]
    
    
    
    
    
    ###########
    ###########
    # multiple modes gates
    
    
    def displace(self, modes, parameters):
        """
        Displace multiple modes.
        
        modes = n or [n_1, ..., n_N],
        parameters = torch tensor with shape (batch, num_para) or (num_para).
        """
        if isinstance(modes, int):
            modes = [modes]
        if not set(modes).issubset(set(range(self._mode_number))):
            raise ValueError('Some modes to be squeezed do not exist!')
        if len(parameters.shape) == 1:
            parameters = torch.stack([parameters] * self._batch_size)
        if not len(modes) == parameters.shape[1]:
            raise ValueError('The number of squeezing parameters must equal the number of modes!')  
    
        # squeeze each mode
        for i in range(len(modes)):
            self.displace_one_mode(modes[i], parameters[:,i])
        
        
        
        
    
    def squeeze(self, modes, r, theta):
        """
        Squeeze some modes with specific parameters.
        
        modes = n or [n_1, ..., n_N],
        r = torch.tensor with shape (batch_size, num_modes) or (num_modes).
        theta = torch.tensor with shape (batch_size, num_modes) or (num_modes).
        """
        if isinstance(modes, int):
            modes = [modes]
        if not set(modes).issubset(set(range(self._mode_number))):
            raise ValueError('There are some modes to be squeezed do not exist!')
        if len(r.shape) == 1:
            r = torch.stack([r] * self._batch_size)
        if not len(modes) == r.shape[1]:
            raise ValueError('The number of r must equal the number of modes!')
        if len(theta.shape) == 1:
            theta = torch.stack([theta] * self._batch_size)
        if not len(modes) == theta.shape[1]:
            raise ValueError('The number of theta must equal the number of modes!')
        
        # squeeze each mode
        for i in range(len(modes)):
            self.squeeze_one_mode(modes[i], r[:,i], theta[:,i])
      
    
            
    def phase_shift(self, modes, parameters):
        """
        Phase shift some modes.
        
        modes = n or [n_1, ..., n_N],
        parameters = real torch tensor with shape (batch_size, num_modes).
        """
        if isinstance(modes, int):
            modes = [modes]
        if not set(modes).issubset(set(range(self._mode_number))):
            raise ValueError('There are some modes to be squeezed do not exist!')
        if len(parameters.shape) == 1:
            parameters = torch.stack([parameters] * self._batch_size)
        if len(modes) != parameters.shape[1]:
            raise ValueError('The number of squeezing parameters must equal the number of modes!')
        
        # squeeze each mode
        for i in range(len(modes)):
            self.phase_shift_one_mode(modes[i], parameters[:,i])
     
    
    
    def beam_splitter(self, modes, r, theta):
        """
        Apply beam splitter to two modes.
        
        modes: python list, two modes of the beam splitter.
        r = real torch.tensor with shape (batch_size, num_modes) or (num_modes).
        theta = real torch.tensor with shape (batch_size, num_modes) or (num_modes).
        """
        if not set(modes).issubset(set(range(self._mode_number))):
            raise ValueError('There are some modes to be squeezed do not exist!')
        if r.shape[0] != self._batch_size and r.shape[0] != 1:
            raise ValueError('The dimension of tensor r should be equal to batch size or 1!')
        if theta.shape[0] != self._batch_size and theta.shape[0] != 1:
            raise ValueError('The dimension of tensor theta should be equal to batch size or 1!')
            
        r = r.type(self._dtype)
        # transformation matrix
        t = torch.zeros(self._batch_size,2,2, dtype=self._dtype)
        t[:,0,0] = torch.cos(r)
        t[:,0,1] = -1j * torch.exp(1j*theta) * torch.sin(r)
        t[:,1,0] = -1j * torch.exp(-1j*theta) * torch.sin(r)
        t[:,1,1] = torch.cos(r)

        # Hermitian conjugate of matrix
        hermitian_t = torch.conj(torch.transpose(t, 1, 2))
        # complex conjugate of t
        conjugate_t = torch.conj(t)
        
        # update displacement vector
        modes2 = [modes[0]+self._mode_number, modes[1]+self._mode_number]
        dis = self._displacement.clone()
        self._displacement[:, modes] = torch.squeeze(t @ torch.unsqueeze(dis[:, modes],2), 2)
        self._displacement[:, modes2] = torch.squeeze(conjugate_t @ torch.unsqueeze(dis[:, modes2], 2), 2)

        
        # update covariance matrix
        rows = torch.tensor([[modes[0], modes[0]], [modes[1], modes[1]]])
        columns = torch.tensor([[modes[0], modes[1]], [modes[0], modes[1]]])
        cov = self._covariance.clone()
        self._covariance[:, rows, columns] = t @ cov[:, rows, columns] @ hermitian_t
        
        rows2 = torch.tensor([[modes2[0], modes2[0]], [modes2[1], modes2[1]]])
        columns2 = torch.tensor([[modes2[0], modes2[1]], [modes2[0], modes2[1]]])
        self._covariance[:, rows2, columns2] = conjugate_t @  cov[:, rows2, columns2] @ torch.transpose(t, 1, 2)
    
    
    
    def unitary_transform(self, u):
        """
        Apply a given unitary transformation realized by a linear optical circuit (consists of beam splitter and phaser shifter) to the gaussian state.
        
        Input
        u: a given unitary matrix, torch tensor with shape [batch_size, mode_number, mode_numbe].
        """
        # construct the symplectic matrix from the given unitary matrix
        symplectic = torch.cat([torch.cat([u, torch.zeros(self._mode_number, self._mode_number)], dim=1), torch.cat([torch.zeros(self._mode_number, self._mode_number), torch.conj(u)], dim=1)], dim=0)
        
        # change displacement
        self._displacement = torch.squeeze(symplectic @ torch.unsqueeze(self._displacement, dim=2), dim=2)
        # change covariance
        self._covariance = symplectic @ self._covariance @ torch.conj(symplectic.T)
        
        
    
    def haar_unitary(self):
        """
        Apply a haar random unitary transformation to the gaussian state.
        The random unitary matrix is generated by function 'scipy.stats.unitary_group'.
        """
        # generate a random unitary matrix [mode_number, mode_number], representing a random linear optical circuit acting on creation operators
        u = torch.tensor(unitary_group.rvs(self._mode_number))
        # change the gaussian state
        self.unitary_transform(u)
    
    
    
    
    

        
    
    
    #############
    #############
    # prepare initial gaussian states
  
    
    def prepare_init_squeeze(self, modes, parameters):
        """
        Prepare an initial squeezed state in some modes.
        
        modes = np.array([n_1, ..., n_N]),
        parameters = np.array([[r_1, theta_1], ..., [r_N, theta_N]]).
        """
        self.squeeze(modes, parameters)
       
            
        
    def prepare_init_coherent(self, modes, parameters):
        """
        Prepare an initial coherent state.
        
        modes = np.array([n_1, ..., n_N]),
        parameters = np.array([alpha_1, ..., alpha_N]).
        """
        self.displace(modes, parameters)
        
    
 

        
    #############
    #############
    # measurements    
    
    def general_dyne_one_mode(self, covariance, mode):
        """
        Apply a general-dyne detection with a specific covariance matrix to a mode.
        
        covariance: the covariance matrix.
        mode: the index of mode to be measureed.
        """
        
        if not set(mode).issubset(set(range(self._mode_number))):
            raise ValueError('There are some modes to be squeezed do not exist!')
        if covariance.shape != (self._batch_size, 2*len(mode), 2*len(mode)):
            raise ValueError("The size of covariance matrix does not match the number of modes to be deteced.")
            
        # get the covariance matrix and mean vector in the quadrature representation
        quad_mean, quad_cov = self.quadrature_rep()
        
        # get the IDs of quadrature operators from ID of mode
        mode = torch.tensor(mode)
        quad_ids = torch.cat([mode, mode + self._mode_number], dim=0)
        
        # split the covariance of quadrature into three parts 
        (cov_a, cov_b, cov_c) = util.split_covariance(quad_cov, quad_ids)
        # split the mean vector
        (mean_a, mean_c) = util.split_mean(quad_mean, quad_ids)
        # sample the measured results 
        res = MultivariateNormal(mean_c.real, (cov_c+covariance).real).sample()
        
        # update the covariance matrix and mean vector of the left state after measurment
        # the covariance matrix
        
        v = cov_a - cov_b @ torch.inverse(cov_c + covariance) @ torch.transpose(cov_b, 1, 2)
        full_v = util.embed_to_covariance(v, quad_ids, self._dtype)
        # mean vector
        w = mean_a + torch.squeeze(cov_b @ torch.inverse(cov_c + covariance) @ torch.unsqueeze((res - mean_c), dim=2), dim=2)
        full_w = util.embed_to_mean(w, quad_ids, self._dtype)
        # update the state
        new_mean, new_cov = self.quadrature_to_annihilation(full_w, full_v)
        self.update(new_mean, new_cov)
        
        return res
    
    

    def heterodyne_one_mode(self, mode):
        """
        Apply heterodyne detection to a specific mode.
        """
        if not isinstance(mode, list):
            mode = [mode]
        
        cov = torch.stack([torch.eye(2*len(mode), dtype=self._dtype)] * self._batch_size)
        res = self.general_dyne_one_mode(cov, mode)
    
        return res[0][0] + 1j * res[0][1]

    


    def homodyne_one_mode(self, mode, eps=0.000001):
        """
        Apply homodyne detection to a specific mode.
        """
        if not isinstance(mode, list):
            mode = [mode]
        cov = torch.stack([torch.tensor([[eps**2, 0], [0, 1/eps**2]])] * self._batch_size)
        
        res = self.general_dyne_one_mode(cov, mode)
        return res
    
    
    def prob_hafnian(self, photon_pattern):
        """
        Calculate the probability of a specific photon pattern restricted in free-collision subspace for a Gaussian state with displacement being zero.
        The function needs the package 'thewalrus'.
        Reference: "Detailed study of Gaussian boson sampling".
        
        Input:
        photon_pattern: torch tensor, [batch_size, mode_number].
        
        Output:
        prob: the probability.
        """
        if len(photon_pattern.shape) == 1:
            photon_pattern = torch.stack([photon_pattern] * self._batch_size)
        if photon_pattern.shape[1] != self._mode_number:
            raise ValueError("The number of modes of photon pattern must match the number of modes.")
        if photon_pattern.shape[0] != self._batch_size:
            raise ValueError("The length of photon pattern must match the batch size.")

        
        # get the bathced matrix A
        iden = torch.eye(2*self._mode_number, dtype=self._dtype)
        cov_q = self._covariance + 1/2 * iden # sigma_Q
        #print(torch.isclose(self._covariance, torch.conj(torch.transpose(self._covariance, 1, 2)), rtol=1e-15, atol=1e-18))
        symplectic = torch.cat([torch.cat((torch.zeros(self._mode_number, self._mode_number, dtype=self._dtype), torch.eye(self._mode_number, dtype=self._dtype)), 1), \
               torch.cat((torch.eye(self._mode_number, dtype=self._dtype), torch.zeros(self._mode_number, self._mode_number, dtype=self._dtype)), 1)], 0)
        matrix_A = symplectic @ (iden - torch.linalg.pinv(cov_q, atol=1e-30, rtol=1e-15, hermitian=True)) # matrix A in eq.(8)

    

        def oper(mat, pattern):
            # get the batched matrix A_S
            ids = torch.nonzero(pattern, as_tuple=True)
            ids = torch.cat([ids[0], ids[0]+self._mode_number], dim=0)
            mat_S = mat[ids][:, ids].detach().numpy()
            #print(mat_S)
            return hafnian(mat_S)
        

        # calculate the hafnian
        haf_batch = torch.tensor([oper(matrix_A[i], photon_pattern[i]) for i in range(self._batch_size)])
        # compute the prefactor
        factor = 1 / torch.sqrt(torch.linalg.det(cov_q).abs())

        return factor * haf_batch
    
    
    
    def prob_gbs(self, paras, u, photon_pattern):
        """
        Calculate the probability of a measured photon pattern of a GBS process.
        The function needs the package 'thewalrus'.
        Reference: "Detailed study of Gaussian boson sampling".
        
        Input:
        photon_pattern: torch tensor, [batch_size, mode_number].
        paras: the squeezing paras, r.
        u: a list of the unitrary matrix of linear optical circuit.
        
        Output:
        prob: the probability.
        """
        if len(photon_pattern.shape) == 1:
            photon_pattern = torch.stack([photon_pattern] * self._batch_size)
        if photon_pattern.shape[1] != self._mode_number:
            raise ValueError("The number of modes of photon pattern must match the number of modes.")
        if photon_pattern.shape[0] != self._batch_size:
            raise ValueError("The length of photon pattern must match the batch size.")
        if len(paras.shape) == 1:
            paras = torch.stack([paras] * self._batch_size)
        
        # according to formula (27)
        diag = torch.zeros(self._batch_size, self._mode_number, self._mode_number, dtype=self._dtype)
        for i in range(paras.shape[1]):
            diag[:, i, i] = torch.tanh(paras[:, i])
        # B matrix in (27)
        t = torch.eye(self._mode_number, dtype=self._dtype)
        if type(u) == list:
            for i in range(len(u)):
                t = t @ u[i]
        else:
            t = u
        B_mat = t @ diag @ t.T

        def oper(mat, pattern):
            # get the batched matrix A_S
            ids = torch.nonzero(pattern, as_tuple=True)[0]
            mat_S = mat[ids][:, ids].detach().numpy()
            return hafnian(mat_S)
        
        # calculate the hafnian
        haf_batch = torch.tensor([oper(B_mat[i], photon_pattern[i]) for i in range(self._batch_size)])
        #print(haf_batch)
        
        # sigma_Q in (26)
        cov_q = self._covariance + torch.eye(2*self._mode_number, dtype=self._dtype) / 2
        # compute the prefactor
        factor = 1 / torch.sqrt(torch.linalg.det(cov_q).abs())

        return factor.detach() * torch.abs(haf_batch).detach()
    
    

    
    def mean_photon_number(self, mode_id):
        """
        Calculate the mean photon number for a specific mode from taking derivatives of characteristic funciton.
        """
        # get the covairance matrix and displacement vector in quadrature representation
        dis, cov = self.annihilation_to_quadrature(self._displacement, self._covariance)
        # get the cov and dis in xpxp ordering
        dis, cov = util.xxpp_to_xpxp(dis, cov, self._mode_number, self._dtype)
        
        # get the commutation matrix in xpxp ordering
        lamb = util.lambda_xpxp(self._mode_number, self._dtype)
        cov = lamb @ cov @ lamb.T
        dis = dis @ lamb.T
        
        # the indices of variables in xpxp ordering of quadrature representation
        i, j = 2*mode_id, 2*mode_id + 1
        t = util.double_partial(dis, cov, i) + util.double_partial(dis, cov, j)
    
        return -(1 + t) / 2
    
    
    def diff_photon_number(self, mode1, mode2):
        """
        Calculate the expectation value of the square of the difference of photon number operator of two modes.
        Ref: Training Gaussian boson sampling by quantum machine learning.
        
        """
        # get the covairance matrix and displacement vector in quadrature representation
        dis, cov = self.annihilation_to_quadrature(self._displacement, self._covariance)
        # get the cov and dis in xpxp ordering
        dis, cov = util.xxpp_to_xpxp(dis, cov, self._mode_number, self._dtype)
        
        # get the commutation matrix in xpxp ordering
        lamb = util.lambda_xpxp(self._mode_number, self._dtype)
        cov = lamb @ cov @ lamb.T
        dis = dis @ lamb.T
        
        # the indices of variables in xpxp ordering of quadrature representation
        i, j, k, l = 2*mode1, 2*mode1 + 1, 2*mode2, 2*mode2 + 1
        # calculate the double derivatives
        t = util.two_double_partial(dis, cov, i, i) + util.two_double_partial(dis, cov, j, j) \
            + util.two_double_partial(dis, cov, k, k) + util.two_double_partial(dis, cov, l, l) \
            + 2 * (util.two_double_partial(dis, cov, i, j) - util.two_double_partial(dis, cov, i, k)\
            - util.two_double_partial(dis, cov, i, l) - util.two_double_partial(dis, cov, j, k)\
            - util.two_double_partial(dis, cov, j, l) + util.two_double_partial(dis, cov, k, l))
        #t = torch.sqrt(t * torch.conj(t)).real
        return t / 4 - 1 / 2
        
    
 





###########################
#### single mode gates ####
###########################


class Displacement(nn.Module):
    """
    Parameters:
        r (tensor): displacement magnitude 
        phi (tesnor): displacement angle 
    """
    def __init__(self, mode=0):
        super().__init__()
        self.mode = mode
        self.is_r_set  = False
        self.is_phi_set  = False
        
        
    def forward(self, state):
        self.auto_params(state)
        distance = self.r * torch.exp(1j * self.phi)
        # if distance is a torch tensor scalar, change it into torch tensor with shape [1]
        if distance.shape == torch.Size([]):
            distance = torch.unsqueeze(distance, dim=0)
        # displace the mode
        state.displace_one_mode(self.mode, distance)
        return state
    
    def set_params(self, r=None, phi=None):
        """set r, phi to tensor independently"""
        if r != None:
            self.register_buffer('r', r)
            self.is_r_set = True
        if phi != None:
            self.register_buffer('phi', phi)
            self.is_phi_set = True

    def auto_params(self, state):
        """automatically set None parameter as nn.Paramter for users"""
        if not self.is_r_set:
            self.register_parameter('r', nn.Parameter(torch.randn([], dtype=torch.float64)))
        if not self.is_phi_set:
            self.register_parameter('phi', nn.Parameter(torch.randn([], dtype=torch.float64)))





class Squeeze(nn.Module):
    """
    Parameters:
        r (tensor): displacement magnitude 
        phi (tesnor): displacement angle 
    """
    def __init__(self, mode=0):
        super().__init__()
        self.mode = mode
        self.is_r_set  = False
        self.is_phi_set  = False
        
        
    def forward(self, state):
        self.auto_params(state)
        if self.r.shape == torch.Size([]):
            r = torch.unsqueeze(self.r, dim=0)
        else:
            r = self.r
        if self.phi.shape == torch.Size([]):
            phi = torch.unsqueeze(self.phi, dim=0)
        else:
            phi = self.phi
        # part of initialization of parameter
        if not self.is_phi_set:
            phi = 2 * torch.pi * phi
        # squeeze the mode
        state.squeeze_one_mode(self.mode, r, phi)
        return state
    
    def set_params(self, r=None, phi=None):
        """set r, phi to tensor independently"""
        if r != None:
            self.register_buffer('r', r)
            self.is_r_set = True
        if phi != None:
            self.register_buffer('phi', phi)
            self.is_phi_set = True

    def auto_params(self, state):
        """automatically set None parameter as nn.Paramter for users"""
        if not self.is_r_set:
            self.register_parameter('r', nn.Parameter(torch.randn([], dtype=torch.float64)))
        if not self.is_phi_set:
            self.register_parameter('phi', nn.Parameter(torch.rand([], dtype=torch.float64)))





class PhaseShifter(nn.Module):
    """
    Parameters:
        r (tensor): displacement magnitude 
        phi (tesnor): displacement angle 
    """
    def __init__(self, mode=0):
        super().__init__()
        self.mode = mode
        self.is_phi_set  = False
        
        
    def forward(self, state):
        self.auto_params(state)
        if self.phi.shape == torch.Size([]):
            phi = torch.unsqueeze(self.phi, dim=0)
        # part of initialization of parameter
        if not self.is_phi_set:
            phi = 2 * torch.pi * phi
        # phase shifter the mode
        state.displace_one_mode(self.mode, phi)
        return state
    
    def set_params(self, phi=None):
        """set phi to tensor independently"""
        if phi != None:
            self.register_buffer('phi', phi)
            self.is_phi_set = True

    def auto_params(self, state):
        """automatically set None parameter as nn.Paramter for users"""
        if not self.is_phi_set:
            self.register_parameter('phi', nn.Parameter(torch.rand([], dtype=torch.float64)))





###########################
#### two mode gates ####
###########################



# :todo:  change the name of the args
class BeamSplitter(nn.Module):
    """
    Parameters:
        r (tensor): displacement magnitude 
        phi (tesnor): displacement angle 
    """
    def __init__(self, mode=0):
        super().__init__()
        self.mode = mode
        self.is_r_set  = False
        self.is_phi_set  = False
        
        
    def forward(self, state):
        self.auto_params(state)
        if self.r.shape == torch.Size([]):
            r = torch.unsqueeze(self.r, dim=0)
        if self.phi.shape == torch.Size([]):
            phi = torch.unsqueeze(self.phi, dim=0)
        # part of initialization of parameter
        if not self.is_phi_set:
            phi = 2 * torch.pi * phi
        # squeeze the mode
        state.beam_splitter(self.mode, r, phi)
        return state
    
    def set_params(self, r=None, phi=None):
        """set r, phi to tensor independently"""
        if r != None:
            self.register_buffer('r', r)
            self.is_r_set = True
        if phi != None:
            self.register_buffer('phi', phi)
            self.is_phi_set = True

    def auto_params(self, state):
        """automatically set None parameter as nn.Paramter for users"""
        if not self.is_r_set:
            self.register_parameter('r', nn.Parameter(torch.randn([], dtype=torch.float64)))
        if not self.is_phi_set:
            self.register_parameter('phi', nn.Parameter(torch.rand([], dtype=torch.float64)))



class RandomUnitary(nn.Module):
    """
    Generate a Haar random unitary matrix.
    """
    def __init__(self, seed):
        super().__init__()
        self.seed = seed
        

    def forward(self, state):
        # produce the generator
        self.generator = unitary_group(state._mode_number, self.seed)
        self.u = torch.tensor(self.generator.rvs())
        # apply unitary transformation
        state.unitary_transform(self.u)
        return state