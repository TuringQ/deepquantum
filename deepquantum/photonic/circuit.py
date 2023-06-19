import numbers
from itertools import product
from string import ascii_lowercase as indices
import numpy as np
import torch
from torch import nn   
import gaussian
import fock
from scipy.stats import unitary_group
from scipy.special import factorial
from torch.distributions.categorical import Categorical





class QumodeCircuit(nn.Module):

    def __init__(self, batch_size, n_modes, backend='fock'):
        super().__init__()
        self.init_state = FockState(batch_size, n_modes) # :bug: how to move init_state.tesnor to GPU
        self.batch_size = batch_size
        self.n_modes = n_modes
        self.backend = backend
        self.operators = nn.ModuleList([])
        # store the squeezing parameters
        self.squeezing_paras = []
        # store the directly applied random unitary matrix
        self.random_u = []



    def __add__(self, rhs):
        # outplace operation
        # https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.compose.html
        cir = QumodeCircuit(self.batch_size, self.n_modes, self.backend)
        cir.operators = self.operators + rhs.operators
        return cir


    def add(self, op):
        self.operators.append(op)


    def forward(self, state=None):
        if state is None:
            state = self.init_state.reset()
        for op in self.operators:
            state = op(state)
            if isinstance(op, gaussian.ops.RandomUnitary):
                self.random_u.append(op.u)
        return state
    

    def displace(self, r=None, phi=None, mode=0):
        if self.backend == 'fock':
            op = fock.ops.Displacement(mode)
            op.set_params(r, phi)
        else:
            op = gaussian.ops.Displacement(mode)
            op.set_params(r, phi)
        
        self.add(op)

    def squeeze(self, r=None, phi=None, mode=0):
        if self.backend == 'fock':
            op = fock.ops.Squeezing(mode)
            op.set_params(r, phi)
        else:
            op = gaussian.ops.Squeeze(mode)
            op.set_params(r, phi)
            self.squeezing_paras.append([r, phi])
        
        self.add(op)
    
    def phase_shift(self, phi=None, mode=0):
        if self.backend == 'fock':
            op = fock.ops.PhaseShifter(mode)
            op.set_params(phi)
        else:
            op = gaussian.ops.PhaseShifter(mode)
            op.set_params(phi)
        
        self.add(op)

    def beam_split(self, theta=None, phi=None, mode1=0, mode2=1):
        if self.backend == 'fock':
            op = fock.ops.BeamSplitter(mode1, mode2)
            op.set_params(theta, phi)
        else:
            op = gaussian.ops.BeamSplitter([mode1, mode2])
            op.set_params(theta, phi)
        
        self.add(op)
    
    def random_unitary(self, seed=123):
        """
        Generate a Haar random unitary matrix U(N).
        """
        if self.backend == 'fock':
            raise ValueError('Fock backend does not support random unitary transformation.')
        else:
            op = gaussian.ops.RandomUnitary(seed)

        self.add(op)

    def kerr(self, kappa=None, mode=0):
        if self.backend == 'fock':
            op = fock.ops.KerrInteraction(mode)
            op.set_params(kappa)
        else:
            raise ValueError('Gaussian backend does not support kerr transformation.')
        
        self.add(op)








class FockState:
    """Instance attributes includes:
    [_n_modes, _cutoff, _batch_size, _hbar, _pure,
    _dtype, state]

    Only here keeps state vector O(M^N)
    """
    
    def __init__(self, batch_size=1, n_modes=1, cutoff=10, hbar=2., pure=True, dtype=torch.complex128):
        self.set(batch_size, n_modes, cutoff, hbar, pure, dtype)

    def set(self, batch_size, n_modes, cutoff, hbar, pure, dtype):
        r"""
        Sets the state of the qumode circuit to have all modes in vacuum.

        Args:
            
            n_modes (int): sets the number of modes in the circuit.
            cutoff (int): new Fock space cutoff dimension to use.
            batch_size (int): number of circuits in a batch, each circuit takes in different x, they share same weights 
            hbar (float): new :math:`\hbar` value.
            pure (bool): if True, the circuit will represent its state as a pure state. If False, the state will be mixed.
        """
        self._cache = {}

        if not isinstance(batch_size, int):
            raise ValueError("Argument 'batch_size' must be a positive integer")
        self._batch_size = batch_size

        if not isinstance(n_modes, int):
            raise ValueError("Argument 'n_modes' must be a positive integer")
        self._n_modes = n_modes

        if not isinstance(cutoff, int) or cutoff < 1:
            raise ValueError("Argument 'cutoff' must be a positive integer")
        self._cutoff = cutoff

        if not isinstance(hbar, numbers.Real) or hbar <= 0:
            raise ValueError("Argument 'hbar' must be a positive number")
        self._hbar = hbar
 
        if not isinstance(pure, bool):
            raise ValueError("Argument 'pure' must be either True or False")
        self._pure = pure

        if dtype not in (torch.complex64, torch.complex128):
            raise ValueError("Argument 'dtype' must be a complex PyTorch data type")
        self._dtype = dtype
        self._dtype2 = torch.float32 if dtype is torch.complex64 else torch.float64
        
        # init vac state
        self._make_vac_states(self._cutoff)
        single_mode_vac = self._single_mode_pure_vac if pure else self._single_mode_mixed_vac
        if self._n_modes == 1:
            vac = single_mode_vac
        else: 
            vac = fock.ops.combine_single_modes([single_mode_vac] * self._n_modes)
        self.tensor = vac
        


    def _make_vac_states(self, cutoff):
        """Make single mode vacuum state tensors"""
        v = torch.zeros(cutoff, dtype=self._dtype)
        v[0] = 1.+0.j
        self._single_mode_pure_vac = v
        self._single_mode_mixed_vac = torch.einsum('i,j->ij', v, v)
        self._single_mode_pure_vac = torch.stack([self._single_mode_pure_vac] * self._batch_size)
        self._single_mode_mixed_vac = torch.stack([self._single_mode_mixed_vac] * self._batch_size)

    def reset(self):
        self.set(self._batch_size, self._n_modes, self._cutoff,
                 self._hbar, self._pure, self._dtype)
        
        return self
        
    def homodyne_measure(self, phi=0., mode=0, shots=1):
        """
        Homodyne measurement on a single mode.
        Measures `mode` in the basis of quadrature eigenstates (rotated by phi)
        and updates remaining modes conditioned on this result.
        After measurement, the states in `mode` are reset to the vacuum.
        Note: This method does not support gradient.
        Args:
            phi (float): phase angle of quadrature to measure
            mode (int): which mode to measure.
            shots (int): the number of times to measure the state, 
                collapse state conditioned on measurement result only when shots == 1,
                otherwise return state before the measurement.

        Returns:
            tensor: shape (batch_size, shots), returns measurements of a single mode for each different state in a batch
        """
        if not isinstance(mode, int):
            raise ValueError("Specified mode are not valid.")
        if mode < 0 or mode >= self._n_modes:
            raise ValueError("Specified mode are not valid.")

        m_omega_over_hbar = 1 / self._hbar
        if self._pure:
            mode_size = 1
        else:
            mode_size = 2
      
        batch_offset = 1
        batch_size = self._batch_size
        
        phi = torch.tensor(phi, dtype=self._dtype2)
        phi = fock.ops.add_batch_dim(phi, self._batch_size)
      
        # create reduced state on the mode to be measured
        reduced_state_tensor = fock.ops.reduced_density_matrix(self.tensor, mode, self._pure)

        # rotate to homodyne basis
        pf_op = fock.ops.PhaseShifter(mode=0)
        pf_op.set_params(phi=-phi)
        reduced_state = FockState(batch_size=self._batch_size, 
                  n_modes=1, cutoff=self._cutoff, pure=False, 
                  dtype=self._dtype) 
        reduced_state.tensor = reduced_state_tensor
        reduced_state = pf_op(reduced_state)
    

        # create pdf for homodyne measurement
        # We use the following quadrature wavefunction for the Fock states:
        # \psi_n(x) 
        # https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator
        q_mag = 10 
        num_bins = 100000
        if "q_tensor" in self._cache:
            # use cached q_tensor
            q_tensor = self._cache["q_tensor"]
        else:
            q_tensor = torch.tensor(np.linspace(-q_mag, q_mag, num_bins), dtype=self._dtype2)
            self._cache["q_tensor"] = q_tensor
        x = np.sqrt(m_omega_over_hbar) * q_tensor
        if "hermite_polys" in self._cache:
            # use cached polynomials
            hermite_polys = self._cache["hermite_polys"]
        else:
            H0 = 0 * x + 1.0
            H1 = 2 * x
            hermite_polys = [H0, H1]
            Hn = H1
            Hn_m1 = H0
            for n in range(1, self._cutoff - 1):
                Hn_p1 = fock.ops.H_n_plus_1(Hn, Hn_m1, n, x)
                hermite_polys.append(Hn_p1)
                Hn_m1 = Hn
                Hn = Hn_p1
            self._cache["hermite_polys"] = hermite_polys

        number_state_indices = list(product(range(self._cutoff), repeat=2))
    
        terms = [
            1
            / np.sqrt(2**n * factorial(n) * 2**m * factorial(m))
            * hermite_polys[n]
            * hermite_polys[m]
            for n, m in number_state_indices
        ]

        hermite_matrix = torch.stack(terms).reshape([self._cutoff, self._cutoff, num_bins])
        hermite_terms = reduced_state.tensor.unsqueeze(-1) * hermite_matrix.unsqueeze(0)
        
        #rho_dist shape (batch_size, num_bins)
        rho_dist = (
            torch.sum(hermite_terms, dim=[1, 2]).real
            * (m_omega_over_hbar / np.pi) ** 0.5
            * torch.exp(-(x**2))
            * (q_tensor[1] - q_tensor[0])
        )  # Delta_q for normalization (only works if the bins are equally spaced)
        rho_dist[rho_dist<0] = 0 # remove negative values
        
        # use torch.distributions.categorical.Categorical to sample
       
        logits = torch.log(rho_dist)
        categorical_dist = Categorical(logits=logits)
        samples_idx = categorical_dist.sample(sample_shape=[shots])  # shape (shots, batch_size)
        samples_idx = samples_idx.T # shape (batch_size, shots)
        meas_result = q_tensor[samples_idx] # shape (batch_size, shots)

        # collapse state conditioned on measurement result only when shots == 1
        if shots == 1:
            meas_result = meas_result.squeeze() # shape (batch_size,)
            # project remaining modes into conditional state
            if self._n_modes == 1:
                self.reset()
            else:
                # only one mode was measured: put unmeasured modes in conditional state, while reseting measured mode to vac
                # prepares quad eigenstate |x>
                inf_squeezed_vac = torch.tensor(
                    [
                        (-0.5) ** (m // 2) * np.sqrt(factorial(m)) / factorial(m // 2)
                        if m % 2 == 0
                        else 0.0
                        for m in range(self._cutoff)
                    ],
                    dtype=self._dtype,
                )
                inf_squeezed_vac = torch.stack([inf_squeezed_vac] * batch_size)
                displacement_size = meas_result * np.sqrt(m_omega_over_hbar / 2)
                inf_squeezed_vac_state = FockState(batch_size=self._batch_size, 
                                  n_modes=1, cutoff=self._cutoff, pure=True, 
                                  dtype=self._dtype) 
                inf_squeezed_vac_state.tensor = inf_squeezed_vac
                
                dis_op = fock.ops.Displacement(mode=0)
                dis_op.set_params(r=torch.abs(displacement_size), 
                                  phi=torch.angle(displacement_size))
                quad_eigenstate = dis_op(inf_squeezed_vac_state)
                

                pf_op = fock.ops.PhaseShifter(mode=0)
                pf_op.set_params(phi=phi)
                homodyne_eigenstate = pf_op(quad_eigenstate)

                # conditional_state is a tensor
                conditional_state = fock.ops.conditional_state(
                    self.tensor, homodyne_eigenstate.tensor, mode, self._pure
                )

                # normalize
                if self._pure:
                    norm = torch.linalg.vector_norm(torch.reshape(conditional_state, [batch_size, -1]), dim=1)
                else:
                    r = conditional_state
                    for _ in range(self._n_modes - 2):
                        r = fock.ops.partial_trace(r, 0, False)
                    norm = r.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
                    
                # for broadcasting
                norm_reshape = [1] * len(conditional_state.shape[batch_offset:])
                norm_reshape = [self._batch_size] + norm_reshape
                normalized_conditional_state = conditional_state / torch.reshape(norm, norm_reshape)

                # reset measured modes into vacuum
                meas_mode_vac = (
                    self._single_mode_pure_vac if self._pure else self._single_mode_mixed_vac
                )
                batch_index = indices[:batch_offset]
                meas_mode_indices = indices[batch_offset : batch_offset + mode_size]
                conditional_indices = indices[
                    batch_offset + mode_size : batch_offset + mode_size * self._n_modes
                ]
                eqn_lhs = batch_index + meas_mode_indices + "," + batch_index + conditional_indices
                eqn_rhs = ""
                meas_ctr = 0
                cond_ctr = 0
                for m in range(self._n_modes):
                    if m == mode:
                        # use measured_indices
                        eqn_rhs += meas_mode_indices[mode_size * meas_ctr : mode_size * (meas_ctr + 1)]
                        meas_ctr += 1
                    
                    else:
                        # use conditional indices
                        eqn_rhs += conditional_indices[
                            mode_size * cond_ctr : mode_size * (cond_ctr + 1)
                        ]
                        cond_ctr += 1
                eqn = eqn_lhs + "->" + batch_index + eqn_rhs
                new_state = torch.einsum(eqn, meas_mode_vac, normalized_conditional_state)

                self.tensor = new_state

        return meas_result
    

    def dm(self):
        r"""
        Return state's density matrix, optionally converts the state from pure state (ket) to mixed state (density matrix). 
        """
        if self._pure:
            self.tensor = fock.ops.mix(self.tensor)
        return self.tensor

    def reduced_dm(self, modes):
        r"""
        Computes the reduced density matrix for modes
        Args:
            modes (int or Sequence[int]): specifies the mode(s) to return the reduced
                                density matrix for.
        Returns:
            Tensor: the reduced density matrix for modes
        """
        if isinstance(modes, int):
            modes = [modes]
        if modes == list(range(self._n_modes)):
            # reduced state is full state
            return self.dm()
        if len(modes) > self._n_modes:
            raise ValueError(
                "The number of specified modes cannot " "be larger than the number of subsystems."
            )
        reduced = fock.ops.reduced_density_matrix(self.tensor, modes, self._pure)
        return reduced


    def quad_expectation(self, phi=0., mode=0):
        """Compute the expectation value of the quadrature operator :math:`\hat{x}_\phi` in single mode
        
        Args:
            phi (float): rotation angle for the quadrature operator
            mode (int): which single mode to take the expectation value of

        Returns:
            Tensor: the expectation value
        """
        rho = self.reduced_dm(mode) 

        if len(phi.shape) == 0:  
            phi = torch.unsqueeze(phi, 0)
        larger_cutoff = self._cutoff + 1  # start one dimension higher to avoid truncation errors
        R = fock.ops.phase_shifter_matrix(phi, larger_cutoff, self._dtype, self._batch_size)

        a, ad = fock.ops.ladder_ops(larger_cutoff)
        x = np.sqrt(self._hbar / 2.0) * (a + ad)
        x = x.to(self._dtype)
        x = torch.unsqueeze(x, 0)  # add batch dimension to x
        quad = torch.conj(R) @ x @ R
        quad2 = (quad @ quad)[:, : self._cutoff, : self._cutoff]
        quad = quad[:, : self._cutoff, : self._cutoff]  # drop highest dimension

        flat_rho = torch.reshape(rho, [-1, self._cutoff ** 2])
        flat_quad = torch.reshape(quad, [1, self._cutoff ** 2])
        flat_quad2 = torch.reshape(quad2, [1, self._cutoff ** 2])

        e = torch.real(
            torch.sum(flat_rho * flat_quad, dim=1)
        )  # implements a batched tr(rho @ x) x.T = x
        e2 = torch.real(
            torch.sum(flat_rho * flat_quad2, dim=1)
        )  # implements a batched tr(rho @ x ** 2)
        v = e2 - e ** 2

        return e, v






class GaussianState(gaussian.ops.Gaussian):
    """
    Class of Gaussian state. Gaussian state is a special kind of quantum states whose Wigner functions are gaussian functions, 
    which is completely determined by a covariance matrix and a displacement vector. 
    
    In our convention, the covariance matrix and displacement vector are defined for creation and annihilation operators.
    """
    def __init__(self, batch_size=1, n_modes=1, h_bar=1/np.sqrt(2), dtype=torch.complex128):
        """
        Initialize a gaussian system in the vaccum state with a specfic number of optical modes.
        """
        super().__init__(n_modes, batch_size, h_bar, dtype)
        # initialize a vaccum state  
        #self.reset(n_modes, batch_size, h_bar, dtype)
        #print(f'Initialize a gaussian vaccum state with {n_modes} modes and batch size being {batch_size}.')
    





#################
#################


# homodyne measurement for one mode
def homodyne_measure(state, phi=0., mode=0, shots=1):
    """
    Make a homodyne measurement of a single mode.
    Args:
        phi (float): phase angle of quadrature to measure, gaussian backend only supports phi=0 and phi=90
        mode (int): which mode to measure.
        shots (int): the number of times to measure the state, gaussian backend only supports shots=1. 
            collapse state conditioned on measurement result only when shots == 1, otherwise return state before the measurement.

    Returns:
        res (tensor): shape (batch_size, shots), returns measurements of a single mode for each different state in a batch,
            gaussian backend returns tensor with shape (batch_size, 2), e.g. [[x, p], ...]
        state (FockState or GaussianState): collapsed state after the measurement. If it's FockState, measured mode will be set to vaccum.
    """
    if isinstance(state, FockState):
        res = state.homodyne_measure(phi=phi, mode=0, shots=1) 
    else:
        # measurement results
        res = state.homodyne_one_mode(mode)
    return res, state


# heterodyne measurement for one mode
def heterodyne_measure(state, mode):
    if isinstance(state, FockState):
        raise ValueError('Fock backend does not support.')
    else:
        # measurement results
        res = state.heterodyne_one_mode(mode)
        return res, state


def prob_gbs(state, cir, pattern):
    """
    Compute the probability of a measured photon pattern of a special GBS process. 
    Special GBS refers to the process where an initial vaccum gaussian state are squeezed by a series of squeezing gates
    and then pass through a linear optical circuit , which realizes a unitary transformation.
    Reference: "Detailed study of Gaussian boson sampling".
    """
    if isinstance(state, FockState):
        raise ValueError('Fock backend does not support.')
    else:
        paras = torch.tensor(cir.squeezing_paras)[:, 0]
        prob = state.prob_gbs(paras, cir.random_u, pattern)
        return prob
    

def mean_photon_number(state, mode):
    """
    Compute the mean photon number of a mode.
    Reference: "Training Gaussian boson sampling by quantum machine learning".
    """
    if isinstance(state, FockState):
        raise ValueError('Fock backend does not support.')
    else:
        return state.mean_photon_number(mode)
    

def diff_photon_number(state, mode1, mode2):
    """
    Calculate the expectation value of the square of the difference of two mode's photon number operator.
    Ref: Training Gaussian boson sampling by quantum machine learning.
    """
    if isinstance(state, FockState):
        raise ValueError('Fock backend does not support.')
    else:
        return state.diff_photon_number(mode1, mode2)
    