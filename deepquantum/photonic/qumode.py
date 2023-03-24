import numbers
import numpy as np
import torch
from torch import nn   
import gaussian.ops
import fock
from scipy.stats import unitary_group







class QumodeCircuit(nn.Module):

    def __init__(self, n_modes, backend='fock'):
        super().__init__()
        self.n_modes = n_modes
        self.backend = backend
        self.operators = nn.ModuleList([])
    

    def __add__(self, rhs):
        # outplace operation
        # https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.compose.html
        cir = QumodeCircuit(self.n_modes, self.backend)
        cir.operators = self.operators + rhs.operators
        return cir


    def add(self, op):
        self.operators.append(op)


    def forward(self, state):
        for op in self.operators:
            state = op(state)
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
            None
        else:
            op = gaussian.ops.Squeeze(mode)
            op.set_params(r, phi)
        
        self.add(op)
    
    def phase_shift(self, phi=None, mode=0):
        if self.backend == 'fock':
            None
        else:
            op = gaussian.ops.PhaseShifter(mode)
            op.set_params(phi)
        
        self.add(op)

    def beam_split(self, r=None, phi=None, mode=0):
        if self.backend == 'fock':
            None
        else:
            op = gaussian.ops.BeamSplitter(mode)
            op.set_params(r, phi)
        
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








class FockState:
    """Instance attributes includes:
    [_n_modes, _cutoff, _batch_size, _hbar, _pure,
    _dtype, state]

    Only here keeps state vector O(M^N)
    """
    
    def __init__(self, batch_size=1, n_modes=1, cutoff=10, hbar=2., pure=True, dtype=torch.complex64):
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
def homodyne_measure(state, mode):
    """
    Make a homodyne measurement of a single mode, return the measurement result and collapse the initial state into 
    final state.
    """
    if isinstance(state, FockState):
        raise ValueError('Fock backend does not support.')
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



def prob_gbs(state, pattern):
    """
    Compute the probability of a measured photon pattern of a special GBS process. 
    Special GBS refers to the process where an initial vaccum gaussian state are squeezed by a series of squeezing gates
    and then pass through a linear optical circuit , which realizes a unitary transformation.
    Reference: "Detailed study of Gaussian boson sampling".
    """
    if isinstance(state, FockState):
        raise ValueError('Fock backend does not support.')
    else:
        
        prob = state.prob_gbs()