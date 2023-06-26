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
from state import FockState, GaussianState




class QumodeCircuit(nn.Module):

    def __init__(self, batch_size, n_modes, backend='fock', cutoff=10, dtype=torch.complex64):
        super().__init__()
        if backend == 'fock':
            self.init_state = FockState(batch_size, n_modes, dtype=dtype) 
        else:
            self.init_state = GaussianState(batch_size, n_modes)
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
        cir = QumodeCircuit(self.batch_size, self.n_modes, self.backend).to(self.init_state.tensor.device)
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






#---------------------------------------------------------------------------------------------------------------------------------deprecated measure and expectation should be defined inside states

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
    