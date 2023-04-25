# https://arxiv.org/pdf/2004.11002.pdf
# Our code is based on this reference with slight modifications.



from string import ascii_lowercase as indices
max_num_indices = len(indices)

import numpy as np
import torch
from torch import nn
from scipy.special import binom, factorial



def conditional_state(system, projector, mode, state_is_pure):
    """Compute the (unnormalized) conditional state of 'system' after applying ket 'projector' to 'mode'."""
    # basic_form (pure states): abc...ijk...xyz,j-> abc...ik...xyz
    # basic_form (mixed states): abcd...ijklmn...wxyz,k,l-> abcd...ijmn...wxyz
    num_indices = system.ndim

    batch_offset = 1
    if state_is_pure:
        mode_size = 1
    else:
        mode_size = 2
    num_modes = (num_indices - batch_offset) // mode_size
    # max_num = (max_num_indices - batch_offset) // num_modes
    # if num_modes > max_num:
    #     raise ValueError(
    #         "Conditional state projection currently only supported for {} modes.".format(max_num)
    #     )
    # get abcstract indices
    batch_index = indices[:batch_offset]
    mode_indices = indices[batch_offset : batch_offset + num_modes * mode_size]
    projector_indices = mode_indices[:mode_size]
    free_mode_indices = mode_indices[mode_size : num_modes * mode_size]
    
    state_lhs = (
        batch_index
        + free_mode_indices[: mode * mode_size]
        + projector_indices
        + free_mode_indices[mode * mode_size :]
    )
    projector_lhs = batch_index + projector_indices[0]
    if mode_size == 2:
        projector_lhs += "," + batch_index + projector_indices[1]
    eqn_lhs = ",".join([state_lhs, projector_lhs])
    eqn_rhs = batch_index + free_mode_indices
    eqn = eqn_lhs + "->" + eqn_rhs
    einsum_args = [system, torch.conj(projector)]
    if not state_is_pure:
        einsum_args.append(projector)
    cond_state = torch.einsum(eqn, *einsum_args)
    
    return cond_state




def H_n_plus_1(H_n, H_n_m1, n, x):
    """Recurrent definition of Hermite polynomials."""
    H_n_p1 = 2 * x * H_n - 2 * n * H_n_m1
    return H_n_p1




def add_batch_dim(tensor, batch_size=1):
    """Adds a batch dimension to the tensor if it does not already have one"""
    if tensor.ndim == 0:
        tensor = torch.stack([tensor] * batch_size)
    return tensor



def mix(pure_state):
    """Converts the state from pure state (ket) to mixed state (density matrix)"""
    
    batch_offset = 1
    num_modes = pure_state.ndim - batch_offset
    max_num = (max_num_indices - batch_offset) // 2

    if num_modes > max_num:
        raise ValueError(
            "Converting state from pure to mixed currently only supported for {} modes.".format(
                max_num
            )
        )
    
    # eqn: 'abc...xyz,ABC...XYZ->aAbBcC...xXyYzZ' (lowercase belonging to 'ket' side, uppercase belonging to 'bra' side)
    batch_index = indices[:batch_offset]  
    bra_indices = indices[batch_offset : batch_offset + num_modes]
    ket_indices = indices[batch_offset + num_modes : batch_offset + 2 * num_modes]
    eqn_lhs = batch_index + bra_indices + "," + batch_index + ket_indices
    eqn_rhs = "".join(bdx + kdx for bdx, kdx in zip(bra_indices, ket_indices))
    eqn = eqn_lhs + "->" + batch_index + eqn_rhs
    mixed_state = torch.einsum(eqn, pure_state, torch.conj(pure_state))

    return mixed_state


def partial_trace(state, mode, state_is_pure):
    """
    Trace out subsystem 'mode' from 'state'.
    This operation always returns a mixed state, since we do not know in advance if a mode is entangled with others.
    """
    

    batch_offset = 1

    if state_is_pure:
        system = mix(state)
    else:
        system = state

    num_indices = len(system.shape)
    num_modes = (num_indices - batch_offset) // 2

    assert mode < num_modes, f"mode index must be [0, 1, ..., num_modes-1], you have mode={mode}, num_modes={num_modes}"
    # batch trace implementation
    # requires subsystem to be traced out to be at end
    # i.e., ab...klmnop...yz goes to ab...klop...yzmn
    indices_list = list(range(batch_offset + 2 * num_modes))


    permuted_indices_list = (
        indices_list[: batch_offset + 2 * mode]  # for density matrix, mode=0,1,2,  tensor index=1,3,5
        + indices_list[batch_offset + 2 * (mode + 1) :]
        + indices_list[batch_offset + 2 * mode : batch_offset + 2 * (mode + 1)]
    )


    permuted_sys = torch.permute(system, permuted_indices_list)
    reduced_state = permuted_sys.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
    return reduced_state



def reduced_density_matrix(state, modes, state_is_pure):
    """
    Trace out all subsystems except those specified in ``modes`` from ``state``. ``modes`` can be either an int or a list.
    This operation always returns a mixed state, since we do not know in advance if a mode is entangled with others.
    """
    if isinstance(modes, int):
        modes = [modes]

    if state_is_pure:
        reduced_state = mix(state)
    else:
        reduced_state = state
        
    num_indices = len(reduced_state.shape)
    batch_offset = 1
    num_modes = (num_indices - batch_offset) // 2  # always mixed
    removed_cnt = 0
    for m in range(num_modes):
        if m not in modes: # keep information in modes
            reduced_state = partial_trace(reduced_state, m-removed_cnt, False)
            removed_cnt += 1
    return reduced_state



def combine_single_modes(modes_list):
    """Group together a list of single modes (each having ndim=1 or ndim=2) into a composite mode system."""
    batch_offset = 1
    num_modes = len(modes_list)
    
    if num_modes <= 1:
        raise ValueError("'modes_list' must have at least two modes")

    ndims = np.array([mode.ndim - batch_offset for mode in modes_list])
    if min(ndims) < 1 or max(ndims) > 2:
        raise ValueError("Each mode in 'modes_list' can only have ndim=1 or ndim=2")

    if np.all(ndims == 1):
        # All modes are represented as pure states.
        # Can return combined state also as pure state.
        # basic form (no batch):
        # 'a,b,c,...,x,y,z->abc...xyz' 
        max_num = max_num_indices - batch_offset
        if num_modes > max_num:
            raise NotImplementedError("The max number of supported modes for this operation with pure states is currently {}".format(max_num))
        batch_index = indices[:batch_offset]                         # 'a'
        out_str = indices[batch_offset : batch_offset + num_modes]   # 'bcde'
        modes_str = ",".join([batch_index + idx for idx in out_str]) # 'ab,ac,ad,ae'
        eqn = "{}->{}".format(modes_str, batch_index + out_str)      # 'ab,ac,ad,ae->abcde'
        einsum_inputs = modes_list
    else:
        # Return combined state as mixed states.
        # basic form:
        # e.g., if first mode is pure and second is mixed...
        # 'ab,cd,...->abcd...'
        # where ab will belong to the first mode (density matrix)
        # and cd will belong to the second mode (density matrix)
        max_num = (max_num_indices - batch_offset) // 2
        if num_modes > max_num:
            raise NotImplementedError("The max number of supported modes for this operation with mixed states is currently {}".format(max_num))
        batch_index = indices[:batch_offset]
        mode_idxs = [indices[slice(batch_offset + idx, batch_offset + idx + 2)] for idx in range(0, 2 * num_modes, 2)] # each mode gets a pair of consecutive indices
        # mode_idxs = ['bc', 'de']
        eqn_rhs = batch_index + "".join(mode_idxs) # 'abcde'
        eqn_idxs = [batch_index + m for m in mode_idxs] # ['abc', 'ade'] 
        eqn_lhs = ",".join(eqn_idxs) #  'abc, ade'
        eqn = eqn_lhs + "->" + eqn_rhs
        einsum_inputs = modes_list
    
    combined_modes = torch.einsum(eqn, *einsum_inputs)
    return combined_modes



def single_mode_gate(matrix, mode, in_modes, pure=True):
    """Basic form:
    'ab,cde...b...xyz->cde...a...xyz' (pure state)      U|Ψ>
    'ab,ef...bc...xyz,cd->ef...ad...xyz' (mixed state)  UρU†
    """

    batch_offset = 1
 
    batch_index = indices[:batch_offset]
    left_gate_str = indices[batch_offset : batch_offset + 2] # |a><b|
    num_indices = len(in_modes.shape)
    if pure:
        num_modes = num_indices - batch_offset
        mode_size = 1
    else:
        right_gate_str = indices[batch_offset + 2 : batch_offset + 4] # |c><d|
        num_modes = (num_indices - batch_offset) // 2
        mode_size = 2
    max_len = len(indices) - 2 * mode_size - batch_offset #26 letters 2*mode_size reserved for gates, 1 for batch
    if num_modes == 0:
        raise ValueError("'in_modes' must have at least one mode")
    if num_modes > max_len:
        raise NotImplementedError("The max number of supported modes for this operation is currently {}".format(max_len))
    if mode < 0 or mode >= num_modes:
        raise ValueError("'mode' argument is not compatible with number of in_modes")
    else:
        other_modes_indices = indices[batch_offset + 2 * mode_size : batch_offset + (1 + num_modes) * mode_size]
        if pure:
            eqn_lhs = "{},{}{}{}{}".format(batch_index + left_gate_str, batch_index, other_modes_indices[:mode * mode_size], left_gate_str[1], other_modes_indices[mode * mode_size:])
            eqn_rhs = "".join([batch_index, other_modes_indices[:mode * mode_size], left_gate_str[0], other_modes_indices[mode * mode_size:]])
        else:
            eqn_lhs = "{},{}{}{}{}{},{}".format(batch_index + left_gate_str, batch_index, other_modes_indices[:mode * mode_size], left_gate_str[1], right_gate_str[0], other_modes_indices[mode * mode_size:], batch_index + right_gate_str)
            eqn_rhs = "".join([batch_index, other_modes_indices[:mode * mode_size], left_gate_str[0], right_gate_str[1], other_modes_indices[mode * mode_size:]])

    eqn = eqn_lhs + "->" + eqn_rhs
    
    einsum_inputs = [matrix, in_modes]
    if not pure:
        transposed_axis = [0, 2, 1]
        einsum_inputs.append(torch.permute(torch.conj(matrix), transposed_axis))
    #print('debug', eqn)
    output = torch.einsum(eqn, *einsum_inputs)
    return output

def two_mode_gate(matrix, mode1, mode2, in_modes, pure=True):
    """Basic form:
    'abcd,efg...b...d...xyz->efg...a...c...xyz' (pure state)
    'abcd,ij...be...dg...xyz,efgh->ij...af...ch...xyz' (mixed state)
    """
    # pylint: disable=too-many-branches,too-many-statements
    
    batch_offset = 1
    batch_index = indices[:batch_offset]
    left_gate_str = indices[batch_offset : batch_offset + 4] # |a><b| |c><d|
    num_indices = len(in_modes.shape)
    if pure:
        num_modes = num_indices - batch_offset
        mode_size = 1
    else:
        right_gate_str = indices[batch_offset + 4 : batch_offset + 8] # |e><f| |g><h|
        num_modes = (num_indices - batch_offset) // 2
        mode_size = 2
    max_len = (len(indices) - 4) // mode_size - batch_offset

    if num_modes == 0:
        raise ValueError("'in_modes' must have at least one mode")
    if num_modes > max_len:
        raise NotImplementedError("The max number of supported modes for this operation is currently {}".format(max_len))
    else:
        min_mode = min(mode1, mode2)
        max_mode = max(mode1, mode2)
        if min_mode < 0 or max_mode >= num_modes or mode1 == mode2:
            raise ValueError("One or more mode numbers are incompatible")
        else:
            other_modes_indices = indices[batch_offset + 4 * mode_size : batch_offset + 4 * mode_size + mode_size * (num_modes - 2)]
            # build equation
            if mode1 == min_mode:
                lhs_min_mode_indices = left_gate_str[1]
                lhs_max_mode_indices = left_gate_str[3]
                rhs_min_mode_indices = left_gate_str[0]
                rhs_max_mode_indices = left_gate_str[2]
            else:
                lhs_min_mode_indices = left_gate_str[3]
                lhs_max_mode_indices = left_gate_str[1]
                rhs_min_mode_indices = left_gate_str[2]
                rhs_max_mode_indices = left_gate_str[0]
            if not pure:
                if mode1 == min_mode:
                    lhs_min_mode_indices += right_gate_str[0]
                    lhs_max_mode_indices += right_gate_str[2]
                    rhs_min_mode_indices += right_gate_str[1]
                    rhs_max_mode_indices += right_gate_str[3]
                else:
                    lhs_min_mode_indices += right_gate_str[2]
                    lhs_max_mode_indices += right_gate_str[0]
                    rhs_min_mode_indices += right_gate_str[3]
                    rhs_max_mode_indices += right_gate_str[1]
            eqn_lhs = "{},{}{}{}{}{}{}".format(batch_index + left_gate_str,
                                               batch_index,
                                               other_modes_indices[:min_mode * mode_size],
                                               lhs_min_mode_indices,
                                               other_modes_indices[min_mode * mode_size : (max_mode - 1) * mode_size],
                                               lhs_max_mode_indices,
                                               other_modes_indices[(max_mode - 1) * mode_size:])
            if not pure:
                eqn_lhs += "," + batch_index + right_gate_str
            eqn_rhs = "".join([batch_index,
                               other_modes_indices[:min_mode * mode_size],
                               rhs_min_mode_indices,
                               other_modes_indices[min_mode * mode_size: (max_mode - 1) * mode_size],
                               rhs_max_mode_indices,
                               other_modes_indices[(max_mode - 1) * mode_size:]
                              ])
            eqn = eqn_lhs + "->" + eqn_rhs
            einsum_inputs = [matrix, in_modes]
            if not pure:
                transpose_list = (0, 2, 1, 4, 3)
                # else:
                #     transpose_list = (0, 2, 1, 4, 3)
                einsum_inputs.append(torch.permute(torch.conj(matrix), transpose_list)) # to do , 有疑问
            #print('debug', eqn)
            output = torch.einsum(eqn, *einsum_inputs)
            return output

#---------------------------------------------------------------------------------------------------------------------------------------


def displacement_matrix(r, phi, cutoff, dtype, batch_size):  # pragma: no cover
    r"""Calculates the matrix elements of the displacement gate using a recurrence relation.
    
    Args:
        r (torch.Tensor): batched displacement magnitude shape = (batch_size,)
        phi (torch.Tensor): batched displacement angle shape = (batch_size,)
        cutoff (int): Fock ladder cutoff
        dtype (data type): Specifies the data type used for the calculation
    
    Returns:
        torch.Tensor: matrix representing the displacement operation.
    """

    # broadcast a scalar to a vector to support batch
    r = add_batch_dim(r, batch_size)
    phi = add_batch_dim(phi, batch_size)

    r = r.to(dtype)
    phi = phi.to(dtype)
    batch_size = r.shape[0]


    D = torch.zeros((batch_size, cutoff, cutoff)).to(dtype)
    sqrt = torch.sqrt(torch.arange(cutoff)).to(dtype)
    
    alpha0 = r * torch.exp(1j * phi)
    alpha1 = -r * torch.exp(-1j * phi)
    

    D[:, 0, 0] = torch.exp(-0.5 * r**2)
    
    for m in range(1, cutoff):

        D[:, m, 0] = alpha0 / sqrt[m] * D[:, m - 1, 0].clone()
 

    for n in range(1, cutoff):
        D[:, 0, n] = alpha1 / sqrt[n] * D[:, 0, n - 1].clone()


    for m in range(1, cutoff):
        for n in range(1, cutoff):
            D[:, m, n] = alpha1 / sqrt[n] * D[:, m, n - 1].clone() + sqrt[m] / sqrt[n] * D[:, m - 1, n - 1].clone()
    
    return D


def squeezing_matrix(r, phi, cutoff, dtype, batch_size):  # pragma: no cover
    r"""Calculates the matrix elements of the squeezing gate using a recurrence relation.

    Args:
        r (torch.Tensor): batched squeezing magnitude shape = (batch_size,)
        phi (torch.Tensor): batched squeezing angle shape = (batch_size,)
        cutoff (int): Fock ladder cutoff
        dtype (data type): Specifies the data type used for the calculation
    
    Returns:
        torch.Tensor: matrix representing the squeezing operation.
    """
    r = add_batch_dim(r, batch_size)
    phi = add_batch_dim(phi, batch_size)

    r = r.to(dtype)
    phi = phi.to(dtype)
    batch_size = r.shape[0]
    

    S = torch.zeros((batch_size, cutoff, cutoff)).to(dtype)
    sqrt = torch.sqrt(torch.arange(cutoff)).to(dtype)

    

    eiphi_tanhr = torch.exp(1j * phi) * torch.tanh(r)
    sechr = 1.0 / torch.cosh(r)

    R_00 = -eiphi_tanhr
    R_01 = sechr
    R_10 = sechr
    R_11 = torch.conj(eiphi_tanhr)

    S[:, 0, 0] = torch.sqrt(sechr)
    for m in range(2, cutoff, 2):
        S[:, m, 0] = sqrt[m - 1] / sqrt[m] * R_00 * S[:, m - 2, 0].clone()

    for n in range(2, cutoff, 2):
        S[:, 0, n] = sqrt[n - 1] / sqrt[n] * R_11 * S[:, 0, n-2].clone() 

    for m in range(1, cutoff):
        for n in range(1, cutoff):
            if (m + n) % 2 == 0:
                S[:, m, n] = (
                    sqrt[n - 1] / sqrt[n] * R_11 * S[:, m, n - 2].clone()
                    + sqrt[m] / sqrt[n] * R_01 * S[:, m - 1, n - 1].clone()  
                )                                                    
    return S                                                        


def beamsplitter_matrix(theta, phi, cutoff, dtype):  # pragma: no cover
    r"""Calculates the matrix elements of the beamsplitter gate using a recurrence relation.
    
    Args:
        theta (torch.Tensor): transmissivity angle of the beamsplitter. The transmissivity is :math:`t=\cos(\theta)`
        phi (torch.Tensor): reflection phase of the beamsplitter
        cutoff (int): Fock ladder cutoff
        dtype (data type): Specifies the data type used for the calculation
    
    Returns:
        torch.Tensor: matrix representing the beamsplitter operation.
    """
    
    theta = theta.to(dtype)
    phi = phi.to(dtype)
    batch_size = theta.shape[0]

    sqrt = torch.sqrt(torch.arange(cutoff)).to(dtype)
    ct = torch.cos(theta)
    st = torch.sin(theta) * torch.exp(1j * phi)
    
    R_02 = ct
    R_12 = st
    R_03 = -torch.conj(st)
    R_13 = ct

    Z = torch.zeros((batch_size, cutoff, cutoff, cutoff, cutoff)).to(dtype)
    Z[:, 0, 0, 0, 0] = 1.0

    # rank 3
    for m in range(cutoff):
        for n in range(cutoff - m):
            p = m + n
            if 0 < p < cutoff:
                Z[:, m, n, p, 0] = (
                    R_02 * sqrt[m] / sqrt[p] * Z[:, m - 1, n, p - 1, 0].clone()
                    + R_12 * sqrt[n] / sqrt[p] * Z[:, m, n - 1, p - 1, 0].clone()
                )
           

    # rank 4
    for m in range(cutoff):
        for n in range(cutoff):
            a = m + n - (cutoff-1)
            if a <= 0:  
                for p in range(0, m+n):
                    q = m + n - p
                    Z[:, m, n, p, q] = (
                        R_03 * sqrt[m] / sqrt[q] * Z[:, m - 1, n, p, q - 1].clone()
                        + R_13 * sqrt[n] / sqrt[q] * Z[:, m, n - 1, p, q - 1].clone()
                    )
                   

            if a > 0:
                for p in range(m+n-cutoff,  cutoff):
                    q = m + n - p
                    if 0 < q < cutoff:
                        Z[:, m, n, p, q] = (
                            R_03 * sqrt[m] / sqrt[q] * Z[:, m - 1, n, p, q - 1].clone()
                            + R_13 * sqrt[n] / sqrt[q] * Z[:, m, n - 1, p, q - 1].clone()
                        )
                   
            
    
    # Z = np.transpose(Z, [0, 1, 3, 2, 4]) 
    Z = torch.permute(Z, (0, 1, 3, 2, 4)) 
    return Z


def phase_shifter_matrix(phi, cutoff, dtype, batch_size):
    """Creates the single mode phase shifter matrix
    
    Args:
        phi (torch.Tensor): batched angle shape = (batch_size,)
    """
    phi = add_batch_dim(phi, batch_size)
    phi = phi.to(dtype)
   
    diag = [torch.exp(1j * phi * n) for n in range(cutoff)]
    diag = torch.stack(diag, dim=1)
    diag_matrix = torch.diag_embed(diag)
    return diag_matrix


def kerr_interaction_matrix(kappa, cutoff, dtype):
    """Creates the single mode Kerr interaction matrix
    
    Args:
        kappa (torch.Tensor): batched angle shape = (batch_size,)
    """
    kappa = kappa.to(dtype)

    diag = [torch.exp(1j * kappa * n ** 2) for n in range(cutoff)]
    diag = torch.stack(diag, dim=1)
    diag_matrix = torch.diag_embed(diag)
    return diag_matrix



def snap_maxtrix(theta, cutoff, dtype):
    """https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.115.137002

    Constructs the matrix for a SNAP gate operation
    that can be applied to a state.
    
    Args:
        cutoff (int): Hilbert space cuttoff
        theta (torch.Tensor): A vector of theta values to 
                apply SNAP operation, shape = (batch_size, cutoff)
    
    Returns:
        torch.Tensor: matrix representing the SNAP gate, shape = (batch_size, cutoff, cutoff)
    """
    theta = theta.to(dtype)

    diag = [torch.exp(1j * theta[:, n]) for n in range(cutoff)]
    diag = torch.stack(diag, dim=1)
    diag_matrix = torch.diag_embed(diag)
    return diag_matrix




#---------------------------------------------------------------------------------------------------------------------------------------deprecated



# deprecated

def displacement(r, phi, mode, in_modes, cutoff, pure=True, dtype=torch.complex64):
    """returns displacement unitary matrix applied on specified input modes"""
    matrix = displacement_matrix(r, phi, cutoff, dtype)
    output = single_mode_gate(matrix, mode, in_modes, pure)

    return output

def squeezing(r, theta, mode, in_modes, cutoff, pure=True, dtype=torch.complex64):
    """returns squeezing unitary matrix applied on specified input modes"""
    matrix = squeezing_matrix(r, theta, cutoff, dtype)
    output = single_mode_gate(matrix, mode, in_modes, pure)

    return output

def beamsplitter(theta, phi, mode1, mode2, in_modes, cutoff, pure=True, dtype=torch.complex64):
    """returns beamsplitter unitary matrix applied on specified input modes"""
    matrix = beamsplitter_matrix(theta, phi, cutoff, dtype)
    output = two_mode_gate(matrix, mode1, mode2, in_modes, pure)

    return output


def phase_shifter(phi, mode, in_modes, cutoff, pure=True, dtype=torch.complex64):
    """returns phase shift unitary matrix applied on specified input modes"""
    matrix = phase_shifter_matrix(phi, cutoff, dtype=dtype)
    output = single_mode_gate(matrix, mode, in_modes, pure)
    return output


def kerr_interaction(kappa, mode, in_modes, cutoff, pure=True, dtype=torch.complex64):
    """returns Kerr unitary matrix applied on specified input modes"""
    matrix = kerr_interaction_matrix(kappa, cutoff, dtype=dtype)
    output = single_mode_gate(matrix, mode, in_modes, pure)
    return output



def snap(theta, mode, in_modes, cutoff, pure=True, dtype=torch.complex64):
    """returns SNAP unitary matrix applied on specified input modes"""
    matrix = snap_maxtrix(theta, cutoff, dtype=dtype)
    output = single_mode_gate(matrix, mode, in_modes, pure)
    return output
    

def coherent_state(r, phi, cutoff, pure=True, dtype=torch.complex64):
    """creates a single mode coherent state"""
    r = r.to(dtype)
    phi = phi.to(dtype)
    alpha = r * torch.exp(1j * phi)
    coh = torch.stack(
        [
            torch.exp(-0.5 * torch.abs(r) ** 2)
            * alpha ** n
            / np.sqrt(factorial(n))
            for n in range(cutoff)
        ],
        dim=-1,
    )
    if not pure:
        coh = mix(coh)
    return coh



def ladder_ops(cutoff):
    """returns the matrix representation of the annihilation and creation operators"""
    vals = torch.tensor([np.sqrt(n) for n in range(1, cutoff)])
    a = torch.diag(vals, 1)
    ad = torch.transpose(torch.conj(a), 1, 0)
    return a, ad





#------------------------------------------------------------------------------------


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
        # state in, state out, this is in-ploace operation
        self.auto_params(dtype=state._dtype2)
        # tensor contraction
        self.matirx = displacement_matrix(self.r, self.phi, state._cutoff, state._dtype, state._batch_size)
        state.tensor = single_mode_gate(self.matirx, self.mode, state.tensor, state._pure)
        return state
    
    def set_params(self, r=None, phi=None):
        """set r, phi to tensor independently"""
        if r != None:
            self.register_buffer('r', r)
            self.is_r_set = True
        if phi != None:
            self.register_buffer('phi', phi)
            self.is_phi_set = True

    def auto_params(self, dtype):
        """automatically set None parameter as nn.Paramter for users"""
        if not self.is_r_set:
            self.register_parameter('r', nn.Parameter(torch.randn([], dtype=dtype)))
        if not self.is_phi_set:
            self.register_parameter('phi', nn.Parameter(torch.randn([], dtype=dtype)))


class Squeezing(nn.Module):
    """
    Parameters:
        r (tensor): squeezing magnitude 
        theta (tesnor): squeezing angle 
    """
    def __init__(self, mode=0):
        super().__init__()
        self.mode = mode
        self.is_r_set  = False
        self.is_theta_set  = False
        
    def forward(self, state):
        # state in, state out, this is in-ploace operation
        self.auto_params(dtype=state._dtype2)
        # tensor contraction
        self.matirx = squeezing_matrix(self.r, self.theta, state._cutoff, state._dtype, state._batch_size)
        state.tensor = single_mode_gate(self.matirx, self.mode, state.tensor, state._pure)
        return state
    
    def set_params(self, r=None, theta=None):
        """set r, theta to tensor independently"""
        if r != None:
            self.register_buffer('r', r)
            self.is_r_set = True
        if theta != None:
            self.register_buffer('theta', theta)
            self.is_theta_set = True

    def auto_params(self, dtype):
        """automatically set None parameter as nn.Paramter for users"""
        if not self.is_r_set:
            self.register_parameter('r', nn.Parameter(torch.randn([], dtype=dtype)))
        if not self.is_theta_set:
            self.register_parameter('theta', nn.Parameter(torch.randn([], dtype=dtype)))

class PhaseShifter(nn.Module):
    """
    Parameters:
        phi (tesnor): phase shift angle 
    """
    def __init__(self, mode=0):
        super().__init__()
        self.mode = mode
        self.is_phi_set  = False
        
    def forward(self, state):
        # state in, state out, this is in-ploace operation
        self.auto_params(dtype=state._dtype2)
        # tensor contraction
        self.matirx = phase_shifter_matrix(self.phi, state._cutoff, state._dtype, state._batch_size)
        state.tensor = single_mode_gate(self.matirx, self.mode, state.tensor, state._pure)
        return state
    
    def set_params(self, phi=None):
        """set phi to tensor independently"""
        if phi != None:
            self.register_buffer('phi', phi)
            self.is_phi_set = True

    def auto_params(self, dtype):
        """automatically set None parameter as nn.Paramter for users"""
        if not self.is_phi_set:
            self.register_parameter('phi', nn.Parameter(torch.randn([], dtype=dtype)))


