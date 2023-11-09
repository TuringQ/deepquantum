"""
Optical quantum gates
"""

from copy import copy
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn
from photonic_qmath import  FockGateTensor

from operation import Gate
# from .qmath import multi_kron, is_unitary, svd


class PhaseShift(Gate): 
    """
    The phaseshifter in the optical quantum gates
    """  
    def __init__(
        self,
        inputs: Any = None,
        nmode: int = None,
        wires: Union[int, List[int], None] = None,
        cutoff: int = None,
        requires_grad: bool = False
    ) -> None:
        super().__init__(name='PhaseShift', nmode=nmode, wires=wires, cutoff=cutoff)

        assert len(wires) == 1, "PS gate acts on single mode"
        self.npara = 1
        self.requires_grad = requires_grad
        self.inv_mode = False
        self.init_para(inputs=inputs)

    



    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Get the local unitary matrix. The matrix here represents the matrix for linear optical elements
         which acts on the creation operator a^dagger"""
        theta = self.inputs_to_tensor(theta)
        return torch.exp(1j * theta)
    

    def update_matrix(self) -> torch.Tensor:
        """Update the local unitary matrix."""
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
        matrix = self.get_matrix(theta)
        self.matrix = matrix.detach()
        return matrix
    
    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        theta = self.inputs_to_tensor(inputs=inputs)
        if self.requires_grad:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('theta', theta)
        self.update_matrix()
    
    def inputs_to_tensor(self, inputs: Any = None) -> torch.Tensor:
        """Convert inputs to torch.Tensor."""
        while isinstance(inputs, list):
            inputs = inputs[0]
        if inputs is None:
            inputs = torch.rand(1)[0] * 4 * torch.pi
        elif not isinstance(inputs, (torch.Tensor, nn.Parameter)):
            inputs = torch.tensor(inputs, dtype=torch.float)
        return inputs
    
    def get_unitary(self) -> torch.Tensor:
        """Get the global unitary matrix."""
        matrix = self.update_matrix()
        identity = [torch.tensor(1, dtype=matrix.dtype, device=matrix.device)]*self.nmode
    
        # identity = torch.eye(self.nmode, dtype=matrix.dtype, device=matrix.device)  ##device?
        idx = self.wires[0]
        identity[idx] = matrix
        mat = torch.stack(identity).reshape(-1).diag_embed()  
        return mat
    
    ##################### here fo the tensor calculation
    def get_op_tensor(self, theta: Any) -> torch.Tensor:
        """Get the local unitary matrix. The matrix here represents the matrix acting on the fock state tensor"""
        theta = self.inputs_to_tensor(theta)
        fock_tensor = FockGateTensor(n_mode=2, cutoff=self.cutoff, parameters=[theta])
        ps_ts = fock_tensor.ps()
        return ps_ts
    
    def update_unitary_tensor(self) -> torch.Tensor:
        """Update the local unitary tensor for operators."""
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
        matrix = self.get_op_tensor(theta)
        self.matrix = matrix.detach()
        return matrix
    
    

    

        
   

class BeamSplitter(Gate):
    r"""
    The beamsplitter in the optical quantum gates
    see https://arxiv.org/abs/2004.11002 eqs 42b 
    
    **Matrix Representation:**
    .. math::
    \text{BS} =
        \begin{pmatrix}
            \cos\left(\theta\right) & -e^(-i\phi) \sin\left(\theta\right) \\
            e^(i\phi) \sin\left(\theta\right) &  \cos\left(\theta\right) \\
        \end{pmatrix}

    Args:
        requires_grad (list of bool): theta or phi
    """
    def __init__(
        self,
        inputs: Any = None,
        nmode: int = None,
        wires: Optional[List[int]] = None,
        cutoff: int = None,
        requires_grad: List[bool] = [False, False] 
    ) -> None:
        super().__init__(name='ReconfigurableBeamSplitter', nmode=nmode, wires=wires, cutoff=cutoff)

        assert(len(wires) == 2), "BS gate must act on two wires"
        assert(self.wires[0]+1 == self.wires[1]), "BS gate must act on the neighbor wires"
        
        self.npara = 2
        self.inv_mode = False
        self.requires_grad = requires_grad
        self.init_para(inputs=inputs)

    def get_matrix(self, theta: Any, phi:Any) -> torch.Tensor:
        """Get the local unitary matrix. The matrix here represents the matrix for linear optical elements
         which acts on the creation operator a^dagger """
        theta, phi = self.inputs_to_tensor([theta, phi])
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        # m1 = torch.eye(1, dtype=theta.dtype, device=theta.device)
        m2 = torch.stack([cos, -torch.exp(-1j*phi)*sin, torch.exp(1j*phi)*sin, cos]).reshape(2, 2) + 0j
        return m2
    
    def inputs_to_tensor(self, inputs: Any = None) -> Tuple[torch.Tensor]:
        """Convert inputs to torch.Tensor."""
        if inputs is None:
            theta = torch.rand(1)[0] * torch.pi
            phi   = torch.rand(1)[0] * 2 * torch.pi

        else:
            theta = inputs[0]
            phi   = inputs[1]
        if not isinstance(theta, (torch.Tensor, nn.Parameter)):
            theta = torch.tensor(theta, dtype=torch.float)
        if not isinstance(phi, (torch.Tensor, nn.Parameter)):
            phi = torch.tensor(phi, dtype=torch.float)
        return theta, phi
    
    def update_matrix(self) -> torch.Tensor:
        """Update the local unitary matrix."""
        if self.inv_mode:
            theta = -self.theta
            phi   = -self.phi
        else:
            theta = self.theta
            phi   = self.phi
        matrix = self.get_matrix(theta, phi)
        self.matrix = matrix.detach()
        return matrix
    
    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        # print(inputs)
        theta, phi = self.inputs_to_tensor(inputs=inputs)

        if self.requires_grad[0]:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('theta', theta)

        if self.requires_grad[1]:
            self.phi = nn.Parameter(phi)
        else:
            self.register_buffer('phi', phi)
        self.update_matrix()


    def get_unitary(self) -> torch.Tensor:
        """Get the global unitary matrix."""
        matrix = self.update_matrix()
        identity = torch.eye(self.nmode, dtype=matrix.dtype, device=matrix.device)
    
        # identity = torch.eye(self.nmode, dtype=matrix.dtype, device=matrix.device)  ##device?
        idx0 = self.wires[0]
        idx1 = self.wires[1]

        identity[idx0: idx1+1, idx0: idx1+1] = matrix
        # mat = identity         
        return identity 
    

   ##################### here fo the tensor calculation
    def get_op_tensor(self, theta, phi) -> torch.Tensor:
        """
        Get the local unitary matrix. 
        The matrix here represents the matrix acting on the fock state tensor
        """
        theta, phi = self.inputs_to_tensor([theta, phi])
        fock_tensor = FockGateTensor(n_mode=2, cutoff=self.cutoff, parameters=[theta, phi])
        bs_ts = fock_tensor.bs()
        return bs_ts 
    

    def update_unitary_tensor(self) -> torch.Tensor:
        """Update the local unitary tensor for operators."""
        if self.inv_mode:
            theta = -self.theta
            phi   = -self.phi
        else:
            theta = self.theta
            phi   = self.phi
        matrix = self.get_op_tensor(theta, phi)
        self.matrix = matrix.detach()
        return matrix
    


class BeamSplitter_1(BeamSplitter):
    r"""
    This type BeamSplitter is fixing phi at pi/2
    
    **Matrix Representation:**
    .. math::
    \text{BS} =
        \begin{pmatrix}
            \cos\left(\theta\right) & i\sin\left(\theta\right) \\
            i\sin\left(\theta\right) &  \cos\left(\theta\right) \\
        \end{pmatrix}
    
    Args:
        requires_grad (bool): for theta
    """
    def __init__(
        self,
        inputs: Any = None,
        nmode: int = None,
        wires: Optional[List[int]] = None,
        cutoff: int = None,
        requires_grad: bool = False 
    ) -> None:
        super().__init__(inputs=[inputs, torch.pi/2], nmode=nmode, wires=wires, cutoff=cutoff, 
                         requires_grad=[requires_grad, False])
        self.npara = 1
        self.inv_mode = False
        self.requires_grad = requires_grad



class BeamSplitter_2(BeamSplitter):
    r"""
    This type BeamSplitter is fixing theta at pi/4
    
    **Matrix Representation:**
    .. math::
    \text{BS} =
        \begin{pmatrix}
            \frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2}e^(-i\phi)  \\
            \frac{\sqrt{2}}{2}e^(i\phi)  &  \frac{\sqrt{2}}{2} \\
        \end{pmatrix}

    Args:
        requires_grad (list of bool): for phi
    """
    def __init__(
        self,
        inputs: Any = None,
        nmode: int = None,
        wires: Optional[List[int]] = None,
        cutoff: int = None,
        requires_grad: bool = False 
    ) -> None:
        super().__init__(inputs=[torch.pi/4, inputs], nmode=nmode, wires=wires, cutoff=cutoff, 
                         requires_grad=[False, requires_grad])
        self.npara = 1
        self.inv_mode = False
        self.requires_grad = requires_grad
    





    
       
        
    # def get_unitary_fock():
    #     v -> unitary_fock
    
