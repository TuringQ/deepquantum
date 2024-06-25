"""
Base classes
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from .qmath import xxpp_to_xpxp, xpxp_to_xxpp
from ..qmath import inverse_permutation, state_to_tensors
from ..state import MatrixProductState


class Operation(nn.Module):
    """A base class for quantum operations.

    Args:
        name (str or None, optional): The name of the quantum operation. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int, optional): The Fock space truncation. Default: 2
    """
    def __init__(
        self,
        name: Optional[str] = None,
        nmode: int = 1,
        wires: Union[int, List, None] = None,
        cutoff: int = 2
    ) -> None:
        super().__init__()
        self.name = name
        self.nmode = nmode
        self.wires = wires
        self.cutoff = cutoff
        self.npara = 0

    def tensor_rep(self, x: torch.Tensor) -> torch.Tensor:
        """Get the tensor representation of the state."""
        return x.reshape([-1] + [self.cutoff] * self.nmode)

    def init_para(self) -> None:
        """Initialize the parameters."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass."""
        return self.tensor_rep(x)

    def _convert_indices(self, indices: Union[int, List[int]]) -> List[int]:
        """Convert and check the indices of the modes."""
        if isinstance(indices, int):
            indices = [indices]
        assert isinstance(indices, list), 'Invalid input type'
        assert all(isinstance(i, int) for i in indices), 'Invalid input type'
        if len(indices) > 0:
            assert min(indices) > -1 and max(indices) < self.nmode, 'Invalid input'
        assert len(set(indices)) == len(indices), 'Invalid input'
        return indices

    def _check_minmax(self, minmax: List[int]) -> None:
        """Check the minimum and maximum indices of the modes."""
        assert isinstance(minmax, list)
        assert len(minmax) == 2
        assert all(isinstance(i, int) for i in minmax)
        assert -1 < minmax[0] <= minmax[1] < self.nmode


class Gate(Operation):
    r"""A base class for photonic quantum gates.

    Args:
        name (str or None, optional): The name of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int, optional): The Fock space truncation. Default: ``None``
    """
    def __init__(
        self,
        name: Optional[str] = None,
        nmode: int = 1,
        wires: Union[int, List[int], None] = None,
        cutoff: Optional[int] = None
    ) -> None:
        self.nmode = nmode
        if wires is None:
            wires = [0]
        wires = self._convert_indices(wires)
        if cutoff is None:
            cutoff = 2
        super().__init__(name=name, nmode=nmode, wires=wires, cutoff=cutoff)

    def update_matrix(self) -> torch.Tensor:
        """Update the local unitary matrix acting on creation operators."""
        return self.matrix

    def get_unitary(self) -> torch.Tensor:
        """Get the global unitary matrix acting on creation operators."""
        matrix = self.update_matrix()
        assert matrix.shape[-2] == matrix.shape[-1] == len(self.wires), 'The matrix may not act on creation operators.'
        nmode1 = min(self.wires)
        nmode2 = self.nmode - nmode1 - len(self.wires)
        m1 = torch.eye(nmode1, dtype=matrix.dtype, device=matrix.device)
        m2 = torch.eye(nmode2, dtype=matrix.dtype, device=matrix.device)
        return torch.block_diag(m1, matrix, m2)

    def get_matrix_state(self, matrix: torch.Tensor) -> torch.Tensor:
        """Get the local transformation matrix acting on Fock state tensors."""
        raise NotImplementedError

    def update_matrix_state(self) -> torch.Tensor:
        """Update the local transformation matrix acting on Fock state tensors."""
        matrix = self.update_matrix()
        return self.get_matrix_state(matrix)

    def op_state_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass for state tensors."""
        nt = len(self.wires)
        matrix = self.update_matrix_state().reshape(self.cutoff ** nt, self.cutoff ** nt)
        wires = [i + 1 for i in self.wires]
        pm_shape = list(range(self.nmode + 1))
        for i in wires:
            pm_shape.remove(i)
        pm_shape = wires + pm_shape
        x = x.permute(pm_shape).reshape(self.cutoff ** nt, -1)
        x = (matrix @ x).reshape([self.cutoff] * nt + [-1] + [self.cutoff] * (self.nmode - nt))
        x = x.permute(inverse_permutation(pm_shape))
        return x

    def update_transform_xp(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the local affine symplectic transformation acting on quadrature operators in xxpp order."""
        return self.matrix_xp, self.vector_xp

    def get_symplectic(self) -> torch.Tensor:
        """Get the global symplectic matrix acting on quadrature operators in xxpp order."""
        matrix, _ = self.update_transform_xp()
        assert matrix.shape[-2] == matrix.shape[-1] == 2 * len(self.wires), 'The matrix may not act on xxpp operators.'
        matrix_xpxp = xxpp_to_xpxp(matrix) # here change order to xpxp
        nmode1 = min(self.wires)
        nmode2 = self.nmode - nmode1 - len(self.wires)
        m1 = torch.eye(2 * nmode1, dtype=matrix.dtype, device=matrix.device)
        m2 = torch.eye(2 * nmode2, dtype=matrix.dtype, device=matrix.device)
        return xpxp_to_xxpp(torch.block_diag(m1, matrix_xpxp, m2)) # here change order to xxpp

    def get_displacement(self) -> torch.Tensor:
        """Get the global displacement vector acting on quadrature operators in xxpp order."""
        _, vector = self.update_transform_xp()
        assert vector.shape[-2] == 2 * len(self.wires), 'The vector may not act on xxpp operators.'
        vector_xpxp = xxpp_to_xpxp(vector) # here change order to xpxp
        nmode1 = min(self.wires)
        nmode2 = self.nmode - nmode1 - len(self.wires)
        v1 = torch.zeros(2 * nmode1, 1, dtype=vector.dtype, device=vector.device)
        v2 = torch.zeros(2 * nmode2, 1, dtype=vector.dtype, device=vector.device)
        return xpxp_to_xxpp(torch.cat([v1, vector_xpxp, v2], dim=-2)) # here change order to xxpp

    def op_gaussian(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Perform a forward pass for Gaussian states."""
        cov, mean = x
        mat_xp, vec_xp = self.update_transform_xp()
        wires_x1 = list(range(self.wires[0]))
        wires_x  = self.wires
        wires_x2 = list(range(self.wires[-1] + 1, self.nmode))
        wires_p1 = [wire + self.nmode for wire in wires_x1]
        wires_p  = [wire + self.nmode for wire in self.wires]
        wires_p2 = [wire + self.nmode for wire in wires_x2]
        cov_row = mat_xp @ cov[..., wires_x + wires_p, :]
        cov_row_x = cov_row[..., :len(self.wires), :]
        cov_row_p = cov_row[..., len(self.wires):, :]
        cov = torch.cat([cov[..., wires_x1, :], cov_row_x, cov[..., wires_x2, :],
                         cov[..., wires_p1, :], cov_row_p, cov[..., wires_p2, :]], dim=-2)
        cov_col = cov[..., wires_x + wires_p] @ mat_xp.mT
        cov_col_x = cov_col[..., :len(self.wires)]
        cov_col_p = cov_col[..., len(self.wires):]
        cov = torch.cat([cov[..., wires_x1], cov_col_x, cov[..., wires_x2],
                         cov[..., wires_p1], cov_col_p, cov[..., wires_p2]], dim=-1)
        mean_local = mat_xp @ mean[..., wires_x + wires_p, :] + vec_xp
        mean_x = mean_local[..., :len(self.wires), :]
        mean_p = mean_local[..., len(self.wires):, :]
        mean = torch.cat([mean[..., wires_x1, :], mean_x, mean[..., wires_x2, :],
                          mean[..., wires_p1, :], mean_p, mean[..., wires_p2, :]], dim=-2)
        return [cov, mean]

    def get_mpo(self) -> Tuple[List[torch.Tensor], int]:
        r"""Convert gate to MPO form with identities at empty sites.

        Note:
            If sites are not adjacent, insert identities in the middle, i.e.,

            >>>      |       |            |   |   |
            >>>    --A---x---B--   ->   --A---I---B--
            >>>      |       |            |   |   |

            where

            >>>         a
            >>>         |
            >>>    --i--I--j--
            >>>         |
            >>>         b

            means :math:`\delta_{i,j} \delta_{a,b}`
        """
        index = self.wires
        index_left = min(index)
        nindex = len(index)
        index_sort = sorted(index)
        mat = self.update_matrix_state()
        # transform gate from (out1, out2, ..., in1, in2 ...) to (out1, in1, out2, in2, ...)
        order = list(np.arange(2 * nindex).reshape((2, nindex)).T.flatten())
        mat = mat.reshape([self.cutoff] * 2 * nindex).permute(order).reshape([self.cutoff ** 2] * nindex)
        main_tensors = state_to_tensors(mat, nsite=nindex, qudit=self.cutoff ** 2)
        # each tensor is in shape of (i, a, b, j)
        tensors = []
        previous_i = None
        for i, main_tensor in zip(index_sort, main_tensors):
            # insert identities in the middle
            if previous_i is not None:
                for _ in range(previous_i + 1, i):
                    chi = tensors[-1].shape[-1]
                    identity = torch.eye(chi * self.cutoff, dtype=self.matrix.dtype, device=self.matrix.device)
                    tensors.append(identity.reshape(chi, self.cutoff, chi, self.cutoff).permute(0, 1, 3, 2))
            nleft, _, nright = main_tensor.shape
            tensors.append(main_tensor.reshape(nleft, self.cutoff, self.cutoff, nright))
            previous_i = i
        return tensors, index_left

    def op_mps(self, mps: MatrixProductState) -> MatrixProductState:
        """Perform a forward pass for the ``MatrixProductState``."""
        mpo_tensors, left = self.get_mpo()
        right = left + len(mpo_tensors) - 1
        diff_left = abs(left - mps.center)
        diff_right = abs(right - mps.center)
        center_left = diff_left < diff_right
        if center_left:
            end1 = left
            end2 = right
        else:
            end1 = right
            end2 = left
        wires = list(range(left, right + 1))
        out = MatrixProductState(nsite=mps.nsite, state=mps.tensors, chi=mps.chi, normalize=mps.normalize)
        out.center_orthogonalization(end1, dc=-1, normalize=out.normalize)
        out.apply_mpo(mpo_tensors, wires)
        out.center_orthogonalization(end2, dc=-1, normalize=out.normalize)
        out.center_orthogonalization(end1, dc=out.chi, normalize=out.normalize)
        return out

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor], MatrixProductState]
    ) -> Union[torch.Tensor, List[torch.Tensor], MatrixProductState]:
        """Perform a forward pass."""
        if isinstance(x, MatrixProductState):
            return self.op_mps(x)
        if isinstance(x, torch.Tensor):
            return self.op_state_tensor(x)
        elif isinstance(x, list):
            return self.op_gaussian(x)
