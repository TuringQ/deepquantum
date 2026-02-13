"""Base classes for photonic quantum operations"""

from typing import Any

import numpy as np
import torch
from torch import nn, vmap

from ..qmath import evolve_den_mat, evolve_state, state_to_tensors
from ..state import MatrixProductState
from .distributed import dist_gate
from .state import DistributedFockState


class Operation(nn.Module):
    """A base class for quantum operations.

    Args:
        name (str or None, optional): The name of the quantum operation. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int, optional): The Fock space truncation. Default: 2
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """

    def __init__(
        self,
        name: str | None = None,
        nmode: int = 1,
        wires: int | list | None = None,
        cutoff: int = 2,
        den_mat: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__()
        self.name = name
        self.nmode = nmode
        self.wires = wires
        self.cutoff = cutoff
        self.den_mat = den_mat
        self.noise = noise
        self.mu = mu
        self.sigma = sigma
        self.npara = 0

    def tensor_rep(self, x: torch.Tensor) -> torch.Tensor:
        """Get the tensor representation of the state."""
        if self.den_mat:
            return x.reshape([-1] + [self.cutoff] * 2 * self.nmode)
        else:
            return x.reshape([-1] + [self.cutoff] * self.nmode)

    def init_para(self) -> None:
        """Initialize the parameters."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass."""
        return self.tensor_rep(x)

    def _convert_indices(self, indices: int | list[int]) -> list[int]:
        """Convert and check the indices of the modes."""
        if isinstance(indices, int):
            indices = [indices]
        assert isinstance(indices, list), 'Invalid input type'
        assert all(isinstance(i, int) for i in indices), 'Invalid input type'
        if len(indices) > 0:
            assert min(indices) > -1 and max(indices) < self.nmode, 'Invalid input'
        assert len(set(indices)) == len(indices), 'Invalid input'
        return indices

    def _check_minmax(self, minmax: list[int]) -> None:
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
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """

    def __init__(
        self,
        name: str | None = None,
        nmode: int = 1,
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        self.nmode = nmode
        if wires is None:
            wires = [0]
        wires = self._convert_indices(wires)
        if cutoff is None:
            cutoff = 2
        super().__init__(
            name=name, nmode=nmode, wires=wires, cutoff=cutoff, den_mat=den_mat, noise=noise, mu=mu, sigma=sigma
        )

    def update_matrix(self) -> torch.Tensor:
        """Update the local unitary matrix acting on creation operators."""
        return self.matrix

    def get_unitary(self) -> torch.Tensor:
        """Get the global unitary matrix acting on creation operators."""
        matrix = self.update_matrix()
        assert matrix.shape[-2] == matrix.shape[-1] == len(self.wires), 'The matrix may not act on creation operators.'
        u = matrix.new_ones(self.nmode)
        u = torch.diag(u)
        u[np.ix_(self.wires, self.wires)] = matrix
        return u

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
        matrix = self.update_matrix_state().reshape(self.cutoff**nt, self.cutoff**nt)
        return evolve_state(x, matrix, self.nmode, self.wires, self.cutoff)

    def op_den_mat(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass for density matrices."""
        nt = len(self.wires)
        matrix = self.update_matrix_state().reshape(self.cutoff**nt, self.cutoff**nt)
        return evolve_den_mat(x, matrix, self.nmode, self.wires, self.cutoff)

    def update_transform_xp(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the local affine symplectic transformation acting on quadrature operators in xxpp order."""
        return self.matrix_xp, self.vector_xp

    def get_symplectic(self) -> torch.Tensor:
        """Get the global symplectic matrix acting on quadrature operators in xxpp order."""
        matrix, _ = self.update_transform_xp()
        assert matrix.shape[-2] == matrix.shape[-1] == 2 * len(self.wires), 'The matrix may not act on xxpp operators.'
        s = matrix.new_ones(2 * self.nmode)
        s = torch.diag(s)
        wires = self.wires + [wire + self.nmode for wire in self.wires]
        s[np.ix_(wires, wires)] = matrix
        return s

    def get_displacement(self) -> torch.Tensor:
        """Get the global displacement vector acting on quadrature operators in xxpp order."""
        _, vector = self.update_transform_xp()
        assert vector.shape[-2] == 2 * len(self.wires), 'The vector may not act on xxpp operators.'
        d = vector.new_zeros(2 * self.nmode, 1)
        wires = self.wires + [wire + self.nmode for wire in self.wires]
        d[np.ix_(wires)] = vector
        return d

    def op_cv(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Perform a forward pass for Gaussian (Bosonic) states."""
        cov, mean = x[:2]
        sp_mat = self.get_symplectic()
        cov = sp_mat @ cov @ sp_mat.mT
        mean = sp_mat.to(mean.dtype) @ mean + self.get_displacement()
        return [cov, mean] + x[2:]

    def get_mpo(self) -> tuple[list[torch.Tensor], int]:
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
        mat = mat.reshape([self.cutoff] * 2 * nindex).permute(order).reshape([self.cutoff**2] * nindex)
        main_tensors = state_to_tensors(mat, nsite=nindex, qudit=self.cutoff**2)
        # each tensor is in shape of (i, a, b, j)
        tensors = []
        previous_i = None
        for i, main_tensor in zip(index_sort, main_tensors, strict=True):
            # insert identities in the middle
            if previous_i is not None:
                for _ in range(previous_i + 1, i):
                    chi = tensors[-1].shape[-1]
                    identity = torch.eye(chi * self.cutoff, dtype=mat.dtype, device=mat.device)
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

    def op_dist_state(self, x: DistributedFockState) -> DistributedFockState:
        """Perform a forward pass for a distributed Fock state tensor."""
        nt = len(self.wires)
        matrix = self.update_matrix_state().reshape(self.cutoff**nt, self.cutoff**nt)
        targets = [self.nmode - wire - 1 for wire in self.wires]
        return dist_gate(x, targets, matrix)

    def forward(
        self, x: torch.Tensor | list[torch.Tensor] | MatrixProductState | DistributedFockState
    ) -> torch.Tensor | list[torch.Tensor] | MatrixProductState | DistributedFockState:
        """Perform a forward pass."""
        if isinstance(x, DistributedFockState):
            return self.op_dist_state(x)
        elif isinstance(x, MatrixProductState):
            return self.op_mps(x)
        elif isinstance(x, torch.Tensor):
            if self.den_mat:
                return self.op_den_mat(x)
            else:
                return self.op_state_tensor(x)
        elif isinstance(x, list):
            return self.op_cv(x)

    def extra_repr(self) -> str:
        return f'wires={self.wires}'


class Channel(Operation):
    r"""A base class for photonic quantum channels.

    Args:
        name (str or None, optional): The name of the channel. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
    """

    def __init__(
        self,
        name: str | None = None,
        nmode: int = 1,
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
    ) -> None:
        self.nmode = nmode
        if wires is None:
            wires = [0]
        wires = self._convert_indices(wires)
        if cutoff is None:
            cutoff = 2
        super().__init__(name=name, nmode=nmode, wires=wires, cutoff=cutoff, den_mat=True, noise=False)

    def update_matrix_state(self) -> torch.Tensor:
        """Update the local Kraus matrices acting on Fock state density matrices."""
        raise NotImplementedError

    def op_den_mat(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass for density matrices."""
        nt = len(self.wires)
        matrix = self.update_matrix_state().reshape(-1, self.cutoff**nt, self.cutoff**nt)
        x = vmap(evolve_den_mat, in_dims=(None, 0, None, None, None))(x, matrix, self.nmode, self.wires, self.cutoff)
        return x.sum(0)

    def update_transform_xy(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the local transformation matrices X and Y acting on Gaussian states."""
        return self.matrix_x, self.matrix_y

    def op_cv(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Perform a forward pass for Gaussian (Bosonic) states.

        See Quantum Continuous Variables: A Primer of Theoretical Methods (2024)
        by Alessio Serafini Eq.(5.35-5.37) in page 90
        """
        cov, mean = x[:2]
        local_x, local_y = self.update_transform_xy()
        assert local_x.shape[-2] == local_x.shape[-1] == 2 * len(self.wires), 'Invalid matrix shape.'
        assert local_y.shape[-2] == local_y.shape[-1] == 2 * len(self.wires), 'Invalid matrix shape.'
        wires = self.wires + [wire + self.nmode for wire in self.wires]
        mat_x = local_x.new_ones(2 * self.nmode)
        mat_x = torch.diag(mat_x)
        mat_x[np.ix_(wires, wires)] = local_x
        mat_y = local_y.new_zeros(2 * self.nmode, 2 * self.nmode)
        mat_y[np.ix_(wires, wires)] = local_y
        cov = mat_x @ cov @ mat_x.mT + mat_y
        mean = mat_x.to(mean.dtype) @ mean
        return [cov, mean] + x[2:]

    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        """Perform a forward pass."""
        if isinstance(x, torch.Tensor):
            return self.op_den_mat(x)
        elif isinstance(x, list):
            return self.op_cv(x)


class Delay(Operation):
    r"""Delay loop.

    Args:
        name (str or None, optional): The name of the quantum operation. Default: ``'Delay'``
        ntau (int, optional): The number of modes in the delay loop. Default: 1
        nmode (int, optional): The number of spatial modes that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """

    def __init__(
        self,
        name='Delay',
        ntau: int = 1,
        nmode: int = 1,
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        self.nmode = nmode
        if wires is None:
            wires = [0]
        wires = self._convert_indices(wires)
        if cutoff is None:
            cutoff = 2
        super().__init__(
            name=name, nmode=nmode, wires=wires, cutoff=cutoff, den_mat=den_mat, noise=noise, mu=mu, sigma=sigma
        )
        assert len(self.wires) == 1, f'{self.name} must act on one mode'
        self.ntau = ntau
        self.gates = nn.Sequential()

    def to(self, arg: Any) -> 'Delay':
        """Set dtype or device of the ``Delay``."""
        for gate in self.gates:
            gate.to(arg)
        return self

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        count = 0
        for gate in self.gates:
            if inputs is None:
                gate.init_para()
            else:
                gate.init_para(inputs[count : count + gate.npara])
            count += gate.npara

    def forward(
        self, x: torch.Tensor | list[torch.Tensor] | MatrixProductState | DistributedFockState
    ) -> torch.Tensor | list[torch.Tensor] | MatrixProductState | DistributedFockState:
        """Perform a forward pass."""
        return self.gates(x)

    def extra_repr(self) -> str:
        return f'wires={self.wires}, ntau={self.ntau}'
