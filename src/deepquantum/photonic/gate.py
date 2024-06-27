"""
Photonic quantum gates
"""

import copy
import itertools
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn

import deepquantum.photonic as dqp
from .operation import Gate
from ..qmath import is_unitary


class PhaseShift(Gate):
    r"""Phase Shifter.

    **Matrix Representation:**

    .. math::

        U_{\text{PS}}(\theta) = e^{i\theta}

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
        inv_mode (bool, optional): Whether the rotation in the phase space is clockwise. Default: ``False``
    """
    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 1,
        wires: Union[int, List[int], None] = None,
        cutoff: Optional[int] = None,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
        inv_mode: bool = False
    ) -> None:
        super().__init__(name='PhaseShift', nmode=nmode, wires=wires, cutoff=cutoff)
        assert len(self.wires) == 1, 'PS gate acts on single mode'
        self.npara = 1
        self.requires_grad = requires_grad
        self.noise = noise
        self.mu = mu
        self.sigma = sigma
        self.inv_mode = inv_mode
        self.init_para(inputs=inputs)

    def inputs_to_tensor(self, inputs: Any = None) -> torch.Tensor:
        """Convert inputs to torch.Tensor."""
        while isinstance(inputs, list):
            inputs = inputs[0]
        if inputs is None:
            inputs = torch.rand(1)[0] * 2 * torch.pi
        elif not isinstance(inputs, (torch.Tensor, nn.Parameter)):
            inputs = torch.tensor(inputs, dtype=torch.float)
        if self.noise:
            inputs = inputs + torch.normal(self.mu, self.sigma, size=(1, )).squeeze()
        return inputs

    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Get the local unitary matrix acting on creation operators."""
        theta = self.inputs_to_tensor(theta)
        if self.inv_mode:
            theta = -theta
        return torch.exp(1j * theta).reshape(1, 1)

    def update_matrix(self) -> torch.Tensor:
        """Update the local unitary matrix acting on creation operators."""
        matrix = self.get_matrix(self.theta)
        self.matrix = matrix.detach()
        return matrix

    def get_matrix_state(self, matrix: torch.Tensor) -> torch.Tensor:
        """Get the local transformation matrix acting on Fock state tensors."""
        return torch.stack([matrix[0, 0] ** n for n in range(self.cutoff)]).reshape(-1).diag_embed()

    def get_transform_xp(self, theta: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        theta = self.inputs_to_tensor(theta)
        if self.inv_mode:
            theta = -theta
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        matrix_xp = torch.stack([cos, -sin, sin, cos]).reshape(2, 2)
        vector_xp = torch.zeros(2, 1, dtype=theta.dtype, device=theta.device)
        return matrix_xp, vector_xp

    def update_transform_xp(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        matrix_xp, vector_xp = self.get_transform_xp(self.theta)
        self.matrix_xp = matrix_xp.detach()
        self.vector_xp = vector_xp.detach()
        return matrix_xp, vector_xp

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        theta = self.inputs_to_tensor(inputs=inputs)
        if self.requires_grad:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('theta', theta)
        self.update_matrix()
        self.update_transform_xp()


class BeamSplitter(Gate):
    r"""Beam Splitter.

    See https://arxiv.org/abs/2004.11002 Eq.(42b)

    **Matrix Representation:**

    .. math::

        U_{\text{BS}}(\theta, \phi) =
            \begin{pmatrix}
                \cos\left(\theta\right)           & -e^{-i\phi} \sin\left(\theta\right) \\
                e^{i\phi} \sin\left(\theta\right) & \cos\left(\theta\right)             \\
            \end{pmatrix}

    **Symplectic Transformation:**

    .. math::

        BS^\dagger(\theta, \phi)
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix}
        BS(\theta, \phi) =
        \begin{pmatrix}
            \mathrm{Re}(U_{\text{BS}}) & -\mathrm{Im}(U_{\text{BS}}) \\
            \mathrm{Im}(U_{\text{BS}}) &  \mathrm{Re}(U_{\text{BS}}) \\
        \end{pmatrix}
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix}

    Args:
        inputs (Any, optional): The parameters of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 2
        wires (List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """
    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 2,
        wires: Optional[List[int]] = None,
        cutoff: Optional[int] = None,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1
    ) -> None:
        if wires is None:
            wires = [0, 1]
        super().__init__(name='BeamSplitter', nmode=nmode, wires=wires, cutoff=cutoff)
        assert len(self.wires) == 2, 'BS gate must act on two wires'
        assert self.wires[0] + 1 == self.wires[1], 'BS gate must act on the neighbor wires'
        self.npara = 2
        self.requires_grad = requires_grad
        self.noise = noise
        self.mu = mu
        self.sigma = sigma
        self.init_para(inputs=inputs)

    def inputs_to_tensor(self, inputs: Any = None) -> Tuple[torch.Tensor]:
        """Convert inputs to torch.Tensor."""
        if inputs is None:
            theta = torch.rand(1)[0] * 2 * torch.pi
            phi   = torch.rand(1)[0] * 2 * torch.pi
        else:
            theta = inputs[0]
            phi   = inputs[1]
        if not isinstance(theta, (torch.Tensor, nn.Parameter)):
            theta = torch.tensor(theta, dtype=torch.float)
        if not isinstance(phi, (torch.Tensor, nn.Parameter)):
            phi = torch.tensor(phi, dtype=torch.float)
        if self.noise:
            theta = theta + torch.normal(self.mu, self.sigma, size=(1, )).squeeze()
            phi = phi + torch.normal(self.mu, self.sigma, size=(1, )).squeeze()
        return theta, phi

    def get_matrix(self, theta: Any, phi: Any) -> torch.Tensor:
        """Get the local unitary matrix acting on creation operators."""
        theta, phi = self.inputs_to_tensor([theta, phi])
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        e_m_ip = torch.exp(-1j * phi)
        e_ip = torch.exp(1j * phi)
        return torch.stack([cos, -e_m_ip * sin, e_ip * sin, cos]).reshape(2, 2)

    def update_matrix(self) -> torch.Tensor:
        """Update the local unitary matrix acting on creation operators."""
        matrix = self.get_matrix(self.theta, self.phi)
        self.matrix = matrix.detach()
        return matrix

    def get_matrix_state(self, matrix: torch.Tensor) -> torch.Tensor:
        """Get the local transformation matrix acting on Fock state tensors.

        See https://arxiv.org/pdf/2004.11002.pdf Eq.(74) and Eq.(75)
        """
        sqrt = torch.sqrt(torch.arange(self.cutoff, device=matrix.device))
        tran_mat = matrix.new_zeros([self.cutoff] * 2 * len(self.wires))
        tran_mat[0, 0, 0, 0] = 1.0
        # rank 3
        for m in range(self.cutoff):
            for n in range(self.cutoff - m):
                p = m + n
                if 0 < p < self.cutoff:
                    tran_mat[m, n, p, 0] = (
                        matrix[0, 0] * sqrt[m] / sqrt[p] * tran_mat[m - 1, n, p - 1, 0].clone()
                        + matrix[1, 0] * sqrt[n] / sqrt[p] * tran_mat[m, n - 1, p - 1, 0].clone()
                    )
        # rank 4
        for m in range(self.cutoff):
            for n in range(self.cutoff):
                for p in range(self.cutoff):
                    q = m + n - p
                    if 0 < q < self.cutoff:
                        tran_mat[m, n, p, q] = (
                            matrix[0, 1] * sqrt[m] / sqrt[q] * tran_mat[m - 1, n, p, q - 1].clone()
                            + matrix[1, 1] * sqrt[n] / sqrt[q] * tran_mat[m, n - 1, p, q - 1].clone()
                        )
        return tran_mat

    def get_transform_xp(self, theta: Any, phi: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        theta, phi = self.inputs_to_tensor([theta, phi])
        matrix = self.get_matrix(theta, phi)          # correspond to: U a^+ U^+ = u^T @ a^+ and U^+ a U = u @ a
        # matrix = matrix.conj() # conflict with vmap # correspond to: U a U^+ = (u^*)^T @ a and U^+ a^+ U = u^* @ a^+
        mat_real = matrix.real
        mat_imag = -matrix.imag
        matrix_xp = torch.cat([torch.cat([mat_real, mat_imag], dim=-1),
                               torch.cat([-mat_imag, mat_real], dim=-1)], dim=-2)
        vector_xp = torch.zeros(4, 1, dtype=theta.dtype, device=theta.device)
        return matrix_xp, vector_xp

    def update_transform_xp(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        matrix_xp, vector_xp = self.get_transform_xp(self.theta, self.phi)
        self.matrix_xp = matrix_xp.detach()
        self.vector_xp = vector_xp.detach()
        return matrix_xp, vector_xp

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        theta, phi = self.inputs_to_tensor(inputs=inputs)
        if self.requires_grad:
            self.theta = nn.Parameter(theta)
            self.phi = nn.Parameter(phi)
        else:
            self.register_buffer('theta', theta)
            self.register_buffer('phi', phi)
        self.update_matrix()
        self.update_transform_xp()


class MZI(BeamSplitter):
    r"""Mach-Zehnder Interferometer.

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        U_{\text{MZI}}(\theta, \phi) = ie^{i\theta/2}
            \begin{pmatrix}
                e^{i\phi} \sin\left(\th\right) &  \cos\left(\th\right) \\
                e^{i\phi} \cos\left(\th\right) & -\sin\left(\th\right) \\
            \end{pmatrix}

    or when ``phi_first`` is ``False``:

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        U_{\text{MZI}}(\theta, \phi) = ie^{i\theta/2}
            \begin{pmatrix}
                e^{i\phi} \sin\left(\th\right) & e^{i\phi} \cos\left(\th\right) \\
                \cos\left(\th\right)           & -\sin\left(\th\right)          \\
            \end{pmatrix}

    **Symplectic Transformation:**

    .. math::

        MZI^\dagger(\theta, \phi)
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix}
        MZI(\theta, \phi) =
        \begin{pmatrix}
            \mathrm{Re}(U_{\text{MZI}}) & -\mathrm{Im}(U_{\text{MZI}}) \\
            \mathrm{Im}(U_{\text{MZI}}) &  \mathrm{Re}(U_{\text{MZI}}) \\
        \end{pmatrix}
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix}

    Args:
        inputs (Any, optional): The parameters of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 2
        wires (List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        phi_first (bool, optional): Whether :math:`\phi` is the first phase shifter. Default: ``True``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """
    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 2,
        wires: Optional[List[int]] = None,
        cutoff: Optional[int] = None,
        phi_first: bool = True,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1
    ) -> None:
        self.phi_first = phi_first
        super().__init__(inputs=inputs, nmode=nmode, wires=wires, cutoff=cutoff, requires_grad=requires_grad,
                         noise=noise, mu=mu, sigma=sigma)
        self.name = 'MZI'

    def get_matrix(self, theta: Any, phi: Any) -> torch.Tensor:
        """Get the local unitary matrix acting on operators."""
        theta, phi = self.inputs_to_tensor([theta, phi])
        cos = torch.cos(theta / 2)
        sin = torch.sin(theta / 2)
        e_it = torch.exp(1j * theta / 2)
        e_ip = torch.exp(1j * phi)
        mat =  1j * e_it * torch.stack([e_ip * sin, cos, e_ip * cos, -sin]).reshape(2, 2)
        if not self.phi_first:
            mat = mat.T
        return mat


class BeamSplitterTheta(BeamSplitter):
    r"""Beam Splitter with fixed :math:`\phi` at :math:`\pi/2`.

    **Matrix Representation:**

    .. math::

        U_{\text{BS}}(\theta) =
            \begin{pmatrix}
                 \cos\left(\theta\right) & i\sin\left(\theta\right) \\
                i\sin\left(\theta\right) &  \cos\left(\theta\right) \\
            \end{pmatrix}

    **Symplectic Transformation:**

    .. math::

        BS^\dagger(\theta)
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix}
        BS(\theta) =
        \begin{pmatrix}
            \mathrm{Re}(U_{\text{BS}}) & -\mathrm{Im}(U_{\text{BS}}) \\
            \mathrm{Im}(U_{\text{BS}}) &  \mathrm{Re}(U_{\text{BS}}) \\
        \end{pmatrix}
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix}

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 2
        wires (List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """
    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 2,
        wires: Optional[List[int]] = None,
        cutoff: Optional[int] = None,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1
    ) -> None:
        super().__init__(inputs=inputs, nmode=nmode, wires=wires, cutoff=cutoff, requires_grad=requires_grad,
                         noise=noise, mu=mu, sigma=sigma)
        self.npara = 1

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        if inputs is None:
            inputs = torch.rand(1)[0] * 2 * torch.pi
        theta, phi = self.inputs_to_tensor(inputs=[inputs, torch.pi / 2])
        if self.requires_grad:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('theta', theta)
        self.register_buffer('phi', phi)
        self.update_matrix()


class BeamSplitterPhi(BeamSplitter):
    r"""Beam Splitter with fixed :math:`\theta` at :math:`\pi/4`.

    **Matrix Representation:**

    .. math::

        U_{\text{BS}}(\phi) =
            \begin{pmatrix}
                \frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2}e^{-i\phi} \\
                \frac{\sqrt{2}}{2}e^{i\phi}  &  \frac{\sqrt{2}}{2} \\
            \end{pmatrix}

    **Symplectic Transformation:**

    .. math::

        BS^\dagger(\phi)
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix}
        BS(\phi) =
        \begin{pmatrix}
            \mathrm{Re}(U_{\text{BS}}) & -\mathrm{Im}(U_{\text{BS}}) \\
            \mathrm{Im}(U_{\text{BS}}) &  \mathrm{Re}(U_{\text{BS}}) \\
        \end{pmatrix}
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix}

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 2
        wires (List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """
    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 2,
        wires: Optional[List[int]] = None,
        cutoff: Optional[int] = None,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1
    ) -> None:
        super().__init__(inputs=inputs, nmode=nmode, wires=wires, cutoff=cutoff, requires_grad=requires_grad,
                         noise=noise, mu=mu, sigma=sigma)
        self.npara = 1

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        if inputs is None:
            inputs = torch.rand(1)[0] * 2 * torch.pi
        theta, phi = self.inputs_to_tensor(inputs=[torch.pi / 4, inputs])
        if self.requires_grad:
            self.phi = nn.Parameter(phi)
        else:
            self.register_buffer('phi', phi)
        self.register_buffer('theta', theta)
        self.update_matrix()


class BeamSplitterSingle(BeamSplitter):
    r"""Beam Splitter with a single parameter.

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        U_{\text{BS-Rx}}(\theta) =
            \begin{pmatrix}
                 \cos\left(\th\right) & i\sin\left(\th\right) \\
                i\sin\left(\th\right) &  \cos\left(\th\right) \\
            \end{pmatrix}

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        U_{\text{BS-Ry}}(\theta) =
            \begin{pmatrix}
                \cos\left(\th\right) & -\sin\left(\th\right) \\
                \sin\left(\th\right) &  \cos\left(\th\right) \\
            \end{pmatrix}

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        U_{\text{BS-H}}(\theta) =
            \begin{pmatrix}
                \cos\left(\th\right) &  \sin\left(\th\right) \\
                \sin\left(\th\right) & -\cos\left(\th\right) \\
            \end{pmatrix}

    **Symplectic Transformation:**

    .. math::

        BS^\dagger(\theta)
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix}
        BS(\theta) =
        \begin{pmatrix}
            \mathrm{Re}(U_{\text{BS}}) & -\mathrm{Im}(U_{\text{BS}}) \\
            \mathrm{Im}(U_{\text{BS}}) &  \mathrm{Re}(U_{\text{BS}}) \\
        \end{pmatrix}
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix}

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 2
        wires (List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        convention (str, optional): The convention of the type of the beam splitter, including ``'rx'``,
            ``'ry'`` and ``'h'``. Default: ``'rx'``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """
    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 2,
        wires: Optional[List[int]] = None,
        cutoff: Optional[int] = None,
        convention: str = 'rx',
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1
    ) -> None:
        self.convention = convention
        super().__init__(inputs=inputs, nmode=nmode, wires=wires, cutoff=cutoff, requires_grad=requires_grad,
                         noise=noise, mu=mu, sigma=sigma)
        self.npara = 1

    def inputs_to_tensor(self, inputs: Any = None) -> torch.Tensor:
        """Convert inputs to torch.Tensor."""
        while isinstance(inputs, list):
            inputs = inputs[0]
        if inputs is None:
            inputs = torch.rand(1)[0] * 4 * torch.pi
        elif not isinstance(inputs, (torch.Tensor, nn.Parameter)):
            inputs = torch.tensor(inputs, dtype=torch.float)
        if self.noise:
            inputs = inputs + torch.normal(self.mu, self.sigma, size=(1, )).squeeze()
        return inputs

    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Get the local unitary matrix acting on creation operators."""
        theta = self.inputs_to_tensor(theta)
        cos = torch.cos(theta / 2) + 0j
        sin = torch.sin(theta / 2) + 0j
        if self.convention == 'rx':
            return torch.stack([cos, 1j * sin, 1j * sin, cos]).reshape(2, 2)
        elif self.convention == 'ry':
            return torch.stack([cos, -sin, sin, cos]).reshape(2, 2)
        elif self.convention == 'h':
            return torch.stack([cos, sin, sin, -cos]).reshape(2, 2)

    def update_matrix(self) -> torch.Tensor:
        """Update the local unitary matrix acting on creation operators."""
        matrix = self.get_matrix(self.theta)
        self.matrix = matrix.detach()
        return matrix

    def get_transform_xp(self, theta: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        theta = self.inputs_to_tensor(theta)
        matrix = self.get_matrix(theta)               # correspond to: U a^+ U^+ = u^T @ a^+ and U^+ a U = u @ a
        # matrix = matrix.conj() # conflict with vmap # correspond to: U a U^+ = (u^*)^T @ a and U^+ a^+ U = u^* @ a^+
        mat_real = matrix.real
        mat_imag = -matrix.imag
        matrix_xp = torch.cat([torch.cat([mat_real, mat_imag], dim=-1),
                               torch.cat([-mat_imag, mat_real], dim=-1)], dim=-2)
        vector_xp = torch.zeros(4, 1, dtype=theta.dtype, device=theta.device)
        return matrix_xp, vector_xp

    def update_transform_xp(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        matrix_xp, vector_xp = self.get_transform_xp(self.theta)
        self.matrix_xp = matrix_xp.detach()
        self.vector_xp = vector_xp.detach()
        return matrix_xp, vector_xp

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        theta = self.inputs_to_tensor(inputs=inputs)
        if self.requires_grad:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('theta', theta)
        self.update_matrix()
        self.update_transform_xp()


class UAnyGate(Gate):
    """Arbitrary unitary gate.

    Args:
        unitary (Any): Any given unitary matrix.
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        minmax (List[int] or None, optional): The minimum and maximum indices of the modes that the quantum
            operation acts on. Only valid when ``wires`` is ``None``. Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        name (str, optional): The name of the gate. Default: ``'UAnyGate'``
    """
    def __init__(
        self,
        unitary: Any,
        nmode: int = 1,
        wires: Optional[List[int]] = None,
        minmax: Optional[List[int]] = None,
        cutoff: Optional[int] = None,
        name: str = 'UAnyGate'
    ) -> None:
        self.nmode = nmode
        if wires is None:
            if minmax is None:
                minmax = [0, nmode - 1]
            self._check_minmax(minmax)
            wires = list(range(minmax[0], minmax[1] + 1))
        super().__init__(name=name, nmode=nmode, wires=wires, cutoff=cutoff)
        self.minmax = [min(self.wires), max(self.wires)]
        for i in range(len(self.wires) - 1):
            assert self.wires[i] + 1 == self.wires[i + 1], 'The wires should be consecutive integers'
        if not isinstance(unitary, torch.Tensor):
            unitary = torch.tensor(unitary, dtype=torch.cfloat).reshape(-1, len(self.wires))
        assert unitary.dtype in (torch.cfloat, torch.cdouble)
        assert unitary.shape[0] == len(self.wires), 'Please check wires'
        assert is_unitary(unitary), 'Please check the unitary matrix'
        self.register_buffer('matrix', unitary)
        self.register_buffer('matrix_state', None)

    def get_matrix_state(self, matrix: torch.Tensor) -> torch.Tensor:
        """Get the local transformation matrix acting on Fock state tensors.

        See https://arxiv.org/pdf/2004.11002.pdf Eq.(71)
        """
        nt = len(self.wires)
        sqrt = torch.sqrt(torch.arange(self.cutoff, device=matrix.device))
        tran_mat = matrix.new_zeros([self.cutoff] *  2 * nt)
        tran_mat[tuple([0] * 2 * nt)] = 1.0
        for rank in range(nt + 1, 2 * nt + 1):
            col_num = rank - nt - 1
            matrix_j = matrix[:, col_num]
            # all combinations of the first `rank-1` modes
            combs = itertools.product(range(self.cutoff), repeat=rank-1)
            for modes in combs:
                mode_out = modes[:nt]
                mode_in_part = modes[nt:]
                # number of photons for the last nonzero mode
                in_rest = sum(mode_out) - sum(mode_in_part)
                if 0 < in_rest < self.cutoff:
                    state = list(modes) + [in_rest] + [0] * (2 * nt - rank)
                    sum_tmp = 0
                    for i in range(nt):
                        state_pre = copy.deepcopy(state)
                        state_pre[i] = state_pre[i] - 1
                        state_pre[len(modes)] = state_pre[len(modes)] - 1
                        sum_tmp += matrix_j[i] * sqrt[modes[i]] * tran_mat[tuple(state_pre)].clone()
                    tran_mat[tuple(state)] = sum_tmp / sqrt[in_rest]
        return tran_mat

    def update_matrix_state(self) -> torch.Tensor:
        """Update the local transformation matrix acting on Fock state tensors."""
        if self.matrix_state is None:
            matrix = self.update_matrix()
            tran_mat = self.get_matrix_state(matrix)
            self.register_buffer('matrix_state', tran_mat)
        else:
            tran_mat = self.matrix_state
        return tran_mat

    def get_transform_xp(self, matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        # correspond to: U a^+ U^+ = u^T @ a^+ and U^+ a U = u @ a
        # correspond to: U a U^+ = (u^*)^T @ a and U^+ a^+ U = u^* @ a^+
        # matrix = matrix.conj() # conflict with vmap
        mat_real = matrix.real
        mat_imag = -matrix.imag
        matrix_xp = torch.cat([torch.cat([mat_real, mat_imag], dim=-1),
                               torch.cat([-mat_imag, mat_real], dim=-1)], dim=-2)
        vector_xp = torch.zeros(2 * self.nmode, 1, dtype=mat_real.dtype, device=mat_real.device)
        return matrix_xp, vector_xp

    def update_transform_xp(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        matrix_xp, vector_xp = self.get_transform_xp(self.matrix)
        self.matrix_xp = matrix_xp.detach()
        self.vector_xp = vector_xp.detach()
        return matrix_xp, vector_xp


class Squeezing(Gate):
    r"""Squeezing gate.

    **Symplectic Transformation:**

    .. math::

        S^{\dagger}(r, \theta)
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix}
        S(r,\theta) =
        \begin{pmatrix}
            \cosh r - \sinh r \cos\theta & -\sinh r \sin\theta \\
            -\sinh r \sin\theta & \cosh r + \sinh r \cos\theta \\
        \end{pmatrix}
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix}

    Args:
        inputs (Any, optional): The parameters of the gate (:math:`r` and :math:`\theta`). Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """
    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 1,
        wires: Union[int, List[int], None] = None,
        cutoff: Optional[int] = None,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1
    ) -> None:
        super().__init__(name='Squeezing', nmode=nmode, wires=wires, cutoff=cutoff)
        assert len(self.wires) == 1, 'Squeezing gate acts on single mode'
        self.npara = 2
        self.requires_grad = requires_grad
        self.noise = noise
        self.mu = mu
        self.sigma = sigma
        self.init_para(inputs=inputs)

    def inputs_to_tensor(self, inputs: Any = None) -> Tuple[torch.Tensor]:
        """Convert inputs to torch.Tensor."""
        if inputs is None:
            r = torch.rand(1)[0]
            theta = torch.rand(1)[0] * 2 * torch.pi
        else:
            r = inputs[0]
            theta = inputs[1]
        if not isinstance(r, (torch.Tensor, nn.Parameter)):
            r = torch.tensor(r, dtype=torch.float)
        if not isinstance(theta, (torch.Tensor, nn.Parameter)):
            theta = torch.tensor(theta, dtype=torch.float)
        if self.noise:
            r = r + torch.normal(self.mu, self.sigma, size=(1, )).squeeze()
            theta = theta + torch.normal(self.mu, self.sigma, size=(1, )).squeeze()
        return r, theta

    def get_matrix(self, r: Any, theta: Any) -> torch.Tensor:
        """Get the local matrix acting on annihilation and creation operators."""
        r, theta = self.inputs_to_tensor([r, theta])
        ch = torch.cosh(r)
        sh = torch.sinh(r)
        e_it = torch.exp(1j * theta)
        e_m_it = torch.exp(-1j * theta)
        return torch.stack([ch, -e_it * sh, -e_m_it * sh, ch]).reshape(2, 2)

    def update_matrix(self) -> torch.Tensor:
        """Update the local matrix acting on annihilation and creation operators."""
        matrix = self.get_matrix(self.r, self.theta)
        self.matrix = matrix.detach()
        return matrix

    def get_matrix_state(self, matrix: torch.Tensor) -> torch.Tensor:
        """Get the local transformation matrix acting on Fock state tensors.

        See https://arxiv.org/pdf/2004.11002.pdf Eq.(51) and Eq.(52)
        """
        sqrt = torch.sqrt(torch.arange(self.cutoff, device=matrix.device))
        tran_mat = matrix.new_zeros([self.cutoff] * 2)
        sech = 1 / matrix[0, 0]
        e_it_tanh = -matrix[0, 1] / matrix[0, 0]
        e_m_it_tanh = -matrix[1, 0] / matrix[0, 0]
        tran_mat[0, 0] = torch.sqrt(sech)
        # rank 1
        for m in range(1, self.cutoff - 1, 2):
            tran_mat[m + 1, 0] = -sqrt[m] / sqrt[m + 1] * tran_mat[m - 1, 0].clone() * e_it_tanh
        # rank 2
        for m in range(self.cutoff):
            for n in range(self.cutoff - 1):
                if (m + n) % 2 == 1:
                    tran_mat[m, n + 1] = (sqrt[m] * tran_mat[m - 1, n].clone() * sech
                                         + sqrt[n] * tran_mat[m, n - 1].clone() * e_m_it_tanh) / sqrt[n + 1]
        return tran_mat

    def get_transform_xp(self, r: Any, theta: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        r, theta = self.inputs_to_tensor([r, theta])
        ch = torch.cosh(r)
        sh = torch.sinh(r)
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        matrix_xp = torch.stack([ch - sh * cos, -sh * sin, -sh * sin, ch + sh * cos]).reshape(2, 2)
        vector_xp = torch.zeros(2, 1, dtype=r.dtype, device=r.device)
        return matrix_xp, vector_xp

    def update_transform_xp(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        matrix_xp, vector_xp = self.get_transform_xp(self.r, self.theta)
        self.matrix_xp = matrix_xp.detach()
        self.vector_xp = vector_xp.detach()
        return matrix_xp, vector_xp

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        r, theta = self.inputs_to_tensor(inputs=inputs)
        if self.requires_grad:
            self.r = nn.Parameter(r)
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('r', r)
            self.register_buffer('theta', theta)
        self.update_matrix()
        self.update_transform_xp()


class Displacement(Gate):
    r"""Displacement gate.

    **Symplectic Transformation:**

    .. math::
        D^\dagger(\alpha)
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix}
        D(\alpha) =
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix} + \frac{\sqrt{\hbar}}{\kappa}
        \begin{pmatrix}
            r\cos\theta \\
            r\sin\theta \\
        \end{pmatrix}

    Args:
        inputs (Any, optional): The parameters of the gate (:math:`r` and :math:`\theta`). Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """
    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 1,
        wires: Union[int, List[int], None] = None,
        cutoff: Optional[int] = None,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1
    ) -> None:
        super().__init__(name='Displacement', nmode=nmode, wires=wires, cutoff=cutoff)
        assert len(self.wires) == 1, 'Displacement gate acts on single mode'
        self.npara = 2
        self.requires_grad = requires_grad
        self.noise = noise
        self.mu = mu
        self.sigma = sigma
        self.init_para(inputs=inputs)

    def inputs_to_tensor(self, inputs: Any = None) -> Tuple[torch.Tensor]:
        """Convert inputs to torch.Tensor."""
        if inputs is None:
            r = torch.rand(1)[0]
            theta = torch.rand(1)[0] * 2 * torch.pi
        else:
            r = inputs[0]
            theta = inputs[1]
        if not isinstance(r, (torch.Tensor, nn.Parameter)):
            r = torch.tensor(r, dtype=torch.float)
        if not isinstance(theta, (torch.Tensor, nn.Parameter)):
            theta = torch.tensor(theta, dtype=torch.float)
        if self.noise:
            r = r + torch.normal(self.mu, self.sigma, size=(1, )).squeeze()
            theta = theta + torch.normal(self.mu, self.sigma, size=(1, )).squeeze()
        return r, theta

    def get_matrix(self, r: Any, theta: Any) -> torch.Tensor:
        """Get the local unitary matrix acting on annihilation and creation operators."""
        return torch.eye(2, dtype=r.dtype, device=r.device) + 0j

    def update_matrix(self) -> torch.Tensor:
        """Update the local unitary matrix acting on annihilation and creation operators."""
        matrix = self.get_matrix(self.r, self.theta)
        self.matrix = matrix.detach()
        return matrix

    def get_matrix_state(self, r: Any, theta: Any) -> torch.Tensor:
        """Get the local transformation matrix acting on Fock state tensors.

        See https://arxiv.org/pdf/2004.11002.pdf Eq.(57) and Eq.(58)
        """
        r, theta = self.inputs_to_tensor(inputs=[r, theta])
        sqrt = torch.sqrt(torch.arange(self.cutoff, device=r.device))
        alpha = r * torch.exp(1j * theta)
        alpha_c = r * torch.exp(-1j * theta)
        tran_mat = alpha.new_zeros([self.cutoff] * 2)
        tran_mat[0, 0] = torch.exp(-(r ** 2) / 2)
        # rank 1
        for m in range(self.cutoff - 1):
            tran_mat[m + 1, 0] = alpha / sqrt[m + 1] * tran_mat[m, 0].clone()
        # rank 2
        for m in range(self.cutoff):
            for n in range(self.cutoff - 1):
                tran_mat[m, n + 1] = (-alpha_c * tran_mat[m, n].clone()
                                     + sqrt[m] * tran_mat[m - 1, n].clone()) / sqrt[n + 1]
        return tran_mat

    def update_matrix_state(self) -> torch.Tensor:
        """Update the local transformation matrix acting on Fock state tensors."""
        return self.get_matrix_state(self.r, self.theta)

    def get_transform_xp(self, r: Any, theta: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        r, theta = self.inputs_to_tensor([r, theta])
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        matrix_xp = torch.eye(2, dtype=r.dtype, device=r.device)
        vector_xp = torch.stack([r * cos, r * sin]).reshape(2, 1) * dqp.hbar ** 0.5 / dqp.kappa
        return matrix_xp, vector_xp

    def update_transform_xp(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        matrix_xp, vector_xp = self.get_transform_xp(self.r, self.theta)
        self.matrix_xp = matrix_xp.detach()
        self.vector_xp = vector_xp.detach()
        return matrix_xp, vector_xp

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        r, theta = self.inputs_to_tensor(inputs=inputs)
        if self.requires_grad:
            self.r = nn.Parameter(r)
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('r', r)
            self.register_buffer('theta', theta)
        self.update_matrix()
        self.update_transform_xp()
