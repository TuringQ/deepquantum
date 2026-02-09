"""
Photonic quantum gates
"""

import copy
import itertools
from typing import Any

import torch
from torch import nn

import deepquantum.photonic as dqp

from ..qmath import is_unitary
from .operation import Delay, Gate
from .qmath import ladder_ops


class SingleGate(Gate):
    """A base class for single-mode photonic quantum gates.

    Args:
        name (str or None, optional): The name of the gate. Default: ``None``
        inputs (Any, optional): The parameters of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """

    def __init__(
        self,
        name: str | None = None,
        inputs: Any = None,
        nmode: int = 1,
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__(
            name=name, nmode=nmode, wires=wires, cutoff=cutoff, den_mat=den_mat, noise=noise, mu=mu, sigma=sigma
        )
        assert len(self.wires) == 1, f'{self.name} must act on one mode'
        self.requires_grad = requires_grad
        self.init_para(inputs)


class DoubleGate(Gate):
    """A base class for two-mode photonic quantum gates.

    Args:
        name (str or None, optional): The name of the gate. Default: ``None``
        inputs (Any, optional): The parameters of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 2
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """

    def __init__(
        self,
        name: str | None = None,
        inputs: Any = None,
        nmode: int = 2,
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        if wires is None:
            wires = [0, 1]
        super().__init__(
            name=name, nmode=nmode, wires=wires, cutoff=cutoff, den_mat=den_mat, noise=noise, mu=mu, sigma=sigma
        )
        assert len(self.wires) == 2, f'{self.name} must act on two modes'
        self.requires_grad = requires_grad
        self.init_para(inputs)


class PhaseShift(SingleGate):
    r"""Phase Shifter.

    **Matrix Representation:**

    .. math::

        U_{\text{PS}}(\theta) = e^{i\theta}

    **Symplectic Transformation:**

    .. math::

        PS^\dagger(\theta)
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix}
        PS(\theta) =
        \begin{pmatrix}
            \cos\theta & -\sin\theta \\
            \sin\theta &  \cos\theta \\
        \end{pmatrix}
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix}

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
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
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
        inv_mode: bool = False,
    ) -> None:
        self.inv_mode = inv_mode
        super().__init__(
            name='PhaseShift',
            inputs=inputs,
            nmode=nmode,
            wires=wires,
            cutoff=cutoff,
            den_mat=den_mat,
            requires_grad=requires_grad,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        self.npara = 1

    def inputs_to_tensor(self, inputs: Any = None) -> torch.Tensor:
        """Convert inputs to torch.Tensor."""
        while isinstance(inputs, list):
            inputs = inputs[0]
        if inputs is None:
            inputs = torch.rand(1)[0] * 2 * torch.pi
        elif not isinstance(inputs, (torch.Tensor, nn.Parameter)):
            inputs = torch.tensor(inputs, dtype=torch.float)
        if self.noise:
            inputs = inputs + torch.normal(self.mu, self.sigma, size=(1,)).squeeze()
        return inputs

    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Get the local unitary matrix acting on creation operators."""
        # correspond to: U a^+ U^+ = u^T @ a^+
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

    def get_transform_xp(self, theta: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        theta = self.inputs_to_tensor(theta)
        if self.inv_mode:
            theta = -theta
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        matrix_xp = torch.stack([cos, -sin, sin, cos]).reshape(2, 2)
        vector_xp = torch.zeros(2, 1, dtype=theta.dtype, device=theta.device)
        return matrix_xp, vector_xp

    def update_transform_xp(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        matrix_xp, vector_xp = self.get_transform_xp(self.theta)
        self.matrix_xp = matrix_xp.detach()
        self.vector_xp = vector_xp.detach()
        return matrix_xp, vector_xp

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        noise = self.noise
        self.noise = False
        theta = self.inputs_to_tensor(inputs)
        self.noise = noise
        if self.requires_grad:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('theta', theta)
        self.update_matrix()
        self.update_transform_xp()

    def extra_repr(self) -> str:
        if self.inv_mode:
            theta = -self.theta
        else:
            theta = self.theta
        return f'wires={self.wires}, theta={theta.item()}'


class BeamSplitter(DoubleGate):
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
            \hat{x}_1 \\
            \hat{x}_2 \\
            \hat{p}_1 \\
            \hat{p}_2 \\
        \end{pmatrix}
        BS(\theta, \phi) =
        \begin{pmatrix}
            \mathrm{Re}(U_{\text{BS}}) & -\mathrm{Im}(U_{\text{BS}}) \\
            \mathrm{Im}(U_{\text{BS}}) &  \mathrm{Re}(U_{\text{BS}}) \\
        \end{pmatrix}
        \begin{pmatrix}
            \hat{x}_1 \\
            \hat{x}_2 \\
            \hat{p}_1 \\
            \hat{p}_2 \\
        \end{pmatrix}

    Args:
        inputs (Any, optional): The parameters of the gate (:math:`\theta` and :math:`\phi`). Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 2
        wires (List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        requires_grad (bool, optional): Whether the parameters are ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """

    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 2,
        wires: list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__(
            name='BeamSplitter',
            inputs=inputs,
            nmode=nmode,
            wires=wires,
            cutoff=cutoff,
            den_mat=den_mat,
            requires_grad=requires_grad,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        self.npara = 2

    def inputs_to_tensor(self, inputs: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert inputs to torch.Tensor."""
        if inputs is None:
            theta = torch.rand(1)[0] * 2 * torch.pi
            phi = torch.rand(1)[0] * 2 * torch.pi
        else:
            theta = inputs[0]
            phi = inputs[1]
        if not isinstance(theta, (torch.Tensor, nn.Parameter)):
            theta = torch.tensor(theta, dtype=torch.float)
        if not isinstance(phi, (torch.Tensor, nn.Parameter)):
            phi = torch.tensor(phi, dtype=torch.float)
        if self.noise:
            theta, phi = self._add_noise(theta, phi)
        return theta, phi

    def _add_noise(self, theta: torch.Tensor, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Add Gaussian noise to the parameters."""
        theta = theta + torch.normal(self.mu, self.sigma, size=(1,)).squeeze()
        phi = phi + torch.normal(self.mu, self.sigma, size=(1,)).squeeze()
        return theta, phi

    def get_matrix(self, theta: Any, phi: Any) -> torch.Tensor:
        """Get the local unitary matrix acting on creation operators."""
        # correspond to: U a^+ U^+ = u^T @ a^+
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
        sqrt = torch.sqrt(torch.arange(self.cutoff, dtype=torch.double, device=matrix.device))
        tran_mat = matrix.new_zeros([self.cutoff] * 4)
        tran_mat[0, 0, 0, 0] = 1.0
        # rank 3
        for m in range(self.cutoff):
            for n in range(self.cutoff - m):
                p = m + n
                if 0 < p < self.cutoff:
                    tran_mat[m, n, p, 0] = (
                        sqrt[m] / sqrt[p] * matrix[0, 0] * tran_mat[m - 1, n, p - 1, 0].clone()
                        + sqrt[n] / sqrt[p] * matrix[1, 0] * tran_mat[m, n - 1, p - 1, 0].clone()
                    )
        # rank 4
        for m in range(self.cutoff):
            for n in range(self.cutoff):
                for p in range(self.cutoff):
                    q = m + n - p
                    if 0 < q < self.cutoff:
                        tran_mat[m, n, p, q] = (
                            sqrt[m] / sqrt[q] * matrix[0, 1] * tran_mat[m - 1, n, p, q - 1].clone()
                            + sqrt[n] / sqrt[q] * matrix[1, 1] * tran_mat[m, n - 1, p, q - 1].clone()
                        )
        return tran_mat

    def get_transform_xp(self, theta: Any, phi: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        theta, phi = self.inputs_to_tensor([theta, phi])
        matrix = self.get_matrix(theta, phi)
        # correspond to: U a^+ U^+ = u^T @ a^+ and U^+ a U = u @ a
        # correspond to: U a U^+ = (u^*)^T @ a and U^+ a^+ U = u^* @ a^+
        matrix_xp = torch.cat(
            [torch.cat([matrix.real, -matrix.imag], dim=-1), torch.cat([matrix.imag, matrix.real], dim=-1)], dim=-2
        ).reshape(4, 4)
        vector_xp = torch.zeros(4, 1, dtype=theta.dtype, device=theta.device)
        return matrix_xp, vector_xp

    def update_transform_xp(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        matrix_xp, vector_xp = self.get_transform_xp(self.theta, self.phi)
        self.matrix_xp = matrix_xp.detach()
        self.vector_xp = vector_xp.detach()
        return matrix_xp, vector_xp

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        noise = self.noise
        self.noise = False
        theta, phi = self.inputs_to_tensor(inputs)
        self.noise = noise
        if self.requires_grad:
            self.theta = nn.Parameter(theta)
            self.phi = nn.Parameter(phi)
        else:
            self.register_buffer('theta', theta)
            self.register_buffer('phi', phi)
        self.update_matrix()
        self.update_transform_xp()

    def extra_repr(self) -> str:
        return f'wires={self.wires}, theta={self.theta.item()}, phi={self.phi.item()}'


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
            \hat{x}_1 \\
            \hat{x}_2 \\
            \hat{p}_1 \\
            \hat{p}_2 \\
        \end{pmatrix}
        MZI(\theta, \phi) =
        \begin{pmatrix}
            \mathrm{Re}(U_{\text{MZI}}) & -\mathrm{Im}(U_{\text{MZI}}) \\
            \mathrm{Im}(U_{\text{MZI}}) &  \mathrm{Re}(U_{\text{MZI}}) \\
        \end{pmatrix}
        \begin{pmatrix}
            \hat{x}_1 \\
            \hat{x}_2 \\
            \hat{p}_1 \\
            \hat{p}_2 \\
        \end{pmatrix}

    Args:
        inputs (Any, optional): The parameters of the gate (:math:`\theta` and :math:`\phi`). Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 2
        wires (List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        phi_first (bool, optional): Whether :math:`\phi` is the first phase shifter. Default: ``True``
        requires_grad (bool, optional): Whether the parameters are ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """

    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 2,
        wires: list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        phi_first: bool = True,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        self.phi_first = phi_first
        super().__init__(
            inputs=inputs,
            nmode=nmode,
            wires=wires,
            cutoff=cutoff,
            den_mat=den_mat,
            requires_grad=requires_grad,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        self.name = 'MZI'

    def get_matrix(self, theta: Any, phi: Any) -> torch.Tensor:
        """Get the local unitary matrix acting on creation operators."""
        # correspond to: U a^+ U^+ = u^T @ a^+
        theta, phi = self.inputs_to_tensor([theta, phi])
        cos = torch.cos(theta / 2)
        sin = torch.sin(theta / 2)
        e_it = torch.exp(1j * theta / 2)
        e_ip = torch.exp(1j * phi)
        mat = 1j * e_it * torch.stack([e_ip * sin, cos, e_ip * cos, -sin]).reshape(2, 2)
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
            \hat{x}_1 \\
            \hat{x}_2 \\
            \hat{p}_1 \\
            \hat{p}_2 \\
        \end{pmatrix}
        BS(\theta) =
        \begin{pmatrix}
            \mathrm{Re}(U_{\text{BS}}) & -\mathrm{Im}(U_{\text{BS}}) \\
            \mathrm{Im}(U_{\text{BS}}) &  \mathrm{Re}(U_{\text{BS}}) \\
        \end{pmatrix}
        \begin{pmatrix}
            \hat{x}_1 \\
            \hat{x}_2 \\
            \hat{p}_1 \\
            \hat{p}_2 \\
        \end{pmatrix}

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 2
        wires (List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
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
        wires: list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__(
            inputs=inputs,
            nmode=nmode,
            wires=wires,
            cutoff=cutoff,
            den_mat=den_mat,
            requires_grad=requires_grad,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        self.npara = 1

    def _add_noise(self, theta: torch.Tensor, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Add Gaussian noise to the parameters."""
        theta = theta + torch.normal(self.mu, self.sigma, size=(1,)).squeeze()
        return theta, phi

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        if inputs is None:
            inputs = torch.rand(1)[0] * 2 * torch.pi
        noise = self.noise
        self.noise = False
        theta, phi = self.inputs_to_tensor([inputs, torch.pi / 2])
        phi = phi.to(theta.dtype).to(theta.device)
        self.noise = noise
        if self.requires_grad:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('theta', theta)
        self.register_buffer('phi', phi)
        self.update_matrix()
        self.update_transform_xp()


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
            \hat{x}_1 \\
            \hat{x}_2 \\
            \hat{p}_1 \\
            \hat{p}_2 \\
        \end{pmatrix}
        BS(\phi) =
        \begin{pmatrix}
            \mathrm{Re}(U_{\text{BS}}) & -\mathrm{Im}(U_{\text{BS}}) \\
            \mathrm{Im}(U_{\text{BS}}) &  \mathrm{Re}(U_{\text{BS}}) \\
        \end{pmatrix}
        \begin{pmatrix}
            \hat{x}_1 \\
            \hat{x}_2 \\
            \hat{p}_1 \\
            \hat{p}_2 \\
        \end{pmatrix}

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 2
        wires (List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
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
        wires: list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__(
            inputs=inputs,
            nmode=nmode,
            wires=wires,
            cutoff=cutoff,
            den_mat=den_mat,
            requires_grad=requires_grad,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        self.npara = 1

    def _add_noise(self, theta: torch.Tensor, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Add Gaussian noise to the parameters."""
        phi = phi + torch.normal(self.mu, self.sigma, size=(1,)).squeeze()
        return theta, phi

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        if inputs is None:
            inputs = torch.rand(1)[0] * 2 * torch.pi
        noise = self.noise
        self.noise = False
        theta, phi = self.inputs_to_tensor([torch.pi / 4, inputs])
        theta = theta.to(phi.dtype).to(phi.device)
        self.noise = noise
        if self.requires_grad:
            self.phi = nn.Parameter(phi)
        else:
            self.register_buffer('phi', phi)
        self.register_buffer('theta', theta)
        self.update_matrix()
        self.update_transform_xp()


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
            \hat{x}_1 \\
            \hat{x}_2 \\
            \hat{p}_1 \\
            \hat{p}_2 \\
        \end{pmatrix}
        BS(\theta) =
        \begin{pmatrix}
            \mathrm{Re}(U_{\text{BS}}) & -\mathrm{Im}(U_{\text{BS}}) \\
            \mathrm{Im}(U_{\text{BS}}) &  \mathrm{Re}(U_{\text{BS}}) \\
        \end{pmatrix}
        \begin{pmatrix}
            \hat{x}_1 \\
            \hat{x}_2 \\
            \hat{p}_1 \\
            \hat{p}_2 \\
        \end{pmatrix}

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 2
        wires (List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
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
        wires: list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        convention: str = 'rx',
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        self.convention = convention
        super().__init__(
            inputs=inputs,
            nmode=nmode,
            wires=wires,
            cutoff=cutoff,
            den_mat=den_mat,
            requires_grad=requires_grad,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
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
            inputs = inputs + torch.normal(self.mu, self.sigma, size=(1,)).squeeze()
        return inputs

    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Get the local unitary matrix acting on creation operators."""
        # correspond to: U a^+ U^+ = u^T @ a^+
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

    def get_transform_xp(self, theta: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        theta = self.inputs_to_tensor(theta)
        matrix = self.get_matrix(theta)
        # correspond to: U a^+ U^+ = u^T @ a^+ and U^+ a U = u @ a
        # correspond to: U a U^+ = (u^*)^T @ a and U^+ a^+ U = u^* @ a^+
        matrix_xp = torch.cat(
            [torch.cat([matrix.real, -matrix.imag], dim=-1), torch.cat([matrix.imag, matrix.real], dim=-1)], dim=-2
        ).reshape(4, 4)
        vector_xp = torch.zeros(4, 1, dtype=theta.dtype, device=theta.device)
        return matrix_xp, vector_xp

    def update_transform_xp(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        matrix_xp, vector_xp = self.get_transform_xp(self.theta)
        self.matrix_xp = matrix_xp.detach()
        self.vector_xp = vector_xp.detach()
        return matrix_xp, vector_xp

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        noise = self.noise
        self.noise = False
        theta = self.inputs_to_tensor(inputs)
        self.noise = noise
        if self.requires_grad:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('theta', theta)
        self.update_matrix()
        self.update_transform_xp()

    def extra_repr(self) -> str:
        return f'wires={self.wires}, theta={self.theta.item()}, convention={self.convention}'


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
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        name (str, optional): The name of the gate. Default: ``'UAnyGate'``
    """

    def __init__(
        self,
        unitary: Any,
        nmode: int = 1,
        wires: list[int] | None = None,
        minmax: list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        name: str = 'UAnyGate',
    ) -> None:
        self.nmode = nmode
        if wires is None:
            if minmax is None:
                minmax = [0, nmode - 1]
            self._check_minmax(minmax)
            wires = list(range(minmax[0], minmax[1] + 1))
        super().__init__(name=name, nmode=nmode, wires=wires, cutoff=cutoff, den_mat=den_mat, noise=False)
        self.minmax = [min(self.wires), max(self.wires)]
        if not isinstance(unitary, torch.Tensor):
            unitary = torch.tensor(unitary, dtype=torch.cfloat).reshape(-1, len(self.wires))
        assert unitary.dtype in (torch.cfloat, torch.cdouble)
        assert unitary.shape[0] == len(self.wires), 'Please check wires'
        assert is_unitary(unitary), 'Please check the unitary matrix'
        self.register_buffer('matrix', unitary)
        self.register_buffer('matrix_state', None)

    def to(self, arg: Any) -> 'UAnyGate':
        """Set dtype or device of the ``UAnyGate``."""
        if arg == torch.float:
            self.matrix = self.matrix.to(torch.cfloat)
            if self.matrix_state is not None:
                self.matrix_state = self.matrix_state.to(torch.cfloat)
        elif arg == torch.double:
            self.matrix = self.matrix.to(torch.cdouble)
            if self.matrix_state is not None:
                self.matrix_state = self.matrix_state.to(torch.cdouble)
        else:
            super().to(arg)
        return self

    def get_matrix_state(self, matrix: torch.Tensor) -> torch.Tensor:
        """Get the local transformation matrix acting on Fock state tensors.

        See https://arxiv.org/pdf/2004.11002.pdf Eq.(71)
        """
        nt = len(self.wires)
        sqrt = torch.sqrt(torch.arange(self.cutoff, dtype=torch.double, device=matrix.device))
        tran_mat = matrix.new_zeros([self.cutoff] * 2 * nt)
        tran_mat[tuple([0] * 2 * nt)] = 1.0
        for rank in range(nt + 1, 2 * nt + 1):
            col_num = rank - nt - 1
            matrix_j = matrix[:, col_num]
            # all combinations of the first `rank-1` modes
            combs = itertools.product(range(self.cutoff), repeat=rank - 1)
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

    def get_transform_xp(self, matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        # correspond to: U a^+ U^+ = u^T @ a^+ and U^+ a U = u @ a
        # correspond to: U a U^+ = (u^*)^T @ a and U^+ a^+ U = u^* @ a^+
        n = len(self.wires)
        matrix_xp = torch.cat(
            [torch.cat([matrix.real, -matrix.imag], dim=-1), torch.cat([matrix.imag, matrix.real], dim=-1)], dim=-2
        ).reshape(2 * n, 2 * n)
        vector_xp = torch.zeros(2 * n, 1, dtype=matrix.real.dtype, device=matrix.real.device)
        return matrix_xp, vector_xp

    def update_transform_xp(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        matrix_xp, vector_xp = self.get_transform_xp(self.matrix)
        self.matrix_xp = matrix_xp.detach()
        self.vector_xp = vector_xp.detach()
        return matrix_xp, vector_xp


class Squeezing(SingleGate):
    r"""Squeezing gate.

    **Symplectic Transformation:**

    .. math::

        S^\dagger(r, \theta)
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix}
        S(r, \theta) =
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
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        requires_grad (bool, optional): Whether the parameters are ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """

    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 1,
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__(
            name='Squeezing',
            inputs=inputs,
            nmode=nmode,
            wires=wires,
            cutoff=cutoff,
            den_mat=den_mat,
            requires_grad=requires_grad,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        self.npara = 2

    def inputs_to_tensor(self, inputs: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
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
            r = r + torch.normal(self.mu, self.sigma, size=(1,)).squeeze()
            theta = theta + torch.normal(self.mu, self.sigma, size=(1,)).squeeze()
        return r, theta

    def get_matrix(self, r: Any, theta: Any) -> torch.Tensor:
        """Get the local symplectic matrix acting on annihilation and creation operators."""
        # correspond to: U^+ (a a^+) U = s @ (a a^+)
        r, theta = self.inputs_to_tensor([r, theta])
        ch = torch.cosh(r)
        sh = torch.sinh(r)
        e_it = torch.exp(1j * theta)
        e_m_it = torch.exp(-1j * theta)
        return torch.stack([ch, -e_it * sh, -e_m_it * sh, ch]).reshape(2, 2)

    def update_matrix(self) -> torch.Tensor:
        """Update the local symplectic matrix acting on annihilation and creation operators."""
        matrix = self.get_matrix(self.r, self.theta)
        self.matrix = matrix.detach()
        return matrix

    def get_matrix_state(self, r: Any, theta: Any) -> torch.Tensor:
        """Get the local transformation matrix acting on Fock state tensors.

        See https://arxiv.org/pdf/2004.11002.pdf Eq.(51) and Eq.(52)
        """
        r, theta = self.inputs_to_tensor([r, theta])
        sqrt = torch.sqrt(torch.arange(self.cutoff, dtype=r.dtype, device=r.device))
        sech = 1 / torch.cosh(r)
        e_it_tanh = torch.exp(1j * theta) * torch.tanh(r)
        e_m_it_tanh = torch.exp(-1j * theta) * torch.tanh(r)
        tran_mat = e_it_tanh.new_zeros([self.cutoff] * 2)
        tran_mat[0, 0] = torch.sqrt(sech)
        # rank 1
        for m in range(1, self.cutoff - 1, 2):
            tran_mat[m + 1, 0] = -sqrt[m] / sqrt[m + 1] * e_it_tanh * tran_mat[m - 1, 0].clone()
        # rank 2
        for m in range(self.cutoff):
            for n in range(self.cutoff - 1):
                if (m + n) % 2 == 1:
                    tran_mat[m, n + 1] = (
                        sqrt[m] / sqrt[n + 1] * sech * tran_mat[m - 1, n].clone()
                        + sqrt[n] / sqrt[n + 1] * e_m_it_tanh * tran_mat[m, n - 1].clone()
                    )
        return tran_mat

    def update_matrix_state(self) -> torch.Tensor:
        """Update the local transformation matrix acting on Fock state tensors."""
        return self.get_matrix_state(self.r, self.theta)

    def get_transform_xp(self, r: Any, theta: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        r, theta = self.inputs_to_tensor([r, theta])
        ch = torch.cosh(r)
        sh = torch.sinh(r)
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        matrix_xp = torch.stack([ch - sh * cos, -sh * sin, -sh * sin, ch + sh * cos]).reshape(2, 2)
        vector_xp = torch.zeros(2, 1, dtype=r.dtype, device=r.device)
        return matrix_xp, vector_xp

    def update_transform_xp(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        matrix_xp, vector_xp = self.get_transform_xp(self.r, self.theta)
        self.matrix_xp = matrix_xp.detach()
        self.vector_xp = vector_xp.detach()
        return matrix_xp, vector_xp

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        noise = self.noise
        self.noise = False
        r, theta = self.inputs_to_tensor(inputs)
        self.noise = noise
        if self.requires_grad:
            self.r = nn.Parameter(r)
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('r', r)
            self.register_buffer('theta', theta)
        self.update_matrix()
        self.update_transform_xp()

    def extra_repr(self) -> str:
        return f'wires={self.wires}, r={self.r.item()}, theta={self.theta.item()}'


class Squeezing2(DoubleGate):
    r"""Two-mode Squeezing gate.

    **Symplectic Transformation:**

    .. math::

        S_2^\dagger(r, \theta)
        \begin{pmatrix}
            \hat{x}_1 \\
            \hat{x}_2 \\
            \hat{p}_1 \\
            \hat{p}_2 \\
        \end{pmatrix}
        S_2(r, \theta) =
        \begin{pmatrix}
            \cosh r & \cos\theta\sinh r & 0 & \sin\theta\sinh r  \\
            \cos\theta\sinh r & \cosh r & \sin\theta\sinh r & 0  \\
            0 & \sin\theta\sinh r & \cosh r & -\cos\theta\sinh r \\
            \sin\theta\sinh r & 0 & -\cos\theta\sinh r & \cosh r \\
        \end{pmatrix}
        \begin{pmatrix}
            \hat{x}_1 \\
            \hat{x}_2 \\
            \hat{p}_1 \\
            \hat{p}_2 \\
        \end{pmatrix}

    Args:
        inputs (Any, optional): The parameters of the gate (:math:`r` and :math:`\theta`). Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 2
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        requires_grad (bool, optional): Whether the parameters are ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """

    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 2,
        wires: list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__(
            name='Squeezing2',
            inputs=inputs,
            nmode=nmode,
            wires=wires,
            cutoff=cutoff,
            den_mat=den_mat,
            requires_grad=requires_grad,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        self.npara = 2

    def inputs_to_tensor(self, inputs: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
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
            r = r + torch.normal(self.mu, self.sigma, size=(1,)).squeeze()
            theta = theta + torch.normal(self.mu, self.sigma, size=(1,)).squeeze()
        return r, theta

    def get_matrix(self, r: Any, theta: Any) -> torch.Tensor:
        """Get the local symplectic matrix acting on annihilation and creation operators."""
        # correspond to: U^+ (a a^+) U = s @ (a a^+)
        r, theta = self.inputs_to_tensor([r, theta])
        ch = torch.cosh(r)
        sh = torch.sinh(r)
        e_sh = torch.exp(1j * theta) * sh
        e_m_sh = torch.exp(-1j * theta) * sh
        m1 = torch.stack([ch, ch, ch, ch]).reshape(-1).diag_embed().reshape(4, 4)
        m2 = torch.stack([e_sh, e_sh, e_m_sh, e_m_sh]).reshape(-1).diag_embed().fliplr().reshape(4, 4)
        return m1 + m2

    def update_matrix(self) -> torch.Tensor:
        """Update the local symplectic matrix acting on annihilation and creation operators."""
        matrix = self.get_matrix(self.r, self.theta)
        self.matrix = matrix.detach()
        return matrix

    def get_matrix_state(self, r: Any, theta: Any) -> torch.Tensor:
        """Get the local transformation matrix acting on Fock state tensors.

        See https://arxiv.org/pdf/2004.11002.pdf Eq.(64-67)
        """
        r, theta = self.inputs_to_tensor([r, theta])
        sqrt = torch.sqrt(torch.arange(self.cutoff, dtype=r.dtype, device=r.device))
        sech = 1 / torch.cosh(r)
        e_it_tanh = torch.exp(1j * theta) * torch.tanh(r)
        e_m_it_tanh = torch.exp(-1j * theta) * torch.tanh(r)
        tran_mat = e_it_tanh.new_zeros([self.cutoff] * 4)
        tran_mat[0, 0, 0, 0] = sech
        # rank 2
        for n in range(1, self.cutoff):
            tran_mat[n, n, 0, 0] = e_it_tanh * tran_mat[n - 1, n - 1, 0, 0].clone()
        # rank 3
        for m in range(1, self.cutoff):
            for n in range(m):
                p = m - n
                if p < self.cutoff:
                    tran_mat[m, n, p, 0] = sech * sqrt[m] / sqrt[p] * tran_mat[m - 1, n, p - 1, 0].clone()
        # rank 4
        for m in range(self.cutoff):
            for n in range(self.cutoff):
                for p in range(self.cutoff):
                    q = p - (m - n)
                    if 0 < q < self.cutoff:
                        tran_mat[m, n, p, q] = (
                            sech * sqrt[n] / sqrt[q] * tran_mat[m, n - 1, p, q - 1].clone()
                            - e_m_it_tanh * sqrt[p] / sqrt[q] * tran_mat[m, n, p - 1, q - 1].clone()
                        )
        return tran_mat

    def update_matrix_state(self) -> torch.Tensor:
        """Update the local transformation matrix acting on Fock state tensors."""
        return self.get_matrix_state(self.r, self.theta)

    def get_transform_xp(self, r: Any, theta: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        r, theta = self.inputs_to_tensor([r, theta])
        ch = torch.cosh(r)
        sh = torch.sinh(r)
        cos_sh = torch.cos(theta) * sh
        sin_sh = torch.sin(theta) * sh
        m1 = torch.stack([ch, ch, ch, ch]).reshape(-1).diag_embed().reshape(4, 4)
        m2 = torch.stack([sin_sh, sin_sh, sin_sh, sin_sh]).reshape(-1).diag_embed().fliplr().reshape(4, 4)
        m3 = (torch.eye(2, dtype=r.dtype, device=r.device) * cos_sh).fliplr().reshape(2, 2)
        matrix_xp = m1 + m2 + torch.block_diag(m3, -m3)
        vector_xp = torch.zeros(4, 1, dtype=r.dtype, device=r.device)
        return matrix_xp, vector_xp

    def update_transform_xp(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        matrix_xp, vector_xp = self.get_transform_xp(self.r, self.theta)
        self.matrix_xp = matrix_xp.detach()
        self.vector_xp = vector_xp.detach()
        return matrix_xp, vector_xp

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        noise = self.noise
        self.noise = False
        r, theta = self.inputs_to_tensor(inputs)
        self.noise = noise
        if self.requires_grad:
            self.r = nn.Parameter(r)
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('r', r)
            self.register_buffer('theta', theta)
        self.update_matrix()
        self.update_transform_xp()

    def extra_repr(self) -> str:
        return f'wires={self.wires}, r={self.r.item()}, theta={self.theta.item()}'


class Displacement(SingleGate):
    r"""Displacement gate.

    **Symplectic Transformation:**

    .. math::

        D^\dagger(r, \theta)
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix}
        D(r, \theta) =
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
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        requires_grad (bool, optional): Whether the parameters are ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """

    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 1,
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__(
            name='Displacement',
            inputs=inputs,
            nmode=nmode,
            wires=wires,
            cutoff=cutoff,
            den_mat=den_mat,
            requires_grad=requires_grad,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        self.npara = 2

    def inputs_to_tensor(self, inputs: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
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
            r, theta = self._add_noise(r, theta)
        return r, theta

    def _add_noise(self, r: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Add Gaussian noise to the parameters."""
        r = r + torch.normal(self.mu, self.sigma, size=(1,)).squeeze()
        theta = theta + torch.normal(self.mu, self.sigma, size=(1,)).squeeze()
        return r, theta

    def get_matrix(self, r: Any, theta: Any) -> torch.Tensor:
        """Get the local symplectic unitary matrix acting on annihilation and creation operators."""
        # correspond to: U^+ (a a^+) U = s @ (a a^+)
        r, theta = self.inputs_to_tensor([r, theta])
        return torch.eye(2, dtype=r.dtype, device=r.device) + 0j

    def update_matrix(self) -> torch.Tensor:
        """Update the local symplectic unitary matrix acting on annihilation and creation operators."""
        matrix = self.get_matrix(self.r, self.theta)
        self.matrix = matrix.detach()
        return matrix

    def get_matrix_state(self, r: Any, theta: Any) -> torch.Tensor:
        """Get the local transformation matrix acting on Fock state tensors.

        See https://arxiv.org/pdf/2004.11002.pdf Eq.(57) and Eq.(58)
        """
        r, theta = self.inputs_to_tensor([r, theta])
        sqrt = torch.sqrt(torch.arange(self.cutoff, dtype=r.dtype, device=r.device))
        alpha = r * torch.exp(1j * theta)
        alpha_c = r * torch.exp(-1j * theta)
        tran_mat = alpha.new_zeros([self.cutoff] * 2)
        tran_mat[0, 0] = torch.exp(-(r**2) / 2)
        # rank 1
        for m in range(self.cutoff - 1):
            tran_mat[m + 1, 0] = alpha / sqrt[m + 1] * tran_mat[m, 0].clone()
        # rank 2
        for m in range(self.cutoff):
            for n in range(self.cutoff - 1):
                tran_mat[m, n + 1] = (
                    -alpha_c / sqrt[n + 1] * tran_mat[m, n].clone() + sqrt[m] / sqrt[n + 1] * tran_mat[m - 1, n].clone()
                )
        return tran_mat

    def update_matrix_state(self) -> torch.Tensor:
        """Update the local transformation matrix acting on Fock state tensors."""
        return self.get_matrix_state(self.r, self.theta)

    def get_transform_xp(self, r: Any, theta: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        r, theta = self.inputs_to_tensor([r, theta])
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        matrix_xp = torch.eye(2, dtype=r.dtype, device=r.device)
        vector_xp = torch.stack([r * cos, r * sin]).reshape(2, 1) * dqp.hbar**0.5 / dqp.kappa
        return matrix_xp, vector_xp

    def update_transform_xp(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        matrix_xp, vector_xp = self.get_transform_xp(self.r, self.theta)
        self.matrix_xp = matrix_xp.detach()
        self.vector_xp = vector_xp.detach()
        return matrix_xp, vector_xp

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        noise = self.noise
        self.noise = False
        r, theta = self.inputs_to_tensor(inputs)
        self.noise = noise
        if self.requires_grad:
            self.r = nn.Parameter(r)
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('r', r)
            self.register_buffer('theta', theta)
        self.update_matrix()
        self.update_transform_xp()

    def extra_repr(self) -> str:
        return f'wires={self.wires}, r={self.r.item()}, theta={self.theta.item()}'


class DisplacementPosition(Displacement):
    r"""Position Displacement gate.

    **Symplectic Transformation:**

    .. math::
        X^\dagger(x)
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix}
        X(x) =
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix} +
        \begin{pmatrix}
            x \\
            0 \\
        \end{pmatrix}

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
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
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__(
            inputs=inputs,
            nmode=nmode,
            wires=wires,
            cutoff=cutoff,
            den_mat=den_mat,
            requires_grad=requires_grad,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        self.name = 'DisplacementPosition'
        self.npara = 1

    def _add_noise(self, r: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Add Gaussian noise to the parameters."""
        r = r + torch.normal(self.mu, self.sigma, size=(1,)).squeeze()
        return r, theta

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        if inputs is None:
            inputs = torch.rand(1)[0] * 2 - 1
        inputs = inputs * dqp.kappa / dqp.hbar**0.5
        noise = self.noise
        self.noise = False
        r, theta = self.inputs_to_tensor([inputs, 0])
        theta = theta.to(r.dtype).to(r.device)
        self.noise = noise
        if self.requires_grad:
            self.r = nn.Parameter(r)
        else:
            self.register_buffer('r', r)
        self.register_buffer('theta', theta)
        self.update_matrix()
        self.update_transform_xp()


class DisplacementMomentum(Displacement):
    r"""Momentum Displacement gate.

    **Symplectic Transformation:**

    .. math::
        Z^\dagger(p)
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix}
        Z(p) =
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix} +
        \begin{pmatrix}
            0 \\
            p \\
        \end{pmatrix}

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
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
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__(
            inputs=inputs,
            nmode=nmode,
            wires=wires,
            cutoff=cutoff,
            den_mat=den_mat,
            requires_grad=requires_grad,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        self.name = 'DisplacementMomentum'
        self.npara = 1

    def _add_noise(self, r: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Add Gaussian noise to the parameters."""
        r = r + torch.normal(self.mu, self.sigma, size=(1,)).squeeze()
        return r, theta

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        if inputs is None:
            inputs = torch.rand(1)[0] * 2 - 1
        inputs = inputs * dqp.kappa / dqp.hbar**0.5
        noise = self.noise
        self.noise = False
        r, theta = self.inputs_to_tensor([inputs, torch.pi / 2])
        theta = theta.to(r.dtype).to(r.device)
        self.noise = noise
        if self.requires_grad:
            self.r = nn.Parameter(r)
        else:
            self.register_buffer('r', r)
        self.register_buffer('theta', theta)
        self.update_matrix()
        self.update_transform_xp()


class QuadraticPhase(SingleGate):
    r"""Quadratic phase gate.

    **Operator Representation:**

    .. math::

        QP(s) = e^{i \frac{s}{2 \hbar} \hat{x}^2}

    **Symplectic Transformation:**

    .. math::

        QP^\dagger(s)
        \begin{pmatrix}
            \hat{a} \\
            \hat{a}^\dagger \\
        \end{pmatrix}
        QP(s) =
        \begin{pmatrix}
            1 + i \frac{s}{2} & i \frac{s}{2}     \\
            -i \frac{s}{2}    & 1 - i \frac{s}{2} \\
        \end{pmatrix}
        \begin{pmatrix}
            \hat{a} \\
            \hat{a}^\dagger \\
        \end{pmatrix}

    .. math::

        QP^\dagger(s)
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix}
        QP(s) =
        \begin{pmatrix}
            1 & 0 \\
            s & 1 \\
        \end{pmatrix}
        \begin{pmatrix}
            \hat{x} \\
            \hat{p} \\
        \end{pmatrix}

    **Decomposition Representation:**

    .. math::

        QP(s) = PS(\theta) S(r e^{i \phi})

    where :math:`\cosh(r) = \sqrt{1 + (\frac{s}{2})^2}, \quad
    \tan(\theta) = \frac{s}{2}, \quad
    \phi = -\sign(s)\frac{\pi}{2} - \theta`.

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        requires_grad (bool, optional): Whether the parameters are ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """

    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 1,
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__(
            name='QuadraticPhase',
            inputs=inputs,
            nmode=nmode,
            wires=wires,
            cutoff=cutoff,
            den_mat=den_mat,
            requires_grad=requires_grad,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        self.npara = 1

    def inputs_to_tensor(self, inputs: Any = None) -> torch.Tensor:
        """Convert inputs to torch.Tensor."""
        while isinstance(inputs, list):
            inputs = inputs[0]
        if inputs is None:
            inputs = torch.rand(1)[0]
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float)
        if self.noise:
            inputs = inputs + torch.normal(self.mu, self.sigma, size=(1,)).squeeze()
        return inputs

    def get_matrix(self, s: Any) -> torch.Tensor:
        """Get the local symplectic matrix acting on annihilation and creation operators."""
        # correspond to: U^+ (a a^+) U = s @ (a a^+)
        s = self.inputs_to_tensor(s)
        x = 1j * s / 2
        return torch.stack([1 + x, x, -x, 1 - x]).reshape(2, 2)

    def update_matrix(self) -> torch.Tensor:
        """Update the local symplectic matrix acting on annihilation and creation operators."""
        matrix = self.get_matrix(self.s)
        self.matrix = matrix.detach()
        return matrix

    def get_matrix_state(self, s: Any) -> torch.Tensor:
        """Get the local transformation matrix acting on Fock state tensors."""
        s = self.inputs_to_tensor(s)
        r = torch.arccosh((1 + s**2 / 4) ** 0.5)
        theta = torch.arctan(s / 2)
        phi = -torch.sign(s) * torch.pi / 2 - theta
        # noise already in s
        mat_s = Squeezing([r, phi], cutoff=self.cutoff).update_matrix_state()
        mat_ps = PhaseShift(theta, cutoff=self.cutoff).update_matrix_state()
        return mat_ps @ mat_s

    def update_matrix_state(self) -> torch.Tensor:
        """Update the local transformation matrix acting on Fock state tensors."""
        return self.get_matrix_state(self.s)

    def get_transform_xp(self, s: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        s = self.inputs_to_tensor(s).reshape(-1)
        one = s.new_ones(1)
        zero = s.new_zeros(1)
        matrix_xp = torch.stack([one, zero, s, one]).reshape(2, 2)
        vector_xp = s.new_zeros(2, 1)
        return matrix_xp, vector_xp

    def update_transform_xp(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        matrix_xp, vector_xp = self.get_transform_xp(self.s)
        self.matrix_xp = matrix_xp.detach()
        self.vector_xp = vector_xp.detach()
        return matrix_xp, vector_xp

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        noise = self.noise
        self.noise = False
        s = self.inputs_to_tensor(inputs)
        self.noise = noise
        if self.requires_grad:
            self.s = nn.Parameter(s)
        else:
            self.register_buffer('s', s)
        self.update_matrix()
        self.update_transform_xp()

    def extra_repr(self) -> str:
        return f'wires={self.wires}, s={self.s.item()}'


class ControlledX(DoubleGate):
    r"""Controlled-X gate.

    **Operator Representation:**

    .. math::

        CX(s) = e^{-i \frac{s}{\hbar} \hat{x}_1 \hat{p}_2}

    **Symplectic Transformation:**

    .. math::

        CX^\dagger(s)
        \begin{pmatrix}
            \hat{a}_1 \\
            \hat{a}_2 \\
            \hat{a}_1^\dagger \\
            \hat{a}_2^\dagger \\
        \end{pmatrix}
        CX(s) =
        \begin{pmatrix}
            1           & -\frac{s}{2} & 0           & \frac{s}{2}  \\
            \frac{s}{2} & 1            & \frac{s}{2} & 0            \\
            0           & \frac{s}{2}  & 1           & -\frac{s}{2} \\
            \frac{s}{2} & 0            & \frac{s}{2} & 1            \\
        \end{pmatrix}
        \begin{pmatrix}
            \hat{a}_1 \\
            \hat{a}_2 \\
            \hat{a}_1^\dagger \\
            \hat{a}_2^\dagger \\
        \end{pmatrix}

    .. math::

        CX^\dagger(s)
        \begin{pmatrix}
            \hat{x}_1 \\
            \hat{x}_2 \\
            \hat{p}_1 \\
            \hat{p}_2 \\
        \end{pmatrix}
        CX(s) =
        \begin{pmatrix}
            1 & 0 & 0 & 0  \\
            s & 1 & 0 & 0  \\
            0 & 0 & 1 & -s \\
            0 & 0 & 0 & 1  \\
        \end{pmatrix}
        \begin{pmatrix}
            \hat{x}_1 \\
            \hat{x}_2 \\
            \hat{p}_1 \\
            \hat{p}_2 \\
        \end{pmatrix}

    **Decomposition Representation:**

    .. math::

        CX(s) = BS(\frac{\pi}{2} + \theta, 0) \left(S(r, 0) \otimes S(-r, 0)\right) BS(\theta, 0)

    where :math:`\sin(2 \theta) = -\frac{1}{\cosh r}, \quad
    \cos(2 \theta) = -\tanh r, \quad
    \sinh(r) = -\frac{s}{2}`.

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 2
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        requires_grad (bool, optional): Whether the parameters are ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """

    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 2,
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__(
            name='ControlledX',
            inputs=inputs,
            nmode=nmode,
            wires=wires,
            cutoff=cutoff,
            den_mat=den_mat,
            requires_grad=requires_grad,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        self.npara = 1

    def inputs_to_tensor(self, inputs: Any = None) -> torch.Tensor:
        """Convert inputs to torch.Tensor."""
        while isinstance(inputs, list):
            inputs = inputs[0]
        if inputs is None:
            inputs = torch.rand(1)[0]
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float)
        if self.noise:
            inputs = inputs + torch.normal(self.mu, self.sigma, size=(1,)).squeeze()
        return inputs

    def get_matrix(self, s: Any) -> torch.Tensor:
        """Get the local symplectic matrix acting on annihilation and creation operators."""
        # correspond to: U^+ (a a^+) U = s @ (a a^+)
        s = self.inputs_to_tensor(s).reshape(-1)
        one = s.new_ones(1)
        zero = s.new_zeros(1)
        x = s / 2
        mat = torch.stack([one, -x, zero, x, x, one, x, zero, zero, x, one, -x, x, zero, x, one]).reshape(4, 4) + 0j
        return mat

    def update_matrix(self) -> torch.Tensor:
        """Update the local symplectic matrix acting on annihilation and creation operators."""
        matrix = self.get_matrix(self.s)
        self.matrix = matrix.detach()
        return matrix

    def get_matrix_state(self, s: Any) -> torch.Tensor:
        """Get the local transformation matrix acting on Fock state tensors."""
        s = self.inputs_to_tensor(s).reshape(-1)
        zero = s.new_zeros(1)
        r = torch.arcsinh(-s / 2)
        theta = torch.atan2(-1 / torch.cosh(r), -torch.tanh(r)) / 2
        # noise already in s
        mat_bs1 = BeamSplitter([theta, zero], cutoff=self.cutoff).update_matrix_state()
        mat_s1 = Squeezing([r, zero], cutoff=self.cutoff).update_matrix_state()
        mat_s2 = Squeezing([-r, zero], cutoff=self.cutoff).update_matrix_state()
        mat_bs2 = BeamSplitter([theta + torch.pi / 2, zero], cutoff=self.cutoff).update_matrix_state()
        mat = torch.einsum('abcd,ce,df,efgh->abgh', mat_bs2, mat_s1, mat_s2, mat_bs1)
        return mat

    def update_matrix_state(self) -> torch.Tensor:
        """Update the local transformation matrix acting on Fock state tensors."""
        return self.get_matrix_state(self.s)

    def get_transform_xp(self, s: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        s = self.inputs_to_tensor(s).reshape(-1)
        one = s.new_ones(1)
        zero = s.new_zeros(1)
        mat1 = torch.stack([one, zero, s, one]).reshape(2, 2)
        mat2 = torch.stack([one, -s, zero, one]).reshape(2, 2)
        matrix_xp = torch.block_diag(mat1, mat2)
        vector_xp = s.new_zeros(4, 1)
        return matrix_xp, vector_xp

    def update_transform_xp(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        matrix_xp, vector_xp = self.get_transform_xp(self.s)
        self.matrix_xp = matrix_xp.detach()
        self.vector_xp = vector_xp.detach()
        return matrix_xp, vector_xp

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        noise = self.noise
        self.noise = False
        s = self.inputs_to_tensor(inputs)
        self.noise = noise
        if self.requires_grad:
            self.s = nn.Parameter(s)
        else:
            self.register_buffer('s', s)
        self.update_matrix()
        self.update_transform_xp()

    def extra_repr(self) -> str:
        return f'wires={self.wires}, s={self.s.item()}'


class ControlledZ(DoubleGate):
    r"""Controlled-Z gate.

    **Operator Representation:**

    .. math::

        CZ(s) = e^{-i \frac{s}{\hbar} \hat{x}_1 \hat{x}_2}

    **Symplectic Transformation:**

    .. math::

        CZ^\dagger(s)
        \begin{pmatrix}
            \hat{a}_1 \\
            \hat{a}_2 \\
            \hat{a}_1^\dagger \\
            \hat{a}_2^\dagger \\
        \end{pmatrix}
        CZ(s) =
        \begin{pmatrix}
            1              & i \frac{s}{2}  & 0              & i \frac{s}{2}  \\
            i \frac{s}{2}  & 1              & i \frac{s}{2}  & 0              \\
            0              & -i \frac{s}{2} & 1              & -i \frac{s}{2} \\
            -i \frac{s}{2} & 0              & -i \frac{s}{2} & 1              \\
        \end{pmatrix}
        \begin{pmatrix}
            \hat{a}_1 \\
            \hat{a}_2 \\
            \hat{a}_1^\dagger \\
            \hat{a}_2^\dagger \\
        \end{pmatrix}

    .. math::

        CZ^\dagger(s)
        \begin{pmatrix}
            \hat{x}_1 \\
            \hat{x}_2 \\
            \hat{p}_1 \\
            \hat{p}_2 \\
        \end{pmatrix}
        CZ(s) =
        \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & s & 1 & 0 \\
            s & 0 & 0 & 1 \\
        \end{pmatrix}
        \begin{pmatrix}
            \hat{x}_1 \\
            \hat{x}_2 \\
            \hat{p}_1 \\
            \hat{p}_2 \\
        \end{pmatrix}

    **Decomposition Representation:**

    .. math::

        CZ(s) = PS_2(\pi/2) CX(s) PS_2^\dagger(\pi/2)

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 2
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        requires_grad (bool, optional): Whether the parameters are ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """

    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 2,
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__(
            name='ControlledZ',
            inputs=inputs,
            nmode=nmode,
            wires=wires,
            cutoff=cutoff,
            den_mat=den_mat,
            requires_grad=requires_grad,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        self.npara = 1

    def inputs_to_tensor(self, inputs: Any = None) -> torch.Tensor:
        """Convert inputs to torch.Tensor."""
        while isinstance(inputs, list):
            inputs = inputs[0]
        if inputs is None:
            inputs = torch.rand(1)[0]
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float)
        if self.noise:
            inputs = inputs + torch.normal(self.mu, self.sigma, size=(1,)).squeeze()
        return inputs

    def get_matrix(self, s: Any) -> torch.Tensor:
        """Get the local symplectic matrix acting on annihilation and creation operators."""
        # correspond to: U^+ (a a^+) U = s @ (a a^+)
        s = self.inputs_to_tensor(s).reshape(-1)
        one = s.new_ones(1)
        zero = s.new_zeros(1)
        x = 1j * s / 2
        mat = torch.stack([one, x, zero, x, x, one, x, zero, zero, -x, one, -x, -x, zero, -x, one]).reshape(4, 4)
        return mat

    def update_matrix(self) -> torch.Tensor:
        """Update the local symplectic matrix acting on annihilation and creation operators."""
        matrix = self.get_matrix(self.s)
        self.matrix = matrix.detach()
        return matrix

    def get_matrix_state(self, s: Any) -> torch.Tensor:
        """Get the local transformation matrix acting on Fock state tensors."""
        s = self.inputs_to_tensor(s)
        theta = torch.tensor(torch.pi / 2, dtype=s.dtype, device=s.device)
        # noise already in s
        mat_ps1 = PhaseShift(-theta, cutoff=self.cutoff).update_matrix_state()
        mat_cx = ControlledX(s, cutoff=self.cutoff).update_matrix_state()
        mat_ps2 = PhaseShift(theta, cutoff=self.cutoff).update_matrix_state()
        mat = torch.einsum('an,mnkl,lb->makb', mat_ps2, mat_cx, mat_ps1)
        return mat

    def update_matrix_state(self) -> torch.Tensor:
        """Update the local transformation matrix acting on Fock state tensors."""
        return self.get_matrix_state(self.s)

    def get_transform_xp(self, s: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        s = self.inputs_to_tensor(s).reshape(-1)
        zero = s.new_zeros(1)
        mat1 = torch.eye(4, dtype=s.dtype, device=s.device)
        mat2 = torch.stack([zero, zero, s, s]).reshape(-1).diag_embed().fliplr().reshape(4, 4)
        matrix_xp = mat1 + mat2
        vector_xp = s.new_zeros(4, 1)
        return matrix_xp, vector_xp

    def update_transform_xp(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the local affine symplectic transformation acting on quadrature operators in ``xxpp`` order."""
        matrix_xp, vector_xp = self.get_transform_xp(self.s)
        self.matrix_xp = matrix_xp.detach()
        self.vector_xp = vector_xp.detach()
        return matrix_xp, vector_xp

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        noise = self.noise
        self.noise = False
        s = self.inputs_to_tensor(inputs)
        self.noise = noise
        if self.requires_grad:
            self.s = nn.Parameter(s)
        else:
            self.register_buffer('s', s)
        self.update_matrix()
        self.update_transform_xp()

    def extra_repr(self) -> str:
        return f'wires={self.wires}, s={self.s.item()}'


class CubicPhase(SingleGate):
    r"""Cubic phase gate.

    **Operator Representation:**

    .. math::

        CP(\gamma) = e^{i \frac{\gamma}{3 \hbar} \hat{x}^3}

    **Operator Transformation:**

    .. math::

        CP^\dagger(\gamma) \hat{a} CP(\gamma) = \hat{a} + i\frac{\gamma(\hat{a} + \hat{a}^\dagger)^2}{2\sqrt{2/\hbar}}

    .. math::

        CP^\dagger(\gamma) \hat{x} CP(\gamma) &= \hat{x}, \\
        CP^\dagger(\gamma) \hat{p} CP(\gamma) &= \hat{p} + \gamma\hat{x}^2.

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        requires_grad (bool, optional): Whether the parameters are ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """

    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 1,
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__(
            name='CubicPhase',
            inputs=inputs,
            nmode=nmode,
            wires=wires,
            cutoff=cutoff,
            den_mat=den_mat,
            requires_grad=requires_grad,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        self.npara = 1

    def inputs_to_tensor(self, inputs: Any = None) -> torch.Tensor:
        """Convert inputs to torch.Tensor."""
        while isinstance(inputs, list):
            inputs = inputs[0]
        if inputs is None:
            inputs = torch.rand(1)[0]
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float)
        if self.noise:
            inputs = inputs + torch.normal(self.mu, self.sigma, size=(1,)).squeeze()
        return inputs

    def get_matrix_state(self, gamma: Any) -> torch.Tensor:
        """Get the local transformation matrix acting on Fock state tensors."""
        gamma = self.inputs_to_tensor(gamma)
        a, ad = ladder_ops(self.cutoff, gamma.dtype, device=gamma.device)
        x = (a + ad) * dqp.hbar**0.5 / (2 * dqp.kappa)
        mat = torch.matrix_exp(1j * gamma * torch.matrix_power(x, 3) / (3 * dqp.hbar)).reshape([self.cutoff] * 2)
        return mat

    def update_matrix_state(self) -> torch.Tensor:
        """Update the local transformation matrix acting on Fock state tensors."""
        return self.get_matrix_state(self.gamma)

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        noise = self.noise
        self.noise = False
        gamma = self.inputs_to_tensor(inputs)
        self.noise = noise
        if self.requires_grad:
            self.gamma = nn.Parameter(gamma)
        else:
            self.register_buffer('gamma', gamma)

    def extra_repr(self) -> str:
        return f'wires={self.wires}, gamma={self.gamma.item()}'


class Kerr(SingleGate):
    r"""Kerr gate.

    **Operator Representation:**

    .. math::

        K(\kappa) = e^{i \kappa \hat{n}^2}

    **Operator Transformation:**

    .. math::

        K^\dagger(\kappa) \hat{a} K(\kappa) &= e^{i \kappa (2 \hat{a}^\dagger \hat{a} + 1)} \hat{a}
                                             = \hat{a} e^{i \kappa (2 \hat{a}^\dagger \hat{a} - 1)}, \\
        K^\dagger(\kappa) \hat{a}^\dagger K(\kappa) &= \hat{a}^\dagger e^{-i \kappa (2 \hat{a}^\dagger \hat{a} + 1)}
                                                     = e^{i \kappa (2 \hat{a}^\dagger \hat{a} + 1)} \hat{a}^\dagger.

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        requires_grad (bool, optional): Whether the parameters are ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """

    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 1,
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__(
            name='Kerr',
            inputs=inputs,
            nmode=nmode,
            wires=wires,
            cutoff=cutoff,
            den_mat=den_mat,
            requires_grad=requires_grad,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        self.npara = 1

    def inputs_to_tensor(self, inputs: Any = None) -> torch.Tensor:
        """Convert inputs to torch.Tensor."""
        while isinstance(inputs, list):
            inputs = inputs[0]
        if inputs is None:
            inputs = torch.rand(1)[0]
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float)
        if self.noise:
            inputs = inputs + torch.normal(self.mu, self.sigma, size=(1,)).squeeze()
        return inputs

    def get_matrix_state(self, kappa: Any) -> torch.Tensor:
        """Get the local transformation matrix acting on Fock state tensors."""
        kappa = self.inputs_to_tensor(kappa)
        n = torch.arange(self.cutoff, dtype=kappa.dtype, device=kappa.device)
        mat = torch.exp(1j * kappa * n**2).diag_embed().reshape([self.cutoff] * 2)
        return mat

    def update_matrix_state(self) -> torch.Tensor:
        """Update the local transformation matrix acting on Fock state tensors."""
        return self.get_matrix_state(self.kappa)

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        noise = self.noise
        self.noise = False
        kappa = self.inputs_to_tensor(inputs)
        self.noise = noise
        if self.requires_grad:
            self.kappa = nn.Parameter(kappa)
        else:
            self.register_buffer('kappa', kappa)

    def extra_repr(self) -> str:
        return f'wires={self.wires}, kappa={self.kappa.item()}'


class CrossKerr(DoubleGate):
    r"""Cross-Kerr gate.

    **Operator Representation:**

    .. math::

        CK(\kappa) = e^{i \kappa \hat{n}_1 \hat{n}_2}

    **Operator Transformation:**

    .. math::

        CK^\dagger(\kappa) \hat{a}_1 CK(\kappa) &= e^{i \kappa \hat{n}_2} \hat{a}_1
                                                 = \hat{a}_1 e^{i \kappa \hat{n}_2}, \\
        CK^\dagger(\kappa) \hat{a}_2 CK(\kappa) &= e^{i \kappa \hat{n}_1} \hat{a}_2
                                                 = \hat{a}_2 e^{i \kappa \hat{n}_1}, \\
        CK^\dagger(\kappa) \hat{a}_1^\dagger CK(\kappa) &= e^{-i \kappa \hat{n}_2} \hat{a}_1^\dagger
                                                         = \hat{a}_1^\dagger e^{-i \kappa \hat{n}_2}, \\
        CK^\dagger(\kappa) \hat{a}_2^\dagger CK(\kappa) &= e^{-i \kappa \hat{n}_1} \hat{a}_2^\dagger
                                                         = \hat{a}_2^\dagger e^{-i \kappa \hat{n}_1}.

    Args:
        inputs (Any, optional): The parameter of the gate. Default: ``None``
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 2
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        requires_grad (bool, optional): Whether the parameters are ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """

    def __init__(
        self,
        inputs: Any = None,
        nmode: int = 2,
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        requires_grad: bool = False,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__(
            name='CrossKerr',
            inputs=inputs,
            nmode=nmode,
            wires=wires,
            cutoff=cutoff,
            den_mat=den_mat,
            requires_grad=requires_grad,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        self.npara = 1

    def inputs_to_tensor(self, inputs: Any = None) -> torch.Tensor:
        """Convert inputs to torch.Tensor."""
        while isinstance(inputs, list):
            inputs = inputs[0]
        if inputs is None:
            inputs = torch.rand(1)[0]
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float)
        if self.noise:
            inputs = inputs + torch.normal(self.mu, self.sigma, size=(1,)).squeeze()
        return inputs

    def get_matrix_state(self, kappa: Any) -> torch.Tensor:
        """Get the local transformation matrix acting on Fock state tensors."""
        kappa = self.inputs_to_tensor(kappa)
        n = torch.arange(self.cutoff, dtype=kappa.dtype, device=kappa.device)
        n1n2 = torch.kron(n, n)
        mat = torch.exp(1j * kappa * n1n2).diag_embed().reshape([self.cutoff] * 4)
        return mat

    def update_matrix_state(self) -> torch.Tensor:
        """Update the local transformation matrix acting on Fock state tensors."""
        return self.get_matrix_state(self.kappa)

    def init_para(self, inputs: Any = None) -> None:
        """Initialize the parameters."""
        noise = self.noise
        self.noise = False
        kappa = self.inputs_to_tensor(inputs)
        self.noise = noise
        if self.requires_grad:
            self.kappa = nn.Parameter(kappa)
        else:
            self.register_buffer('kappa', kappa)

    def extra_repr(self) -> str:
        return f'wires={self.wires}, kappa={self.kappa.item()}'


class DelayBS(Delay):
    r"""Delay loop with ``BeamSplitterTheta`` and ``PhaseShift``.

    Args:
        inputs (Any, optional): The parameters of the gate. Default: ``None``
        ntau (int, optional): The number of modes in the delay loop. Default: 1
        nmode (int, optional): The number of spatial modes that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        requires_grad (bool, optional): Whether the parameters are ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        loop_gates (List[Gate] or None, optional): The list of gates added to the delay loop. Default: ``None``
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """

    def __init__(
        self,
        inputs: Any = None,
        ntau: int = 1,
        nmode: int = 1,
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        requires_grad: bool = False,
        loop_gates: list | None = None,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__(
            name='DelayBS',
            ntau=ntau,
            nmode=nmode,
            wires=wires,
            cutoff=cutoff,
            den_mat=den_mat,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        self.requires_grad = requires_grad
        bs = BeamSplitterTheta(
            inputs=None,
            nmode=2,
            wires=None,
            cutoff=cutoff,
            den_mat=den_mat,
            requires_grad=requires_grad,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        ps = PhaseShift(
            inputs=None,
            nmode=1,
            wires=None,
            cutoff=cutoff,
            den_mat=den_mat,
            requires_grad=requires_grad,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        self.gates.append(bs)
        self.gates.append(ps)
        self.npara = 2
        self.init_para(inputs)
        if loop_gates is not None:
            for gate in loop_gates:
                self.gates.append(gate)
                self.npara += gate.npara

    @property
    def theta(self):
        return self.gates[0].theta

    @property
    def phi(self):
        return self.gates[1].theta

    def extra_repr(self) -> str:
        return f'wires={self.wires}, ntau={self.ntau}, theta={self.theta.item()}, phi={self.phi.item()}'


class DelayMZI(Delay):
    r"""Delay loop with MZI.

    Args:
        inputs (Any, optional): The parameters of the gate. Default: ``None``
        ntau (int, optional): The number of modes in the delay loop. Default: 1
        nmode (int, optional): The number of spatial modes that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        den_mat (bool, optional): Whether to use density matrix representation. Default: ``False``
        requires_grad (bool, optional): Whether the parameters are ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
        loop_gates (List[Gate] or None, optional): The list of gates added to the delay loop. Default: ``None``
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """

    def __init__(
        self,
        inputs: Any = None,
        ntau: int = 1,
        nmode: int = 1,
        wires: int | list[int] | None = None,
        cutoff: int | None = None,
        den_mat: bool = False,
        requires_grad: bool = False,
        loop_gates: list | None = None,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__(
            name='DelayMZI',
            ntau=ntau,
            nmode=nmode,
            wires=wires,
            cutoff=cutoff,
            den_mat=den_mat,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        self.requires_grad = requires_grad
        mzi = MZI(
            inputs=inputs,
            nmode=2,
            wires=None,
            cutoff=cutoff,
            den_mat=den_mat,
            phi_first=False,
            requires_grad=requires_grad,
            noise=noise,
            mu=mu,
            sigma=sigma,
        )
        self.gates.append(mzi)
        self.npara = 2
        if loop_gates is not None:
            for gate in loop_gates:
                self.gates.append(gate)
                self.npara += gate.npara

    @property
    def theta(self):
        return self.gates[0].theta

    @property
    def phi(self):
        return self.gates[0].phi

    def extra_repr(self) -> str:
        return f'wires={self.wires}, ntau={self.ntau}, theta={self.theta.item()}, phi={self.phi.item()}'


class Barrier(Gate):
    """Barrier.

    Args:
        nmode (int, optional): The number of modes that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the modes that the quantum operation acts on.
            Default: ``None``
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
    """

    def __init__(self, nmode: int = 1, wires: int | list[int] | None = None, cutoff: int | None = None) -> None:
        if wires is None:
            wires = list(range(nmode))
        super().__init__(name='Barrier', nmode=nmode, wires=wires, cutoff=cutoff)

    def forward(self, x: Any) -> Any:
        """Perform a forward pass."""
        return x

    def extra_repr(self) -> str:
        return f'wires={self.wires}'
