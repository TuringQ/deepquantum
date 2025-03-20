"""
Quantum channels
"""

from typing import Any, List, Union

import torch

from .gate import Identity, PauliX, PauliY, PauliZ
from .operation import Channel


mat_i = Identity().matrix
mat_x = PauliX().matrix
mat_y = PauliY().matrix
mat_z = PauliZ().matrix


class BitFlip(Channel):
    r"""Bit flip channel.

    The bit flip channel is defined as:

    .. math::
        \rho \Rightarrow (1-p) \rho + p X \rho X^{\dagger}

    Args:
        inputs (Any, optional): The parameter of the channel. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`. Default: ``False``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(inputs=inputs, name='BitFlip', nqubit=nqubit, wires=wires, tsr_mode=tsr_mode,
                         requires_grad=requires_grad)

    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Get the local Kraus matrices acting on density matrices."""
        theta = self.inputs_to_tensor(theta).reshape(-1)
        prob = torch.sin(theta) ** 2
        mat1 = torch.sqrt(1 - prob) * mat_i.to(prob.device)
        mat2 = torch.sqrt(prob) * mat_x.to(prob.device)
        return torch.stack([mat1, mat2])

    def _qasm(self) -> str:
        return self._qasm_customized('Bflip')


class PhaseFlip(Channel):
    r"""Phase flip channel.

    The phase flip channel is defined as:

    .. math::
        \rho \Rightarrow (1-p) \rho + p Z \rho Z^{\dagger}

    Args:
        inputs (Any, optional): The parameter of the channel. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`. Default: ``False``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(inputs=inputs, name='PhaseFlip', nqubit=nqubit, wires=wires, tsr_mode=tsr_mode,
                         requires_grad=requires_grad)

    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Get the local Kraus matrices acting on density matrices."""
        theta = self.inputs_to_tensor(theta).reshape(-1)
        prob = torch.sin(theta) ** 2
        mat1 = torch.sqrt(1 - prob) * mat_i.to(prob.device)
        mat2 = torch.sqrt(prob) * mat_z.to(prob.device)
        return torch.stack([mat1, mat2])

    def _qasm(self) -> str:
        return self._qasm_customized('Pflip')


class Depolarizing(Channel):
    r"""Depolarizing channel.

    The depolarizing channel is defined as:

    .. math::
        \rho \Rightarrow (1-p) \rho
            + p/3 X \rho X^{\dagger}
            + p/3 Y \rho Y^{\dagger}
            + p/3 Z \rho Z^{\dagger}

    Args:
        inputs (Any, optional): The parameter of the channel. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`. Default: ``False``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(inputs=inputs, name='Depolarizing', nqubit=nqubit, wires=wires, tsr_mode=tsr_mode,
                         requires_grad=requires_grad)

    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Get the local Kraus matrices acting on density matrices."""
        theta = self.inputs_to_tensor(theta).reshape(-1)
        prob = torch.sin(theta) ** 2
        mat1 = torch.sqrt(1 - prob) * mat_i.to(prob.device)
        mat2 = torch.sqrt(prob / 3) * mat_x.to(prob.device)
        mat3 = torch.sqrt(prob / 3) * mat_y.to(prob.device)
        mat4 = torch.sqrt(prob / 3) * mat_z.to(prob.device)
        return torch.stack([mat1, mat2, mat3, mat4])

    def _qasm(self) -> str:
        return self._qasm_customized('Dp')


class Pauli(Channel):
    r"""Pauli channel.

    The Pauli channel is defined as:

    .. math::
        \rho \Rightarrow (1-px-py-pz) \rho
            + px X \rho X^{\dagger}
            + py Y \rho Y^{\dagger}
            + pz Z \rho Z^{\dagger}

    Args:
        inputs (Any, optional): The parameters of the channel. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`. Default: ``False``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(inputs=inputs, name='Pauli', nqubit=nqubit, wires=wires, tsr_mode=tsr_mode,
                         requires_grad=requires_grad)
        self.npara = 4

    @property
    def prob(self):
        """The error probabilities."""
        prob = torch.sin(self.theta) ** 2
        return prob / prob.sum()

    def inputs_to_tensor(self, inputs: Any = None) -> torch.Tensor:
        """Convert inputs to torch.Tensor."""
        if inputs is None:
            inputs = torch.rand(4) * torch.pi
        elif not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float).reshape(-1)[:4]
        return inputs

    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Get the local Kraus matrices acting on density matrices."""
        theta = self.inputs_to_tensor(theta).reshape(-1)
        prob = torch.sin(theta) ** 2
        prob = prob / prob.sum()
        mat1 = torch.sqrt(prob[0:1]) * mat_i.to(prob.device) # avoid scalar Tensor
        mat2 = torch.sqrt(prob[1:2]) * mat_x.to(prob.device)
        mat3 = torch.sqrt(prob[2:3]) * mat_y.to(prob.device)
        mat4 = torch.sqrt(prob[3:4]) * mat_z.to(prob.device)
        return torch.stack([mat1, mat2, mat3, mat4])

    def extra_repr(self) -> str:
        return f'wires={self.wires}, px={self.prob[1].item()}, py={self.prob[2].item()}, pz={self.prob[3].item()}'


class AmplitudeDamping(Channel):
    r"""Amplitude damping channel.

    The amplitude damping channel is defined as:

    .. math::
        \rho \Rightarrow K_0 \rho K_0^{\dagger} + K_1 \rho K_1^{\dagger},
        K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - p} \end{pmatrix},
        K_1 = \begin{pmatrix} 0 & \sqrt{p} \\ 0 & 0 \end{pmatrix}

    Args:
        inputs (Any, optional): The parameter of the channel. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`. Default: ``False``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(inputs=inputs, name='AmplitudeDamping', nqubit=nqubit, wires=wires, tsr_mode=tsr_mode,
                         requires_grad=requires_grad)

    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Get the local Kraus matrices acting on density matrices."""
        theta = self.inputs_to_tensor(theta).reshape(-1)
        prob = torch.sin(theta) ** 2
        m0 = torch.tensor([0.], dtype=prob.dtype, device=prob.device)
        m1 = torch.tensor([1.], dtype=prob.dtype, device=prob.device)
        mat1 = torch.stack([m1, m0, m0, torch.sqrt(1 - prob)]).reshape(2, 2)
        mat2 = torch.stack([m0, torch.sqrt(prob), m0, m0]).reshape(2, 2)
        return torch.stack([mat1, mat2]) + 0j

    def _qasm(self) -> str:
        return self._qasm_customized('Adamp')


class PhaseDamping(Channel):
    r"""Phase damping channel.

    The phase damping channel is defined as:

    .. math::
        \rho \Rightarrow K_0 \rho K_0^{\dagger} + K_1 \rho K_1^{\dagger},
        K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - p} \end{pmatrix},
        K_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{p} \end{pmatrix}

    Args:
        inputs (Any, optional): The parameter of the channel. The Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`. Default: ``False``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(inputs=inputs, name='PhaseDamping', nqubit=nqubit, wires=wires, tsr_mode=tsr_mode,
                         requires_grad=requires_grad)

    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Get the local Kraus matrices acting on density matrices."""
        theta = self.inputs_to_tensor(theta).reshape(-1)
        prob = torch.sin(theta) ** 2
        m0 = torch.tensor([0.], dtype=prob.dtype, device=prob.device)
        m1 = torch.tensor([1.], dtype=prob.dtype, device=prob.device)
        mat1 = torch.stack([m1, m0, m0, torch.sqrt(1 - prob)]).reshape(2, 2)
        mat2 = torch.stack([m0, m0, m0, torch.sqrt(prob)]).reshape(2, 2)
        return torch.stack([mat1, mat2]) + 0j

    def _qasm(self) -> str:
        return self._qasm_customized('Pdamp')


class GeneralizedAmplitudeDamping(Channel):
    r"""Generalized amplitude damping channel.

    The generalized amplitude damping channel is defined as:

    .. math::
        \rho \Rightarrow K_0 \rho K_0^{\dagger} + K_1 \rho K_1^{\dagger}
            + K_2 \rho K_2^{\dagger} + K_3 \rho K_3^{\dagger},
        K_0 = \sqrt{p} \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - \gamma} \end{pmatrix},
        K_1 = \sqrt{p} \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix},
        K_2 = \sqrt{1 - p} \begin{pmatrix} \sqrt{1 - \gamma} & 0 \\ 0 & 1 \end{pmatrix},
        K_3 = \sqrt{1 - p} \begin{pmatrix} 0 & 0 \\ \sqrt{\gamma} & 0 \end{pmatrix}

    Args:
        inputs (Any, optional): The parameters of the channel. The first parameter is the probability
            of amplitude damping error, and the second parameter is the damping rate. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 1
        wires (int, List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`. Default: ``False``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """
    def __init__(
        self,
        inputs: Any = None,
        nqubit: int = 1,
        wires: Union[int, List[int], None] = None,
        tsr_mode: bool = False,
        requires_grad: bool = False
    ) -> None:
        super().__init__(inputs=inputs, name='GeneralizedAmplitudeDamping', nqubit=nqubit, wires=wires,
                         tsr_mode=tsr_mode, requires_grad=requires_grad)
        self.npara = 2

    def inputs_to_tensor(self, inputs: Any = None) -> torch.Tensor:
        """Convert inputs to torch.Tensor."""
        if inputs is None:
            inputs = torch.rand(2) * torch.pi
        elif not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float).reshape(-1)[:2]
        return inputs

    def get_matrix(self, theta: Any) -> torch.Tensor:
        """Get the local Kraus matrices acting on density matrices."""
        theta = self.inputs_to_tensor(theta).reshape(-1)
        prob = torch.sin(theta) ** 2
        m0 = torch.tensor(0., dtype=prob.dtype, device=prob.device)
        m1 = torch.tensor(1., dtype=prob.dtype, device=prob.device)
        mat1 = torch.sqrt(prob[0]) * torch.stack([m1, m0, m0, torch.sqrt(1 - prob[1])]).reshape(2, 2)
        mat2 = torch.sqrt(prob[0]) * torch.stack([m0, torch.sqrt(prob[1]), m0, m0]).reshape(2, 2)
        mat3 = torch.sqrt(1 - prob[0]) * torch.stack([torch.sqrt(1 - prob[1]), m0, m0, m1]).reshape(2, 2)
        mat4 = torch.sqrt(1 - prob[0]) * torch.stack([m0, m0, torch.sqrt(prob[1]), m0]).reshape(2, 2)
        return torch.stack([mat1, mat2, mat3, mat4]) + 0j

    def extra_repr(self) -> str:
        return f'wires={self.wires}, probability={self.prob[0].item()}, rate={self.prob[1].item()}'

    def _qasm(self) -> str:
        return self._qasm_customized('Gad')
