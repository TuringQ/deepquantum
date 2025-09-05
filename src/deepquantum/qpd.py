"""
Quasiprobability decomposition gates
"""

from typing import List, Optional, Tuple

from torch import nn

from .gate import PauliX, Hadamard, SGate, SDaggerGate
from .operation import GateQPD, MeasureQPD


class DoubleGateQPD(GateQPD):
    r"""A base class for two-qubit quasiprobability decomposition gates.

    Args:
        bases (List[Tuple[nn.Sequential, ...]]): A list of tuples describing the operations probabilistically used to
            simulate an ideal quantum operation.
        coeffs (List[float]): The coefficients for quasiprobability representation.
        name (str or None, optional): The name of the quantum operation. Default: ``None``
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 2
        wires (List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`. Default: ``False``
    """
    def __init__(
        self,
        bases: List[Tuple[nn.Sequential, ...]],
        coeffs: List[float],
        name: Optional[str] = None,
        nqubit: int = 2,
        wires: Optional[List[int]] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        if wires is None:
            wires = [0, 1]
        assert len(wires) == 2
        for basis in bases:
            assert len(basis) == 2
        super().__init__(bases=bases, coeffs=coeffs, name=name, nqubit=nqubit, wires=wires,
                         den_mat=den_mat, tsr_mode=tsr_mode)


class MoveQPD(DoubleGateQPD):
    r"""QPD of the move operation.

    Args:
        nqubit (int, optional): The number of qubits that the quantum operation acts on. Default: 2
        wires (List[int] or None, optional): The indices of the qubits that the quantum operation acts on.
            Default: ``None``
        den_mat (bool, optional): Whether the quantum operation acts on density matrices or state vectors.
            Default: ``False`` (which means state vectors)
        tsr_mode (bool, optional): Whether the quantum operation is in tensor mode, which means the input
            and output are represented by a tensor of shape :math:`(\text{batch}, 2, ..., 2)`. Default: ``False``
    """
    def __init__(
        self,
        nqubit: int = 2,
        wires: Optional[List[int]] = None,
        den_mat: bool = False,
        tsr_mode: bool = False
    ) -> None:
        if wires is None:
            wires = [0, 1]
        h1 = Hadamard(nqubit=nqubit, wires=wires[0], den_mat=den_mat, tsr_mode=True)
        m1 = MeasureQPD(nqubit=nqubit, wires=wires[0])
        sdg1 = SDaggerGate(nqubit=nqubit, wires=wires[0], den_mat=den_mat, tsr_mode=True)
        x2 = PauliX(nqubit=nqubit, wires=wires[1], den_mat=den_mat, tsr_mode=True)
        h2 = Hadamard(nqubit=nqubit, wires=wires[1], den_mat=den_mat, tsr_mode=True)
        s2 = SGate(nqubit=nqubit, wires=wires[1], den_mat=den_mat, tsr_mode=True)

        measure_i = nn.Sequential()
        measure_x = nn.Sequential(h1, m1)
        measure_y = nn.Sequential(sdg1, h1, m1)
        measure_z = nn.Sequential(m1)

        prep_0 = nn.Sequential()
        prep_1 = nn.Sequential(x2)
        prep_plus = nn.Sequential(h2)
        prep_minus = nn.Sequential(x2, h2)
        prep_iplus = nn.Sequential(h2, s2)
        prep_iminus = nn.Sequential(x2, h2, s2)

        bases = [(measure_i, prep_0),
                (measure_i, prep_1),
                (measure_x, prep_plus),
                (measure_x, prep_minus),
                (measure_y, prep_iplus),
                (measure_y, prep_iminus),
                (measure_z, prep_0),
                (measure_z, prep_1)]
        coeffs = [0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5]
        super().__init__(bases=bases, coeffs=coeffs, name='MoveQPD', nqubit=nqubit, wires=wires,
                         den_mat=den_mat, tsr_mode=tsr_mode)
