"""
Basic gate in MBQC pattern
"""
import torch
from typing import Any, List, Optional, Tuple, Union

from .operation import Operation
from .mbqc import MBQC

class CNOT(MBQC):
    def __init__(
        self,
        control: int,
        target: int,
        ancilla: List[int],
        init_state: Any=None
    ) -> None:
        assert len(ancilla) == 2
        self._bg_qubit = 4
        self.control = control
        self.target = target
        self.ancilla = ancilla
        super().__init__(nqubit=2, init_state=init_state, name='cnot')

        self.node(ancilla[0])
        self.node(ancilla[1])
        self.entanglement([target, ancilla[0]])
        self.entanglement([control, ancilla[0]])
        self.entanglement(ancilla)
        self.measurement(target)
        self.measurement(ancilla[0])
        self.X(ancilla[1], signal_domain=[ancilla[0]])
        self.Z(ancilla[1], signal_domain=[target])
        self.Z(control, signal_domain=[target])

class X(MBQC):
    def __init__(
        self,
        input_node: int,
        ancilla: List[int],
        init_state: Any=None
    ) -> None:
        assert len(ancilla) == 2
        self._bg_qubit = 3
        self.input_node = input_node
        self.ancilla = ancilla
        super().__init__(nqubit=1, init_state=init_state, name='x')

        self.node(ancilla[0])
        self.node(ancilla[1])
        self.entanglement([input_node, ancilla[0]])
        self.entanglement(ancilla)
        self.measurement(input_node)
        self.measurement(ancilla[0], angle=-torch.pi)
        self.X(ancilla[1], signal_domain=[ancilla[0]])
        self.Z(ancilla[1], signal_domain=[input_node])



