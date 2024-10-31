"""
Basic gate in MBQC pattern
"""
import torch
from typing import Any, List, Optional, Tuple, Union

from .operation import Operation
from .mbqc import Pattern

class CNOT(Pattern):
    def __init__(
        self,
        control: int,
        target: int,
        ancilla: List[int],
        init_state: Any=None
    ) -> None:
        assert len(ancilla) == 2
        self.control = control
        self.target = target
        self.ancilla = ancilla
        super().__init__(n_input_nodes=2, init_state=init_state, name='cnot')

        self.n(ancilla[0])
        self.n(ancilla[1])
        self.e([target, ancilla[0]])
        self.e([control, ancilla[0]])
        self.e(ancilla)
        self.m(target)
        self.m(ancilla[0])
        self.x(ancilla[1], signal_domain=[ancilla[0]])
        self.z(ancilla[1], signal_domain=[target])
        self.z(control, signal_domain=[target])

class X(Pattern):
    def __init__(
        self,
        input_node: int,
        ancilla: List[int],
        init_state: Any=None
    ) -> None:
        assert len(ancilla) == 2
        self.input_node = input_node
        self.ancilla = ancilla
        super().__init__(n_input_nodes=1, init_state=init_state, name='x')

        self.n(ancilla[0])
        self.n(ancilla[1])
        self.e([input_node, ancilla[0]])
        self.e(ancilla)
        self.m(input_node)
        self.m(ancilla[0], angle=-torch.pi)
        self.x(ancilla[1], signal_domain=[ancilla[0]])
        self.z(ancilla[1], signal_domain=[input_node])



