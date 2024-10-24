"""
Measurement based quantum circuit
"""
from typing import Any, List, Optional, Tuple, Union
from copy import copy
import torch
from networkx import Graph, draw_networkx
from torch import nn
from operation import Operation, Node, Entanglement
from qmath import kron


class MBQC(Operation):
    r"""Measurement based quantum circuit.
    nqubit: the number of input qubits
    """
    def __init__(
        self,
        nqubit: int,
        init_state: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__(nqubit=nqubit, wires=list(range(nqubit)))
        self._bg_state = None
        self._bg_qubit = nqubit
        self.nqubit = nqubit
        self._graph = None
        self.pattern = None
        self.operators = nn.Sequential()
        self.encoders = [ ]
        self.npara = 0
        self.ndata = 0
        self._node_list = list(range(self.nqubit))
        self._edge_list = [ ]

        if init_state is None:
            plus_state = torch.sqrt(torch.tensor(2))*torch.tensor([1,1])/2
            init_state = kron([plus_state] * nqubit)
        self.init_state = init_state

    def set_graph(self, graph: List[List]):
        vertices, edges = graph
        assert len(vertices) > self.nqubit
        for i in vertices:
            if i not in self._node_list:
                self.node(i)
        for edge in edges:
            self.entanglement(edge)
        return

    def get_graph(self):
        assert len(self._node_list) == self._bg_qubit
        g = Graph()
        g.add_nodes_from(self._node_list)
        g.add_edges_from(self._edge_list)
        self._graph = g
        return g

    def add(
        self,
        op: Operation,
        encode: bool = False,
        wires: Union[int, List[int], None] = None
    ) -> None:
        """A method that adds an operation to the photonic quantum circuit.

        The operation can be a gate or another photonic quantum circuit. The method also updates the
        attributes of the photonic quantum circuit. If ``wires`` is specified, the parameters of gates
        are shared.

        Args:
            op (Operation): The operation to add. It is an instance of ``Operation`` class or its subclasses,
                such as ``Gate``, or ``QumodeCircuit``.
            encode (bool): Whether the gate is to encode data. Default: ``False``
            wires (Union[int, List[int], None]): The wires to apply the gate on. It can be an integer
                or a list of integers specifying the indices of the wires. Default: ``None`` (which means
                the gate has its own wires)

        Raises:
            AssertionError: If the input arguments are invalid or incompatible with the quantum circuit.
        """
        assert isinstance(op, Operation)
        if wires is not None:
            wires = self._convert_indices(wires)
            assert len(wires) == len(op.wires), 'Invalid input'
            op = copy(op)
            op.wires = wires
        self.operators.append(op)
        if encode:
            assert not op.requires_grad, 'Please set requires_grad of the operation to be False'
            self.encoders.append(op)
            self.ndata += op.npara
        else:
            self.npara += op.npara

    def node(self, wires: Union[int, List[int]] = None):
        node_ = Node(wires=wires)
        assert node_.wires[0] not in self._node_list, 'node already exists'
        self._node_list.append(node_.wires[0])
        self.add(node_)
        self._bg_qubit += 1

    def entanglement(self, wires: List[int] = None):
        assert wires[0] in self._node_list and wires[1] in self._node_list
        entang_ = Entanglement(wires=wires)
        self._edge_list.append(wires)
        self.add(entang_)

    def forward(self):
        state = self.init_state
        for op in self.operators:
            state = op.forward(state)
        self._bg_state = state
        return state

    def draw(self, wid: int=3):
        g = self.get_graph()
        pos = {}
        for i in self._node_list:
            pos_x = i % wid
            pos_y = i // wid
            pos[i] = (pos_x, -pos_y)
        draw_networkx(g, pos=pos)
        return