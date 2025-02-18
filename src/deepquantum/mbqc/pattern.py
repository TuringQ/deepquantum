"""
Measurement pattern
"""

from copy import copy, deepcopy
from typing import Any, Dict, Iterable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from networkx import MultiDiGraph, draw_networkx_nodes, draw_networkx_edges, draw_networkx_labels, multipartite_layout
from torch import nn

from .command import Node, Entanglement, Measurement, Correction
from .operation import Operation
from .state import SubGraphState, GraphState


class Pattern(Operation):
    """Measurement-based quantum computing (MBQC) pattern.

    A pattern represents a measurement-based quantum computation, which consists of a sequence of
    commands (node preparation, entanglement, measurement, and correction) applied to qubits arranged
    in a graph structure.

    Args:
        nodes_state (int, List[int] or None, optional): The nodes of the input state in the initial graph state.
            It can be an integer representing the number of nodes or a list of node indices. Default: ``None``
        state (Any, optional): The input state of the initial graph state. The string representation of state
            could be ``'plus'``, ``'minus'``, ``'zero'``, and ``'one'``. Default: ``'plus'``
        edges (List or None, optional): Additional edges connecting the nodes in the initial graph state.
            Default: ``None``
        nodes (int, List[int] or None, optional): Additional nodes to include in the initial graph state.
            Default: ``None``
        name (str or None, optional): The name of the pattern. Default: ``None``
        reupload (bool, optional): Whether to use data re-uploading. Default: ``False``

    Ref: V. Danos, E. Kashefi and P. Panangaden. J. ACM 54.2 8 (2007)
    """
    def __init__(
        self,
        nodes_state: Union[int, List[int], None] = None,
        state: Any = 'plus',
        edges: Optional[List] = None,
        nodes: Union[int, List[int], None] = None,
        name: Optional[str] = None,
        reupload: bool = False
    ) -> None:
        super().__init__(name=name, nodes=None)
        self.reupload = reupload
        self.init_state = GraphState(nodes_state, state, edges, nodes)
        self.commands = nn.Sequential()
        self.encoders = []
        self.state = None
        self.ndata = 0
        self.nodes_out_seq = None

    def forward(self, data: Optional[torch.Tensor] = None, state: Optional[GraphState] = None) -> GraphState:
        """Perform a forward pass of the MBQC pattern and return the final graph state.

        Args:
            data (torch.Tensor or None, optional): The input data for the ``encoders``. Default: ``None``
            state (GraphState or None, optional): The initial graph state for the pattern. Default: ``None``

        Returns:
            GraphState: The final graph state of the pattern after applying the ``commands``.
        """
        if state is None:
            self.state = deepcopy(self.init_state)
        else:
            self.state = state
        self.encode(data)
        self.state = self.commands(self.state)
        self.state.set_nodes_out_seq(self.nodes_out_seq)
        if data is not None:
            if data.ndim == 2:
                # for plotting the last data
                self.encode(data[-1])
        return self.state

    def encode(self, data: Optional[torch.Tensor]) -> None:
        """Encode the input data into the measurement angles as parameters.

        This method iterates over the ``encoders`` of the MBQC pattern and initializes their parameters
        with the input data. If ``reupload`` is ``False``, the input data must be at least as long as the number
        of parameters in the ``encoders``. If ``reupload`` is ``True``, the input data can be repeated to fill up
        the parameters.

        Args:
            data (torch.Tensor or None): The input data for the ``encoders``, could be a 1D or 2D tensor.

        Raises:
            AssertionError: If input data is shorter than the number of parameters in the ``encoders``.
        """
        if data is None:
            return
        if not self.reupload:
            assert data.shape[-1] >= self.ndata, 'The pattern needs more data, or consider data re-uploading'
        count = 0
        if self.reupload and self.ndata > data.shape[-1]:
            n = int(np.ceil(self.ndata / data.shape[-1]))
            data = torch.cat([data] * n, dim=-1)
        for op in self.encoders:
            count_up = count + op.npara
            if data.ndim == 2:
                op.init_para(data[:, count:count_up])
            else:
                op.init_para(data[count:count_up])
            count = count_up

    def add_graph(self,
        nodes_state: Union[int, List[int], None] = None,
        state: Any = 'plus',
        edges: Optional[List] = None,
        nodes: Union[int, List[int], None] = None,
        index: Optional[int] = None
    ) -> None:
        """Add a subgraph state to the graph state.

        Args:
            nodes_state (int, List[int] or None, optional): The nodes of the input state in the subgraph state.
                It can be an integer representing the number of nodes or a list of node indices. Default: ``None``
            state (Any, optional): The input state of the subgraph state. The string representation of state
                could be ``'plus'``, ``'minus'``, ``'zero'``, and ``'one'``. Default: ``'plus'``
            edges (List or None, optional): Additional edges connecting the nodes in the subgraph state.
                Default: ``None``
            nodes (int, List[int] or None, optional): Additional nodes to include in the subgraph state.
                Default: ``None``
            index (int or None, optional): The index where to insert the subgraph state. Default: ``None``
        """
        self.init_state.add_subgraph(nodes_state=nodes_state, state=state, edges=edges, nodes=nodes, index=index)

    @property
    def graph(self) -> SubGraphState:
        """The combined graph state of the initial or final graph state."""
        if self.state is None:
            return self.init_state.graph
        else:
            return self.state.graph

    def set_nodes_out_seq(self, nodes: Optional[List[int]] = None) -> None:
        """Set the output sequence of the nodes."""
        self.nodes_out_seq = nodes

    def add(
        self,
        op: Operation,
        encode: bool = False
    ) -> None:
        """A method that adds an operation to the MBQC pattern.

        Args:
            op (Operation): The operation to add. It is an instance of ``Operation`` class or its subclasses,
                such as ``Node``, ``Entanglement``, ``Measurement``, or ``Correction``.
            encode (bool): Whether the command is to encode data. Default: ``False``
        """
        assert isinstance(op, Operation)
        self.commands.append(op)
        if encode:
            assert not op.requires_grad, 'Please set requires_grad of the operation to be False'
            self.encoders.append(op)
            self.ndata += op.npara
        else:
            self.npara += op.npara

    def n(self, nodes: Union[int, List[int]]) -> None:
        """Add a node command."""
        n = Node(nodes=nodes)
        self.add(n)

    def e(self, node1: int, node2: int) -> None:
        """Add an entanglement command."""
        e = Entanglement(node1=node1, node2=node2)
        self.add(e)

    def m(
        self,
        node: int,
        angle: float = 0.,
        plane: str = 'xy',
        t_domain: Union[int, Iterable[int], None] = None,
        s_domain: Union[int, Iterable[int], None] = None,
        encode: bool = False
    ) -> None:
        """Add a measurement command."""
        requires_grad = not encode
        if angle is not None:
            requires_grad = False
        m = Measurement(nodes=node, angle=angle, plane=plane, t_domain=t_domain, s_domain=s_domain,
                        requires_grad=requires_grad)
        self.add(m, encode=encode)

    def x(self, node: int, domain: Union[int, Iterable[int], None] = None) -> None:
        """Add an X-correction command."""
        x = Correction(nodes=node, basis='x', domain=domain)
        self.add(x)

    def z(self, node: int, domain: Union[int, Iterable[int], None] = None) -> None:
        """Add a Z-correction command."""
        z = Correction(nodes=node, basis='z', domain=domain)
        self.add(z)

    def draw(self, width: int = 4):
        """Draw the MBQC pattern."""
        g = MultiDiGraph(self.init_state.graph.graph)
        nodes_init = deepcopy(g.nodes())
        for i in nodes_init:
            g.nodes[i]['layer'] = 0
        nodes_measured = []
        edges_t_domain = []
        edges_s_domain = []
        for op in self.commands:
            if isinstance(op, Node):
                g.add_nodes_from(op.nodes, layer=2)
            elif isinstance(op, Entanglement):
                g.add_edge(*op.nodes)
            elif isinstance(op, Measurement):
                nodes_measured.append(op.nodes[0])
                if op.nodes[0] not in nodes_init:
                    g.nodes[op.nodes[0]]['layer'] = 1
                for i in op.t_domain:
                    edges_t_domain.append(tuple([i, op.nodes[0]]))
                for i in op.s_domain:
                    edges_s_domain.append(tuple([i, op.nodes[0]]))
        pos = multipartite_layout(g, subset_key='layer')
        draw_networkx_nodes(g, pos, nodelist=nodes_init, node_color='#1f78b4', node_shape='s')
        draw_networkx_nodes(g, pos, nodelist=nodes_measured, node_color='#1f78b4')
        draw_networkx_nodes(g, pos, nodelist=list(set(g.nodes()) - set(nodes_measured)),
                            node_color='#d7dde0', node_shape='o')
        draw_networkx_edges(g, pos, g.edges(), arrows=False)
        draw_networkx_edges(g, pos, edges_t_domain, arrows=True, style=':',
                            edge_color='#4cd925', connectionstyle='arc3,rad=-0.2')
        draw_networkx_edges(g, pos, edges_s_domain, arrows=True, style=':',
                            edge_color='#db1d2c', connectionstyle='arc3,rad=0.2')
        draw_networkx_labels(g, pos)
        plt.plot([], [], color='k',label='graph edge')
        plt.plot([], [], ':', color='#4cd925', label='zflow')
        plt.plot([], [], ':', color='#db1d2c', label='xflow')
        plt.plot([], [], 's', color='#1f78b4', label='input nodes')
        plt.plot([], [], 'o', color='#d7dde0', label='output nodes')
        plt.xlim(-width / 2, width / 2)
        plt.ylim(-width / 2, width / 2)
        plt.legend(loc='upper right', fontsize=10)
        plt.tight_layout()
        plt.show()

    def is_standard(self) -> bool:
        """Determine whether the command sequence is standard.

        Returns:
            bool: ``True`` if the pattern follows NEMC standardization, ``False`` otherwise
        """
        it = iter(self.commands)
        try:
            # Check if operations follow NEMC order
            op = next(it)
            while isinstance(op, Node):  # First all Node operations
                op = next(it)
            while isinstance(op, Entanglement):  # Then all Entanglement operations
                op = next(it)
            while isinstance(op, Measurement):  # Then all Measurement operations
                op = next(it)
            while isinstance(op, Correction):  # Finally all Correction operations
                op = next(it)
            return False  # If we get here, there were operations after NEMC sequence
        except StopIteration:
            return True  # If we run out of operations, pattern is standard

    def standardize(self) -> None:
        """Standardize the command sequence into NEMC form.

        This function reorders operations into the standard form:
        - Node preparations (N)
        - Entanglement operations (E)
        - Measurement operations (M)
        - Correction operations (C)

        It handles the propagation of correction operations by:
        1. Moving X-corrections through entanglements (generating Z-corrections)
        2. Moving corrections through measurements (modifying measurement signal domains)
        3. Collecting remaining corrections at the end

        See https://arxiv.org/pdf/0704.1263 Ch.(5.4)
        """
        # Initialize lists for each operation type
        n_list = []  # Node operations
        e_list = []  # Entanglement operations
        m_list = []  # Measurement operations
        z_dict = {}  # Tracks Z corrections by node
        x_dict = {}  # Tracks X corrections by node

        def add_correction_domain(domain_dict: Dict, node, domain) -> None:
            """Helper function to update correction domains with XOR operation"""
            if previous_domain := domain_dict.get(node):
                previous_domain ^= domain
            else:
                domain_dict[node] = domain.copy()

        # Process each operation and reorganize into standard form
        for op in self.commands:
            if isinstance(op, Node):
                n_list.append(op)
            elif isinstance(op, Entanglement):
                for side in (0, 1):
                    # Propagate X corrections through entanglement (generates Z corrections)
                    if s_domain := x_dict.get(op.nodes[side], None):
                        add_correction_domain(z_dict, op.nodes[1 - side], s_domain)
                e_list.append(op)
            elif isinstance(op, Measurement):
                # Apply pending corrections to measurement parameters
                new_op = copy(op)
                if t_domain := z_dict.pop(op.nodes[0], None):
                    new_op.t_domain = new_op.t_domain ^ t_domain
                if s_domain := x_dict.pop(op.nodes[0], None):
                    new_op.s_domain = new_op.s_domain ^ s_domain
                m_list.append(new_op)
            elif isinstance(op, Correction):
                if op.basis == 'z':
                    add_correction_domain(z_dict, op.nodes[0], op.domain)
                elif op.basis == 'x':
                    add_correction_domain(x_dict, op.nodes[0], op.domain)

        # Reconstruct command sequence in standard order
        self.commands = nn.Sequential(
                    *n_list,
                    *e_list,
                    *m_list,
                    *(Correction(nodes=node, basis='z', domain=domain) for node, domain in z_dict.items()),
                    *(Correction(nodes=node, basis='x', domain=domain) for node, domain in x_dict.items())
        )

    def shift_signals(self) -> Dict:
        """Perform signal shifting procedure.

        This allows one to dispose of dependencies induced by the Z-action,
        and obtain sometimes standard patterns with smaller computational depth complexity.

        It handles the propagation of signal shifting commands by:
        1. Extracting signals via t_domain (in XY plane cases) of measurements.
        2. Moving signals to the left, through modifying other measurements and corrections.

        Returns:
            Dict: A signal dictionary including all the signal shifting commands.

        See https://arxiv.org/pdf/0704.1263 Ch.(5.5)
        """
        signal_dict = {}

        def expand_domain(domain: set[int]) -> None:
            for node in domain & signal_dict.keys():
                domain ^= signal_dict[node]

        for op in self.commands:
            if isinstance(op, Measurement):
                s_domain = set(op.s_domain)
                t_domain = set(op.t_domain)
                expand_domain(s_domain)
                expand_domain(t_domain)
                if op.plane in ['xy', 'yx']:
                    # M^{XY,α} X^s Z^t = M^{XY,(-1)^s·α+tπ}
                    #                  = S^t M^{XY,(-1)^s·α}
                    #                  = S^t M^{XY,α} X^s
                    if t_domain:
                        signal_dict[op.nodes[0]] = t_domain
                        t_domain = set()
                elif op.plane in ['zx', 'xz']:
                    # M^{XZ,α} X^s Z^t = M^{XZ,(-1)^t((-1)^s·α+sπ)}
                    #                  = M^{XZ,(-1)^{s+t}·α+(-1)^t·sπ}
                    #                  = M^{XZ,(-1)^{s+t}·α+sπ         (since (-1)^t·π ≡ π (mod 2π))
                    #                  = S^s M^{XZ,(-1)^{s+t}·α}
                    #                  = S^s M^{XZ,α} Z^{s+t}
                    if s_domain:
                        signal_dict[op.nodes[0]] = s_domain
                        t_domain ^= s_domain
                        s_domain = set()
                elif op.plane in ['yz', 'zy']:
                    # positive Y axis as 0 angle
                    # M^{YZ,α} X^s Z^t = M^{YZ,(-1)^t·α+(s+t)π)}
                    #                  = S^s M^{YZ,(-1)^t·α+tπ}
                    #                  = S^s M^{YZ,α} Z^t
                    # still remains M^{YZ,(-1)^t·α+tπ)} after signal shifting,
                    # but dependency on s_domain has been reduced
                    if s_domain:
                        signal_dict[op.nodes[0]] = s_domain
                        s_domain = set()
                op.s_domain = s_domain
                op.t_domain = t_domain
            elif isinstance(op, Correction):
                domain = set(op.domain)
                expand_domain(domain)
                op.domain = domain
        return signal_dict
