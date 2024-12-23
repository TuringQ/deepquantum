"""
Measurement pattern
"""

from copy import copy, deepcopy
from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt
from networkx import MultiDiGraph, draw_networkx_nodes, draw_networkx_edges, draw_networkx_labels, multipartite_layout
import torch
from torch import nn

from .command import Node, Entanglement, Measurement, Correction
from .operation import Operation
from .state import GraphState


class Pattern(Operation):
    """Measurement-based quantum computing (MBQC) pattern.

    A Pattern represents a measurement-based quantum computation, which consists of a sequence of
    operations (preparation, entanglement, measurement, and correction) applied to qubits arranged
    in a graph structure.

    Args:
        name (str or None, optional): The name of the pattern. Default: ``None``
    """
    def __init__(
        self,
        nodes_state: Union[int, List[int], None] = None,
        state: Any = 'plus',
        edges: Optional[List] = None,
        nodes: Union[int, List[int], None] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name, nodes=None)
        self.state = GraphState(nodes_state, state, edges, nodes)
        self.init_graph = deepcopy(self.state.graph.graph)
        self.commands = nn.Sequential()
        self.encoders = []
        self.npara = 0
        self.ndata = 0

    def forward(self):
        """Perform a forward pass of the MBQC pattern and return the final state."""
        self.state = self.commands(self.state)
        return self.state.graph.full_state

    def add_graph(self,
        nodes_state: Union[int, List[int], None] = None,
        state: Any = 'plus',
        edges: Optional[List] = None,
        nodes: Union[int, List[int], None] = None,
        index: Optional[int] = None
    ) -> None:
        """
        Sets the underlying graph structure of the MBQC pattern.

        Takes a graph specification in the form of [vertices, edges] and constructs
        the corresponding graph structure by adding nodes and edges to the pattern.

        Args:
            graph (List[List]): A list containing two elements:
                - vertices: List of vertex indices
                - edges: List of pairs representing edges
        """
        self.state.add_subgraph(nodes_state=nodes_state, state=state, edges=edges, nodes=nodes, index=index)

    def get_graph(self):
        return self.state.graph

    def add(
        self,
        op: Operation,
        encode: bool = False
    ) -> None:
        """A method that adds an operation to the mbqc pattern.

        Args:
            op (Operation): The operation to add. It is an instance of ``Operation`` class or its subclasses,
                such as ``Node``, or ``Measurement``.
            encode (bool): Whether the gate is to encode data. Default: ``False``
        """
        assert isinstance(op, Operation)
        self.commands.append(op)
        if encode:
            assert not op.requires_grad, 'Please set requires_grad of the operation to be False'
            self.encoders.append(op)
            self.ndata += op.npara
        else:
            self.npara += op.npara

    def n(self, node: Union[int, List[int]] = None):
        """
        Add a new node to the pattern.

        Args:
            node (Union[int, List[int]], optional): Index or list of indices for the new node.
                Default: ``None``
        """
        node_ = Node(nodes=node)
        self.add(node_)

    def e(self, node: List[int] = None):
        """
        Add an entanglement edge between two nodes in the pattern.

        Args:
            node (List[int], optional): A list of two integers specifying the nodes to entangle.
                Must reference existing nodes in the pattern.
        """
        entang_ = Entanglement(node1=node[0], node2=node[1])
        self.add(entang_)

    def m(
        self,
        node: int = None,
        plane: Optional[str] = 'XY',
        angle: float = 0,
        t_domain: Union[int, List[int]] = [],
        s_domain: Union[int, List[int]] = []
    ):
        """
        Add a measurement operation to the pattern.

        Args:
            node (Optional[int]): The node to measure.
            plane (Optional[str]): Measurement plane ('XY', 'YZ', or 'XZ'). Defaults to 'XY'.
            angle (float): Measurement angle in radians. Defaults to 0.
            t_domain (Union[int, List[int]]): List of nodes that contribute to the Z correction.
                Defaults to empty list.
            s_domain (Union[int, List[int]]): List of nodes that contribute to the X correction.
                Defaults to empty list.
        """
        mea_op = Measurement(nodes=node, plane=plane, angle=angle, t_domain=t_domain, s_domain=s_domain)
        self.add(mea_op)

    def c_x(self, node: int = None, domain: List[int] = None):
        """
        Add an X correction operation to the pattern.

        Args:
            node (int, optional): The node to apply the X correction to.
            domain (List[int], optional): List of measurement results that determine
                if the correction should be applied.
        """
        c_x = Correction(nodes=node, basis='x', domain=domain)
        self.add(c_x)

    def c_z(self, node: int = None, domain: List[int] = None):
        """Add a Z correction operation to the pattern.

        Args:
            node (int, optional): The node to apply the Z correction to.
            domain (List[int], optional): List of measurement results that determine
                if the correction should be applied.
        """
        c_z = Correction(nodes=node, basis='z', domain=domain)
        self.add(c_z)

    def draw(self, width: int=4):
        """
        Draw MBQC pattern
        """
        g = MultiDiGraph(self.init_graph)
        nodes_init = deepcopy(g.nodes())
        for i in nodes_init:
            g.nodes[i]['layer'] = 0
        nodes_measured = []
        edges_t_domain = []
        edges_s_domain = []
        for op in self.commands:
            if isinstance(op, Node):
                g.add_node(*op.nodes,layer=2)
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
        draw_networkx_nodes(g,pos,nodelist=nodes_init, node_color='#1f78b4', node_shape='s')
        draw_networkx_nodes(g,pos,nodelist=nodes_measured, node_color='#1f78b4')
        draw_networkx_nodes(g,pos,nodelist=list(set(g.nodes()) - set(nodes_measured)), node_color='#d7dde0', node_shape='o')
        draw_networkx_edges(g, pos, g.edges(), arrows=False)
        draw_networkx_edges(g, pos, edges_t_domain, arrows=True, style=':', edge_color='#4cd925', connectionstyle='arc3,rad=-0.2')
        draw_networkx_edges(g, pos, edges_s_domain, arrows=True, style=':', edge_color='#db1d2c', connectionstyle='arc3,rad=0.2')
        draw_networkx_labels(g, pos)
        plt.plot([], [], color="k",label="graph edge")
        plt.plot([], [], ':', color="#4cd925", label="xflow")
        plt.plot([], [], ':', color="#db1d2c", label="zflow")
        plt.plot([], [], 's', color="#1f78b4", label="input nodes")
        plt.plot([], [], 'o', color="#d7dde0", label="output nodes")
        plt.xlim(-width/2,width/2)
        plt.legend(loc="upper right", fontsize=10)
        plt.tight_layout()
        plt.show()

    def is_standard(self) -> bool:
        """Determine whether the command sequence is standard.

        Returns
        -------
        is_standard : bool
            True if the pattern follows NEMC standardization, False otherwise
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

    def standardize(self):
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

        See https://arxiv.org/pdf/0704.1263 ch.(5.4)
        """
        # Initialize lists for each operation type
        n_list = []  # Node operations
        e_list = []  # Entanglement operations
        m_list = []  # Measurement operations
        z_dict = {}  # Tracks Z corrections by node
        x_dict = {}  # Tracks X corrections by node

        def add_correction_domain(domain_dict: dict, node, domain) -> None:
            """Helper function to update correction domains with XOR operation"""
            if previous_domain := domain_dict.get(node):
                previous_domain = previous_domain ^ domain
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

    def signal_shifting(self):
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

        See https://arxiv.org/pdf/0704.1263 ch.(5.5)
        """
        # Initialize lists for each operation type
        n_list = []  # Node operations
        e_list = []  # Entanglement operations
        m_list = []  # Measurement operations
        z_dict = {}  # Tracks Z corrections by node
        x_dict = {}  # Tracks X corrections by node

        def add_correction_domain(domain_dict: dict, node, domain) -> None:
            """Helper function to update correction domains with XOR operation"""
            if previous_domain := domain_dict.get(node):
                previous_domain = previous_domain ^ domain
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
