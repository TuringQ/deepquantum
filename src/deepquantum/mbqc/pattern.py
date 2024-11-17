"""
Measurement pattern
"""

from copy import copy, deepcopy
from typing import Any, List, Optional, Tuple, Union

import torch
from networkx import Graph, draw_networkx
from torch import nn

from . import gate
from .command import Node, Entanglement, Measurement, Correction, CorrectionX, CorrectionZ
from .operation import Operation
from .qmath import list_xor
from ..qmath import multi_kron, inverse_permutation


class Pattern(Operation):
    """Measurement-based quantum computing (MBQC) pattern.

    A Pattern represents a measurement-based quantum computation, which consists of a sequence of
    operations (preparation, entanglement, measurement, and corrections) applied to qubits arranged
    in a graph structure.

    Args:
        n_input_nodes (int): The number of input qubits in the pattern
        init_state (Any, optional): Initial quantum state. If None, initializes to |+⟩^⊗n state
            Default: ``None``.
        name (str, optional): Name identifier for the pattern
    """
    def __init__(
        self,
        n_input_nodes: int,
        init_state: Any = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name, n_input_nodes=n_input_nodes, node=list(range(n_input_nodes)))
        self._bg_state = None
        self._tot_qubit = n_input_nodes
        self.n_input_nodes = n_input_nodes
        self._graph = None
        self.commands = nn.Sequential()
        self.encoders = []
        self.npara = 0
        self.ndata = 0
        self._node_list = list(range(self.n_input_nodes))
        self._edge_list = [ ]
        self.measured_dic = {}
        self.unmeasured_list = list(range(self.n_input_nodes))
        self.nout_wire_dic = {}

        if init_state is None:
            plus_state = torch.sqrt(torch.tensor(2))*torch.tensor([1,1])/2
            init_state = multi_kron([plus_state] * n_input_nodes)
        if not isinstance(init_state, torch.Tensor):
            init_state = torch.tensor(init_state)
        if init_state.ndim == 1:
            init_state = init_state.unsqueeze(0)
        assert init_state[0].numel() == 2 ** n_input_nodes
        self.init_state = init_state

    def set_graph(self, graph: List[List]):
        """
        Sets the underlying graph structure of the MBQC pattern.

        Takes a graph specification in the form of [vertices, edges] and constructs
        the corresponding graph structure by adding nodes and edges to the pattern.

        Args:
            graph (List[List]): A list containing two elements:
                - vertices: List of vertex indices
                - edges: List of pairs representing edges
        """
        vertices, edges = graph
        assert len(vertices) > self.n_input_nodes
        for i in vertices:
            if i not in self._node_list:
                self.n(i)
        for edge in edges:
            self.e(edge)
        return

    def _get_graph(self):
        assert len(self._node_list) == self._tot_qubit
        g = Graph()
        g.add_nodes_from(self._node_list)
        g.add_edges_from(self._edge_list)
        self._graph = g
        return g

    def add(
        self,
        op: Operation,
        encode: bool = False,
        node: Union[int, List[int], None] = None
    ) -> None:
        """A method that adds an operation to the mbqc pattern.

        Args:
            op (Operation): The operation to add. It is an instance of ``Operation`` class or its subclasses,
                such as ``Node``, or ``Measurement``.
            encode (bool): Whether the gate is to encode data. Default: ``False``
            node (Union[int, List[int], None]): The node to apply the gate on. It can be an integer
                or a list of integers specifying the indices of the node. Default: ``None``
        """
        assert isinstance(op, Operation)
        if node is not None:
            node = self._convert_indices(node)
            assert len(node) == len(op.node), 'Invalid input'
            op = copy(op)
            op.node = node
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
        node_ = Node(node=node)
        assert node_.node[0] not in self._node_list, 'node already exists'
        self._node_list.append(node_.node[0])
        self.add(node_)
        self._tot_qubit += 1
        self.unmeasured_list.append(node_.node[0])

    def e(self, node: List[int] = None):
        """
        Add an entanglement edge between two nodes in the pattern.

        Args:
            node (List[int], optional): A list of two integers specifying the nodes to entangle.
                Must reference existing nodes in the pattern.
        """
        assert node[0] in self._node_list and node[1] in self._node_list, \
            'no command acts on a qubit not yet prepared, unless it is an input qubit'
        entang_ = Entanglement(node=node)
        self._edge_list.append(node)
        self.add(entang_)

    def m(
        self,
        node: Optional[int] = None,
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
        mea_op = Measurement(node=node, plane=plane, angle=angle, t_domain=t_domain, s_domain=s_domain)
        self.add(mea_op)

    def x(self, node: int = None, signal_domain: List[int] = None):
        """
        Add an X correction operation to the pattern.

        Args:
            node (int, optional): The node to apply the X correction to.
            signal_domain (List[int], optional): List of measurement results that determine
                if the correction should be applied.
        """
        assert node in self._node_list, 'no command acts on a qubit not yet prepared, unless it is an input qubit'
        x_ = CorrectionX(node=node, signal_domain=signal_domain)
        self.add(x_)

    def z(self, node: int = None, signal_domain: List[int] = None):
        """Add a Z correction operation to the pattern.

        Args:
            node (int, optional): The node to apply the Z correction to.
            signal_domain (List[int], optional): List of measurement results that determine
                if the correction should be applied.
        """
        assert node in self._node_list, 'no command acts on a qubit not yet prepared, unless it is an input qubit'
        z_ = CorrectionZ(node=node, signal_domain=signal_domain)
        self.add(z_)

    def forward(self):
        """
         Execute the MBQC pattern by applying commands in sequence
         to the quantum state
        """
        state = self.init_state
        for op in self.commands:
            self._check_measured(op.node)
            if isinstance(op, Measurement):
                node = self.unmeasured_list.index(op.node[0])
                state = op.forward(node, state, self.measured_dic)
                self.measured_dic[op.node[0]] = op.sample
                del self.unmeasured_list[node]
            elif isinstance(op, (CorrectionX, CorrectionZ)):
                node = self.unmeasured_list.index(op.node[0])
                state = op.forward(node, state, self.measured_dic)
            elif isinstance(op, Entanglement):
                node = [self.unmeasured_list.index(op.node[0]), self.unmeasured_list.index(op.node[1])]
                state = op.forward(node, state)
            else:
                state = op.forward(state)
        self._bg_state = state
        if self.nout_wire_dic != {}:
            index = torch.argsort(torch.tensor(list(self.nout_wire_dic.values()))).tolist()
            batch_size = state.size(0)
            state = state.reshape(batch_size, -1)
            n_node =  int(torch.log2(torch.tensor(state.size(1))))
            idx2 = [0] + [i+1 for i in index]
            inv_perm = inverse_permutation(idx2)
            state = state.reshape([batch_size] + [2] * n_node).permute(*inv_perm).reshape(batch_size, -1)
        return state.squeeze()

    def _check_measured(self, node):
        """
        check if the qubit already measured.
        """
        measured_list = list(self.measured_dic.keys())
        for i in node:
            if i in measured_list:
                raise ValueError (f'qubit {i} already measured')
        return

    def draw(self, wid: int=3):
        """
        Draw MBQC pattern
        """
        g = self._get_graph()
        pos = {}
        for i in self._node_list:
            pos_x = i % wid
            pos_y = i // wid
            pos[i] = (pos_x, -pos_y)
        measured_nq = list(self.measured_dic.keys())
        node_colors = ['gray' if i in measured_nq else 'green' for i in self._node_list]
        node_edge_colors = ['red' if i < self.n_input_nodes else 'black' for i in self._node_list]
        draw_networkx(g, pos=pos,
                      node_color=node_colors,
                      edgecolors=node_edge_colors,
                      node_size=500,
                      width=2)
        return

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
                previous_domain = list_xor(previous_domain, domain)
            else:
                domain_dict[node] = domain.copy()

        # Process each operation and reorganize into standard form
        for op in self.commands:
            if isinstance(op, Node):
                n_list.append(op)
            elif isinstance(op, Entanglement):
                for side in (0, 1):
                    # Propagate X corrections through entanglement (generates Z corrections)
                    if s_domain := x_dict.get(op.node[side], None):
                        add_correction_domain(z_dict, op.node[1 - side], s_domain)
                e_list.append(op)
            elif isinstance(op, Measurement):
                # Apply pending corrections to measurement parameters
                new_op = deepcopy(op)
                if t_domain := z_dict.pop(op.node[0], None):
                    new_op.t_domain = list_xor(new_op.t_domain, t_domain)
                if s_domain := x_dict.pop(op.node[0], None):
                    new_op.s_domain = list_xor(new_op.s_domain, s_domain)
                m_list.append(new_op)
            elif isinstance(op, CorrectionZ):
                add_correction_domain(z_dict, op.node[0], op.signal_domain)
            elif isinstance(op, CorrectionX):
                add_correction_domain(x_dict, op.node[0], op.signal_domain)

        # Reconstruct command sequence in standard order
        self.commands = nn.Sequential(
                    *n_list,
                    *e_list,
                    *m_list,
                    *(CorrectionZ(node=node, signal_domain=domain) for node, domain in z_dict.items()),
                    *(CorrectionX(node=node, signal_domain=domain) for node, domain in x_dict.items())
        )

    def _update(self):
        if len(self.measured_dic) == 0:
            self._tot_qubit = len(self._node_list)
            self.unmeasured_list = list(range(self._tot_qubit))
        return

    def _apply_single(
        self,
        gate_func,
        input_node: int,
        required_ancilla: int,
        ancilla: Optional[List[int]]=None,
        **kwargs
    ):
        """
        Helper method to apply quantum gate patterns.

        Args:
            gate: Gate function to apply (h, pauli_x, pauli_y, etc.)
            input_node: Input qubit node
            required_ancilla: Number of required ancilla qubits
            ancilla: Optional ancilla qubits
        """
        if ancilla is None:
            ancilla = list(range(self._tot_qubit, self._tot_qubit + required_ancilla))
        pattern = gate_func(input_node, ancilla, **kwargs)
        self.commands += pattern[0]
        self._node_list += pattern[1]
        self._edge_list += pattern[2]
        self._update()

    def h(self, input_node: int, ancilla: Optional[List[int]]=None):
        """Apply Hadamard gate to specified input node.

        Args:
            input_node (int): Index of the node to apply the Hadamard gate to
            ancilla (Optional[List[int]], optional): List of ancilla node indices. If None,
                a new ancilla node will be allocated. Defaults to None.
        """
        self._apply_single(gate.h, input_node, 1, ancilla)

    def pauli_x(self, input_node: int, ancilla: Optional[List[int]]=None):
        """Apply Pauli-X gate to specified input node.

        Args:
            input_node (int): Index of the node to apply the Hadamard gate to
            ancilla (Optional[List[int]], optional): List of ancilla node indices. If None,
                two new ancilla nodes will be allocated. Defaults to None.
        """
        self._apply_single(gate.pauli_x, input_node, 2, ancilla)

    def pauli_y(self, input_node: int, ancilla: Optional[List[int]]=None):
        """Apply Pauli-Y gate to specified input node.

        Args:
            input_node (int): Index of the node to apply the Hadamard gate to
            ancilla (Optional[List[int]], optional): List of ancilla node indices. If None,
                4 new ancilla nodes will be allocated. Defaults to None.
        """
        self._apply_single(gate.pauli_y, input_node, 4, ancilla)

    def pauli_z(self, input_node: int, ancilla: Optional[List[int]]=None):
        """Apply Pauli-Z gate to specified input node.

        Args:
            input_node (int): Index of the node to apply the Hadamard gate to
            ancilla (Optional[List[int]], optional): List of ancilla node indices. If None,
                two new ancilla nodes will be allocated. Defaults to None.
        """
        self._apply_single(gate.pauli_z, input_node, 2, ancilla)

    def s(self, input_node: int, ancilla: Optional[List[int]]=None):
        """Apply S gate to specified input node.

        Args:
            input_node (int): Index of the node to apply the Hadamard gate to
            ancilla (Optional[List[int]], optional): List of ancilla node indices. If None,
                two new ancilla nodes will be allocated. Defaults to None.
        """
        self._apply_single(gate.s, input_node, 2, ancilla)

    def rx(
        self,
        input_node: int,
        theta: Optional[torch.Tensor]=None,
        ancilla: Optional[List[int]]=None
    ):
        """Apply rotation around X-axis to specified input node.

        Args:
            input_node (int): Index of the node to apply the RX gate to
            theta (Optional[torch.Tensor], optional): Rotation angle. Defaults to None.
            ancilla (Optional[List[int]], optional): List of ancilla node indices. If None,
                two new ancilla nodes will be allocated. Defaults to None.
        """
        self._apply_single(gate.rx, input_node, required_ancilla=2, ancilla=ancilla, theta=theta)

    def ry(
        self,
        input_node: int,
        theta: Optional[torch.Tensor]=None,
        ancilla: Optional[List[int]]=None
    ):
        """Apply rotation around Y-axis to specified input node.

        Args:
            input_node (int): Index of the node to apply the RY gate to
            theta (Optional[torch.Tensor], optional): Rotation angle. Defaults to None.
            ancilla (Optional[List[int]], optional): List of ancilla node indices. If None,
                4 new ancilla nodes will be allocated. Defaults to None.
        """
        self._apply_single(gate.ry, input_node, required_ancilla=4, ancilla=ancilla, theta=theta)

    def rz(
        self,
        input_node: int,
        theta: Optional[torch.Tensor]=None,
        ancilla: Optional[List[int]]=None
    ):
        """Apply rotation around Z-axis to specified input node.

        Args:
            input_node (int): Index of the node to apply the RZ gate to
            theta (Optional[torch.Tensor], optional): Rotation angle. Defaults to None.
            ancilla (Optional[List[int]], optional): List of ancilla node indices. If None,
                two new ancilla nodes will be allocated. Defaults to None.
        """
        self._apply_single(gate.rz, input_node, required_ancilla=2, ancilla=ancilla, theta=theta)

    def cnot(self, control_node: int, target_node: int, ancilla: Optional[List[int]]=None):
        """Apply CNOT (controlled-NOT) gate between control and target nodes.

        Args:
            control_node (int): Index of the control node
            target_node (int): Index of the target node
            ancilla (Optional[List[int]], optional): List of ancilla node indices. If None,
                new ancilla nodes will be allocated. Defaults to None.
        """
        if ancilla is None:
            ancilla = [self._tot_qubit, self._tot_qubit+1]
        pattern_cnot = gate.cnot(control_node, target_node, ancilla)
        self.commands += pattern_cnot[0]
        self._node_list += pattern_cnot[1]
        self._edge_list += pattern_cnot[2]
        self._update()
