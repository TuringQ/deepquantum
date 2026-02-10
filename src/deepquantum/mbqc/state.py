"""
Quantum states
"""

from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch import nn, vmap

from ..circuit import QubitCircuit
from ..qmath import inverse_permutation, multi_kron
from ..state import QubitState


class SubGraphState(nn.Module):
    """A subgraph state of a quantum state.

    Args:
        nodes_state (int, List[int] or None, optional): The nodes of the input state in the subgraph state.
            It can be an integer representing the number of nodes or a list of node indices. Default: ``None``
        state (Any, optional): The input state of the subgraph state. The string representation of state
            could be ``'plus'``, ``'minus'``, ``'zero'``, and ``'one'``. Default: ``'plus'``
        edges (List or None, optional): Additional edges connecting the nodes in the subgraph state. Default: ``None``
        nodes (int, List[int] or None, optional): Additional nodes to include in the subgraph state. Default: ``None``
    """

    def __init__(
        self,
        nodes_state: int | list[int] | None = None,
        state: Any = 'plus',
        edges: list | None = None,
        nodes: int | list[int] | None = None,  # primarily, for the single-node case
    ) -> None:
        super().__init__()
        self.nodes_out_seq = None
        self.set_graph(nodes_state, edges, nodes)
        self.set_state(state)
        self.measure_dict = defaultdict(list)  # record the measurement results: {node: batched_bit}

    def to(self, arg: Any) -> 'SubGraphState':
        """Set dtype or device of the ``SubGraphState``."""
        if arg == torch.float:
            self.state = self.state.to(torch.cfloat)
        elif arg == torch.double:
            self.state = self.state.to(torch.cdouble)
        else:
            self.state = self.state.to(arg)
        return self

    @property
    def nodes(self, **kwargs):
        """Nodes of the graph."""
        return self.graph.nodes(**kwargs)

    @property
    def edges(self, **kwargs):
        """Edges of the graph."""
        return self.graph.edges(**kwargs)

    @property
    def full_state(self) -> torch.Tensor:
        """Compute and return the full quantum state of the subgraph state."""
        nqubit = len(self.nodes)
        nodes_bg = list(self.nodes)
        for i in self.nodes_state:
            nodes_bg.remove(i)
        nodes = self.nodes_state + nodes_bg
        wires = [0] + list(map(lambda node: self.node2wire_dict[node] + 1, nodes))  # [0] for batch
        plus = torch.tensor([[1], [1]], dtype=self.state.dtype, device=self.state.device) / 2**0.5
        init_state = multi_kron([self.state] + [plus] * len(nodes_bg)).reshape([-1] + [2] * nqubit)
        init_state = init_state.permute(inverse_permutation(wires)).reshape([-1, 2**nqubit])
        cir = QubitCircuit(nqubit=nqubit, init_state=init_state)
        edges = list(filter(lambda edge: edge[2]['cz'], self.edges(data=True)))
        for edge in edges:
            cir.cz(self.node2wire_dict[edge[0]], self.node2wire_dict[edge[1]])
        cir.to(init_state.real.dtype).to(init_state.device)
        return cir()

    def set_graph(
        self,
        nodes_state: int | list[int] | None = None,
        edges: list | None = None,
        nodes: int | list[int] | None = None,
    ) -> None:
        """Set the graph structure for the subgraph state."""
        if nodes_state is None:
            nodes_state = []
        elif isinstance(nodes_state, int):
            nodes_state = list(range(nodes_state))
        if edges is None:
            edges = []
        if nodes is None:
            nodes = []
        elif isinstance(nodes, int):
            nodes = [nodes]
        graph = nx.Graph()
        if len(nodes_state) > 1:
            nx.add_cycle(graph, nodes_state, cz=False)  # 'cz' is the label for entanglement
        else:
            graph.add_nodes_from(nodes_state)
        graph.add_edges_from(edges, cz=True)
        graph.add_nodes_from(nodes)
        self.graph = graph
        self.nodes_state = nodes_state
        self.update_node2wire_dict()

    def set_state(self, state: Any = 'plus') -> None:
        """Set the input state of the subgraph state."""
        nqubit = len(self.nodes_state)
        if isinstance(state, str):
            if state == 'plus':
                state = torch.tensor([1, 1]) / 2**0.5 + 0j
            elif state == 'minus':
                state = torch.tensor([1, -1]) / 2**0.5 + 0j
            elif state == 'zero':
                state = torch.tensor([1, 0]) + 0j
            elif state == 'one':
                state = torch.tensor([0, 1]) + 0j
            if nqubit > 0:
                state = multi_kron([state] * nqubit)
        elif not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.cfloat)
        if nqubit > 0:
            self.register_buffer('state', QubitState(nqubit, state).state)
        else:
            self.register_buffer('state', torch.tensor(1, dtype=state.dtype, device=state.device))

    def set_nodes_out_seq(self, nodes: list[int] | None = None) -> None:
        """Set the output sequence of the nodes."""
        if nodes is not None:
            assert len(nodes) == len(self.nodes)
            assert set(nodes) == set(self.nodes)
        self.nodes_out_seq = nodes
        self.update_node2wire_dict()

    def add_nodes(self, nodes: int | list[int]) -> None:
        """Add nodes to the subgraph state."""
        if isinstance(nodes, int):
            nodes = [nodes]
        self.graph.add_nodes_from(nodes)
        self.update_node2wire_dict()

    def add_edges(self, edges: list) -> None:
        """Add edges to the subgraph state."""
        self.graph.add_edges_from(edges, cz=True)
        self.update_node2wire_dict()

    def shift_labels(self, n: int) -> None:
        """Shift the labels of the nodes in the graph by a given integer."""
        self.graph = nx.relabel_nodes(self.graph, lambda x: x + n)
        self.nodes_state = (np.array(self.nodes_state) + n).tolist()
        self.measure_dict = {k + n: v for k, v in self.measure_dict.items()}
        self.update_node2wire_dict()

    def compose(self, other: 'SubGraphState', relabel: bool = True) -> 'SubGraphState':
        """Compose this subgraph state with another subgraph state.

        Args:
            other (SubGraphState): The other subgraph state to compose with.
            relabel (bool, optional): Whether to relabel nodes to avoid conflicts. Default: ``True``

        Returns:
            SubGraphState: A new subgraph state that is the composition of the two.
        """
        if relabel and (set(self.nodes) & set(other.nodes)):
            shift = max(self.nodes) - min(other.nodes) + 1
            other.shift_labels(shift)
        graph = nx.compose(self.graph, other.graph)
        for i in other.nodes_state:
            assert i not in self.nodes_state, 'Do NOT use repeated nodes for states'
        nodes_state = self.nodes_state + other.nodes_state
        if self.state.ndim == other.state.ndim == 3:
            if self.state.shape[0] == 1 or other.state.shape[0] == 1:
                state = torch.kron(self.state, other.state)
            else:
                state = vmap(torch.kron)(self.state, other.state)
        else:
            state = torch.kron(self.state, other.state)
        sgs = SubGraphState(nodes_state, state, graph.edges(data=True), graph.nodes)
        sgs.measure_dict = defaultdict(list)
        sgs.measure_dict.update(self.measure_dict)
        sgs.measure_dict.update(other.measure_dict)
        return sgs

    def update_node2wire_dict(self) -> dict:
        """Update the mapping from nodes to wire indices.

        Returns:
            Dict: A dictionary mapping nodes to their corresponding wire indices.
        """
        if self.nodes_out_seq is None:
            wires = inverse_permutation(np.argsort(self.nodes).tolist())
            self.node2wire_dict = {node: wire for node, wire in zip(self.nodes, wires, strict=True)}
        else:
            self.node2wire_dict = {node: i for i, node in enumerate(self.nodes_out_seq)}
        return self.node2wire_dict

    def draw(self, **kwargs):
        """Draw the graph using NetworkX."""
        nx.draw(self.graph, with_labels=True, **kwargs)

    def extra_repr(self) -> str:
        return f'nodes_state={self.nodes_state}, nodes={self.nodes}'


class GraphState(nn.Module):
    """A graph state composed by several SubGraphStates.

    Args:
        nodes_state (int, List[int] or None, optional): The nodes of the input state in the initial graph state.
            It can be an integer representing the number of nodes or a list of node indices. Default: ``None``
        state (Any, optional): The input state of the initial graph state. The string representation of state
            could be ``'plus'``, ``'minus'``, ``'zero'``, and ``'one'``. Default: ``'plus'``
        edges (List or None, optional): Additional edges connecting the nodes in the initial graph state.
            Default: ``None``
        nodes (int, List[int] or None, optional): Additional nodes to include in the initial graph state.
            Default: ``None``
    """

    def __init__(
        self,
        nodes_state: int | list[int] | None = None,
        state: Any = 'plus',
        edges: list | None = None,
        nodes: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        sgs = SubGraphState(nodes_state, state, edges, nodes)
        self.subgraphs = nn.ModuleList([sgs])
        self.nodes_out_seq = None

    def to(self, arg: Any) -> 'GraphState':
        """Set dtype or device of the ``GraphState``."""
        for sgs in self.subgraphs:
            sgs.to(arg)
        return self

    def add_subgraph(
        self,
        nodes_state: int | list[int] | None = None,
        state: Any = 'plus',
        edges: list | None = None,
        nodes: int | list[int] | None = None,
        measure_dict: dict | None = None,
        index: int | None = None,
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
            measure_dict (Dict or None, optional): A dictionary containing all measurement results. Default: ``None``
            index (int or None, optional): The index where to insert the subgraph state. Default: ``None``
        """
        sgs = SubGraphState(nodes_state, state, edges, nodes)
        if index is None:
            dtype = self.subgraphs[0].state.real.dtype
            device = self.subgraphs[0].state.device
            sgs.to(dtype).to(device)
        if measure_dict is not None:
            sgs.measure_dict = measure_dict
        if index is None:
            self.subgraphs.append(sgs)
        else:
            self.subgraphs.insert(index, sgs)

    @property
    def graph(self) -> SubGraphState:
        """The combined graph state of all subgraph states."""
        graph = None
        for subgraph in self.subgraphs:
            graph = subgraph if graph is None else graph.compose(subgraph, relabel=True)
        graph.set_nodes_out_seq(self.nodes_out_seq)
        return graph

    @property
    def full_state(self) -> torch.Tensor:
        """Compute and return the full quantum state of the graph state."""
        return self.graph.full_state

    @property
    def measure_dict(self) -> dict:
        """A dictionary containing all measurement results for the graph state."""
        return self.graph.measure_dict

    def set_nodes_out_seq(self, nodes: list[int] | None = None) -> None:
        """Set the output sequence of the nodes."""
        self.nodes_out_seq = nodes
