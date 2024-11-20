"""
Quantum states
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import networkx as nx
import numpy as np
import torch
from torch import nn

from ..circuit import QubitCircuit
from ..qmath import multi_kron, inverse_permutation
from ..state import QubitState


class SubGraphState(nn.Module):
    """A subgraph state of a quantum state.

    Args:
    """
    def __init__(
        self,
        nodes_state: Union[int, List[int], None] = None,
        state: Any = 'plus',
        edges: Optional[List] = None,
        nodes: Union[int, List[int], None] = None # primarily, for the single-node case
    ) -> None:
        super().__init__()
        self.set_graph(nodes_state, edges, nodes)
        self.set_state(state)
        self.measure_dict = defaultdict(list) # record the measurement results: {node: batched_bit}

    @property
    def nodes(self, **kwargs):
        return self.graph.nodes(**kwargs)

    @property
    def edges(self, **kwargs):
        return self.graph.edges(**kwargs)

    @property
    def full_state(self) -> torch.Tensor:
        nqubit = len(self.nodes)
        nodes_bg = list(self.nodes)
        for i in self.nodes_state:
            nodes_bg.remove(i)
        nodes = self.nodes_state + nodes_bg
        wires = [0] + list(map(lambda node: self.node2wire_dict[node] + 1, nodes)) # [0] for batch
        plus = torch.tensor([[1], [1]], dtype=self.state.dtype, device=self.state.device) / 2 ** 0.5
        init_state = multi_kron([self.state] + [plus] * len(nodes_bg)).reshape([-1] + [2] * nqubit)
        init_state = init_state.permute(inverse_permutation(wires)).reshape([-1, 2 ** nqubit])
        cir = QubitCircuit(nqubit=nqubit, init_state=init_state)
        edges = list(filter(lambda edge: edge[2]['cz'], self.edges(data=True)))
        for edge in edges:
            cir.cz(self.node2wire_dict[edge[0]], self.node2wire_dict[edge[1]])
        cir.to(init_state.real.dtype).to(init_state.device)
        return cir()

    def set_graph(
        self,
        nodes_state: Union[int, List[int], None] = None,
        edges: Optional[List] = None,
        nodes: Union[int, List[int], None] = None
    ) -> None:
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
            nx.add_cycle(graph, nodes_state, cz=False) # 'cz' is the label for entanglement
        else:
            graph.add_nodes_from(nodes_state)
        graph.add_edges_from(edges, cz=True)
        graph.add_nodes_from(nodes)
        self.graph = graph
        self.nodes_state = nodes_state
        self.update_node2wire_dict()

    def set_state(self, state: Any = 'plus') -> None:
        nqubit = len(self.nodes_state)
        if state == 'plus':
            state = torch.tensor([1, 1]) / 2 ** 0.5 + 0j
            if nqubit > 0:
                state = multi_kron([state] * nqubit)
        elif not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.cfloat)
        if nqubit > 0:
            self.register_buffer('state', QubitState(nqubit, state).state)
        else:
            self.register_buffer('state', torch.tensor(1))

    def add_nodes(self, nodes: Union[int, List[int]]) -> None:
        if isinstance(nodes, int):
            nodes = [nodes]
        self.graph.add_nodes_from(nodes)
        self.update_node2wire_dict()

    def add_edges(self, edges: List) -> None:
        self.graph.add_edges_from(edges, cz=True)
        self.update_node2wire_dict()

    def shift_labels(self, n: int) -> None:
        self.graph = nx.relabel_nodes(self.graph, lambda x: x + n)
        self.nodes_state = (np.array(self.nodes_state) + n).tolist()
        self.measure_dict = {k + n: v for k, v in self.measure_dict.items()}
        self.update_node2wire_dict()

    def compose(self, other: 'SubGraphState', relabel: bool = True) -> 'SubGraphState':
        if relabel and (set(self.nodes) & set(other.nodes)):
            shift = max(self.nodes) - min(other.nodes) + 1
            other.shift_labels(shift)
        graph = nx.compose(self.graph, other.graph)
        for i in other.nodes_state:
            assert i not in self.nodes_state, 'Do NOT use repeated nodes for states'
        nodes_state = self.nodes_state + other.nodes_state
        state = torch.kron(self.state, other.state)
        sgs = SubGraphState(nodes_state, state, graph.edges(data=True), graph.nodes)
        sgs.measure_dict = defaultdict(list)
        sgs.measure_dict.update(self.measure_dict)
        sgs.measure_dict.update(other.measure_dict)
        return sgs

    def update_node2wire_dict(self) -> Dict:
        wires = inverse_permutation(np.argsort(self.nodes).tolist())
        self.node2wire_dict = {node: wire for node, wire in zip(self.nodes, wires)}
        return self.node2wire_dict

    def draw(self, **kwargs):
        nx.draw(self.graph, with_labels=True, **kwargs)

    def extra_repr(self) -> str:
        return f'nodes_state={self.nodes_state}, nodes={self.nodes}'


class GraphState(nn.Module):
    """A graph state of a quantum state.

    Args:
    """
    def __init__(
        self,
        nodes_state: Union[int, List[int], None] = None,
        state: Any = 'plus',
        edges: Optional[List] = None,
        nodes: Union[int, List[int], None] = None
    ) -> None:
        super().__init__()
        if nodes_state is None and edges is None and nodes is None:
            self.subgraphs = nn.ModuleList()
        else:
            sgs = SubGraphState(nodes_state, state, edges, nodes)
            self.subgraphs = nn.ModuleList([sgs])

    def add_subgraph(
        self,
        nodes_state: Union[int, List[int], None] = None,
        state: Any = 'plus',
        edges: Optional[List] = None,
        nodes: Union[int, List[int], None] = None,
        measure_dict: Dict = None,
        index: Optional[int] = None
    ) -> None:
        sgs = SubGraphState(nodes_state, state, edges, nodes)
        if measure_dict is not None:
            sgs.measure_dict = measure_dict
        if index is None:
            self.subgraphs.append(sgs)
        else:
            self.subgraphs.insert(index, sgs)

    @property
    def graph(self) -> SubGraphState:
        graph = None
        for subgraph in self.subgraphs:
            if graph is None:
                graph = subgraph
            else:
                graph = graph.compose(subgraph, relabel=True)
        if graph is None:
            return SubGraphState()
        else:
            return graph

    @property
    def measure_dict(self) -> Dict:
        return self.graph.measure_dict
