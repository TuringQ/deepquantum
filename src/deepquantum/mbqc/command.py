"""
MBQC commands
"""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from ..circuit import QubitCircuit
from .operation import Command
from .state import GraphState


class Node(Command):
    """Node (qubit) preparation command.

    Args:
        nodes (int or List[int]): The indices of the nodes to prepare.
    """

    def __init__(self, nodes: int | list[int]) -> None:
        super().__init__(name='Node', nodes=nodes)

    def forward(self, x: GraphState) -> GraphState:
        """Perform a forward pass by adding `SubGraphState` in the `GraphState`."""
        x = super().forward(x)
        nodes = x.graph.nodes
        for node in self.nodes:
            assert node not in nodes, f'Node {node} already exists'
            x.add_subgraph(nodes=node)
        return x


class Entanglement(Command):
    """Entanglement command.

    Args:
        node1 (int): The first node index.
        node2 (int): The second node index.
    """

    def __init__(self, node1: int, node2: int) -> None:
        super().__init__(name='Entanglement', nodes=[node1, node2])

    def forward(self, x: GraphState) -> GraphState:
        """Perform a forward pass by adding an edge in the `GraphState`."""
        x = super().forward(x)
        idx1 = None
        idx2 = None
        for i, sgs in enumerate(x.subgraphs):
            if idx1 is not None and idx2 is not None:
                break
            if self.nodes[0] in sgs.graph:
                idx1 = i
            if self.nodes[1] in sgs.graph:
                idx2 = i
        assert idx1 is not None and idx2 is not None, f'Nodes {self.nodes} not found in the GraphState'
        if idx1 == idx2:
            x.subgraphs[idx1].add_edges([(self.nodes[0], self.nodes[1])])
        else:
            subgraph = x.subgraphs[idx1].compose(x.subgraphs[idx2])
            subgraph.add_edges([(self.nodes[0], self.nodes[1])])
            for i in sorted([idx1, idx2], reverse=True):
                x.subgraphs.pop(i)
            x.subgraphs.insert(0, subgraph)
        return x


class Measurement(Command):
    """Measurement command.

    Args:
        nodes (int or List[int]): The indices of the nodes to measure.
        angle (Any, optional): The measurement angle in radians. Default: 0.
        plane (str, optional): The measurement plane (``'xy'``, ``'yz'`` or ``'zx'``). Default: ``'xy'``
        s_domain (int, Iterable[int] or None, optional): The indices of the nodes that contribute to signal domain s.
            Default: ``None``
        t_domain (int, Iterable[int] or None, optional): The indices of the nodes that contribute to signal domain t.
            Default: ``None``
        requires_grad (bool, optional): Whether the parameter is ``nn.Parameter`` or ``buffer``.
            Default: ``False`` (which means ``buffer``)
    """

    def __init__(
        self,
        nodes: int | list[int],
        angle: Any = 0.0,
        plane: str = 'xy',
        s_domain: int | Iterable[int] | None = None,
        t_domain: int | Iterable[int] | None = None,
        requires_grad: bool = False,
    ) -> None:
        super().__init__(name='Measurement', nodes=nodes)
        self.plane = plane.lower()
        if s_domain is None:
            s_domain = []
        elif isinstance(s_domain, int):
            s_domain = [s_domain]
        if t_domain is None:
            t_domain = []
        elif isinstance(t_domain, int):
            t_domain = [t_domain]
        self.s_domain = set(s_domain)
        self.t_domain = set(t_domain)
        self.requires_grad = requires_grad
        self.init_para(angle)
        self.npara = 1

    def inputs_to_tensor(self, inputs: Any = None) -> torch.Tensor:
        """Convert inputs to torch.Tensor."""
        while isinstance(inputs, list):
            inputs = inputs[0]
        if inputs is None:
            inputs = torch.rand(1)[0] * 2 * torch.pi
        elif not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float)
        return inputs

    def forward(self, x: GraphState) -> GraphState:
        """Perform a forward pass by measuring the `GraphState`."""
        x = super().forward(x)
        idx = None
        for i, sgs in enumerate(x.subgraphs):
            if idx is not None:
                break
            if self.nodes[0] in sgs.graph:
                idx = i
        assert idx is not None, f'Node {self.nodes[0]} not found in the GraphState'
        sgs = x.subgraphs[idx]
        nqubit = len(sgs.nodes)
        init_state = sgs.full_state
        if init_state.ndim == 2:
            init_state = init_state.unsqueeze(0)
        wire = sgs.node2wire_dict[self.nodes[0]]
        if self.angle.ndim == 0:
            angle = self.angle.unsqueeze(0)
        else:
            angle = self.angle
        if len(self.s_domain) != 0:
            qs = sum(map(lambda s: torch.tensor(sgs.measure_dict[s], device=angle.device), self.s_domain))
            qs = qs.reshape(-1, 1)
        else:
            qs = torch.zeros(init_state.shape[0], device=angle.device).reshape(-1, 1)
        if len(self.t_domain) != 0:
            qt = sum(map(lambda t: torch.tensor(sgs.measure_dict[t], device=angle.device), self.t_domain))
            qt = qt.reshape(-1, 1)
        else:
            qt = torch.zeros(init_state.shape[0], device=angle.device).reshape(-1, 1)
        if self.plane in ['xy', 'yx']:
            alpha = (-1) ** qs * angle + torch.pi * qt
            # M^{XY,α} X^s Z^t = M^{XY,(-1)^s·α+tπ}
        elif self.plane in ['zx', 'xz']:
            alpha = (-1) ** (qs + qt) * angle + torch.pi * qs
            # M^{XZ,α} X^s Z^t = M^{XZ,(-1)^t((-1)^s·α+sπ)}
            #                  = M^{XZ,(-1)^{s+t}·α+(-1)^t·sπ}
            #                  = M^{XZ,(-1)^{s+t}·α+sπ}  (since (-1)^t·π ≡ π (mod 2π))
        elif self.plane in ['yz', 'zy']:
            alpha = (-1) ** qt * angle + torch.pi * (qs + qt)
            # positive Y axis as 0 angle
            # M^{YZ,α} X^s Z^t = M^{YZ,(-1)^t·α+(s+t)π)}
        cir = QubitCircuit(nqubit=nqubit)
        cir.j(wires=wire, plane=self.plane, encode=True)
        final_state = cir(data=alpha, state=init_state.squeeze(0))
        rst = cir.measure(shots=1, wires=wire)
        state = []
        if isinstance(rst, list):
            for i, d in enumerate(rst):
                ((k, _),) = d.items()
                state.append(cir._slice_state_vector(state=final_state[i], wires=wire, bits=k))
                sgs.measure_dict[self.nodes[0]].append(int(k))
        else:
            ((k, _),) = rst.items()
            state.append(cir._slice_state_vector(state=final_state[0], wires=wire, bits=k))
            sgs.measure_dict[self.nodes[0]].append(int(k))
        state = torch.stack(state)
        nodes_state = sorted(list(sgs.nodes))
        nodes_state.remove(self.nodes[0])
        x.subgraphs.pop(idx)
        x.add_subgraph(nodes_state=nodes_state, state=state, measure_dict=sgs.measure_dict, index=0)
        return x

    def init_para(self, angle: Any = None) -> None:
        """Initialize the parameters."""
        angle = self.inputs_to_tensor(angle)
        if self.requires_grad:
            self.angle = nn.Parameter(angle)
        else:
            self.register_buffer('angle', angle)

    def extra_repr(self) -> str:
        s = super().extra_repr() + f', plane={self.plane.upper()}, angle={self.angle.item()}'
        return s + f', s_domain={self.s_domain}, t_domain={self.t_domain}'


class Correction(Command):
    """Correction command.

    Args:
        nodes (int or List[int]): The indices of the nodes to correct.
        basis (str, optional): The type of correction (``'x'`` or ``'z'``). Default: ``'x'``
        domain (Union[int, Iterable[int], None], optional): The indices of the nodes that contribute to signal domain s.
            Default: ``None``
    """

    def __init__(self, nodes: int | list[int], basis: str = 'x', domain: int | Iterable[int] | None = None) -> None:
        super().__init__(name='Correction', nodes=nodes)
        self.basis = basis.lower()
        if domain is None:
            domain = []
        elif isinstance(domain, int):
            domain = [domain]
        self.domain = set(domain)

    def forward(self, x: GraphState) -> GraphState:
        """Perform a forward pass by correcting the `GraphState`."""
        x = super().forward(x)
        idx = None
        for i, sgs in enumerate(x.subgraphs):
            if idx is not None:
                break
            if self.nodes[0] in sgs.graph:
                idx = i
        assert idx is not None, f'Node {self.nodes[0]} not found in the GraphState'
        sgs = x.subgraphs[idx]
        nqubit = len(sgs.nodes)
        init_state = sgs.full_state
        if init_state.ndim == 2:
            init_state = init_state.unsqueeze(0)
        wire = sgs.node2wire_dict[self.nodes[0]]
        if len(self.domain) != 0:
            qs = sum(map(lambda s: torch.tensor(sgs.measure_dict[s], device=init_state.device), self.domain))
        else:
            qs = torch.zeros(init_state.shape[0], device=init_state.device)
        theta = torch.pi * qs.to(init_state.real.dtype)
        cir = QubitCircuit(nqubit=nqubit)
        if self.basis == 'x':
            cir.rx(wires=wire, encode=True)  # global phase
        elif self.basis == 'z':
            cir.rz(wires=wire, encode=True)  # global phase
        else:
            raise ValueError(f'Invalid basis {self.basis}')
        state = cir(data=theta.reshape(-1, 1).squeeze(0), state=init_state.squeeze(0))
        nodes_state = sorted(list(sgs.nodes))
        x.subgraphs.pop(idx)
        x.add_subgraph(nodes_state=nodes_state, state=state, measure_dict=sgs.measure_dict, index=0)
        return x

    def extra_repr(self) -> str:
        return f'basis={self.basis}, nodes={self.nodes}, domain={self.domain}'
