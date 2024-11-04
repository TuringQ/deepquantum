"""
Basic gate in MBQC pattern
"""
import torch
from torch import nn
from typing import Any, List, Optional, Tuple, Union

from .operation import Operation, Node, Entanglement, Measurement, XCorrection, ZCorrection

def h(input_node: int, ancilla: List[int]):
    """
    Hadamard gate in MBQC pattern, measure node: input_node, out node: ancilla[0]
    """
    assert len(ancilla) == 1
    cmds = nn.Sequential()
    cmds.append(Node(ancilla[0]))
    cmds.append(Entanglement([input_node, ancilla[0]]))
    cmds.append(Measurement([input_node]))
    cmds.append(XCorrection(ancilla[0], signal_domain=[input_node]))
    node_list = ancilla
    edge_list = [input_node, ancilla[0]]
    return cmds, node_list, edge_list

def pauli_y(input_node: int, ancilla: List[int]):
    """
    Pauli Y gate in MBQC pattern, measure node: input node, ancilla[0,1,2],
    out node: ancilla[3]
    """
    assert len(ancilla) == 4
    cmds = nn.Sequential()
    for i in ancilla:
        cmds.append(Node(i))
    all_nodes = [input_node] + ancilla
    edge_list = []
    for i in range(len(all_nodes)-1):
        cmds.append(Entanglement([all_nodes[i], all_nodes[i+1]]))
        edge_list.extend([all_nodes[i], all_nodes[i+1]])
    cmds.append(Measurement(input_node, angle=torch.pi/2))
    cmds.append(Measurement(ancilla[0], angle=-torch.pi, s_domain=[input_node]))
    cmds.append(Measurement(ancilla[1], angle=-torch.pi/2, s_domain=[input_node]))
    cmds.append(Measurement(ancilla[2]))
    cmds.append(XCorrection(ancilla[3], signal_domain=[ancilla[0], ancilla[2]]))
    cmds.append(ZCorrection(ancilla[3], signal_domain=[ancilla[0], ancilla[1]]))
    node_list = ancilla
    return cmds, node_list, edge_list

def pauli_x(input_node: int, ancilla: List[int]):
    """
    Pauli X gate in MBQC pattern, measure node: input_node, ancilla[0],
    out_node: ancilla[1]
    """
    assert len(ancilla) == 2
    cmds = nn.Sequential()
    cmds.append(Node(ancilla[0]))
    cmds.append(Node(ancilla[1]))
    cmds.append(Entanglement([input_node, ancilla[0]]))
    cmds.append(Entanglement(ancilla))
    cmds.append(Measurement(input_node))
    cmds.append(Measurement(ancilla[0], angle=-torch.pi))
    cmds.append(XCorrection(ancilla[1], signal_domain=[ancilla[0]]))
    cmds.append(ZCorrection(ancilla[1], signal_domain=[input_node]))
    node_list = ancilla
    edge_list = [[input_node, ancilla[0]], ancilla]
    return cmds, node_list, edge_list

def pauli_z(input_node: int, ancilla: List[int]):
    """
    Pauli Z gate in MBQC pattern, measure node: input_node, ancilla[0],
    out_node: ancilla[1]
    """
    assert len(ancilla) == 2
    cmds = nn.Sequential()
    cmds.append(Node(ancilla[0]))
    cmds.append(Node(ancilla[1]))
    cmds.append(Entanglement([input_node, ancilla[0]]))
    cmds.append(Entanglement(ancilla))
    cmds.append(Measurement(input_node, angle=-torch.pi))
    cmds.append(Measurement(ancilla[0]))
    cmds.append(XCorrection(ancilla[1], signal_domain=[ancilla[0]]))
    cmds.append(ZCorrection(ancilla[1], signal_domain=[input_node]))
    node_list = ancilla
    edge_list = [[input_node, ancilla[0]], ancilla]
    return cmds, node_list, edge_list

def s(input_node: int, ancilla: List[int]):
    """
    S gate in MBQC pattern, measure node: input_node, ancilla[0],
    out_node: ancilla[1]
    """
    assert len(ancilla) == 2
    cmds = nn.Sequential()
    cmds.append(Node(ancilla[0]))
    cmds.append(Node(ancilla[1]))
    cmds.append(Entanglement([input_node, ancilla[0]]))
    cmds.append(Entanglement(ancilla))
    cmds.append(Measurement(input_node, angle=-torch.pi/2))
    cmds.append(Measurement(ancilla[0]))
    cmds.append(XCorrection(ancilla[1], signal_domain=[ancilla[0]]))
    cmds.append(ZCorrection(ancilla[1], signal_domain=[input_node]))
    node_list = ancilla
    edge_list = [[input_node, ancilla[0]], ancilla]
    return cmds, node_list, edge_list

def rx(input_node: int, ancilla: List[int], theta: Optional[torch.Tensor]=None):
    """
    Rx gate in MBQC pattern, measure node: input_node, ancilla[0],
    out_node: ancilla[1]
    """
    assert len(ancilla) == 2
    if theta is None:
        theta = torch.rand(1)[0] * torch.pi
    cmds = nn.Sequential()
    cmds.append(Node(ancilla[0]))
    cmds.append(Node(ancilla[1]))
    cmds.append(Entanglement([input_node, ancilla[0]]))
    cmds.append(Entanglement(ancilla))
    cmds.append(Measurement(input_node))
    cmds.append(Measurement(ancilla[0], angle=-theta, s_domain=[input_node]))
    cmds.append(XCorrection(ancilla[1], signal_domain=[ancilla[0]]))
    cmds.append(ZCorrection(ancilla[1], signal_domain=[input_node]))
    node_list = ancilla
    edge_list = [[input_node, ancilla[0]], ancilla]
    return cmds, node_list, edge_list

def ry(input_node: int, ancilla: List[int], theta: Optional[torch.Tensor]=None):
    """
    Ry gate in MBQC pattern, measure node: input node, ancilla[0,1,2],
    out node: ancilla[3]
    """
    assert len(ancilla) == 4
    if theta is None:
        theta = torch.rand(1)[0] * torch.pi
    cmds = nn.Sequential()
    for i in ancilla:
        cmds.append(Node(i))
    all_nodes = [input_node] + ancilla
    edge_list = []
    for i in range(len(all_nodes)-1):
        cmds.append(Entanglement([all_nodes[i], all_nodes[i+1]]))
        edge_list.extend([all_nodes[i], all_nodes[i+1]])
    cmds.append(Measurement(input_node, angle=torch.pi/2))
    cmds.append(Measurement(ancilla[0], angle=-theta, s_domain=[input_node]))
    cmds.append(Measurement(ancilla[1], angle=-torch.pi/2, s_domain=[input_node]))
    cmds.append(Measurement(ancilla[2]))
    cmds.append(XCorrection(ancilla[3], signal_domain=[ancilla[0], ancilla[2]]))
    cmds.append(ZCorrection(ancilla[3], signal_domain=[ancilla[0], ancilla[1]]))
    node_list = ancilla
    return cmds, node_list, edge_list

def rz(input_node: int, ancilla: List[int], theta: Optional[torch.Tensor]=None):
    """
    Rz gate in MBQC pattern, measure node: input_node, ancilla[0],
    out node: ancilla[1]
    """
    assert len(ancilla) == 2
    if theta is None:
        theta = torch.rand(1)[0] * torch.pi
    cmds = nn.Sequential()
    cmds.append(Node(ancilla[0]))
    cmds.append(Node(ancilla[1]))
    cmds.append(Entanglement([input_node, ancilla[0]]))
    cmds.append(Entanglement(ancilla))
    cmds.append(Measurement(input_node, angle=-theta))
    cmds.append(Measurement(ancilla[0]))
    cmds.append(XCorrection(ancilla[1], signal_domain=[ancilla[0]]))
    cmds.append(ZCorrection(ancilla[1], signal_domain=[input_node]))
    node_list = ancilla
    edge_list = [[input_node, ancilla[0]], ancilla]
    return cmds, node_list, edge_list

def cnot(control_node: int, target_node: int, ancilla:List[int]):
    """
    CNOT gate in MBQC pattern, measure node: target_node, ancilla[0],
    out_node: control_node, ancilla[1]
    """
    assert len(ancilla) == 2
    cmds = nn.Sequential()
    cmds.append(Node(ancilla[0]))
    cmds.append(Node(ancilla[1]))
    cmds.append(Entanglement([target_node, ancilla[0]]))
    cmds.append(Entanglement([control_node, ancilla[0]]))
    cmds.append(Entanglement(ancilla))
    cmds.append(Measurement(target_node))
    cmds.append(Measurement(ancilla[0]))
    cmds.append(XCorrection(ancilla[1], signal_domain=[ancilla[0]]))
    cmds.append(ZCorrection(ancilla[1], signal_domain=[target_node]))
    cmds.append(ZCorrection(control_node, signal_domain=[target_node]))
    node_list = ancilla
    edge_list = [[target_node, ancilla[0]], [control_node, ancilla[0]], ancilla]
    return cmds, node_list, edge_list
