"""
Basic gate in MBQC pattern
"""
import torch
from torch import nn
from typing import Any, List, Optional, Tuple, Union

from .operation import Operation, Node, Entanglement, Measurement, XCorrection, ZCorrection

def cnot(control_node: int, target_node: int, ancilla:List[int]):
    """CNOT gate in MBQC pattern"""
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

def pauli_x(input_node: int, ancilla: List[int]):
    """Pauli X gate in MBQC pattern"""
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



