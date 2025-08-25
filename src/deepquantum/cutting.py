"""
Circuit cutting
"""

from collections import defaultdict
from collections.abc import Sequence, Hashable
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from networkx import Graph, connected_components
from torch import nn

from .circuit import QubitCircuit
from .gate import Barrier


def partition_labels(
    operators: nn.Sequential,
    ignore: Callable = lambda x: False,
    keep_idle_wires: bool = False
) -> List[Optional[int]]:
    """Generate partition labels from the connectivity of a quantum circuit."""
    nqubit = operators[0].nqubit
    graph = Graph()
    graph.add_nodes_from(range(nqubit))
    for op in operators:
        if ignore(op):
            continue
        wires = op.wires + op.controls
        for i, wire1 in enumerate(wires):
            for wire2 in wires[i+1:]:
                graph.add_edge(wire1, wire2)
    qubit_subsets = list(connected_components(graph))
    qubit_subsets.sort(key=min)
    if not keep_idle_wires:
        idle_wires = set(range(nqubit))
        for op in operators:
            wires = op.wires + op.controls
            for wire in wires:
                idle_wires.discard(wire)
        qubit_subsets = [
            subset
            for subset in qubit_subsets
            if not (len(subset) == 1 and next(iter(subset)) in idle_wires)
        ]
    qubit_labels = [None] * nqubit
    for i, subset in enumerate(qubit_subsets):
        for qubit in subset:
            qubit_labels[qubit] = i
    return qubit_labels


def map_qubit(qubit_labels: Sequence[Hashable]) -> Tuple[List[Tuple], Dict[Hashable, List]]:
    """Generate a qubit map given a qubit partitioning."""
    qubit_map = []
    label2qubits_dict = defaultdict(list)
    for i, label in enumerate(qubit_labels):
        if label is None:
            qubit_map.append((None, None))
        else:
            qubits = label2qubits_dict[label]
            qubit_map.append((label, len(qubits)))
            qubits.append(i)
    return qubit_map, dict(label2qubits_dict)


def label_operators(operators: nn.Sequential, qubit_map: Sequence[Tuple]) -> Dict[Hashable, List]:
    """Generate a list of operators for each partition of the circuit."""
    unique_labels = set([label for label, _ in qubit_map if label is not None])
    label2ops_dict = {label: [] for label in unique_labels}
    for i, op in enumerate(operators):
        labels = set()
        wires = op.wires + op.controls
        for wire in wires:
            label = qubit_map[wire][0]
            assert label is not None, f'The {wire}-th qubit is provided a partition label of `None`'
            labels.add(label)
        assert len(labels) == 1
        label = labels.pop()
        label2ops_dict[label].append(i)
    return label2ops_dict


def split_barriers(operators: nn.Sequential) -> nn.Sequential:
    """Mutate operators to split barriers into single-qubit barriers."""
    operators = list(operators)
    for i, op in enumerate(operators):
        wires = op.wires + op.controls
        nwire = len(wires)
        if nwire == 1 or (not type(op) is Barrier):
            continue
        barrier_uuid = f'Barrier_uuid={uuid4()}'
        operators[i] = Barrier(op.nqubit, wires[0], barrier_uuid)
        for j in range(1, nwire):
            operators.insert(i + j, Barrier(op.nqubit, wires[j], barrier_uuid))
    return nn.Sequential(operators)


def combine_barriers(operators: nn.Sequential) -> nn.Sequential:
    """Mutate operators to combine barriers with common names into a single barrier."""
    nqubit = operators[0].nqubit
    uuid2idx_dict = defaultdict(list)
    for i, op in enumerate(operators):
        if type(op) is Barrier and len(op.wires) == 1 and 'Barrier_uuid=' in op.name:
            uuid2idx_dict[op.name].append(i)
    cleanup_lst = []
    for indices in uuid2idx_dict.values():
        wires = [operators[i].wires[0] for i in indices]
        new_barrier = Barrier(nqubit, wires)
        operators[indices[0]] = new_barrier
        cleanup_lst.extend(indices[1:])
    cleanup_lst = sorted(cleanup_lst, reverse=True)
    for i in cleanup_lst:
        del operators[i]


def separate_operators(operators: nn.Sequential, qubit_labels: Optional[Sequence[Hashable]] = None) -> Dict:
    """Separate the circuit into its disconnected components."""
    nqubit = operators[0].nqubit
    new_ops = split_barriers(deepcopy(operators))
    if qubit_labels is None:
        qubit_labels = partition_labels(new_ops)
    assert len(qubit_labels) == nqubit
    qubit_map, label2qubits_dict = map_qubit(qubit_labels)
    label2ops_dict = label_operators(new_ops, qubit_map)
    label2sub_dict = {}
    for label, indices in label2ops_dict.items():
        sub_ops = nn.Sequential()
        nqubit_sub = len(label2qubits_dict[label])
        for i in indices:
            new_ops[i].nqubit = nqubit_sub
            wires = [qubit_map[wire][1] for wire in new_ops[i].wires]
            new_ops[i].wires = wires
            sub_ops.append(new_ops[i])
        combine_barriers(sub_ops)
        label2sub_dict[label] = sub_ops
    return label2sub_dict
