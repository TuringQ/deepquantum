"""
Circuit cutting
"""

import bisect
from collections import defaultdict
from collections.abc import Sequence, Hashable
from typing import Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from networkx import Graph, connected_components
from torch import nn

from .gate import Barrier, WireCut, Move
from .layer import Observable
from .operation import GateQPD
from .qpd import DoubleGateQPD


def transform_cut2move(
    operators: nn.Sequential,
    cut_lst: List[Tuple[int, int]],
    observables: Optional[nn.ModuleList] = None
) -> Tuple[nn.Sequential, Optional[nn.ModuleList]]:
    """Transform ``WireCut`` to ``Move`` and expand the observables accordingly."""
    nqubit = operators[0].nqubit
    cuts_per_qubit = defaultdict(list)
    for idx, wire in cut_lst:
        cuts_per_qubit[wire].append(idx)
    ncut_cum_lst = [] # ncut before the current qubit
    ncut = 0
    for i in range(nqubit):
        ncut_cum_lst.append(ncut)
        ncut += len(cuts_per_qubit[i])
    new_nqubit = nqubit + ncut
    for i, op in enumerate(operators):
        op.nqubit = new_nqubit
        new_wires = []
        for wire in op.wires:
            ncut_before = bisect.bisect_left(cuts_per_qubit, i)
            new_wires.append(wire + ncut_cum_lst[wire] + ncut_before)
        op.wires = new_wires
        if isinstance(op, WireCut):
            operators[i] = Move(nqubit=new_nqubit, wires=[op.wires[0], op.wires[0] + 1], tsr_mode=op.tsr_mode)
    if observables is not None:
        for ob in observables:
            ob.nqubit = new_nqubit
            new_wires_ob = []
            for gate in ob.gates:
                gate.nqubit = new_nqubit
                new_wires = []
                for wire in gate.wires:
                    new_wires.append(wire + ncut_cum_lst[wire + 1])
                gate.wires = new_wires
                new_wires_ob.append(new_wires)
            ob.wires = new_wires_ob
    return operators, observables


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
    return nn.Sequential(*operators)


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


def get_qpd_operators(operators: nn.Sequential, qubit_labels: Sequence[Hashable]) -> nn.Sequential:
    """Replace all nonlocal gates belonging to more than one partition with two-qubit QPD gates."""
    nqubit = operators[0].nqubit
    assert len(qubit_labels) == nqubit
    for i, op in enumerate(operators):
        if isinstance(op, (Barrier, GateQPD)):
            continue
        wires = op.wires + op.controls
        if len(wires) < 2:
            continue
        label_set = {qubit_labels[wire] for wire in wires}
        if len(label_set) == 1:
            continue
        assert len(wires) == 2, 'Decomposition is only supported for two-qubit gates.'
        operators[i] = op.qpd()
    return operators


def separate_operators(operators: nn.Sequential, qubit_labels: Optional[Sequence[Hashable]] = None) -> Dict:
    """Separate the circuit into its disconnected components."""
    nqubit = operators[0].nqubit
    operators = split_barriers(operators)
    if qubit_labels is None:
        qubit_labels = partition_labels(operators)
    assert len(qubit_labels) == nqubit
    qubit_map, label2qubits_dict = map_qubit(qubit_labels)
    label2ops_dict = label_operators(operators, qubit_map)
    label2sub_dict = {}
    for label, indices in label2ops_dict.items():
        sub_ops = nn.Sequential()
        nqubit_sub = len(label2qubits_dict[label])
        for i in indices:
            operators[i].nqubit = nqubit_sub
            wires = [qubit_map[wire][1] for wire in operators[i].wires]
            operators[i].wires = wires
            sub_ops.append(operators[i])
        combine_barriers(sub_ops)
        label2sub_dict[label] = sub_ops
    return label2sub_dict


def decompose_observables(observables: nn.ModuleList, qubit_labels: Sequence[Hashable]) -> Dict:
    """Decompose the observables with respect to qubit partition labels."""
    qubit_map, label2qubits_dict = map_qubit(qubit_labels)
    label2obs_dict = {}
    for label, qubits in label2qubits_dict.items():
        sub_obs = nn.ModuleList()
        nqubit_sub = len(qubits)
        for ob in observables:
            new_ob = Observable(nqubit_sub, [], den_mat=ob.den_mat, tsr_mode=ob.tsr_mode)
            for gate in ob.gates:
                wire = gate.wires[0]
                if wire in qubits:
                    new_wire = qubit_map[wire][1]
                    new_ob.wires.append(new_wire)
                    new_ob.basis += ob.basis[ob.wires.index(wire)]
                    gate.nqubit = nqubit_sub
                    gate.wires = [new_wire]
                    new_ob.gates.append(gate)
            sub_obs.append(new_ob)
        label2obs_dict[label] = sub_obs
    return label2obs_dict


def partition_problem(
    operators: nn.Sequential,
    qubit_labels: Optional[Sequence[Hashable]] = None,
    observables: Optional[nn.ModuleList] = None
) -> Tuple[Dict, Optional[Dict]]:
    """Separate the circuit and observables."""
    if qubit_labels is None:
        qubit_labels = partition_labels(operators, lambda op: isinstance(op, DoubleGateQPD))
    operators_qpd = list(get_qpd_operators(operators, qubit_labels))
    gate_label = 0
    for i, op in enumerate(operators_qpd):
        if isinstance(op, DoubleGateQPD):
            op.label = gate_label
            gate1, gate2 = op.decompose()
            operators_qpd[i] = gate1
            operators_qpd.insert(i + 1, gate2)
            gate_label += 1
    label2sub_dict = separate_operators(nn.Sequential(*operators_qpd), qubit_labels)
    if observables is not None:
        label2obs_dict = decompose_observables(observables, qubit_labels)
    else:
        label2obs_dict = None
    return label2sub_dict, label2obs_dict
