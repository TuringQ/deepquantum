
from deepquantum.gate import SingleGate, ParametricSingleGate, DoubleGate, PauliX, PauliY, PauliZ, CNOT, Rx, Ry, Rz, Hadamard, SGate
from .mbqc import Pattern


def transpile(cir):
    gate_to_str = {
                PauliX: "pauli_x",
                PauliY: "pauli_y",
                PauliZ: "pauli_z",
                Hadamard: "h",
                SGate: "s",
                Rx: "rx",
                Ry: "ry",
                Rz: "rz",
                CNOT: "cnot"
            }

    pattern = Pattern(n_input_nodes=cir.init_state.nqubit, init_state=cir.init_state.state.flatten())
    for op in cir.operators:
        gate_to_pattern(pattern, op,  gate_to_str[type(op)])
    return pattern

def gate_to_pattern(pattern, op, op_str):

    if isinstance(op, SingleGate):
        if isinstance(op, ParametricSingleGate):
            getattr(pattern, op_str)(input_node = pattern.nout_wire_dic[op.wires[0]], theta = op.theta)
        else:
            getattr(pattern, op_str)(input_node = pattern.nout_wire_dic[op.wires[0]])
        pattern.nout_wire_dic[op.wires[0]] = pattern._bg_qubit-1
    elif isinstance(op, DoubleGate):
        if isinstance(op, CNOT):
            getattr(pattern, op_str)(control_node = pattern.nout_wire_dic[op.wires[0]],
                                    target_node = pattern.nout_wire_dic[op.wires[1]])
            pattern.nout_wire_dic[op.wires[1]] = pattern._bg_qubit-1