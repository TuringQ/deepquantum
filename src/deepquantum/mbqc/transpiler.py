
from deepquantum.gate import SingleGate, ParametricSingleGate, DoubleGate, PauliX, CNOT, Rx, Ry, Rz, Hadamard, SGate

def transpile(cir):
    from deepquantum import Pattern
    pattern = Pattern(n_input_nodes=cir.init_state.nqubit, init_state=cir.init_state.state.flatten())
    for op in cir.operators:
        gate_to_pattern(pattern, op)
    return pattern

def gate_to_pattern(pattern, op):
    if isinstance(op, SingleGate):
        if isinstance(op, ParametricSingleGate):
            # Create a mapping from class types to their string names
            gate_to_str = {
                Rx: "rx",
                Ry: "ry",
                Rz: "rz"
            }
            getattr(pattern, gate_to_str[type(op)])(input_node = pattern.nout_wire_dic[op.wires[0]], theta = op.theta)
        else:
            gate_to_str = {
                PauliX: "pauli_x",
                PauliY: "pauli_y",
                PauliZ: "pauli_z",
                Hadamard: "h",
                SGate: "s"
            }
            getattr(pattern, gate_to_str[type(op)])(input_node = pattern.nout_wire_dic[op.wires[0]])
        pattern.nout_wire_dic[op.wires[0]] = pattern._bg_qubit-1
    elif isinstance(op, DoubleGate):
        if isinstance(op, CNOT):
            pattern.cnot(control_node = pattern.nout_wire_dic[op.wires[0]],
            target_node = pattern.nout_wire_dic[op.wires[1]])
            pattern.nout_wire_dic[op.wires[1]] = pattern._bg_qubit-1