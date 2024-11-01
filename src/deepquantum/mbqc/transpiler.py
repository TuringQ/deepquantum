
from deepquantum.gate import SingleGate, DoubleGate, PauliX, CNOT

def transpile(cir):
    from deepquantum import Pattern
    pattern = Pattern(n_input_nodes=cir.init_state.nqubit, init_state=cir.init_state.state.flatten())
    for op in cir.operators:
        gate_to_pattern(pattern, op)
    return pattern

def gate_to_pattern(pattern, op):
    if isinstance(op, SingleGate):
        if isinstance(op, PauliX):
            pattern.pauli_x(input_node = pattern.nout_wire_dic[op.wires[0]])
            pattern.nout_wire_dic[op.wires[0]] = pattern._bg_qubit-1

    if isinstance(op, DoubleGate):
        if isinstance(op, CNOT):
            pattern.cnot(control_node = pattern.nout_wire_dic[op.wires[0]],
            target_node = pattern.nout_wire_dic[op.wires[1]])
            pattern.nout_wire_dic[op.wires[1]] = pattern._bg_qubit-1