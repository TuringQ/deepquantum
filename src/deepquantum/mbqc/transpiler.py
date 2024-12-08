"""
Transpiles QubitCircuit into an MBQC pattern
"""

from deepquantum.gate import SingleGate, ParametricSingleGate, DoubleGate
from deepquantum.gate import PauliX, PauliY, PauliZ, CNOT, Rx, Ry, Rz, Hadamard, SGate
from .pattern import Pattern


def transpile(cir) -> Pattern:
    """
    Transpiles QubitCircuit into an MBQC pattern.

    Args:
        cir: QubitCircuit to be transpiled
    Returns:
        Pattern: The resulting MBQC pattern
    """
    # Dictionary mapping gate classes to their corresponding string representations
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
    # Initialize a new Pattern with the circuit's input state
    if cir.init_state.state.ndim == 2:
        pattern = Pattern(n_input_nodes=cir.init_state.nqubit, init_state=cir.init_state.state.flatten())
    else:
        pattern = Pattern(n_input_nodes=cir.init_state.nqubit, init_state=cir.init_state.state.flatten(start_dim = 1))
    pattern.nout_wire_dic = {i:i for i in range(pattern.n_input_nodes)}
    # Convert each operator in the circuit to its corresponding pattern
    for op in cir.operators:
        gate_to_pattern(pattern, op, gate_to_str[type(op)])
    return pattern

def gate_to_pattern(pattern, op, op_str):
    """
    Converts a QubitCircuit gate to its corresponding MBQC pattern representation.

    Args:
        pattern: The MBQC pattern being constructed
        op: The quantum gate operator to convert
        op_str: String representation in mbqc.py of the gate
    """
    # Handle single-qubit gates
    if isinstance(op, SingleGate):
        if isinstance(op, ParametricSingleGate):
            # For parametric gates (Rx, Ry, Rz), include the rotation angle theta
            getattr(pattern, op_str)(input_node = pattern.nout_wire_dic[op.wires[0]], theta = op.theta)
        else:
            # For non-parametric gates (X, Y, Z, H, S)
            getattr(pattern, op_str)(input_node = pattern.nout_wire_dic[op.wires[0]])
        # Update the wire dictionary to point to the newest qubit as output
        pattern.nout_wire_dic[op.wires[0]] = pattern._tot_qubit-1
    # Handle two-qubit gates (currently only CNOT)
    elif isinstance(op, DoubleGate):
        if isinstance(op, CNOT):
            getattr(pattern, op_str)(control_node = pattern.nout_wire_dic[op.wires[0]],
                                    target_node = pattern.nout_wire_dic[op.wires[1]])
            pattern.nout_wire_dic[op.wires[1]] = pattern._tot_qubit-1
