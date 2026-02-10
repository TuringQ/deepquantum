import re
from typing import Any

import numpy as np
import torch

from .circuit import QubitCircuit
from .gate import (
    Barrier,
    CNOT,
    Fredkin,
    Hadamard,
    PauliX,
    PauliY,
    PauliZ,
    PhaseShift,
    Rx,
    Rxx,
    Ry,
    Ryy,
    Rz,
    Rzz,
    SDaggerGate,
    SGate,
    Swap,
    TDaggerGate,
    TGate,
    Toffoli,
    U3Gate,
)
from .operation import Channel, Gate, Layer, Operation

# ==============================================================================
#                 DeepQuantum Circuit to OpenQASM 3.0 Converter
# ==============================================================================


def _op_to_qasm3(op: Operation) -> str:
    """
    Helper function to convert a single deepquantum operation to an OpenQASM 3.0 string.
    """
    if isinstance(op, Layer):
        return '\n'.join([_op_to_qasm3(gate) for gate in op.gates])

    if isinstance(op, Barrier):
        qubits_str = ', '.join([f'q[{w}]' for w in op.wires])
        return f'barrier {qubits_str};'

    if not isinstance(op, Gate):
        return f'// Unsupported operation type: {op.__class__.__name__}'

    if isinstance(op, Channel):
        return f'// Quantum channels like {op.name} are not part of the OpenQASM 3.0 core specification.'

    # Gate name mapping
    name_map = {
        U3Gate: 'u',
        PhaseShift: 'p',
        PauliX: 'x',
        PauliY: 'y',
        PauliZ: 'z',
        Hadamard: 'h',
        SGate: 's',
        SDaggerGate: 'sdg',
        TGate: 't',
        TDaggerGate: 'tdg',
        Rx: 'rx',
        Ry: 'ry',
        Rz: 'rz',
        Swap: 'swap',
        CNOT: 'cx',
        Toffoli: 'ccx',
        Fredkin: 'cswap',
        Rxx: 'rxx',
        Ryy: 'ryy',
        Rzz: 'rzz',
    }

    qasm_name = name_map.get(type(op))
    if not qasm_name:
        return f'// Unsupported gate: {op.name}'

    # Parameters
    param_str = ''
    if hasattr(op, 'npara') and op.npara > 0:
        params = []
        if isinstance(op, U3Gate):
            params = [op.theta.item(), op.phi.item(), op.lambd.item()]
        elif hasattr(op, 'theta'):
            # Works for Rx, Ry, Rz, PhaseShift, Rxx, Ryy, Rzz
            params = [op.theta.item()]

        # Handle inverse operation for parametric gates
        if hasattr(op, 'inv_mode') and op.inv_mode:
            params = [-p for p in params]

        param_str = f'({", ".join(map(str, params))})'

    # Qubits and Controls
    controls = op.controls
    targets = op.wires

    # Implicit controls in gates like CNOT, Toffoli
    if isinstance(op, (CNOT, Toffoli, Fredkin)):
        qubits_str = ', '.join([f'q[{w}]' for w in targets])
        return f'{qasm_name} {qubits_str};'

    all_qubits = controls + targets
    qubits_str = ', '.join([f'q[{w}]' for w in all_qubits])

    # Control modifiers
    ctrl_modifiers = 'ctrl @ ' * len(controls)

    return f'{ctrl_modifiers}{qasm_name}{param_str} {qubits_str};'


def cir_to_qasm3(circuit: QubitCircuit) -> str:
    """Converts ``QubitCircuit`` to an OpenQASM 3.0 string.

    Args:
        circuit (QubitCircuit): The DeepQuantum circuit to convert.

    Returns:
        str: A string containing the OpenQASM 3.0 representation of the circuit.
    """
    qasm_parts = ['OPENQASM 3.0;', 'include "stdgates.inc";']

    num_qubits = circuit.nqubit
    qasm_parts.append(f'qubit[{num_qubits}] q;')

    # Declare classical bits if any measurements are defined
    if circuit.wires_measure:
        # Declare enough bits to cover all measured qubits
        max_measured_wire = max(circuit.wires_measure) if circuit.wires_measure else -1
        num_classical_bits = max_measured_wire + 1
        if num_classical_bits > 0:
            qasm_parts.append(f'bit[{num_classical_bits}] c;')

    # Convert operations
    for op in circuit.operators:
        qasm_line = _op_to_qasm3(op)
        if qasm_line:
            qasm_parts.append(qasm_line)

    # Add measurements
    if circuit.wires_measure:
        qasm_parts.append('\n// Measurements')
        for wire in sorted(circuit.wires_measure):
            qasm_parts.append(f'c[{wire}] = measure q[{wire}];')

    return '\n'.join(qasm_parts)


# ==============================================================================
#                 OpenQASM 3.0 to DeepQuantum Circuit Converter
# ==============================================================================


class GateDefinition:
    def __init__(self, name: str, params: list[str], qubits: list[str], body: list[str]):
        self.name, self.params, self.qubits, self.body = name, params, qubits, body


def qasm3_to_cir(qasm_string: str) -> QubitCircuit:
    """Converts a full-featured OpenQASM 3.0 string to ``QubitCircuit``.
    Supports: `def`, `inv @`, `ctrl @`, and floating-point/negative `pow() @`.
    """
    lines = [line.split('//')[0].strip() for line in qasm_string.strip().splitlines() if line.strip()]

    if not any(line.startswith('OPENQASM 3.0') for line in lines):
        raise ValueError('Input is not a valid OpenQASM 3.0 string (Header missing).')

    gate_definitions: dict[str, GateDefinition] = {}
    main_body_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('def '):
            header_line = line
            open_braces = header_line.count('{')
            body_start_line_offset = 1
            if open_braces == 0:
                if i + 1 < len(lines) and lines[i + 1].strip() == '{':
                    open_braces = 1
                    body_start_line_offset = 2
                else:
                    main_body_lines.append(header_line)
                    i += 1
                    continue
            body_start_index = i + body_start_line_offset
            brace_scan_index = i + body_start_line_offset - 1
            while brace_scan_index < len(lines) and open_braces > 0:
                brace_scan_index += 1
                if brace_scan_index >= len(lines):
                    break
                open_braces += lines[brace_scan_index].count('{')
                open_braces -= lines[brace_scan_index].count('}')
            body_lines = [l_.strip() for l_ in lines[body_start_index:brace_scan_index]]
            i = brace_scan_index + 1
            try:
                header_content = header_line[3:].strip()
                if header_content.endswith('{'):
                    header_content = header_content[:-1].strip()
                paren_match = re.match(r'(\w+)\s*\((.*?)\)\s*(.*)', header_content)
                if paren_match:
                    name, params_str, qubits_str = paren_match.groups()
                else:
                    name_match = re.match(r'(\w+)\s*(.*)', header_content)
                    name, qubits_str = name_match.groups()
                    params_str = ''
                params = [p.strip() for p in params_str.split(',')] if params_str else []
                qubits = [q.strip() for q in qubits_str.strip().split(',')] if qubits_str.strip() else []
                qubits = [q for q in qubits if q]
                gate_definitions[name] = GateDefinition(name, params, qubits, body_lines)
            except Exception:
                print(f"Warning: Could not parse gate definition header: '{header_line}'")
        else:
            main_body_lines.append(line)
            i += 1

    num_qubits = 0
    for line in main_body_lines:
        match = re.search(r'qubit\[(\d+)\]', line)
        if match:
            num_qubits = int(match.group(1))
            break
    if num_qubits == 0:
        raise ValueError('Qubit declaration not found or zero qubits specified.')
    circuit = QubitCircuit(nqubit=num_qubits)

    # --- Helper Function to get gate matrix ---
    def get_gate_matrix(
        gate_name: str, params_str: str, gate_qubits_str: list[str], scope: dict[str, Any]
    ) -> torch.Tensor:
        """Dynamically builds the unitary matrix for a given gate call."""
        n_gate_qubits = len(gate_qubits_str)
        # Create a fake QASM program to parse
        local_qubit_def = f'qubit[{n_gate_qubits}] q;'
        local_qubits_str = ', '.join([f'q[{i}]' for i in range(n_gate_qubits)])
        param_part = f'({params_str})' if params_str else ''
        fake_line = f'{gate_name}{param_part} {local_qubits_str};'

        # Build a temporary full QASM string for the sub-parser
        temp_qasm = ['OPENQASM 3.0;', local_qubit_def]
        # We need the definitions to be available to the sub-parser
        for name, definition in gate_definitions.items():
            params_def = f'({",".join(definition.params)})' if definition.params else ''
            qubits_def = ','.join(definition.qubits)
            body_def = '\n  '.join(definition.body)
            temp_qasm.append(f'def {name}{params_def} {qubits_def} {{\n  {body_def}\n}}')
        temp_qasm.append(fake_line)

        temp_circ = qasm3_to_cir('\n'.join(temp_qasm))

        return temp_circ.get_unitary()

    def _process_qasm_lines(
        lines_to_process: list[str],
        circuit_obj: QubitCircuit,
        scope: dict[str, Any] | None = None,
        external_controls: list[int] | None = None,
        is_inverted: bool = False,
    ):
        if scope is None:
            scope = {}
        if external_controls is None:
            external_controls = []
        gate_pattern = re.compile(r'((?:(?:inv|ctrl|pow\s*\(.*?\))\s*@\s*)*)(\w+)(?:\((.*?)\))?\s+(.*?);')
        processing_order = reversed(lines_to_process) if is_inverted else lines_to_process
        for line in processing_order:
            line = line.strip()
            if not line or line.startswith(('OPENQASM', 'include', 'qubit', 'bit', 'defcal')):
                continue
            if 'measure' in line:
                for m in re.findall(r'q\[(\d+)\]', line):
                    if int(m) not in circuit_obj.wires_measure:
                        circuit_obj.wires_measure.append(int(m))
                continue
            if line.startswith('barrier'):
                qubits_str = line.replace('barrier', '').replace(';', '').strip()
                wires = [int(q.strip()[2:-1]) for q in qubits_str.split(',')] if qubits_str else []
                circuit_obj.barrier(wires=wires if wires else None)
                continue

            match = gate_pattern.match(line)
            if not match:
                print(f"Warning: Could not parse line: '{line}'")
                continue
            modifiers_str, gate_name, params_str, qubits_str = match.groups()

            num_inv = modifiers_str.count('inv')
            num_ctrls = modifiers_str.count('ctrl')
            pow_match = re.search(r'pow\s*\((.*?)\)', modifiers_str)
            effectively_inverted = is_inverted ^ (num_inv % 2 == 1)

            power = 1.0
            if pow_match:
                try:
                    power = float(eval(pow_match.group(1), {'pi': np.pi, 'np': np}))
                except Exception:
                    print(f"Warning: Could not parse power expression '{pow_match.group(1)}'.")
                    continue
            if effectively_inverted:
                power = -power

            call_qubits_all_str = [q.strip() for q in qubits_str.split(',')]

            if int(power) != power:
                inline_controls_str = call_qubits_all_str[:num_ctrls]
                gate_qubits_actual_str = call_qubits_all_str[num_ctrls:]
                total_controls = external_controls + [int(q[2:-1]) for q in inline_controls_str]
                gate_qubits = [int(q[2:-1]) for q in gate_qubits_actual_str]

                try:
                    base_unitary = get_gate_matrix(gate_name, params_str, gate_qubits_actual_str, scope)
                    eigvals, eigvecs = torch.linalg.eig(base_unitary.to(dtype=torch.cfloat))
                    eigvals_pow = eigvals**power
                    final_unitary = eigvecs @ torch.diag(eigvals_pow) @ torch.linalg.inv(eigvecs)
                    circuit_obj.any(final_unitary, wires=gate_qubits, controls=total_controls)
                except Exception as e:
                    print(f"Warning: Cannot apply non-integer power to gate from line '{line}'. Error: {e}")
                    continue
                continue

            for _ in range(int(power)) if power >= 0 else range(int(-power)):
                loop_inverted = effectively_inverted if power >= 0 else not effectively_inverted

                if gate_name in gate_definitions:
                    definition = gate_definitions[gate_name]
                    inline_controls_str = call_qubits_all_str[:num_ctrls]
                    gate_qubits_actual_str = call_qubits_all_str[num_ctrls:]
                    inline_controls = [int(q[2:-1]) for q in inline_controls_str]
                    total_external_controls = external_controls + inline_controls
                    if len(gate_qubits_actual_str) != len(definition.qubits):
                        print(f"Warning: Mismatched qubit count for gate '{gate_name}'.")
                        continue
                    qubit_map = dict(zip(definition.qubits, gate_qubits_actual_str, strict=True))
                    eval_scope = {'pi': np.pi, 'np': np}
                    eval_scope.update(scope)
                    call_params_evaluated = (
                        [eval(p.strip(), eval_scope) for p in params_str.split(',')] if params_str else []
                    )
                    if len(call_params_evaluated) != len(definition.params):
                        print(f"Warning: Mismatched parameter count for gate '{gate_name}'.")
                        continue
                    new_scope = scope.copy()
                    new_scope.update(zip(definition.params, call_params_evaluated, strict=True))
                    expanded_body = []
                    for body_line in definition.body:
                        new_line = body_line
                        for formal_param in definition.params:
                            new_line = re.sub(
                                r'\b' + re.escape(formal_param) + r'\b',
                                str(new_scope.get(formal_param, formal_param)),
                                new_line,
                            )
                        for formal_qubit, actual_qubit in qubit_map.items():
                            new_line = re.sub(r'\b' + re.escape(formal_qubit) + r'\b', actual_qubit, new_line)
                        expanded_body.append(new_line)
                    _process_qasm_lines(
                        expanded_body,
                        circuit_obj,
                        new_scope,
                        external_controls=total_external_controls,
                        is_inverted=loop_inverted,
                    )
                else:
                    _apply_builtin_gate(
                        circuit_obj,
                        gate_name,
                        params_str,
                        call_qubits_all_str,
                        num_ctrls,
                        external_controls,
                        loop_inverted,
                    )

    def _apply_builtin_gate(
        circuit_obj, gate_name, params_str, call_qubits_all_str, num_ctrls, external_controls, is_inverted
    ):
        try:
            qubit_indices = [int(q[2:-1]) for q in call_qubits_all_str]
            params = [float(eval(p, {'pi': np.pi})) for p in params_str.split(',')] if params_str else []
            inline_controls = qubit_indices[:num_ctrls]
            all_qubits_after_mods = qubit_indices[num_ctrls:]
            total_controls = external_controls + inline_controls
            if is_inverted:
                if gate_name in ['rx', 'ry', 'rz', 'p', 'rxx', 'ryy', 'rzz']:
                    params = [-p for p in params]
                elif gate_name == 'u':
                    params = [-params[0], -params[2], -params[1]]
                elif gate_name == 's':
                    gate_name = 'sdg'
                elif gate_name == 'sdg':
                    gate_name = 's'
                elif gate_name == 't':
                    gate_name = 'tdg'
                elif gate_name == 'tdg':
                    gate_name = 't'
            if gate_name == 'cx':
                final_controls = total_controls + [all_qubits_after_mods[0]]
                target = all_qubits_after_mods[1]
                if len(final_controls) == 1:
                    circuit_obj.cnot(control=final_controls[0], target=target)
                else:
                    circuit_obj.x(wires=target, controls=final_controls)
            elif gate_name == 'cz':
                final_controls = total_controls + [all_qubits_after_mods[0]]
                target = all_qubits_after_mods[1]
                circuit_obj.z(wires=target, controls=final_controls)
            elif gate_name == 'ccx':
                final_controls = total_controls + all_qubits_after_mods[0:2]
                target = all_qubits_after_mods[2]
                if len(final_controls) == 2:
                    circuit_obj.toffoli(control1=final_controls[0], control2=final_controls[1], target=target)
                else:
                    circuit_obj.x(wires=target, controls=final_controls)
            elif gate_name == 'cswap':
                final_controls = total_controls + [all_qubits_after_mods[0]]
                targets = all_qubits_after_mods[1:3]
                if len(final_controls) == 1:
                    circuit_obj.fredkin(control=final_controls[0], target1=targets[0], target2=targets[1])
                else:
                    circuit_obj.swap(wires=targets, controls=final_controls)
            else:
                targets = all_qubits_after_mods
                target_arg = targets[0] if len(targets) == 1 else targets
                if gate_name == 'h':
                    circuit_obj.h(wires=target_arg, controls=total_controls)
                elif gate_name == 'x':
                    circuit_obj.x(wires=target_arg, controls=total_controls)
                elif gate_name == 'y':
                    circuit_obj.y(wires=target_arg, controls=total_controls)
                elif gate_name == 'z':
                    circuit_obj.z(wires=target_arg, controls=total_controls)
                elif gate_name == 's':
                    circuit_obj.s(wires=target_arg, controls=total_controls)
                elif gate_name == 'sdg':
                    circuit_obj.sdg(wires=target_arg, controls=total_controls)
                elif gate_name == 't':
                    circuit_obj.t(wires=target_arg, controls=total_controls)
                elif gate_name == 'tdg':
                    circuit_obj.tdg(wires=target_arg, controls=total_controls)
                elif gate_name == 'swap':
                    circuit_obj.swap(wires=target_arg, controls=total_controls)
                elif gate_name == 'rx':
                    circuit_obj.rx(wires=target_arg, inputs=params, controls=total_controls)
                elif gate_name == 'ry':
                    circuit_obj.ry(wires=target_arg, inputs=params, controls=total_controls)
                elif gate_name == 'rz':
                    circuit_obj.rz(wires=target_arg, inputs=params, controls=total_controls)
                elif gate_name == 'p':
                    circuit_obj.p(wires=target_arg, inputs=params, controls=total_controls)
                elif gate_name == 'u':
                    circuit_obj.u3(wires=target_arg, inputs=params, controls=total_controls)
                elif gate_name == 'rxx':
                    circuit_obj.rxx(wires=target_arg, inputs=params, controls=total_controls)
                elif gate_name == 'ryy':
                    circuit_obj.ryy(wires=target_arg, inputs=params, controls=total_controls)
                elif gate_name == 'rzz':
                    circuit_obj.rzz(wires=target_arg, inputs=params, controls=total_controls)
                else:
                    print(f"Warning: Unsupported built-in gate '{gate_name}'")
        except Exception as e:
            print(f"Warning: Error applying gate '{gate_name}'. Qubits: {call_qubits_all_str}. Error: {e}")

    _process_qasm_lines(main_body_lines, circuit)
    circuit.wires_measure.sort()
    return circuit
