# Single-GPU execution with ixquantum

`deepquantum.QubitCircuitGPU` compiles a circuit into a native executor for one Iluvatar (天数) GPU.
It reuses DeepQuantum's gate construction API and supports statevector execution, exact expectations, terminal sampling, and first-order adjoint gradients.

## Installation

The configuration validated for this backend is Linux **x86_64**, an Iluvatar BI-V150 GPU, **CoreX 5.0.0**, **CoreX PyTorch 2.10.0**, and **ixquantum 1.8.0**.
Other CoreX/PyTorch combinations, including CoreX 4.4.0, have not been validated.
Install the matching Iluvatar driver, CoreX development toolkit, and vendor PyTorch distribution first.
A CPU-only or NVIDIA PyTorch distribution cannot execute this backend.

The native build needs a C++17 host compiler, Python development headers, and the CoreX device compiler and headers.
The supplied ixquantum library depends on CoreX's CUDA-compatible runtime (`libcudart.so.10.2`), BLAS (`libcublas.so.10`, `libcublasLt.so.10`), and sparse (`libcusparse.so`) libraries.
These come from the CoreX toolkit/runtime, not from the Python extra.
Use the vendor's environment setup so these libraries are visible to the dynamic linker.

Request the matching **`custatevec.h` and `libcustatevec.so.1.8.0`** from Iluvatar (天数).
These vendor files are external dependencies and are not distributed with DeepQuantum.
Place both files in a directory, then set:

```bash
export IXQUANTUM_ROOT=/path/to/ixquantum
python -m pip install 'deepquantum[ixquantum]'
```

For a source checkout, use `python -m pip install -e '.[ixquantum]'`.
Install into the environment containing the compatible vendor PyTorch; do not replace it with a stock PyTorch build.
`IXQUANTUM_ROOT/include/custatevec.h` with the library under `lib` or `lib64` is also accepted.
If unset, the SDK search defaults to `/usr/local/corex`.

The `ixquantum` extra declares the Python build helpers (`ninja`, `setuptools`, and `packaging`).
Compilation happens on first GPU use through `torch.utils.cpp_extension.load`, using the installed CoreX PyTorch compiler support.
Subsequent processes reuse its build cache.
`TORCH_EXTENSIONS_DIR` selects the cache directory and `MAX_JOBS` limits build parallelism.
Normal DeepQuantum installation and import do not require this SDK or compile an extension.

## Forward execution and sampling

```python
import deepquantum as dq

circuit = dq.QubitCircuitGPU(3, device='cuda:0')
circuit.h(0)
circuit.cnot(0, 1)
circuit.cnot(1, 2)
state = circuit()                  # complex64 CUDA tensor, shape (8, 1)
counts = circuit.measure(1024)      # approximately equal '000' and '111' counts
circuit.close()
```

Logical qubit 0 is the most significant statevector bit, matching `QubitCircuit`.
The returned tensor borrows the executor's storage; use `state.clone()` before another execution if the previous result must be retained.
`close()` releases native resources; a later execution recompiles the circuit.
Choose the execution device in the constructor; moving a circuit to another device is unsupported.
Parameter dtype conversions and moving parameters to that same GPU remain supported.

## Differentiation

```python
import torch
import deepquantum as dq

circuit = dq.QubitCircuitGPU(2, diff_method='adjoint', device='cuda:0')
circuit.ry(0)                      # trainable gate parameter
circuit.cnot(0, 1)
circuit.observable(1, 'z')
optimizer = torch.optim.Adam(circuit.parameters(), lr=0.05)
for _ in range(10):
    optimizer.zero_grad()
    loss = circuit.expectation().sum()
    loss.backward()
    optimizer.step()
circuit.close()
```

In adjoint mode, `expectation()` performs the forward pass itself.
The reverse pass reuses that final state, so complete `backward()` before executing, recompiling, or closing the same circuit again.
Trainable scalar parameters shared between gates accumulate their contributions into one gradient.
For a packed CUDA float32 parameter vector and observable coefficients, `value_and_grad(parameters, coefficients)` returns a weighted value and its gradient directly.
`adjoint_vjp(parameters, coefficients)` returns just the gradient.
Parameter order is first occurrence in the compiled gate sequence, with one entry per distinct trainable scalar.
These direct gradient methods consume the final state and do not construct a higher-order autograd graph.

## Supported behavior

- One pure complex64 statevector, from 1 to 30 qubits subject to device memory, with a fixed initial state.
- Fixed unitary matrix gates, controlled gates, Pauli rotations, and phase gates.
  `multirz(wires)` supplies a multi-wire Z rotation without constructing its dense matrix.
- Exact Pauli-string expectations and first-order gradients for supported scalar rotations and phase gates.
  One fixed Hermitian matrix observable can be registered with `observable_matrix(matrix, wires)` without gradients.
  It cannot be combined with the Pauli observables on the same circuit.
- `expectation_from_state(state, observables)` evaluates DeepQuantum `Observable` Pauli strings on an existing CUDA state without running or compiling gates.
  This readout is non-differentiable and does not copy or retain the input state.
- Terminal sampling with `with_prob=False`; no state collapse.
- Runtime data encoding, batches, density matrices, noise channels, mid-circuit measurement, conditional operations, differentiable initial states, and differentiation of the returned statevector are unsupported.
  Whole-circuit `inverse()` and circuit addition with `+` are also unsupported; individual inverse gates are supported.
  Barriers are ignored during compilation.
  Unsupported operations raise errors rather than silently changing execution backends.

The compiler selects ixquantum APIs automatically: generalized permutations for permutation/diagonal gates,
Pauli rotations for supported rotations, and dense matrices for other supported gates.
Adjoint gradients combine generator action with the inner-product reduction in `adjoint_kernels.cu`.
Phase gradients use the projector onto the active target/control sector directly.
These kernels avoid allocating an additional full statevector for intermediate generator results.
They also accumulate weighted Pauli observables when preparing the adjoint state.
Gate execution uses ixquantum; the custom kernels are limited to adjoint calculations.

| Operation | Implementation |
|---|---|
| X/Y/Z, S/T and inverses, CX/CY/CZ, SWAP/iSWAP, Toffoli/Fredkin | `ApplyGeneralizedPermutationMatrix` |
| PhaseShift/CP | `ApplyGeneralizedPermutationMatrix` with a diagonal |
| Rx/Ry/Rz, Rxx/Ryy/Rzz, MultiRZ and controlled forms | `ApplyPauliRotation` |
| Rxy | Two half-angle XX/YY rotations |
| Other fixed matrices | `ApplyMatrix` |
| Rotation and phase gradients | Fused generator contraction |

Fixed generalized permutations are recognized before GPU upload.
API selection follows gate structure; it does not guarantee the fastest kernel for every size and bit position.

Adding gates or observables, resetting the circuit, and loading a state dictionary automatically invalidate the compiled plan.
Call `invalidate()` after directly changing gate structure, fixed parameters, or fixed matrices.
Updating existing trainable parameter values does not require recompilation.
Compilation and first-use kernel initialization should be excluded from steady-state performance measurements and reported separately.
