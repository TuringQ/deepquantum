# Single-GPU execution with ixquantum

`deepquantum.QubitCircuitGPU` compiles a circuit into a native executor for one Iluvatar CoreX (天数智芯) GPU.
It reuses DeepQuantum's gate construction API and supports statevector execution, exact expectations, terminal sampling, and first-order adjoint gradients.

## Installation

Request the matching GPU driver, **CoreX development toolkit**, **CoreX-compatible PyTorch**, and **ixquantum headers and shared library** from **Iluvatar CoreX (天数智芯)**.
The ixquantum files needed by this backend are `custatevec.h` and `libcustatevec.so.1.8.0`.

The validated configuration is Linux **x86_64**, an Iluvatar BI-V150 GPU, **CoreX 5.0.0**, **CoreX PyTorch 2.10.0**, and **ixquantum 1.8.0**.
Install the vendor packages and activate the Python environment containing CoreX-compatible PyTorch.
The native build also needs a C++17 host compiler and Python development headers.

If CoreX is installed under `/usr/local/corex`, configure the compiler and runtime paths as follows.
Adjust the paths for your installation, and set `IXQUANTUM_ROOT` to the directory containing the two ixquantum files.

```bash
export PATH="/usr/local/corex/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/corex/lib64:/usr/local/corex/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export IXQUANTUM_ROOT=/path/to/ixquantum
python -m pip install 'deepquantum[ixquantum]'
```

For example, use `export IXQUANTUM_ROOT=/root` if both vendor files are stored directly in `/root`.
An SDK layout with `include/custatevec.h` and the library under `lib` or `lib64` is also accepted.
If `IXQUANTUM_ROOT` is unset, the SDK search defaults to `/usr/local/corex`.
For a source checkout, use `python -m pip install -e '.[ixquantum]'`.

The `[ixquantum]` extra installs Python build tools (`ninja`, `setuptools`, and `packaging`); the vendor packages come from 天数智芯.
The extension compiles on first GPU use through `torch.utils.cpp_extension.load` and reuses the build cache thereafter.
`TORCH_EXTENSIONS_DIR` selects the cache directory and `MAX_JOBS` limits build parallelism.
Normal DeepQuantum installation and import do not require the GPU SDK or compile an extension.

## Forward execution and sampling

```python
import deepquantum as dq

circuit = dq.QubitCircuitGPU(3, device='cuda:0')
circuit.h(0)
circuit.cnot(0, 1)
circuit.cnot(1, 2)
state = circuit()                  # complex64 CUDA tensor, shape (8, 1)
counts = circuit.measure(shots=1024)
print(state.shape)                # torch.Size([8, 1])
print(counts)                     # approximately equal '000' and '111' counts
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

The compiler selects ixquantum APIs automatically: generalized permutations for permutation/diagonal gates,
Pauli rotations for supported rotations, and dense matrices for other supported gates.
Adjoint gradients combine generator action with the inner-product reduction in `adjoint_kernels.cu`.
Phase gradients use the projector onto the active target/control sector directly.
These kernels avoid allocating an additional full statevector for intermediate generator results.
They also accumulate weighted Pauli observables when preparing the adjoint state.

| Operation | Implementation |
|---|---|
| X/Y/Z, S/T and inverses, CX/CY/CZ, SWAP/iSWAP, Toffoli/Fredkin | `ApplyGeneralizedPermutationMatrix` |
| PhaseShift/CP | `ApplyGeneralizedPermutationMatrix` with a diagonal |
| Rx/Ry/Rz, Rxx/Ryy/Rzz, MultiRZ and controlled forms | `ApplyPauliRotation` |
| Rxy | Two half-angle XX/YY rotations |
| Other fixed matrices | `ApplyMatrix` |
| Rotation and phase gradients | Fused generator contraction |

Fixed generalized permutations are recognized before GPU upload.
API selection follows gate structure.

Adding gates or observables, resetting the circuit, and loading a state dictionary automatically invalidate the compiled plan.
Call `invalidate()` after directly changing gate structure, fixed parameters, or fixed matrices.
Updating existing trainable parameter values does not require recompilation.
Compilation and first-use kernel initialization should be excluded from steady-state performance measurements and reported separately.
