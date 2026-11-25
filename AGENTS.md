# AGENTS.md

Guidance for AI agents using or modifying this repository.

## Project Overview

This repository is the `deepquantum` Python package. It is a PyTorch-based quantum computing simulation library for quantum circuits, quantum machine learning, photonic quantum computing, measurement-based quantum computation (MBQC), tensor-network/MPS simulation, and distributed simulation.

The core package source is under `src/deepquantum/`. Prefer public APIs exported from `deepquantum` or `deepquantum.photonic` before reaching into private helpers or implementation details.

## Repository Layout

- `src/deepquantum/`: qubit state/circuit simulation, gates, layers, observables, channels, ansatz classes, QASM helpers, circuit cutting, distributed state-vector simulation, and shared math utilities.
- `src/deepquantum/photonic/`: photonic states, `QumodeCircuit`, Fock/Gaussian/Bosonic/Bargmann backends, photonic gates/channels/measurements, Clements/GBS ansatzes, TDM circuits, unitary decomposition/mapping, and photonic distributed simulation.
- `src/deepquantum/mbqc/`: MBQC `Pattern`, `GraphState`, N/E/M/C commands, standardization, signal shifting, and pattern execution.
- `tests/`: source of truth for supported behavior and external compatibility checks.
- `tutorials/`: user-facing notebook tutorials for qubit circuits, photonic circuits, and MBQC.
- `examples/`: runnable notebooks/scripts for algorithms, benchmarks, Clements/unitary mapping, QAOA/HHL/VQE, MBQC, and photonic workflows.
- `docs/`: Sphinx/MyST documentation, quick-start pages, demos, and generated autosummary API pages.

When answering simulation questions, inspect the closest combination of source, tests, tutorial, and example first. Start with `README.md`, then use `tutorials/` or `docs/source/quick_start/`, then match to a nearby `tests/test_*.py` file before writing new code.

## Simulation Usage Guidance

General workflow:

- Import public APIs with `import deepquantum as dq`; use `import deepquantum.photonic as dqp` for photonic-only helpers.
- Start with minimal examples from `README.md`, `tutorials/`, `examples/`, and matching tests. Keep qubit counts, cutoff, bond dimension, batch size, device, and dtype small until correctness is confirmed.
- Use PyTorch tensors for data, parameters, device placement, dtype changes, and autograd. Circuits and states are `torch.nn.Module` objects.
- Prefer public circuit methods such as `cir.h`, `cir.rx`, `cir.cnot`, `cir.observable`, `cir.measure`, `cir.expectation`, `cir.get_prob`, `cir.get_amplitude`, `cir.pattern`, `cir.to(...)`, and photonic equivalents on `QumodeCircuit`.

Qubit circuits:

- Use `dq.QubitCircuit(nqubit, init_state='zeros', den_mat=False, reupload=False, mps=False, chi=None, shots=1024)`.
- State-vector simulation is the default. Call `cir()` or `cir(data=..., state=...)` to run the circuit.
- Add observables with `cir.observable(wires=None, basis='z')`; `basis` supports strings such as `'x'`, `'y'`, `'z'`, or multi-wire strings like `'xy'`.
- Call `cir.expectation()` for exact differentiable expectation values, or `cir.expectation(shots=...)` for sampled estimates.
- Call `cir.measure(shots=..., wires=..., with_prob=True)` after a forward pass for sampled bit-string counts and optional probabilities.
- Use `encode=True` on parameterized gates/layers to consume entries from the `data` tensor. `reupload=True` repeats data when a circuit needs more encoded values than provided.
- `data` may be 1D for one sample or 2D for batch execution; tests verify batched qubit forward, MBQC transpilation, and expectation behavior.
- Density-matrix noise channels are available when `den_mat=True`, including bit flip, phase flip, depolarizing, Pauli, amplitude damping, phase damping, and generalized amplitude damping.

MPS and large qubit simulation:

- Use `dq.QubitCircuit(nqubit, mps=True, chi=...)` for matrix product state simulation. Larger `chi` improves accuracy and increases cost.
- `dq.MatrixProductState(...).full_tensor()` can convert small MPS states back to dense tensors for validation.
- Tests compare MPS against dense simulation for probabilities, amplitudes, and circuit outputs.

Photonic circuits:

- Build a `dq.QumodeCircuit(nmode, init_state, backend=..., cutoff=...)`, add preparations and gates, and call `cir()`. The backend and state representation determine which operations and readouts are available.
- Use `'vac'` for vacuum, or prepare inputs with circuit methods such as `thermal` and `single_photon` on their supported backends.
- On the Fock basis backend, `cir()` returns the unitary, `cir(is_prob=False)` gives transition amplitudes and `cir(is_prob=True)` gives probabilities. Use `basis=False` for superpositions, `den_mat=True` with `basis=False` for density tensors, and `mps=True` with `basis=False` for Fock MPS.
- Gaussian circuits evolve `[cov, mean]` through gates such as `s`, `s2`, `d` and `bs`. Use `get_symplectic` for a unitary Gaussian circuit's symplectic matrix or `cir(is_prob=True)` for photon-count probabilities.
- Bosonic circuits evolve `[cov, mean, weight]` for linear combinations of Gaussians, including cat and GKP states. See `tests/test_photonic_bosonic.py` for preparation and quadrature readout.
- Bargmann circuits evolve a native `BargmannState`. Use its moments and conditional states directly, or call `to_fock(cutoff)` for a finite Fock representation.
- For interferometer decomposition and GBS examples, start with `dq.Clements`, `dq.UnitaryDecomposer`, `dq.GaussianBosonSampling` and `dqp.GraphGBS`. Use `cir.any(...)` for a differentiable arbitrary unitary.

Photonic hardware and submodes:

- Attach a `dqp.HardwareModel` with `cir.set_hardware_model(model)` to model device controls, loss and readout on the same circuit. The hardware examples in `tutorials/photonic_basics.ipynb` and `tests/test_photonic_hardware.py` cover individual calibrations.
- With orthogonal basis modes, circuit wires still label spatial ports. `SubmodeResult.to_joint()` gives the full joint state. A photon counter on a spatial port sums over its basis modes.

Photonic TDM:

- Use `global_circuit(nstep)` to unroll delay loops, or `tdm_substate(...)` for selected unconditional outputs and windows.
- For a Gaussian delay circuit without submodes, `tdm_memory(state)` extracts the loop modes and `cir(memory=...)` takes the next step with fresh inputs. `QumodeCircuitTDM` also supports stepwise homodyne measurements. See `tests/test_photonic_tdm.py` for examples.

Photonic measurement and probabilities:

- After a forward pass, use `get_prob` for an event, `measure` to draw photon counts, and `post_select` to condition on a specified count. `measure(with_state=True)` pairs each sampled record with its conditional state on supported backends.
- Use `quadrature_mean`, `quadrature_moments` or `photon_number_mean_var` for moments and `measure_homodyne` for quadrature samples. Consult the owning method's docstring for detector support, normalization and batch shapes.

MBQC:

- Use `dq.Pattern(nodes_state=..., state='plus')` and add commands with `n`, `e`, `m`, `x`, and `z`.
- `Pattern()` execution returns a `GraphState`; use `.full_state` and `pattern.state.measure_dict`.
- `Pattern.m(..., encode=True)` consumes data as measurement angles; batch data and batch initial graph states are tested.
- Use `pattern.standardize()` and `pattern.shift_signals()` for NEMC standardization and signal shifting.
- `QubitCircuit.pattern()` transpiles supported qubit circuits to MBQC patterns. Tests cover random circuits, encoded parameters, batched data, and data reuploading. Source asserts this is not supported for density matrices or MPS.

Distributed simulation:

- Public helpers are `dq.setup_distributed(backend)` and `dq.cleanup_distributed()`.
- Qubit distributed simulation uses `dq.DistributedQubitCircuit`; photonic distributed Fock tensor simulation uses `dq.DistributedQumodeCircuit`.
- README examples use `torchrun` with `backend='gloo'` for CPU or `backend='nccl'` for GPU. Test or demonstrate distributed code with very small circuits first.

## Environment Setup

Python `>=3.10` is required. Install PyTorch appropriately for the local CPU/CUDA environment before relying on GPU or distributed workflows.

Recommended development setup:

```bash
uv pip install -e ".[dev]"
```

Fallback setup:

```bash
pip install -e .
pip install -r requirements-dev.txt
```

Docs dependencies are in `requirements-docs.txt` / the `docs` extra. Development dependencies include `pytest`, `ruff`, `pre-commit`, `jupytext`, `graphix`, `qutip`, `pennylane-sf`, `perceval-quandela`, `strawberryfields`, and `thewalrus`; `pyproject.toml` and `requirements-dev.txt` use GitHub `master` sources for `strawberryfields` and `thewalrus`.

## Common Commands

```bash
pytest
pytest tests/test_circuit.py
pytest tests/test_photonic_fock.py
pytest tests/test_photonic_bosonic.py
pytest tests/test_photonic_hardware.py tests/test_photonic_tdm.py
pytest tests/test_mbqc_transpile.py
ruff check .
ruff format .
pre-commit run --all-files
```

## Coding Standards

- Ruff is the linter and formatter.
- Line length is 120.
- Single quotes are preferred.
- Imports are sorted by Ruff/isort.
- Google-style docstrings are used; `src/` enforces pydocstyle through Ruff with selected missing-docstring ignores.
- Document nontrivial public APIs in their owning docstrings. Keep this guide focused on core entry points, state representations and workflows that span features. Put device-specific arguments, return shapes and calibration examples in docstrings, tutorials and tests.
- Keep changes consistent with the surrounding PyTorch `nn.Module` patterns, buffer/parameter handling, dtype/device behavior, and existing test tolerances.

## Testing Guidance

- For simulation changes, run the most relevant targeted tests first. Benchmark performance changes end to end with the same physical inputs, requested outputs and timing boundaries on both revisions.
- For qubit circuit changes, inspect and run relevant tests such as `tests/test_circuit.py`, `tests/test_mps.py`, `tests/test_auto_grad.py`, `tests/test_get_amplitude.py`, `tests/test_channel.py`, and `tests/test_module.py`.
- For photonic changes, inspect and run relevant `tests/test_photonic_*.py` files plus compatibility tests such as `tests/test_with_xanadu*.py`, `tests/test_with_perceval.py`, `tests/test_with_qutip.py`, and `tests/test_with_pennylane.py` when applicable.
- For MBQC changes, inspect and run `tests/test_mbqc_transpile.py` and `tests/test_with_graphix.py`.
- For broad behavioral changes, run `pytest`, `ruff check .`, and `pre-commit run --all-files` when practical.
- Numerical tests should use appropriate tolerances and compare against existing expected behavior or trusted reference implementations already present in tests.

## Notebook and Documentation Rules

- Use self-contained DeepQuantum examples in tutorials and PRs.
- Tutorials, examples, and docs demos use Jupytext.
- `.ipynb` files are the source of truth for tutorial/example notebook content.
- Write notebook prose with one sentence per source line and a blank line between paragraphs.
- Paired `.py:percent` files are generated or synchronized by Jupytext/pre-commit.
- Avoid editing generated notebook companion files directly unless the workflow requires it.
- Sphinx docs live under `docs/source/`; notebook execution is disabled in `docs/source/conf.py`.

## Files to Avoid Editing

Do not edit generated caches or build artifacts unless explicitly required. Avoid touching:

- `.venv/`
- `.ruff_cache/`
- `.pytest_cache/`
- `__pycache__/`
- docs build output such as `docs/_build/` or `docs/source/_build/`

Do not modify `src/deepquantum/photonic/cache/` unless the task specifically concerns those cached photonic mapper data files.

## Git and PR Review Rules

- For GitHub PR reviews, judge changes against the PR's actual `base...head` diff or GitHub "Files changed", not the local current `main..head`.
- Separate branch drift from actual review findings.
- Switch to the requested feature branch before editing. Preserve user changes and unrelated local work.
- Keep project lessons in this file. Add a global rule only for high-impact behavior that clearly recurs across projects.

## Agent Workflow Expectations

- Before answering simulation questions, inspect the relevant source, tests, tutorials, or examples.
- Prefer tested examples over speculative API usage.
- When writing simulation snippets, include imports, minimal parameters, and expected output shape or behavior when clear.
- When changing code, keep edits focused and consistent with existing patterns.
- After changes, report what changed and which commands were run.
- If a command was not run, say so clearly.
