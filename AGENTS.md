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

When answering simulation questions, inspect the closest combination of source, tests, tutorial, and example first. Start with `README.md`, then use `tutorials/` or `docs/source/quick_start/` and find an existing test for the public API.

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

- Construct `dq.QumodeCircuit(nmode, init_state, backend=backend, cutoff=cutoff)`. Start from vacuum (`'vac'`) or a state in the chosen representation. Register thermal or single-photon inputs with `cir.thermal` or `cir.single_photon` on supported backends.
- Build passive optics with `cir.ps`, `cir.bs` and `cir.mzi`. Use `cir.s` and `cir.s2` for squeezing, `cir.d` for displacement and `cir.loss_t` for loss specified by power transmittance. Use `cir.add(op)` to compose operations or circuits.
- Call `cir()` to run the circuit and `cir.draw()` to display its optical elements. Mark data-dependent gates with `encode=True` and pass their parameters in gate order through `cir(data=data)`.
- Use `cir.any(unitary)` to apply an interferometer matrix or `cir.clements(unitary)` to synthesize it with optical elements. Inspect the ideal passive transformation with `cir.get_unitary()`. Use `dq.UnitaryDecomposer` for Clements or Reck decomposition parameters.
- For circuit templates, `dq.Clements` builds a programmable mesh, `dq.GaussianBosonSampling` combines squeezing with an interferometer and `dqp.GraphGBS` builds a sampling circuit from a graph adjacency matrix.

**Photonic backends and forward results**

- **Fock** (`backend='fock'`, the default)

  - With input occupations or a basis `dq.FockState`, `basis=True` gives the interferometer unitary from `cir()`. Request amplitudes with `cir(is_prob=False)` or probabilities with `cir(is_prob=True)`. Use `cir.get_amplitude(pattern)` or `cir.get_prob(pattern)` for one output pattern.
  - Use `basis=False` for a Fock state tensor with a per-mode `cutoff`. Superpositions use ket amplitudes, `den_mat=True` selects a density matrix and `mps=True, chi=chi` selects a Fock MPS.

- **Gaussian** (`backend='gaussian'`). A `dq.GaussianState` or `[cov, mean]` specifies the input moments in `xxpp` order. `cov, mean = cir()` returns the output moments. Use `cir.get_symplectic()` for the symplectic matrix of a Gaussian unitary, `cir.get_prob(pattern)` for one counting event or `cir(is_prob=True, detector=detector)` for a probability table. Ordinary Gaussian circuits support `'pnrd'`, `'threshold'` and `'click'` detectors.
- **Bosonic** (`backend='bosonic'`). `cov, mean, weight = cir()` evolves a linear combination of Gaussians. Prepare cat and GKP inputs with `cir.cat` and `cir.gkp`. A `dqp.BosonicState` provides `wigner` and `marginal` for phase-space readout.
- **Bargmann** (`backend='bargmann'`). A circuit without submodes returns a `dq.BargmannState` from `state = cir()`. Work with its ket or density kernel directly, use `state.partial_trace(wires)` to trace out modes, and call `state.to_fock(cutoff)` for a finite Fock expansion.

**Measurement and conditional states**

- After a forward pass, `cir.measure(shots=shots, wires=wires)` samples photon-count histograms on Fock, Gaussian and Bargmann circuits. Set `with_prob=True` to include event probabilities. Fock basis and Gaussian circuits can use `mcmc=True` to sample when enumerating all patterns is expensive. A terminal `detector` override applies to that readout.
- On Fock state tensors, Gaussian and Bargmann circuits, `cir.measure(wires=wires, with_state=True)` pairs each sampled count record with its conditional state on the remaining modes. Use `cir.post_select(wires, photons)` for the conditional state and event weight of a chosen record.
- Use `cir.photon_number_mean_var()` for photon-number statistics, `cir.quadrature_mean(wires, phi=phi)` for mean quadratures at selected angles and `cir.quadrature_moments()` for quadrature covariance and mean.
- On Fock state tensors, Gaussian and Bosonic circuits, `cir.measure_homodyne(wires=wire, phi=phi)` samples a quadrature. For a conditional measurement, register `cir.homodyne(wire, phi)`, run the circuit, then call `cir.measure_homodyne()` and read `cir.state_measured`.

**Hardware and orthogonal submodes**

- Attach a `dqp.HardwareModel` with `cir.set_hardware_model(model)`. Gate selectors configure control noise and optical loss, and `model.homodyne(wire)` configures detector efficiency and readout.
- For a supplied input state, set its mode profile with `model.input(wires).submodes(coefficients)`. A registered single-photon source takes its profile in `cir.single_photon(wires, coefficients=coefficients)`. Circuit wires label spatial ports, and a photon counter sums over the orthogonal basis modes in its port. Use `result.to_joint()` on a `dqp.SubmodeResult` for the joint backend state.

**Time-domain multiplexing**

- Add delay loops with `cir.delay`. Use `cir.global_circuit(nstep)` to unroll the circuit. On Gaussian and Bargmann circuits, `cir.tdm_substate(time_steps, wires=wires)` returns the joint state of selected unconditional outputs. Pass `window=window` to iterate over consecutive output windows.
- Use `cir(memory=memory)` for a Gaussian or Bargmann step with fresh spatial inputs and an incoming loop state. On Gaussian circuits without submodes, `cir.tdm_memory(state)` extracts the loop modes from a full step in next-step order.
- For stepwise homodyne conditioning, build a `dq.QumodeCircuitTDM` and register homodyne measurements on its spatial outputs. Run `tdm(nstep=nstep)` and read the measurement record from `tdm.samples`.

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
pytest tests
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

- Search `tests/` for the affected public method, backend or observable and run the matching cases first. Use pytest's `-k` option to select tests by name and `--collect-only -q` to preview the selection.
- Compare simulation results with an analytic value, an equivalent circuit or an independent simulator. Check gradients, batch behavior, dtype and device where the changed path supports them.
- Benchmark performance changes end to end with the same physical inputs, requested outputs and timing boundaries on both revisions.
- For broad behavioral changes, run `pytest tests`, `ruff check .`, and `pre-commit run --all-files` when practical.

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
