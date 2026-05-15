# AGENTS.md

Guidance for AI agents using or modifying this repository.

## Project Overview

This repository is the `deepquantum` Python package. It is a PyTorch-based quantum computing simulation library for quantum circuits, quantum machine learning, photonic quantum computing, measurement-based quantum computation (MBQC), tensor-network/MPS simulation, and distributed simulation.

The core package source is under `src/deepquantum/`. Prefer public APIs exported from `deepquantum` or `deepquantum.photonic` before reaching into private helpers or implementation details.

## Repository Layout

- `src/deepquantum/`: qubit state/circuit simulation, gates, layers, observables, channels, ansatz classes, QASM helpers, circuit cutting, distributed state-vector simulation, and shared math utilities.
- `src/deepquantum/photonic/`: photonic states, `QumodeCircuit`, Fock/Gaussian/Bosonic backends, photonic gates/channels/measurements, Clements/GBS ansatzes, TDM circuits, unitary decomposition/mapping, and photonic distributed simulation.
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

- Use `dq.QumodeCircuit(nmode, init_state, cutoff=None, backend='fock', basis=True, den_mat=False, detector='pnrd', mps=False, chi=None, noise=False, mu=0, sigma=0.1)`.
- Fock backend:
  - `basis=True` represents Fock basis states such as `[1, 0, 1]`; `cir(is_prob=None)` returns a unitary, `cir(is_prob=True)` returns probabilities, and `cir(is_prob=False)` returns amplitudes.
  - `basis=False` represents Fock state tensors or superpositions such as `[(amp, [n0, n1])]`.
  - Use `cir.measure(...)`, `cir.get_amplitude(final_state)`, and `cir.get_prob(final_state)`.
  - For Fock MPS, use `backend='fock', basis=False, mps=True, chi=..., cutoff=...`.
- Gaussian backend:
  - Use `backend='gaussian'` with `'vac'` or `[cov, mean]`; output is `[cov, mean]` unless `is_prob=True`.
  - Supported workflows include Gaussian gates (`s`, `s2`, `d`, `r`, `bs`, etc.), `get_symplectic`, `photon_number_mean_var`, Fock probabilities with `detector='pnrd'` or `'threshold'`, and `measure_homodyne`.
- Bosonic backend:
  - Use `backend='bosonic'` with `'vac'`, `[cov, mean, weight]`, a list of local `BosonicState`s, or prepared `cat`/`gkp` states.
  - Output is `[cov, mean, weight]`; `CatState`, `GKPState`, `FockStateBosonic`, Wigner functions, marginals, and homodyne workflows are tested.
  - `QumodeCircuit.measure()` asserts Fock/standard sampling is only supported for Fock and Gaussian backends. Use `measure_homodyne` or documented Bosonic measurement classes for Bosonic workflows.
- Clements and GBS:
  - `dq.Clements`, `dq.UnitaryDecomposer`, `dq.GaussianBosonSampling`, and `dqp.GraphGBS` are documented through tutorials/examples.
  - `QumodeCircuit.any(...)` supports arbitrary unitary gates; tutorials note `clements(...)` performs a physical Clements decomposition and is not used as the differentiable arbitrary-unitary path.

Photonic measurement and probabilities:

- `QumodeCircuit.measure(shots=1024, with_prob=False, wires=None, detector=None, mcmc=False)` returns Fock-state keyed dictionaries, or a list for batched state/data.
- Gaussian `measure(..., mcmc=True)` and Fock `measure(..., mcmc=True)` are documented for sampling when full probabilities are expensive.
- `measure_homodyne(shots=..., wires=...)` returns quadrature samples and stores collapsed states in `state_measured` when measurement operations were added with `homodyne`, `homodyne_x`, or `homodyne_p`.

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
- New or changed nontrivial public APIs in `src/` should include clear docstrings.
- Keep changes consistent with the surrounding PyTorch `nn.Module` patterns, buffer/parameter handling, dtype/device behavior, and existing test tolerances.

## Testing Guidance

- For simulation changes, run the most relevant targeted tests first.
- For qubit circuit changes, inspect and run relevant tests such as `tests/test_circuit.py`, `tests/test_mps.py`, `tests/test_auto_grad.py`, `tests/test_get_amplitude.py`, `tests/test_channel.py`, and `tests/test_module.py`.
- For photonic changes, inspect and run relevant `tests/test_photonic_*.py` files plus compatibility tests such as `tests/test_with_xanadu*.py`, `tests/test_with_perceval.py`, `tests/test_with_qutip.py`, and `tests/test_with_pennylane.py` when applicable.
- For MBQC changes, inspect and run `tests/test_mbqc_transpile.py` and `tests/test_with_graphix.py`.
- For broad behavioral changes, run `pytest`, `ruff check .`, and `pre-commit run --all-files` when practical.
- Numerical tests should use appropriate tolerances and compare against existing expected behavior or trusted reference implementations already present in tests.

## Notebook and Documentation Rules

- Tutorials, examples, and docs demos use Jupytext.
- `.ipynb` files are the source of truth for tutorial/example notebook content.
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
- Do not revert user changes or unrelated local changes.
- When a conversation reveals a reusable workflow mistake or correction, distill it into a short global `AGENTS.md` rule so the same process error is less likely to recur.

## Agent Workflow Expectations

- Before answering simulation questions, inspect the relevant source, tests, tutorials, or examples.
- Prefer tested examples over speculative API usage.
- When writing simulation snippets, include imports, minimal parameters, and expected output shape or behavior when clear.
- When changing code, keep edits focused and consistent with existing patterns.
- After changes, report what changed and which commands were run.
- If a command was not run, say so clearly.
