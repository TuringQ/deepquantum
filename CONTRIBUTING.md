This is a comprehensive, professional `CONTRIBUTING.md` tailored for your project.
It covers the specific tools you've configured and adds industry-standard best practices for a seamless open-source contribution experience.

---

# Contributing to DeepQuantum

First off, thank you for considering contributing to DeepQuantum!
It’s people like you who make the open-source community such an amazing place to learn, inspire, and create.

To maintain high code quality and a consistent developer experience, we have established a modern workflow using **Ruff**, **Jupytext**, and **pre-commit hooks**.
Please follow these guidelines to get started.

---

## Setup Development Environment

Before you start coding, ensure you have the development dependencies installed.
We recommend using a virtual environment (Conda or venv).

1. **Install Dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Install pre-commit hooks**:
   We use `pre-commit` to automate code linting and formatting.
   This ensures your code is compliant before every commit.
   ```bash
   pre-commit install
   ```

3. **Initialize Environment**:
   It is a good practice to run the hooks against all files once to ensure your local environment is in sync:
   ```bash
   pre-commit run --all-files
   ```

---

## Coding Standards

We use **Ruff** as our primary linter and formatter.
Our configuration includes:
- **Line Length**: 120 characters.
- **Quote Style**: Single quotes (`'`).
- **Imports**: Sorted automatically (isort rules).

### Linting Rules
We enforce a strict set of rules, including Python upgrade suggestions (`UP`), naming conventions (`N`), and code simplifications (`SIM`).
- **In the `src/` directory**: We additionally enforce **Google-style docstrings**.
Please ensure your functions and classes are well-documented.

### IDE Integration
We recommend installing the **Ruff extension** for VS Code or your preferred IDE.
Enable **Format on Save** to handle most styling issues automatically.

---

## Working with Notebooks (Jupytext)

Our tutorials and examples are managed using **Jupytext**.
This allows us to maintain Jupyter Notebooks (`.ipynb`) while tracking version-control-friendly Python scripts (`.py:percent`).

### The Rule of Thumb:
**Maintain the `.ipynb` files only.**
The paired `.py` files are automatically generated or updated via pre-commit hooks.

### IDE Integration:
We recommend installing the **Jupytext extension** for VS Code or your preferred IDE to sync files automatically.

> [!IMPORTANT]
> **Version Consistency**: The VS Code Jupytext extension often points to a global path rather than your active Conda environment.
Please ensure that the Jupytext version used by your IDE matches the one in `requirements-dev.txt` to avoid formatting discrepancies.

---

## Maintenance & Updates

To keep the development tools up to date with the latest security fixes and features, please occasionally update the pre-commit hook versions:

```bash
pre-commit autoupdate
```

---

## Pull Request Process

1. **Create a Branch**: Use a descriptive name like `feat/new-model` or `fix/issue-123`.
2. **Commit Your Changes**:
   - If `pre-commit` fails during commit, it will often automatically fix the issues.
   - Simply **re-add (`git add`)** the fixed files and commit again.
3. **Reference Issues**: In your PR description, use keywords like `Closes #123` to link the PR to an issue.
4. **Clean History**: We prefer a clean commit history.
Please consider squashing your commits or rebasing on `main` before the final review.

---

## Documentation Guidelines

For source code, we follow the **Google Docstring Convention**.
- **`D102` / `D105` / `D107`**: We currently ignore missing docstrings in public methods and magic methods to reduce boilerplate, but we highly encourage documenting any complex logic.

---

## Need Help?

If you have questions about the setup or a specific feature, feel free to:
- Open an **Issue** on GitHub.

Thank you for contributing to the future of quantum computing!
