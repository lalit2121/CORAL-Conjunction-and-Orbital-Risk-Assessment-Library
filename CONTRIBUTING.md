# Contributing to CORAL

Thank you for your interest in improving CORAL! This document provides guidelines for contributing code, reporting issues, and proposing features.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Testing Requirements](#testing-requirements)
- [C++ Extension Development](#c-extension-development)

---

## Code of Conduct

This project adheres to a standard of professional courtesy. Be respectful, constructive, and assume good intent in all interactions.

---

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/CORAL.git
   cd CORAL
   ```
3. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

---

## Development Setup

### Python Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
pip install -e .  # if setup.py exists
```

### C++ Extension (Optional)

If your changes touch `bindings.cpp` or `casas_da.hpp`:

```bash
bash build.sh
# Verify
python -c "import casas_cpp; print(casas_cpp.__version__)"
```

---

## Coding Standards

### Python

- **PEP 8** compliance (run `flake8` or `ruff` before committing).
- **Type hints** for all public function signatures.
- **Docstrings** in Google style:
  ```python
  def compute_all_pc(rec: CDMRecord, fast: bool = False) -> PcResult:
      """Compute collision probability using four independent methods.

      Args:
          rec: Parsed CDM record with geometry and covariance.
          fast: If True, skips Monte Carlo to save ~0.5s per call.

      Returns:
          PcResult containing Alfano, Foster, Chan, MC, and consensus values.
      """
  ```
- **Maximum line length:** 100 characters.
- **Imports:** Group as stdlib → third-party → local, alphabetized within groups.

### C++

- **Standard:** C++17 minimum.
- **Formatting:** 4-space indentation, `UpperCamelCase` for classes, `snake_case` for functions.
- **Headers:** Use `#pragma once` or include guards.
- **pybind11:** Expose clear Pythonic names; document with `m.def("name", &fn, py::arg("x"), "docstring");`.

---

## Commit Messages

Use structured commit messages:

```
type(scope): concise description

Optional longer explanation body.

- type: feat, fix, docs, style, refactor, test, chore
- scope: parser, db, pc, propagator, api, dashboard, gmat, viz
- description: imperative mood, no period at end
```

Examples:
```
feat(pc): add Chan series fallback for high-eccentricity orbits
fix(db): resolve WAL mode locking during batch insert
refactor(propagator): extract RK78 coefficients to constexpr array
docs(readme): update GMAT interface examples
```

---

## Pull Request Process

1. **Update documentation** if you change APIs or add features.
2. **Add tests** for new functionality (see [Testing Requirements](#testing-requirements)).
3. **Ensure all tests pass:**
   ```bash
   python test.py
   python test_integration.py
   ```
4. **Update `CHANGELOG.md`** (if present) with a summary of your changes.
5. **Submit PR** with a clear title and description referencing any related issues.
6. **Wait for review.** Maintainers will respond within 5 business days.

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests added and passing
- [ ] C++ extension compiles (if modified)
- [ ] Docstrings updated
- [ ] No breaking changes without discussion
- [ ] Commit messages follow convention

---

## Testing Requirements

### Minimum Coverage

All new Python modules must include:

1. **Unit tests** for pure functions (e.g., covariance reconstruction, Pc formulas).
2. **Integration tests** for database operations and API endpoints.
3. **Smoke tests** for C++ ↔ Python round-trips.

### Running Tests

```bash
# Full suite
pytest tests/ -v

# Specific module
pytest tests/test_collision_probability.py -v

# With coverage
pytest --cov=. --cov-report=html

# C++ integration only
python test_integration.py
```

### Test Data

If your tests require CDM data, add minimal CSV fixtures to `tests/fixtures/` rather than committing large production datasets.

---

## C++ Extension Development

### Adding a New Propagator Feature

1. Implement in `casas_da.hpp` with clear mathematical comments.
2. Add pybind11 bindings in `bindings.cpp`.
3. Expose a Pythonic wrapper in `casas_propagator.py` with fallback logic.
4. Add an integration test in `test_integration.py`.

### ABI Compatibility

- Rebuild the C++ extension after any change to `bindings.cpp` or header files.
- The Python layer detects version mismatches via `casas_cpp.__version__`.

---

## Questions?

Open a [Discussion](https://github.com/yourusername/CORAL/discussions) or reach out via the issue tracker.

---

<div align="center">
  <p><strong>Thank you for helping make space safer! 🛰</strong></p>
</div>
