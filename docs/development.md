# Development & Testing Guide

This guide covers developer workflows, running test suites, and maintaining contract invariants.

---

## 1. Setting Up Development Environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

---

## 2. Running Test Suites

The project maintains 141 automated unit, parity, and golden tests across `tests/`:

```bash
# Run full test suite
pytest tests/

# Run unit tests only
pytest tests/unit/

# Run historical parity tests
pytest tests/parity/

# Run golden artifact tests
pytest tests/golden/
```

---

## 3. Core Contract Protection Rules

When contributing to this repository:
1. **Never mutate `comics_panel_extraction/historical_legacy` parameters**: The constants in `config.py` and `pipeline.py` are strictly frozen to preserve historical source truth.
2. **Never change evaluator semantics**: Greedy Dice matching at $0.90$ Dice threshold is the empirical oracle for published tables.
3. **Always run clean-install verification** before committing release updates.
