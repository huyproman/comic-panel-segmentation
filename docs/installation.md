# Installation Guide

## 1. System Requirements
- **Operating System**: Linux (Ubuntu 20.04/22.04 LTS tested) or POSIX-compliant environment
- **Python Version**: Python 3.10 (3.10.x validated; strict requirement `3.10 <= python < 3.11`)
- **Memory**: Minimum 8 GB RAM (16 GB recommended for full batch replay)
- **Disk Space**: ~2 GB for virtualenv and dependencies; ~500 MB for checkpoints/fixtures

---

## 2. Setting Up the Environment

We recommend creating an isolated virtual environment using `python3.10`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

---

## 3. Installing the Package

### Standard Installation
From the repository root:
```bash
pip install -e .
```

### Development Installation
To include test runners and linting tools (`pytest`, `ruff`):
```bash
pip install -e ".[dev]"
```

---

## 4. Verifying the Installation

Verify the CLI is registered and operational:
```bash
comics-panel-extraction --help
```

Run the unit and regression test suite:
```bash
pytest tests/
```
All 141 tests should pass cleanly.
