# Physics-Oriented Tests for Cavity Wiring

This test suite validates the transmon+cavity wiring and state generation functionality using feature branches from `quam-builder` and `py-qua-tools`.

## Dependencies

The tests require specific feature branches that add cavity support:

- `quam-builder@feature/add-bosonic-mode-qpu` - Adds `FixedFrequencyTransmonSingleCavityQuam` class
- `qualang-tools@add-cavity-lines` - Adds `add_cavity_lines()` function

## Installation

### Option 1: Install from requirements.txt

```bash
pip install -r tests/requirements.txt
```

### Option 2: Install in development mode

If you're developing and need to use local clones of the repositories:

```bash
# Clone repositories
git clone https://github.com/qua-platform/quam-builder.git
cd quam-builder
git checkout feature/add-bosonic-mode-qpu
pip install -e .

cd ..
git clone https://github.com/qua-platform/py-qua-tools.git
cd py-qua-tools
git checkout add-cavity-lines
pip install -e .
```

## Running Tests

From the repository root:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_cavity_wiring.py -v

# Run with output (generates state.json and wiring.json in tests/output/)
pytest tests/test_state_wiring_generation.py -v

# Run with coverage
pytest tests/ --cov=qualibration_graphs/superconducting/quam_config --cov-report=html
```

## Test Structure

- `test_cavity_wiring.py` - Tests wiring allocation and connectivity
- `test_cavity_physics.py` - Tests physics parameters and constraints
- `test_state_wiring_generation.py` - Tests generation of state.json and wiring.json files

## Generated Output Files

Tests in `test_state_wiring_generation.py` generate output files in `tests/output/`:

- `wiring.json` - Hardware connectivity mapping
- `state.json` - QUAM state with cavity parameters
- `complete_wiring.json` - Complete wiring setup
- `complete_state.json` - Complete state with populated parameters

These files are git-ignored but can be inspected after test runs.

## Test Categories

### Wiring Allocation Tests
- Single cavity line allocation
- Multi-qubit cavity allocation
- Cavity-transmon association
- Channel specification validation

### Physics Parameter Tests
- MW-FEM band selection (1: 50MHz-5.5GHz, 2: 4.5-7.5GHz, 3: 6.5-10.5GHz)
- Intermediate frequency constraints (|IF| < 400 MHz)
- Power and amplitude calculations
- Cavity coherence times (T1, T2ramsey, T2echo)
- Cavity vs transmon pulse types (SquarePulse vs DRAG)

### State/Wiring Generation Tests
- wiring.json structure validation
- state.json structure validation
- Complete QUAM setup workflow
- Cavity parameter population
