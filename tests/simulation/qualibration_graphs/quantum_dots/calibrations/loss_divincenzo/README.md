# Simulation Testing (Loss-DiVincenzo)

This directory contains local simulation tests for Loss-DiVincenzo calibration nodes.

## Overview

Simulation tests allow you to:
- Verify that calibration nodes produce valid QUA programs
- Generate visual artifacts showing simulated waveforms
- Create documentation for each calibration node
- Detect changes in calibration behavior during code review

## Simulation Mode

### Local Mode (Only)

Uses the hardware cluster's built-in OPX+ simulator via `node.run(simulate=True)`.

## Setup

### 1. Configure Qualibrate (One-Time)

Set the calibration library folder:

```bash
qualibrate config --runner-calibration-library-folder qualibration_graphs/quantum_dots/calibrations/loss_divincenzo
qualibrate config --runner-calibration-library-resolver qualibrate.QualibrationLibrary
```

### 2. Configure Cluster Access (Local Simulator)

Provide the QM host so `node.machine.connect()` can reach the cluster:

```bash
export QM_HOST=YOUR_CLUSTER_IP
export QM_CLUSTER_NAME=YOUR_CLUSTER_NAME  # optional
```

Or create `tests/.qm_cluster_config.json`:

```json
{
  "host": "YOUR_CLUSTER_IP",
  "cluster_name": "YOUR_CLUSTER_NAME"
}
```

### 3. Install Dependencies

Option A: Use `uv` with the Quantum Dots `pyproject.toml`:

```bash
cd qualibration_graphs/quantum_dots
uv sync
source .venv/bin/activate
```

Option B: Use pip directly:

```bash
pip install qm-qua matplotlib pytest
```

## Running Tests

```bash
RUN_SIM_TESTS=1 pytest tests/simulation/qualibration_graphs/quantum_dots/calibrations/loss_divincenzo/ -v
```

Run a specific test:

```bash
RUN_SIM_TESTS=1 pytest tests/simulation/qualibration_graphs/quantum_dots/calibrations/loss_divincenzo/test_09b_time_rabi_chevron_parity_diff_sim.py -v
```

## Adding a New Simulation Test

1) Copy the template:

```bash
cp tests/simulation/test_template.py.example \
  tests/simulation/qualibration_graphs/quantum_dots/calibrations/loss_divincenzo/test_<node_name>_sim.py
```

2) Update `NODE_NAME` and optional overrides in the new test file.

3) Run the test and verify artifacts are produced under:

```
tests/simulation/artifacts/<node_name>/
```

### Common Overrides

Use `param_overrides` to shorten simulations and avoid timeouts:

```python
param_overrides={
    "num_shots": 10,
    "simulation_duration_ns": 10_000,
    "timeout": 30,
}
```

## Artifacts

Each simulation test generates:

- `simulation.png`: Simulated waveforms plot
- `README.md`: Auto-generated documentation with parameters and a plot link

## Directory Structure

```
tests/simulation/qualibration_graphs/quantum_dots/calibrations/loss_divincenzo/
├── conftest.py
├── macros.py
├── quam_factory.py
├── README.md
└── test_*_sim.py
```

Artifacts are written to:

```
tests/simulation/artifacts/<node_name>/
```

## Troubleshooting

### "Simulation tests disabled"

Set the environment variable: `export RUN_SIM_TESTS=1`

### Connection Errors

Verify cluster network access and that your environment can reach the QM hardware cluster.
