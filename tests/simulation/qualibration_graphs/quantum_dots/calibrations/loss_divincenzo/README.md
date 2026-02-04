# Simulation Testing (Loss-DiVincenzo)

This directory contains local simulation tests for Loss-DiVincenzo calibration nodes.

## 1) Create an environment with uv

Use the Quantum Dots `pyproject.toml` to create the env:

```bash
cd qualibration_graphs/quantum_dots
uv sync
source .venv/bin/activate
```

## 2) Configure Qualibrate (one-time)

Set the calibration library folder and resolver:

```bash
qualibrate config --runner-calibration-library-folder qualibration_graphs/quantum_dots/calibrations/loss_divincenzo
qualibrate config --runner-calibration-library-resolver qualibrate.QualibrationLibrary
```

## 3) Configure cluster access (local simulator)

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

## 4) Run a node simulation test

Run all Loss-DiVincenzo simulation tests:

```bash
RUN_SIM_TESTS=1 pytest tests/simulation/qualibration_graphs/quantum_dots/calibrations/loss_divincenzo/ -v
```

Run a single node:

```bash
RUN_SIM_TESTS=1 pytest tests/simulation/qualibration_graphs/quantum_dots/calibrations/loss_divincenzo/test_09b_time_rabi_chevron_parity_diff_sim.py -v
```

## 5) Artifacts

Generated artifacts are written to:

```
tests/simulation/artifacts/<node_name>/
```

Each test generates:
- `simulation.png`: simulated waveforms plot
- `README.md`: auto-generated documentation with parameters and a plot link

## 6) Add a test for a new node

1) Copy the template:

```bash
cp tests/simulation/test_template.py.example \
  tests/simulation/qualibration_graphs/quantum_dots/calibrations/loss_divincenzo/test_<node_name>_sim.py
```

2) Update `NODE_NAME` in the new test file.

3) (Optional) Override node parameters:

```python
param_overrides={
    "num_shots": 10,
    "simulation_duration_ns": 10_000,
    "timeout": 30,
}
```

4) Run the test and verify artifacts in `tests/simulation/artifacts/<node_name>/`.
