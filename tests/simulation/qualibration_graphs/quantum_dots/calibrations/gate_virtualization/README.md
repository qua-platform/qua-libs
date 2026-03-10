# Simulation Testing (Gate Virtualization)

This directory contains local simulation tests for gate-virtualization calibration nodes.

## 1) Create an environment

```bash
cd qualibration_graphs/quantum_dots
uv sync
source .venv/bin/activate
```

## 2) Configure Qualibrate (one-time)

```bash
qualibrate config --runner-calibration-library-folder qualibration_graphs/quantum_dots/calibrations/gate_virtualization
qualibrate config --runner-calibration-library-resolver qualibrate.QualibrationLibrary
```

## 3) Configure cluster access (local simulator)

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

## 4) Run tests

Run all gate-virtualization simulation tests:

```bash
RUN_SIM_TESTS=1 pytest tests/simulation/qualibration_graphs/quantum_dots/calibrations/gate_virtualization/ -v
```

Run a single node:

```bash
RUN_SIM_TESTS=1 pytest tests/simulation/qualibration_graphs/quantum_dots/calibrations/gate_virtualization/test_01_sensor_gate_compensation_sim.py -v
```

## 5) Artifacts

Generated artifacts are written to:

```
tests/simulation/artifacts/<node_name>/
```

Each test generates:
- `simulation.png`
- `README.md`
