# Gate Virtualization: Barrier-Barrier Experiment and Analysis Guide

This document explains the intended **barrier-barrier virtualization experiment** (node 03),
how the current analysis implements it, and where the current simulation is not yet physically
faithful.

Scope:
- Experiment flow for `03_barrier_compensation`
- Optional PAT lever-arm node (`02a_pat_lever_arm_calibration`)
- Analysis pipeline in `barrier_compensation_analysis.py`
- Matrix update behavior in `analysis.py`
- Current hybrid simulator behavior used by analysis tests

Reference method:
- T.-K. Hsiao et al., *Efficient orthogonal control of tunnel couplings in a quantum dot array* (arXiv:2001.07671)

## 1. Physical Goal

After dot-potential virtualization is already in place (`P'`, `B'`), we want new virtual barriers
`B†` such that:

- changing `B†_i` changes only tunnel coupling `t_i`
- all other tunnel couplings `t_j!=i` stay (approximately) fixed
- dot potentials stay fixed (already handled by prior virtual-gate layers)

In paper notation:

- tunnel coupling model is approximately exponential in local gate coordinates:
  `t_i = t0_i * exp(Phi_i)`
- exponent is linear in gates near operating point:
  `Phi_i = sum_j Gamma_ij * B'_j + ...`
- therefore local slope ratios from small sweeps give crosstalk ratios:
  `Gamma_ij / Gamma_ii = (dt_i/dB'_j) / (dt_i/dB'_i)`

## 2. Experimental Measurement Concept

This section is the key point behind:
`Fit t_i vs drive voltage to get slope m_ij = dt_i/dB_j`.

## 2.1 What one matrix element means

`m_ij` means:

- `i` = which tunnel coupling you care about (target coupling), for example `t23`
- `j` = which barrier you deliberately move (drive barrier), for example `B12`
- `m_ij` = local derivative of the extracted tunnel coupling with respect to that barrier:
  `m_ij = dt_i/dB_j`

Interpretation:

- `m_ii` (diagonal) is the intended control strength of the "correct" barrier on its own tunnel coupling.
- `m_ij` for `j != i` is unwanted crosstalk.
- ratio `m_ij / m_ii` is what matters for virtualization row coefficients.

## 2.2 How one `m_ij` is measured in practice

To get one slope `m_ij`, the experiment is:

1. Choose target coupling `t_i`.
2. Sweep drive barrier `B_j` through small setpoints near the operating point.
3. At each fixed `B_j` setpoint, perform a detuning scan through the inter-dot transition used to read out `t_i`.
4. Fit that detuning trace with a finite-temperature two-level model to extract one scalar tunnel coupling value.
5. Repeat for all drive setpoints to get a curve:
   `t_i(B_j1), t_i(B_j2), ..., t_i(B_jN)`
6. Fit this curve linearly around the operating point:
   `t_i ≈ intercept + m_ij * B_j`

The fitted slope is the estimated local derivative `m_ij`.

## 2.3 Why this works for the paper method

Paper assumption near the operating point:

- `t_i = t0_i * exp(Phi_i)`
- `Phi_i` linear in gates

Therefore:

- `dt_i/dB_j = t_i * dPhi_i/dB_j = t_i * Gamma_ij`
- and ratio:
  `(dt_i/dB_j) / (dt_i/dB_i) = Gamma_ij / Gamma_ii`

So measuring local derivatives is enough to recover crosstalk ratios used to define virtual barriers.

## 2.4 Concrete toy example

Suppose target coupling is `t23` and drive is `B12`.

Measured/fitted `t23` values from detuning traces:

- at `B12=-10 mV`: `t23=22.8`
- at `B12=  0 mV`: `t23=23.2`
- at `B12=+10 mV`: `t23=23.6`

Linear slope around 0 mV:

- `m_23,12 = dt23/dB12 ≈ (23.6 - 22.8)/(20 mV) = 0.04 per mV`

If self-slope from sweeping `B23` is:

- `m_23,23 = dt23/dB23 = 0.20 per mV`

then normalized crosstalk coefficient used in row 23 is:

- `r_23,12 = m_23,12 / m_23,23 = 0.20`

This is exactly what enters the row update for stepwise `B* -> B†`.

## 2.5 From all slopes to virtualization matrix

After extracting all required `m_ij`:

- assemble slope matrix `M = [m_ij]`
- for each row `i`, normalize by self slope `m_ii`
- compose stepwise updates in calibration order
- enforce diagonal to exactly 1 after each row
- final transform is `B†`

## 3. Current Code Mapping

## 3.1 Node 03 action flow

File:
- `qualibration_graphs/quantum_dots/calibrations/gate_virtualization/03_barrier_compensation.py`

High-level flow:

1. `create_qua_program`
   - creates per-pair 2D scans for `target_barrier_vs_drive_barrier`
2. `execute_qua_program`
   - acquires and stores datasets in `node.results["ds_raw_all"]`
3. `analyse_data`
   - preprocesses data (`process_raw_dataset`)
   - extracts pair slopes (`extract_barrier_compensation_coefficients`)
   - builds `slope_matrix_raw`
   - runs stepwise composition (`calibrate_stepwise_barrier_virtualization`)
4. `plot_data`
   - pair scans, pair fits, transform history, final matrix plots
5. `update_virtual_gate_matrix`
   - writes only barrier sub-block into active matrix layer

Outputs written into `node.results` include:

- `slope_matrix_raw`
- `barrier_transform_history`
- `barrier_transform_final`
- `residual_crosstalk`
- `fit_results`

## 3.1b Optional PAT node (current placeholder)

File:
- `qualibration_graphs/quantum_dots/calibrations/gate_virtualization/02a_pat_lever_arm_calibration.py`

Purpose:

- Provide PAT-derived lever-arm mappings (dot-pair keyed and barrier keyed)
- Feed barrier analysis with calibrated detuning scale factors when available
- Fall back to `default_lever_arm = 1.0` when unavailable

Current status:

- Hardware PAT acquisition/fitting is not implemented yet.
- Node currently ingests provided mappings and stores them in `node.results`.
- Machine-state persistence is intentionally left as TODO (no machine lever-arm schema yet).

## 3.2 QUA program logic in detail

Files:
- `qualibration_graphs/quantum_dots/calibrations/gate_virtualization/03_barrier_compensation.py`
- `qualibration_graphs/quantum_dots/calibration_utils/gate_virtualization/scan_utils.py`

### 3.2.1 Pair-program generation

`create_qua_program` creates one scan program per pair key:

- `pair_key = "{target_barrier}_vs_{drive_barrier}"`
- sets:
  - `x_axis_name = drive_barrier`
  - `y_axis_name = target_barrier`
- then calls `create_2d_scan_program(...)`

Current behavior note:

- The present scan builder interprets this as a generic 2D gate-voltage scan (`x_volts`, `y_volts`).
- In strict physics terms, the second axis should ideally be detuning for the relevant dot pair.
- So current node-03 data path is an offline-compatible proxy for full detuning-controlled acquisition.

### 3.2.2 Inside `create_2d_scan_program`

Common setup:

1. Resolve gate objects from names.
2. Build voltage arrays `x_volts`, `y_volts` from parameters (`x_span`, `x_points`, etc.).
3. Configure optional QDAC lists if axes are external.
4. Build sweep-axis metadata used by `XarrayDataFetcher`.

Then compile one of four QUA loop topologies:

1. Both axes OPX
2. X from QDAC, Y from OPX
3. X from OPX, Y from QDAC
4. Both axes QDAC with trigger counter

### 3.2.3 Core OPX/OPX loop structure (conceptual)

The most important case (both OPX) is:

1. Outer averaging loop over `num_shots`
2. Loop over X points (`drive barrier` setpoints in node 03)
3. Loop over Y points (`target/proxy axis` in node 03)
4. Ramp/set gate voltages for current pixel
5. Optional pre-measure delay
6. Read I/Q from sensor resonator
7. Save streams and average in stream processing

Pseudo-structure:

```python
for shot in range(num_shots):
    for x in x_volts:
        for y in y_volts:
            seq.ramp_to_voltages({x_gate: x, y_gate: y}, ...)
            measure_sensor_IQ()
        if per_line_compensation:
            seq.apply_compensation_pulse()
    seq.apply_compensation_pulse()
```

Output shape after stream processing:

- `I` and `Q` buffered as `[len(x_volts), len(y_volts)]` per sensor after averaging.

### 3.2.4 Execution and dataset assembly

In `execute_qua_program`:

1. Iterate over all pair programs.
2. Execute each program in `qm_session`.
3. Fetch data via `XarrayDataFetcher` with per-pair sweep axes.
4. Store as:
   `node.results["ds_raw_all"][pair_key] = dataset`

So each pair gets its own 2D dataset.

## 3.3 Tunnel extraction and slope fitting

File:
- `qualibration_graphs/quantum_dots/calibration_utils/gate_virtualization/barrier_compensation_analysis.py`

Key steps:

1. `fit_finite_temperature_two_level(detuning, signal)`
   - grid-search fit for finite-T two-level transition
   - uses the paper-style sensor model (Fig. 2 caption, arXiv:1803.10352):
     `V(eps) = V0 + dV*Q + [s0 + (s1-s0)*Q]*eps`
   - returns extracted tunnel coupling per detuning trace
2. `extract_tunnel_coupling_vs_drive(ds, drive_axis, detuning_axis)`
   - applies the above fit at each drive setpoint
3. `fit_linear_slope(x, y)`
   - linear fit of extracted `t_i` vs drive
4. `fit_barrier_cross_talk(...)`
   - wrapper returning `dt_i/dB_j`, fit quality, stderr
5. `assemble_slope_matrix(...)`
   - constructs `M`
6. `calibrate_stepwise_barrier_virtualization(...)`
   - computes `B* -> B†` transform sequence

## 3.4 Analysis dataflow from one pair dataset

Given one pair dataset:

1. `process_raw_dataset`
   - computes amplitude/phase from I/Q if needed
2. `extract_barrier_compensation_coefficients`
   - resolves drive/detuning axes
   - calls `fit_barrier_cross_talk`
3. `extract_tunnel_coupling_vs_drive`
   - for each drive setpoint:
     - take 1D trace along detuning axis
     - fit finite-T transition and extract `t_i`
4. `fit_linear_slope`
   - regress extracted `t_i` values vs drive values
   - return slope (`m_ij`), stderr, R², point count

That one slope becomes one entry in `slope_matrix_raw`.

## 3.5 Matrix writeback

File:
- `qualibration_graphs/quantum_dots/calibration_utils/gate_virtualization/analysis.py`

`update_compensation_submatrix(...)`:

- resolves active virtual-gate-set and layer
- maps gate names to row/col indices
- updates only selected submatrix entries
- preserves all unrelated matrix entries

For node 03, only barrier rows/cols are overwritten by final barrier transform.

## 4. What the Current Hybrid Simulator Does

Test-only simulator file:
- `tests/analysis/qualibration_graphs/quantum_dots/calibration_utils/gate_virtualization/hybrid_barrier_virtualization_simulator.py`

Current hybrid model per pair scan:

1. Defines tunnel-vs-drive from a local barrier model (currently generated in a small-sweep regime)
2. Generates detuning traces from finite-T two-level response
3. Optionally mixes small `qarray` sensor background
4. Adds small noise and produces synthetic I/Q maps

E2E artifact test:
- `tests/analysis/qualibration_graphs/quantum_dots/calibrations/gate_virtualization/test_03_barrier_compensation_analysis.py`

Artifacts written to:
- `tests/analysis/artifacts/03_barrier_compensation/`

## 5. Why the Simulation Is Not Yet Physically Correct

The current simulator is useful for wiring/integration validation, but it is not yet a fully faithful
device model for barrier virtualization. Main gaps:

1. `qarray` does not natively model inter-dot tunnel coupling `t_ij`
   - it models charge/sensor electrostatics, not direct `t_ij` extraction physics
2. Detuning axis is synthetic
   - real experiment needs physically consistent detuning for a specific dot pair and lever arms
3. Transition-shape fit can quantize/clip extracted tunnel values
   - current finite-T grid fit can introduce discretization artifacts in `t` vs drive
4. Slope matrix may look numerically stable but not represent realistic cross-capacitive physics
   - especially for non-nearest-neighbor entries and sign/magnitude hierarchy
5. Stepwise update currently uses one measured slope matrix for composition/refinement
   - paper procedure is tune-and-remeasure at each `B*` stage

## 6. Recommended Iteration Path (Next)

If we want physically meaningful simulation for method validation, iterate in this order:

1. Build a dedicated tunnel-physics forward model
   - explicit `t_i(B)` with controlled nearest-neighbor `Gamma` structure
   - include realistic operating-point dependence and mild nonlinearity
2. Couple it to a detuning readout model
   - generate charge-transition traces with lever-arm-consistent detuning
3. Calibrate fit discretization
   - tighten fit grids or move to continuous optimizer for `t` extraction
4. Add staged remeasurement loop
   - emulate `B' -> B*1 -> B*2 -> ... -> B†` by regenerating data after each transform
5. Validate against paper-like signatures
   - strong self response, reduced off-diagonal residuals, stable performance over range

## 7. Quick Pointers for Review

If you want to inspect one complete run quickly:

1. Open node logic:
   - `qualibration_graphs/quantum_dots/calibrations/gate_virtualization/03_barrier_compensation.py`
2. Open analysis math:
   - `qualibration_graphs/quantum_dots/calibration_utils/gate_virtualization/barrier_compensation_analysis.py`
3. Open latest test artifacts:
   - `tests/analysis/artifacts/03_barrier_compensation/README.md`
   - `tests/analysis/artifacts/03_barrier_compensation/simulation.png`
   - `tests/analysis/artifacts/03_barrier_compensation/slope_fit_example.png`
   - `tests/analysis/artifacts/03_barrier_compensation/matrix_final.png`

---

If useful, the next step can be a second document that defines a concrete, parameterized
"physics-grounded synthetic device" spec (states, lever arms, barrier map, noise terms),
so simulator and analysis expectations are explicit before code changes.
