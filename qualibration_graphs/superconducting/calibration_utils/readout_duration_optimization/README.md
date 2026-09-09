# Readout duration optimization

Entry points in `calibrations/1Q_calibrations`:

- `08c_readout_duration_optimization.py`: GE readout.
- `14a_gef_readout_duration_optimization.py`: GEF readout.

Both follow the QUAlibrate run-action lifecycle used by 08b: create, simulate or
execute, load historical data, analyze, plot, update successful qubits, and save.
The implementation acquires separate single-shot measurements at each actual
pulse duration. It does not use the accumulated-demodulation SNR estimator from
the standalone 10c example.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `num_shots` | 2000 | Shots per state and duration |
| `min_duration_in_ns` | 200 | First duration, inclusive |
| `max_duration_in_ns` | 2000 | Last duration, inclusive |
| `duration_step_in_ns` | 200 | Duration increment |
| `outliers_threshold` | 0.98 | Minimum non-outlier fraction, using the 08b density criterion |
| `qubits` | None | All active qubits |
| `multiplexed` | False | Measure qubits sequentially or in one batch |
| `reset_type` | thermal | Thermal reset only; GEF waits twice the thermalization time |

Durations must be at least 16 ns and multiples of 4 ns; the step must divide the
range exactly. Adjust the range around the current calibrated readout duration.
Set parameters through the GUI/graph or the node's `custom_param` action.
GEF requires calibrated `x180`, `EF_x180`, and `GEF_frequency_shift` (None means zero).
Only `SquareReadoutPulse` with default rectangular integration weights is supported.
Custom optimized weights must be recalibrated for the selected duration.

## Analysis and state

Each point is converted to volts using its own integration duration. A spherical
Gaussian mixture is fitted to the prepared-state IQ samples, as in 08b. Component
labels are matched to prepared states. Nonconvergent/nonfinite points are excluded.
Each qubit independently selects maximum assignment fidelity among points passing
the non-outlier criterion; ties select the shorter duration. Reported fidelities
are in-sample assignment estimates and include state-preparation errors.

GE selects using GMM classification, then invokes the existing IQ-blob analysis
on the selected shots to obtain rotation, thresholds and the final confusion
matrix. It updates `readout.length`, `integration_weights_angle`, `threshold`,
`rus_exit_threshold`, and the resonator's GE confusion matrix.

GEF obtains three centers from the GMM and scores nearest-center classification,
matching the hardware discriminator. It updates `readout_GEF.length` and
`resonator.gef_centers`, converting centers back to raw demodulation units using
the selected length. If `readout_GEF` does not exist, `readout` supplies its
template and the new operation is persisted only after successful analysis.
The GEF confusion matrix is saved in results; it does not replace the GE matrix.

Results include `ds_raw`, `ds_fit`, `ds_iq_blobs`, per-qubit `fit_results`, and
duration, IQ-blob and confusion-matrix figures. Failed qubits retain their state.
Sweep operations exist only in the execution config and are removed from QUAM
after construction, including on errors. Simulation and `load_data_id` replay
never apply calibration state updates.

## Offline tests

From the superconducting project directory, using its dependency environment:

```sh
PYTHONPATH=. MPLBACKEND=Agg python -m pytest tests/test_readout_duration_optimization.py -q
```

These cover GE/GEF multi-qubit selection, invalid data, ties, conversion,
state updates, plots, duration validation and matching integration weights.
QOP simulation and device validation still require an available controller.
