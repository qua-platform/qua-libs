# TimestampRecorder Protocol Validation Report

Generated: 2026-07-30 02:16 UTC
Hardware phase: disabled

## Executive summary

- Protocol scripts scanned: **46**
- Programs loaded for instrumentation: **41/46**
- Instrumentation succeeded: **41** program(s)
- Manual-timestamp guard (08d): **expected rejection**

### Tool changes made during validation

- Migrated protobuf cloning/walking to betterproto (`copy.deepcopy`, dataclass field walk).
- Added `wait_for_timestamps()` and `timeout_s` to `TimestampRecorder.fetch()` so timestamp-only reads do not block on experiment result streams.
- Added `test_timestamp_tools_protocols.py` batch harness for repeatable validation.

### Scripts not batch-loadable (protocol structure / prerequisites, not tool bugs)

- `04c_*`, `06e_*`: open `qm` and run Octave calibration before the QUA program is traced.
- `17_DRAG_*`: require `drag_coef != 0` in `configuration.py`.
- `20_frequency_tracking.py`: multi-step workflow with programs created inside runtime loops.

## Detailed results

| Script | Category | Load | Instrument | Compile | Hardware | Ops | Timestamps |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| 00_hello_qua.py | hardware_skip | pass | pass | skipped | skipped | 1 | - |
| 01_manual_mixer_calibration.py | hardware_skip | pass | pass | skipped | skipped | 1 | - |
| 02_raw_adc_traces.py | standard | pass | pass | skipped | skipped | 1 | - |
| 02_raw_adc_traces_mw_fem.py | hardware_skip | pass | pass | skipped | skipped | 1 | - |
| 03_time_of_flight.py | standard | pass | pass | skipped | skipped | 1 | - |
| 03_time_of_flight_mw_fem.py | hardware_skip | pass | pass | skipped | skipped | 1 | - |
| 04a_resonator_spectroscopy.py | standard | pass | pass | skipped | skipped | 1 | - |
| 04b_resonator_spectroscopy_wide_range_octave.py | hardware_skip | pass | pass | skipped | skipped | 1 | - |
| 04c_resonator_spectroscopy_wide_range_octave_update_IF.py | hardware_skip | fail | - | - | - | - | - |
| 05_resonator_spectroscopy_vs_amplitude.py | standard | pass | pass | skipped | skipped | 1 | - |
| 06a_qubit_spectroscopy.py | standard | pass | pass | skipped | skipped | 2 | - |
| 06b_qubit_spectroscopy_wide_range_outer_loop.py | standard | pass | pass | skipped | skipped | 2 | - |
| 06c_qubit_spectroscopy_wide_range_inner_loop.py | standard | pass | pass | skipped | skipped | 2 | - |
| 06d_qubit_spectroscopy_wide_range_octave.py | hardware_skip | pass | pass | skipped | skipped | 2 | - |
| 06e_qubit_spectroscopy_wide_range_octave_update_IF.py | hardware_skip | fail | - | - | - | - | - |
| 07a_rabi_chevron_duration.py | standard | pass | pass | skipped | skipped | 2 | - |
| 07b_rabi_chevron_amplitude.py | standard | pass | pass | skipped | skipped | 2 | - |
| 08a_time_rabi.py | standard | pass | pass | skipped | skipped | 2 | - |
| 08b_power_rabi.py | standard | pass | pass | skipped | skipped | 2 | - |
| 08c_power_rabi_error_amplification.py | standard | pass | pass | skipped | skipped | 2 | - |
| 08d_power_rabi_single_shot_timing.py | manual_timestamps | pass | expected_fail | skipped | skipped | 0 | - |
| 08e_power_rabi_single_shot_timing_with_tool.py | reference | pass | pass | skipped | skipped | 2 | - |
| 09a_IQ_blobs.py | standard | pass | pass | skipped | skipped | 3 | - |
| 09b_resonator_depletion_time.py | standard | pass | pass | skipped | skipped | 4 | - |
| 09c_active_reset.py | standard | pass | pass | skipped | skipped | 7 | - |
| 10a_readout_frequency_optimization.py | standard | pass | pass | skipped | skipped | 3 | - |
| 10b_readout_amplitude_optimization.py | standard | pass | pass | skipped | skipped | 3 | - |
| 10c_readout_duration_optimization.py | standard | pass | pass | skipped | skipped | 3 | - |
| 10d_readout_weights_optimization.py | standard | pass | pass | skipped | skipped | 3 | - |
| 11_T1.py | standard | pass | pass | skipped | skipped | 2 | - |
| 12_ramsey_chevron.py | standard | pass | pass | skipped | skipped | 5 | - |
| 13a_ramsey_w_virtual_rotation.py | standard | pass | pass | skipped | skipped | 3 | - |
| 13b_ramsey_w_detuning.py | standard | pass | pass | skipped | skipped | 3 | - |
| 14_echo.py | standard | pass | pass | skipped | skipped | 4 | - |
| 15_allxy.py | standard | pass | pass | skipped | skipped | 57 | - |
| 15_cpmg.py | standard | pass | pass | skipped | skipped | 4 | - |
| 16_xy8.py | standard | pass | pass | skipped | skipped | 11 | - |
| 16a_randomized_benchmarking.py | standard | pass | pass | skipped | skipped | 45 | - |
| 16b_randomized_benchmarking_interleaved.py | standard | pass | pass | skipped | skipped | 45 | - |
| 16c_randomized_benchmarking_20ns.py | standard | pass | pass | skipped | skipped | 85 | - |
| 16d_randomized_benchmarking_interleaved_20ns.py | standard | pass | pass | skipped | skipped | 85 | - |
| 17_DRAG_calibration_Google.py | hardware_skip | fail | - | - | - | - | - |
| 17_DRAG_calibration_Yale.py | hardware_skip | fail | - | - | - | - | - |
| 18_AC_Stark_calibration_Google.py | standard | pass | pass | skipped | skipped | 3 | - |
| 19_state_tomography.py | standard | pass | pass | skipped | skipped | 5 | - |
| 20_frequency_tracking.py | hardware_skip | fail | - | - | - | - | - |

## Failures

- **04c_resonator_spectroscopy_wide_range_octave_update_IF.py** (load): NameError: name 'qm' is not defined
- **06e_qubit_spectroscopy_wide_range_octave_update_IF.py** (load): No completed Program objects found before QOP connection.
- **17_DRAG_calibration_Google.py** (load): AssertionError: The DRAG coefficient 'drag_coef' must be different from 0 in the config.
- **17_DRAG_calibration_Yale.py** (load): AssertionError: The DRAG coefficient 'drag_coef' must be different from 0 in the config.
- **20_frequency_tracking.py** (load): No completed Program objects found before QOP connection.
