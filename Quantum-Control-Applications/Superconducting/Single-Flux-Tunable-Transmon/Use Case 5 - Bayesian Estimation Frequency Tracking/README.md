# Bayesian Qubit Frequency Tracking (Bayesian-only)

_Important note:_ Documentation only; this describes `FrequencyBayesian_full.py` as written (Qualibrate/QUA/QuAM + QM OPX). It is not a drop-in recipe.

## Overview
This use case demonstrates Bayesian estimation of a Ramsey fringe frequency (effective detuning) using single-shot measurements on a flux-tunable transmon controlled by the OPX.

The protocol performs a Ramsey-style experiment and updates a discrete posterior distribution over a frequency grid after each measurement outcome. At the end of each repetition, the posterior mean provides an estimate of the effective detuning.

## The goal
Estimate a Ramsey fringe frequency (effective detuning) on a discrete grid `vf` (MHz) by repeatedly running a Ramsey-style measurement and performing Bayesian updates from single-shot outcomes.

This does NOT directly output an absolute qubit frequency. Converting `vf` to an absolute frequency requires an external reference and a sign convention (assumption).

## The device
A flux tunable transmon qubit with single-shot readout.

## Prerequisites
Before running this experiment, ensure the following calibrations are complete.
- Readout Chain
  - Time of flight calibration
  - Readout IQ mixer calibration
  - Resonator spectroscopy
  - Readout pulse amplitude and duration 
  - Resonator depletion time 
  - Single-shot discrimination
  - SPAM confusion matrix extraction
- Qubit Drive
  - Qubit IQ mixer calibration
  - Qubit spectroscopy
  - π-pulse calibration
- Flux line
  - Flux amplitude calibration
  - Flux-to-frequency conversion calibration

## Methods and results
### Step 1: Measurement (Ramsey-style acquisition)
- Sweep idle times `t` from `min_wait_time_in_ns` to `max_wait_time_in_ns` in steps of `wait_time_step_in_ns` (quantized to 4 ns clock ticks).
- Per `t`: run `x90 - frame_rotation_2pi(phase) - wait(t) - x90`, measure single-shot `m in {0,1}`; optionally save per-shot `state` when `keep_shot_data=True`.
- During wait(t) time, play a flux pulse for `t` duration and the correct flux amplitude. 
- Virtual phase uses `detuning_eff = detuning - physical_detuning` (input) with `phase = detuning_eff * t_s` inside `frame_rotation_2pi(...)`.
- Unit convention used by the Bayesian cosine (exactly as coded): `t_us = t_ns * 1e-3`, `f` in MHz, so `2*pi*f*t_us` is dimensionless.

### Step 2: Bayesian update (posterior over `vf`)
- Grid (MHz): `vf = arange(f_min, f_max + 0.5*df, df)`. Prior is uniform at the start of each repetition.
- Bayes update per shot (multiply then normalize):
  `P_next(vf) = P(vf) * P(m | vf, tau_us) / sum_vf [P(vf) * P(m | vf, tau_us)]`.
- Likelihood includes readout SPAM via `qubit.resonator.confusion_matrix` (folded into the cosine term via `alpha` and `beta`, as implemented).

Implemented likelihood (as coded, including the `0.99` contrast factor):
`P(m | f, tau_us) = 0.5 + (m - 0.5) * (alpha + beta*cos(2*pi*f*tau_us)) * 0.99`.

### Step 3: Outputs (estimate vs time)
- After finishing the `tau` sweep in one repetition, compute the posterior-mean estimate (MHz):
  `f_hat = sum_vf vf * P(vf)` (implemented as `Math.dot(frequencies, Pf)`).

## Parameters & outputs

| Item | Key(s) in `Parameters` / outputs | Units / type | Notes (checkable in `FrequencyBayesian_full.py`) |
|---|---|---|---|
| Map (estimated vs input) | output: `estimated_frequency`; inputs: `detuning`, `physical_detuning` | MHz (out), int (Hz, in) | Output is the posterior mean over `vf` and corresponds to the effective Ramsey detuning used in the Bayes cosine; it is not an absolute qubit frequency. |
| Repetitions | `num_repetitions` | int | One posterior + one `estimated_frequency` per repetition; posterior is reset after each repetition. |
| Ramsey t grid | `min_wait_time_in_ns`, `max_wait_time_in_ns`, `wait_time_step_in_ns` | ns (inputs) | QUA loop uses 4 ns ticks via `// 4`; Bayes cosine uses `t_us = tau_ns*1e-3`. |
| Virtual phase + flux point | `detuning`, `physical_detuning` | int (typically set via `u.MHz`) | Virtual phase uses `detuning - physical_detuning`; `physical_detuning` also enters the flux pulse conversion. |
| Frequency grid | `f_min`, `f_max`, `df` | MHz | `vf = arange(f_min, f_max + 0.5*df, df)`; intended ~0-8 MHz due to QUA `fixed` limitations. |
| SPAM model | `confusion_matrix` | 2x2 floats | Used as `alpha` and `beta` inside the per-shot likelihood. |
| Per-shot state | `keep_shot_data` | bool | If `True`, saves `state`|


Knobs (heuristics):
- `t` grid: choose span/step to resolve the expected fringe; longer spans cost runtime.
- `vf` grid: set `[f_min, f_max]` to bracket expected detuning; `df` trades resolution vs FPGA workload.
- SPAM: if `qubit.resonator.confusion_matrix` drifts, the posterior can bias; recalibrate as needed.

## Limitations
- The posterior is reset to uniform after every repetition (no carry-over tracking filter across repetitions).
- Likelihood is a pure cosine with SPAM; no explicit decay, offset, or unknown phase parameter is inferred.
- QUA `fixed` arithmetic constrains the usable `vf` range and introduces fixed-point normalization/rounding effects.
- `physical_detuning` -> flux amplitude uses a calibration-dependent square root and can be invalid if its argument is negative.

## References (local)
- [`FrequencyBayesianEstimation.py`](FrequencyBayesianEstimation.py)
- [`configuration.py`](configuration.py)
