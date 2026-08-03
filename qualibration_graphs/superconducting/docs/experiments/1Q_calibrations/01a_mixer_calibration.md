# Mixer (Octave) Calibration

[`01a_mixer_calibration.py`](../../../../../calibrations/1Q_calibrations/01a_mixer_calibration.py) · **Targets:** qubits · **Category:** 1Q_calibrations

Runs the QOP's automatic Octave mixer calibration for each qubit's resonator and/or drive upconverter, minimizing LO leakage and image-sideband leakage at the currently configured LO/IF pair.

## Purpose

Every upconversion mixer in the signal chain (resonator readout line, qubit drive line) is driven by an IQ mixer that combines a local-oscillator (LO) tone with an intermediate-frequency (IF) baseband signal from the OPX to produce a single-sideband (SSB) RF tone at $\omega_{RF} = \omega_{LO} \pm \omega_{IF}$. Real mixers deviate from this ideal in three largely independent ways **[GRTW2021]**:

1. **Mixer nonlinearity** — if the IF/AWG drive power into the mixer exceeds its 1-dB compression point (a datasheet-specified value), the SSB output distorts. This is a hardware headroom issue, checked against the mixer's datasheet rather than something this node calibrates away.
2. **LO leakage** — a parasitic, direct LO→RF coupling inside the mixer produces a continuous, always-on tone at $\omega_{LO}$ regardless of what IF signal is applied. Left uncorrected, this tone sits near the qubit or resonator's own frequency and causes unwanted AC Stark shifts or an elevated effective device temperature. It is minimized by applying small DC offset voltages to the mixer's I and Q inputs, found by sweeping the 2D DC-offset plane while monitoring LO-frequency power (on real hardware, with a spectrum analyzer; here, automated by the QOP's calibration routine).
3. **Unwanted-sideband (image) leakage** — caused by *skewness* (I and Q not exactly 90° apart) and *amplitude imbalance* (equal electrical input power at I/Q not producing equal RF output power). It is minimized by jointly tuning a relative I/Q phase offset and an amplitude-scaling correction factor to suppress the image tone at $\omega_{LO} \mp \omega_{IF}$ (the sideband opposite the intended one).

This is why the conventional choice for the qubit drive tone is the **lower sideband** (LSB, $\omega_{LO} - \omega_{IF}$, with $|\omega_{IF}|$ typically 50–100 MHz): since a transmon's anharmonicity is negative ($f_{12} < f_{01}$) **[Koc+2007]**, **[Kra+2019]**, placing the drive on the sideband that keeps residual LO leakage and the (imperfectly suppressed) upper image spectrally far from $f_{12}$ avoids inadvertently driving leakage transitions out of the computational subspace. This node's job is precisely to push the residual LO and image tones down as far as possible so that, whichever sideband convention the drive line uses, neither imperfection is large enough to matter for later calibrations.

## Mechanism

For each qubit, `execute_qua_program` (`calibrations/1Q_calibrations/01a_mixer_calibration.py:47`) calls `qubit.calibrate_octave(qm, calibrate_drive=..., calibrate_resonator=...)`, which is a thin wrapper (`quam_builder`'s `base_transmon.py`) around the QOP's own `QuantumMachine.calibrate_element(...)`:

1. If `calibrate_resonator` is `True` and the qubit's resonator has a `frequency_converter_up` (i.e. is Octave-connected), calibrate it at its **currently configured** LO frequency and intermediate frequency — `QM.calibrate_element(resonator.name, {LO_frequency: (intermediate_frequency,)})`. If the resonator has no `frequency_converter_up`, this raises a `RuntimeError` rather than silently skipping.
2. If `calibrate_drive` is `True` and the qubit's `xy` element is Octave-connected, calibrate it the same way at its own LO/IF.
3. Internally, the QOP's Octave calibration routine performs exactly the DC-offset sweep (LO leakage) and phase/amplitude sweep (image rejection) described above, entirely on the instrument side — no QUA program is written by this node itself.
4. By default (`save_to_db=True` on the underlying `QM.calibrate_element`, not overridden anywhere in this stack), the resulting correction parameters are written to the **Octave calibration database**, not to QUAM state — see State Updates below for why this matters.

Analysis (`calibration_utils/mixer_calibration/analysis.py`):

1. `extract_relevant_fit_parameters` wraps each qubit's raw `MixerCalibrationResults` in a `CalibrationResultPlotter` and reads off `get_lo_leakage_rejection()` and `get_image_rejection()` — both are **dB improvements** (before-vs-after calibration), one pair for the resonator branch and one for the xy-drive branch, whichever were requested.
2. `log_fitted_results` prints these dB figures per qubit.
3. `plot_data` renders the QOP's own before/after calibration figures (`show_lo_leakage_calibration_result`, `show_image_rejection_calibration_result`) via `CalibrationResultPlotter`.

> **`success` is always `True` — there is no real success criterion.** `extract_relevant_fit_parameters` constructs every qubit's `FitParameters` with `success=True` unconditionally (`calibration_utils/mixer_calibration/analysis.py:72`), regardless of how poor the measured LO-leakage/image-rejection numbers actually are. `node.outcomes` will therefore read `"successful"` for every qubit that didn't raise an exception during `calibrate_octave` — a genuinely bad calibration (weak rejection) will *not* be flagged as `"failed"`; only an outright exception (e.g. a non-Octave element) surfaces as a hard failure. Always read the logged dB numbers directly rather than trusting the outcome label.

## Prerequisites

- `00_hello_qua` (or equivalent) confirming basic QUA execution/connectivity — this node needs a working QM session just like any other.
- Each targeted qubit's `resonator`/`xy` element must have a `frequency_converter_up` attribute, i.e. be wired to an Octave in the QUAM state — running this on non-Octave hardware (e.g. an MW-FEM-only setup) will raise `RuntimeError` for the corresponding branch.
- The qubit's LO and intermediate frequencies must already be set to their intended operating values in QUAM state — calibration is performed *at* the currently configured (LO, IF) pair, not searched for.

## Parameters

Inherits the common parameter set — see [Common node parameters](_common_parameters.md) — plus the node-specific parameters below (`calibration_utils/mixer_calibration/parameters.py`).

| Parameter | Type | Default | Unit | Description | Effect |
|---|---|---|---|---|---|
| `calibrate_resonator` | `bool` | `True` | – | Whether to run Octave calibration on the qubit's `resonator` element. | `False` skips the resonator branch entirely — no `resonator` result, no resonator figures, `fit_results[q]["resonator"]` is `None`. |
| `calibrate_drive` | `bool` | `True` | – | Whether to run Octave calibration on the qubit's `xy` element. | `False` skips the xy-drive branch entirely — no `xy_drive` result, no drive figures, `fit_results[q]["xy_drive"]` is `None`. |

> Both parameters have no effect on the *other* branch — they are independent switches, and both default to `True`, i.e. the default run calibrates both lines for every targeted qubit.

## Outputs

**Measured/computed (per qubit, per requested branch):** `lo_leakage` (dB rejection) and `image_rejection` (dB rejection), for `resonator` and/or `xy_drive`.

| Fitted quantity | Unit | Written to state? | Description |
|---|---|---|---|
| `resonator.lo_leakage` | dB | – (see below) | LO-leakage power suppression achieved by calibration, resonator upconverter. Only present if `calibrate_resonator=True`. |
| `resonator.image_rejection` | dB | – (see below) | Image-sideband suppression achieved, resonator upconverter. Only present if `calibrate_resonator=True`. |
| `xy_drive.lo_leakage` | dB | – (see below) | LO-leakage power suppression achieved, xy-drive upconverter. Only present if `calibrate_drive=True`. |
| `xy_drive.image_rejection` | dB | – (see below) | Image-sideband suppression achieved, xy-drive upconverter. Only present if `calibrate_drive=True`. |

**Success criterion:** none, in the meaningful sense — `success` is hardcoded `True` for every qubit that completed `calibrate_octave` without raising (see the callout in Mechanism above).

## State Updates

**No QUAM attribute is written by this node** — there is no `update_state` run action in `01a_mixer_calibration.py`, and the node's docstring (a plain one-liner, `"A simple program to calibrate Octave mixers for all qubits and resonators"`) doesn't claim one either, unlike most other nodes' docstrings which explicitly list a "State update:" section. This is not an oversight: the actual correction values (DC I/Q offsets for LO-leakage suppression, and the phase/gain correction matrix for image suppression) are written by the QOP itself into the **Octave mixer calibration database** (keyed by element, LO frequency, and IF frequency — `save_to_db=True` is the default on `QM.calibrate_element`), not into QUAM's `state.json`. The next time a QM is opened at the same (LO, IF), the QOP automatically retrieves and applies the stored correction. QUAM's own `frequency_converter_up`/mixer-correction fields are not touched by this flow.

## Troubleshooting

1. **`RuntimeError: <element> doesn't have a 'frequency_converter_up' attribute`** → the targeted qubit's `resonator` or `xy` element is not wired to an Octave in the QUAM state (e.g. it's on an MW-FEM or bare LF-FEM path instead). Set `calibrate_resonator`/`calibrate_drive` to `False` for that branch, or fix the QUAM connectivity if an Octave is actually present but misconfigured.
2. **Calibration "succeeds" (`node.outcomes` says `"successful"`) but qubit spectroscopy or readout still shows a spurious tone near the expected frequency** → remember `success=True` is unconditional here. Check the logged `lo_leakage`/`image_rejection` dB numbers directly (`log_fitted_results` output, or `plot_data`'s before/after figures) — a low suppression number (e.g. single-digit dB) means the calibration ran but converged poorly, which this node's outcome will never surface on its own.
3. **LO-leakage suppression is poor even after calibration** → the DC-offset search converges around the currently configured Octave gain/attenuation working point; if the IF drive amplitude is set unusually high or low relative to the mixer's linear range, the offset optimum can be harder to locate cleanly. Also check that nothing else was actively driving the same element (e.g. a leftover job from another QM) during calibration, since the routine assumes a quiet line while it sweeps.
4. **You changed `qubit.xy.intermediate_frequency` or `qubit.resonator.intermediate_frequency` after running this node, and now see LO leakage/image tones again** → the calibration database is keyed by (LO, IF) pair. A correction found for the *old* IF does not automatically apply to a *new* IF at the same LO — re-run this node after any IF (or LO) change, not just once at the start of a bring-up session.
5. **Calibration for the drive line seems to succeed, but downstream spectroscopy still shows power at the wrong sideband contaminating a nearby transition** → double-check which sideband convention is actually configured (upper vs. lower). Per **[GRTW2021]**, the lower sideband is the conventional choice for transmon drives specifically because it keeps residual LO/image leakage away from $f_{12}$ (transmon anharmonicity is negative, so $f_{12} < f_{01}$ **[Koc+2007]**); if the configuration uses the upper sideband instead, even a well-calibrated mixer's residual leakage sits closer to a real leakage transition and is more consequential.

## Parameter Tuning Heuristics

1. **Image rejection is poor even after calibration** → this targets I/Q skew and amplitude imbalance, which are frequency-dependent properties of the physical mixer and cabling; if `|IF|` is very large or very small relative to what the mixer/Octave was characterized for, image suppression can be fundamentally worse regardless of how well the calibration converges. Try calibrating at the actual operating IF you intend to use — a calibration done at a much smaller test IF will not transfer well.
2. **Mixer calibration behaves inconsistently or the underlying instrument reports saturation/distortion** → this is the mixer-nonlinearity failure mode, which this node's DC-offset/phase-amplitude search cannot fix by construction: it only compensates leakage and imbalance, not compression. Check the IF/AWG output power feeding the Octave's mixer against the mixer's datasheet 1-dB compression point **[GRTW2021]** and reduce drive power if it's being exceeded.

## Next Steps

Not part of the automated bring-up calibration graphs (both `80_calibration_graph_bringup_flux_tunable_transmon.py` and `90_calibration_graph_bringup_fixed_frequency_transmon.py` start at `02a_resonator_spectroscopy`) — this is a manual pre-flight step. Next: time-of-flight calibration (`01a_time_of_flight` for OPX+/LF-FEM hardware, or `01b_time_of_flight_mw_fem` for MW-FEM hardware), both of which are themselves prerequisites for `02a_resonator_spectroscopy`.

## References

**[GRTW2021]** Y. Y. Gao, M. A. Rol, S. Touzard, and C. Wang, "Practical guide for building superconducting quantum devices," *PRX Quantum*, vol. 2, p. 040202, 2021.

**[Koc+2007]** J. Koch, T. M. Yu, J. Gambetta, A. A. Houck, D. I. Schuster, J. Majer, A. Blais, M. H. Devoret, S. M. Girvin, and R. J. Schoelkopf, "Charge-insensitive qubit design derived from the Cooper pair box," *Phys. Rev. A*, vol. 76, no. 4, p. 042319, 2007.

**[Kra+2019]** P. Krantz, M. Kjaergaard, F. Yan, T. P. Orlando, S. Gustavsson, and W. D. Oliver, "A quantum engineer's guide to superconducting qubits," *Appl. Phys. Rev.*, vol. 6, no. 2, p. 021318, 2019.
