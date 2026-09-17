# Single-qubit calibrations

This folder takes a qubit from a raw QUAM machine to individually calibrated single-qubit operations, readout, and flux control.

Every node here runs standalone; the order below is what matters, since each step consumes what the previous one wrote into QUAM.

---

# Bring-up flow

The goal of bring-up is a qubit you can drive with calibrated π / π/2 pulses, read out with state discrimination, and (if flux-tunable) park at a known idle bias. Coherence and RB at the end are the check that this actually worked.

Do time-of-flight ([`01a`](./01a_time_of_flight.py) / [`01b`](./01b_time_of_flight_mw_fem.py)) once per setup so the readout window is aligned, then:

### Flux-tunable transmon

```text
  02a  resonator spectroscopy
    |
  02c  resonator vs flux                         idle / sweet-spot bias
    |
  02b  resonator vs power                        readout power below bifurcation
    |
  03a  qubit spectroscopy
    |
  03b  qubit spectroscopy vs flux                frequency-vs-flux map
    |
  04a  Rabi chevron  -->  04b  power Rabi        coarse then π-pulse amplitude
    |
  08b  readout power  -->  08a  readout freq  -->  07  IQ blobs
    |
  09a  Ramsey vs flux                            finer idle point + df/dΦ
    |
  04b  error-amp x180  -->  04b  error-amp x90   π then π/2, now with discrimination
    |
  06a  Ramsey  -->  05  T1  -->  06b  echo       coherence
    |
  10b  DRAG  -->  11a  RB                        gates, then a fidelity number
```

| Step                 | Why it is there                                                                                                |
| -------------------- | -------------------------------------------------------------------------------------------------------------- |
| `02a` → `02c`        | Find the resonator, then park the flux where the qubit (and resonator) should idle.                            |
| `02b`                | Map the resonator vs readout power to stay below bifurcation. Gives 08b a safe starting range.                 |
| `03a` → `03b`        | Find the qubit, then map frequency vs flux. Later nodes (including cryoscope **17c**) consume this conversion. |
| `04a` → `04b`        | Chevron sets a usable drive frequency and a rough amplitude; power Rabi sets the π pulse.                      |
| `08b` → `08a` → `07` | Only after you can flip the qubit is readout worth optimizing. IQ blobs turn on state discrimination.          |
| `09a`                | Ramsey vs flux refines the idle point and the df/dΦ used by flux pulses.                                       |
| error-amp `04b`      | Tight π and π/2 amplitudes now that discrimination is on.                                                      |
| `06a` → `05` → `06b` | T2\*, T1, T2-echo. Not used to set pulses; they tell you whether the qubit is worth gating.                    |
| `10b` → `11a`        | DRAG kills phase error on the drive; RB is the bring-up figure of merit.                                       |

### Fixed-frequency transmon

Same chain with the flux nodes dropped: no `02c`, `03b`, or `09a`. After IQ blobs it goes straight to error-amplified Rabi.

```text
  02a → 02b → 03a → 04a → 04b → 08b → 08a → 07
       → 04b x180 → 04b x90 → 06a → 05 → 06b → 10b → 11a
```

### Retuning

Once bring-up has succeeded, day-to-day drift does not need spectroscopy again. Re-run readout discrimination, re-find the idle point, tighten the pulse amplitudes, and re-measure RB:

- **Flux-tunable:** `07` → `09a` → `04b` error-amp x180 → `04b` error-amp x90 → `11a`
- **Fixed-frequency:** `07` → `06a` → `04b` error-amp x180 → `04b` error-amp x90 → `11a`

Go back to spectroscopy only if the qubit is genuinely lost.

---

# After bring-up

Needed before a CZ. For the full two-qubit calibration pipeline see the [CZ calibration README](../CZ_calibrations/README.md).

| Stage                     | Nodes                                                                                                                                                         | Why                                                                                                                                                                        |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GEF (three-state) readout | [`12`](./12_Qubit_Spectroscopy_E_to_F.py) → [`13`](./13_power_rabi_ef.py) → [`14`](./14_gef_readout_frequency_optimization.py) → [`15`](./15_iq_blobs_gef.py) | CZ leakage nodes measure ǀf⟩ population. Order matters: 12 finds the EF frequency so 13 can drive it, 14 re-optimizes readout at three states, 15 rebuilds the classifier. |
| Timing                    | [`16a`](./16a_xyz_delay.py), [`16b`](./16b_xy_coupler_z_delay.py)                                                                                             | Align XY with qubit / coupler flux pulses.                                                                                                                                 |
| Flux-line distortions     | `17a` / `17b` → `17c`                                                                                                                                         | Predistort the flux line so downstream CZ amplitudes are calibrated against a stable target. See the [visual guide](#flux-line-distortions-17a--17b--17c) below.           |
| Always-on ZZ              | [`19`](./19_zz_off_jazz.py)                                                                                                                                   | Park at the zero-ZZ bias. This keeps single-qubit RB clean and prevents always-on ZZ from biasing the CZ conditional phase.                                                |

Optional checks: [`20`](./20_all_xy.py) (AllXY), [`11b`](./11b_single_qubit_randomized_benchmarking_interleaved.py) (interleaved RB).

Tunable-coupler hardware additionally has coupler spectroscopy ([`02d`](./02d_resonator_spectroscopy_vs_coupler_flux.py), [`03c`](./03c_qubit_spectroscopy_vs_coupler_flux.py), [`09b`](./09b_ramsey_vs_coupler_flux.py), [`22a`](./22a_three_tone_coupler_spectroscopy_flux_pulse.py) / [`22b`](./22b_three_tone_coupler_spectroscopy_vs_coupler_flux.py)) to set a `decouple_offset` before the CZ bootstrap.

---

# Flux-line distortions (17a / 17b / 17c)

Fitting flux-line distortions is more involved than the rest of bring-up — multiple timescales, two probe methods, and a GUI fit before the filters are committed — so this section is a visual guide to these calibrations.

Room-temperature electronics and the cryostat wiring distort the flux pulse, so the waveform the qubit sees is not the one you programmed. Predistortion filters must be fitted **before** any two-qubit tuning, otherwise every amplitude you calibrate downstream is calibrated against a moving target. On a CZ pair this is the **moving-qubit** flux line; the same nodes apply to any flux-tunable qubit.

| Timescale                | Node                                                                                                         | Method                                    |
| ------------------------ | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------- |
| Long (ns → tens of µs)   | [`17a`](./17a_qubit_flux_long_distortion_qubitspec.py) / [`17b`](./17b_qubit_flux_long_distortion_ramsey.py) | Qubit spectroscopy / Ramsey vs flux delay |
| Short (~1 ns resolution) | [`17c`](./17c_qubit_flux_short_distortion.py)                                                                | Cryoscope, optional FIR                   |

## Long-timescale distortions (17a / 17b)

Detune the qubit with a flux pulse and probe its frequency with a delayed microwave pulse. Reconstruct pulse amplitude vs. time and fit exponential filters.

**Ref:** Hellings et al., _arXiv_ (2025), _Calibrating Magnetic Flux Control in Superconducting Circuits by Compensating Distortions on Time Scales from Nanoseconds up to Tens of Microseconds_

<p align="center">
   <img src="../.img/long_distortions_method.png" width="420" alt="Method diagram">
</p>

<p align="center">
   <img src="../.img/long_distortions_fit.png" width="800" alt="Fit result">
</p>

## Cryoscope (17c)

Sweep square-pulse duration inside a Ramsey sequence to reconstruct the pulse shape at ~1 ns resolution and fit short-timescale corrections. Note that 17c consumes the frequency-to-flux conversion from **09a**, referenced by run ID rather than retyped.

**Ref:** Rol et al., _Appl. Phys. Lett._ (2019), _Time-domain Characterization and Correction of On-chip Distortion of Control Pulses in a Quantum Processor_

<p align="center">
   <img src="../.img/cryoscope_fit.png" width="800" alt="Cryoscope fit">
</p>

## GUI fitting (17a / 17b / 17c)

Acquire with `update_state=False`, reload by `load_data_id`, tune fit parameters in the GUI, then set `update_state_from_GUI=True` and re-run to commit filters to QUAM.

<p align="center">
   <img src="../.img/cs_fit_operation.png" width="420" alt="GUI operation">
</p>

**Done when:** the reconstructed step response is flat to within your target over the relevant pulse duration, and the fitted filters are committed to QUAM.

---

# Orchestrated graphs

These graphs run the chains above end to end in Qualibrate. Each is a single script you can execute once the QUAM machine is set up.

| Graph                    | Script                                                              | Chain                                                                                                      |
| ------------------------ | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Flux-tunable bring-up    | [`80`](./80_calibration_graph_bringup_flux_tunable_transmon.py)     | 02a → 02c → 03a → 03b → 04a → 04b → 08b → 08a → 07 → 09a → 04b x180 → 04b x90 → 06a → 05 → 06b → 10b → 11a |
| Flux-tunable retuning    | [`81`](./81_calibration_graph_retuning_flux_tunable_transmon.py)    | 07 → 09a → 04b x180 → 04b x90 → 11a                                                                        |
| Fixed-frequency bring-up | [`90`](./90_calibration_graph_bringup_fixed_frequency_transmon.py)  | 02a → 03a → 04a → 04b → 08b → 08a → 07 → 04b x180 → 04b x90 → 06a → 05 → 06b → 10b → 11a                   |
| Fixed-frequency retuning | [`91`](./91_calibration_graph_retuning_fixed_frequency_transmon.py) | 07 → 06a → 04b x180 → 04b x90 → 11a                                                                        |

The retuning graphs use tighter error-amplification sweeps (`amp_factor` 0.98–1.02, step 0.002, 200 pulses) and higher RB statistics (`num_random_sequences=500`) than bring-up, since you are refining rather than finding from scratch.

---

# Bring-up is complete when

- Single-qubit RB (`11a`) returns a fidelity you are happy with
- T1 and T2 are consistent with what the qubit should give
- State discrimination is on and the confusion matrix is clean
- (Flux-tunable) the idle point is set and the frequency-to-flux conversion is committed

**Next step:** for two-qubit gates, complete the "After bring-up" prerequisites above (GEF, timing, flux distortions, ZZ), then proceed to the [CZ calibration pipeline](../CZ_calibrations/README.md).
