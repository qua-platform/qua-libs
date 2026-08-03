# Ramsey vs Flux Calibration

[`09a_ramsey_vs_flux_calibration.py`](../../../../../calibrations/1Q_calibrations/09a_ramsey_vs_flux_calibration.py) · **Targets:** qubits · **Category:** 1Q_calibrations

Repeats a virtual-Z Ramsey sequence at many small flux-pulse amplitudes to precisely map the qubit's transition frequency vs. a *pulsed* flux excursion around its current operating point, refining the idle-flux offset and extracting the local frequency-flux curvature (`freq_vs_flux_01_quad_term`).

## Purpose

`03b_qubit_spectroscopy_vs_flux` locates the sweet spot coarsely, by sweeping a *static* DC flux offset over a wide span (default 0.05 V) and re-running spectroscopy at each point — precision there is limited by the spectroscopy line's power-broadened width. This node is the precise, local companion: it sweeps a much narrower flux range (default 0.01 V total) and, at each point, replaces the spectroscopy measurement with a Ramsey sequence — an interferometric technique whose frequency precision is set by $T_2^*$ and the idle-time span rather than a line width, and which is a prerequisite step in the standard flux-tunable-transmon calibration graph precisely because of that precision gain.

Two other differences matter as much as "which technique is used":

- **The flux here is a pulse, not a DC offset.** The swept flux level is only applied *during the Ramsey idle window* (see Mechanism) — a short excursion around whatever static/DC point `03b` (or the current QUAM state) has already set. `03b` characterizes the DC dispersion relation; this node characterizes the qubit's response to a *pulsed* flux step of the kind used by pulsed-flux control, which is exactly what the downstream pulsed-flux-distortion nodes (`17_pi_vs_flux_long_distortions`, `18_cryoscope`) need.
- **It fits a local parabola, not a cosine.** Because the swept range is small, the flux dispersion $f_{01}(\Phi)$ **[Koc+2007]**, **[Kra+2019]** is well-approximated by its second-order Taylor expansion around the current point. Fitting a degree-2 polynomial to frequency vs. flux yields both the vertex (a refined estimate of the sweet spot, the flux-noise-insensitive point first exploited on the quantronium **[Vio+2002]**) and the curvature itself — the quantity later nodes need to convert a target detuning into a flux-pulse amplitude ($\Delta f \propto \text{quad\_term}\cdot V^2$).

The Ramsey sequence itself uses **virtual Z rotations** rather than a physically detuned drive: the second $\pi/2$ pulse's frame is rotated by a phase proportional to the idle time, mimicking an artificial detuning while keeping every played pulse resonant. An artificial detuning is not optional here — a real detuning near zero produces a trace indistinguishable from pure $T_2$ decay, so `frequency_detuning_in_mhz` intentionally offsets the observed oscillation away from that degenerate case **[GRTW2021]**.

## Mechanism

For each (flux, idle-time) point in the 2D sweep, for every qubit in a batch:

1. `node.machine.initialize_qpu` sets each qubit's static/DC flux point once per batch; a single `qubit.readout_state` call (before the averaging loop) seeds a per-qubit `init_state` variable.
2. Inside the averaging loop, for each swept flux level and idle time `t`, within a `strict_timing_()` block:
   - Play `x90` on `qubit.xy`, then apply `frame_rotation_2pi(phi)` where `phi` is computed from `frequency_detuning_in_mhz` and `t` (the virtual-Z artificial detuning).
   - `qubit.xy` waits `t + 1` cycles while, concurrently, `qubit.z` waits out the `x90` pulse length and then plays the `"const"` operation for duration `t`, with `amplitude_scale = flux / qubit.z.operations["const"].amplitude` — i.e. a flux **pulse**, confined to the idle window, at the current swept level.
   - Play the second `x90`.
3. Measure via `qubit.readout_state` (see callout below), XOR the result against the previous shot's post-measurement state, save that XOR as `state`, and carry the new state forward as the next shot's `init_state`. Reset the `xy` frame afterward so virtual rotations don't accumulate across shots.

> **No reset is ever issued in this node's QUA program.** There is no `qubit.reset_qubit_thermal()`/`qubit.reset()` call anywhere in `create_qua_program` — consecutive shots free-run, chained together via the init/current-state XOR scheme above instead of an explicit inter-shot reset. The declared `reset_type` common parameter therefore has **no effect** in this node, in either its `"thermal"`, `"active"`, or `"active_gef"` setting.

> **`use_state_discrimination` also has no effect.** The sequence unconditionally calls `qubit.readout_state` (discriminated single-shot 0/1 state) — there is no raw-I/Q branch. Only a `state` stream is saved; no `I`/`Q` streams exist in `ds_raw` at all. A calibrated IQ-blob threshold (`07_iq_blobs`) is therefore a hard prerequisite, not an optional SNR boost — consistent with this node following `IQ_blobs` directly in the bring-up graph.

Analysis (`fit_raw_data` in `calibration_utils/ramsey_versus_flux_calibration/analysis.py`; `process_raw_dataset` is a no-op pass-through):

1. Fit each (qubit, flux) trace over `idle_times` to a fixed-exponent decaying cosine $a\,e^{-t\cdot\text{decay}}\cos(2\pi f t+\phi)+\text{offset}$ (`fit_oscillation_decay_exp`/`oscillation_decay_exp`), extracting the oscillation frequency `f` and decay rate (giving $T_2^*=1/\text{decay}$).
2. Discard any (qubit, flux) point whose fitted `f` came out negative (`frequency.where(frequency > 0, drop=True)`).
3. Fit the surviving `f` vs. `flux_bias`, per qubit, to a degree-2 polynomial — the local-parabola approximation described in Purpose.
4. From the parabola's coefficients, compute the vertex flux (`flux_offset`), the parabola's value there relative to the artificial detuning (`freq_offset`), and the curvature (`quad_term`, which becomes `freq_vs_flux_01_quad_term`).

> **The fit never actually fails.** `FitParameters.success` is hardcoded to `True` for every qubit in `fit_raw_data` — there is no NaN check, no minimum-valid-points check, nothing. `_extract_relevant_fit_parameters` (present in this module, mirroring the pattern in `03b`) is an unused no-op (`pass`), and `log_fitted_results` is also a no-op. `node.outcomes` will read `"successful"` for every qubit unless step 2/3 above throws an exception (e.g. too few surviving flux points for a degree-2 fit) — see Troubleshooting.

## Prerequisites

- Mixer/Octave calibration (`01a_mixer_calibration`).
- Time of flight calibrated (`01a_time_of_flight` or `01b_time_of_flight_mw_fem`).
- Readout calibrated (`02a_resonator_spectroscopy`, and/or `02b`/`02c`).
- A rough qubit frequency and sweet spot already found (`03a_qubit_spectroscopy`, `03b_qubit_spectroscopy_vs_flux`) — this node refines locally, it does not search for the sweet spot from scratch.
- `x90` (and, indirectly, `x180`) pulse amplitude calibrated (`04a_rabi_chevron`, `04b_power_rabi`).
- **`07_iq_blobs` calibrated** — required, not optional, because readout is unconditionally discriminated (see Mechanism callout).
- `qubit.z.flux_point` already set to `"independent"` or `"joint"` — determines which state attribute gets updated (see State Updates), and any other value crashes `update_state`.

## Parameters

Inherits the common parameter set — see [Common node parameters](_common_parameters.md) — plus the node-specific parameters below.

| Parameter | Type | Default | Unit | Description | Effect |
|---|---|---|---|---|---|
| `num_shots` | `int` | 100 | – | Averages per (flux, idle-time) point. | More shots lower noise on the extracted oscillation at each point; linear cost in run time. |
| `frequency_detuning_in_mhz` | `float` | 1.0 | MHz | Artificial detuning applied via the virtual-Z rotation of the second `x90`. | Must be large enough that the resulting oscillation is distinguishable from pure decay **[GRTW2021]**, but well under the Nyquist limit set by `wait_time_step_in_ns` (≈ $1/(2\times\text{step})$) or the fit aliases. |
| `min_wait_time_in_ns` | `int` | 16 | ns | Shortest Ramsey idle time. | Rounded down to a multiple of 4 ns (clock cycles) internally. |
| `max_wait_time_in_ns` | `int` | 5000 | ns | Longest Ramsey idle time. | Longer idle times resolve $T_2^*$ better but cost more if the qubit has already dephased — diminishing returns past a few $T_2^*$. |
| `wait_time_step_in_ns` | `int` | 60 | ns | Idle-time step size. | Sets the Nyquist limit on resolvable detuning (`frequency_detuning_in_mhz` must stay well under it); one fixed sweep is reused at every flux point (no adaptive coarse/fine pass). |
| `flux_span` | `float` | 0.01 | V | Full symmetric span of the **pulsed** flux sweep, centered on the qubit's current static flux point. | Deliberately narrow — this is a local-curvature probe, not a sweet-spot search; too wide a span breaks the local-parabola approximation used in the fit. |
| `flux_num` | `int` | 21 | – | Number of points across `flux_span`. | More points resolve the curvature better; too few relative to how much the frequency actually curves over the span risks a poorly-conditioned parabola fit. |

> **`reset_type` and `use_state_discrimination` are declared but silently ignored by this node** — see the Mechanism callouts above. Setting either has no observable effect on this node's own behavior; they only matter because they're inherited fields on `Parameters`.

## Outputs

**Measured:** `state` only (discriminated 0/1, XOR of consecutive shots) — no raw `I`/`Q` is ever collected by this node, regardless of `use_state_discrimination`.

| Fitted quantity | Unit | Written to state? | Description |
|---|---|---|---|
| `flux_offset` | V | ✅ (as an increment) | Vertex of the fitted local parabola — the flux correction toward the local extremum. |
| `freq_offset` | Hz (combined with the configured detuning in `update_state`) | ✅ (as an increment) | The configured artificial detuning minus the parabola's fitted value at its vertex — the residual physical frequency correction. |
| `quad_term` | Hz/V² | ✅ (`freq_vs_flux_01_quad_term`, replaced) | Local curvature of $f_{01}$ vs. the swept flux pulse; consumed by `18_cryoscope` and `17_pi_vs_flux_long_distortions` to convert a target detuning into a flux-pulse amplitude via $V=\sqrt{-\Delta f/\text{quad\_term}}$. |
| `t2_star` | ns | – | $1/\text{decay}$ from the fixed-exponent (`n=1`) fit, per (qubit, flux) point. |
| `success` | bool | – | Hardcoded `True` — not a real signal (see Mechanism callout). |

> **Dataset unit labels are unreliable for `t2_star` and the raw `frequency` fit variable.** The `idle_times` sweep axis is registered in ns, and the decay/frequency fit operates directly on those ns values, so the raw numeric `t2_star` is in **ns**, and the raw fitted `f` is numerically in cycles/ns (≡ GHz) — but their `xarray` attrs claim `"uSec"` and `"MHz"` respectively (a mislabeling; the plotting code separately applies the correct ×1000 conversion where it matters for display, and the state-update formulas apply their own correct, self-consistent conversions — see source). Trust the numeric value and this doc, not the attrs, when inspecting `ds_fit` directly.

**Success criterion:** none, effectively — `success` is unconditionally `True` for every qubit that survives the fit without an exception (see Troubleshooting #1–2).

## State Updates

Applied only when `node.outcomes[qubit] != "failed"` (which, per the Mechanism callout, is essentially always, absent an exception during fitting):

| Attribute | New value | Mode | Condition |
|---|---|---|---|
| `qubit.z.independent_offset` | `+ flux_offset` | **increment (`+=`)** | outcome successful **and** `qubit.z.flux_point == "independent"` |
| `qubit.z.joint_offset` | `+ flux_offset` | **increment (`+=`)** | outcome successful **and** `qubit.z.flux_point == "joint"` |
| `qubit.f_01` | `+ freq_offset` | increment (`+=`) | outcome successful |
| `qubit.xy.RF_frequency` | `+ freq_offset` | increment (`+=`) | outcome successful |
| `qubit.freq_vs_flux_01_quad_term` | fitted `quad_term` | **replace** | outcome successful |

Any other value of `qubit.z.flux_point` raises `RuntimeError("Unknown flux_point")`, aborting the rest of `update_state` for the qubits not yet processed in that call.

> **Unlike `03b`, there is no replace-vs-increment asymmetry between `"independent"` and `"joint"` here — both branches increment.** `qubit.f_01` and `qubit.xy.RF_frequency` are also incremented, not replaced. A well-converged qubit should see these increments shrink toward zero on repeated runs; if they don't, see Troubleshooting #5.

## Troubleshooting

See also: [General troubleshooting](_general_troubleshooting.md#general-troubleshooting) for issues common to most nodes in this library. Below are issues specific to this node.

1. **Every qubit reports `"successful"`, even ones whose parabola plot clearly looks wrong** → `success` is hardcoded `True` in `fit_raw_data`; there is no automated NaN/quality guard in this node (`_extract_relevant_fit_parameters` is an unused stub). Always visually check `plot_parabolas_with_fit`'s dashed lines against a believable vertex before trusting an unattended re-run; use `load_data_id` to re-inspect a suspicious run's `ds_fit` without re-measuring.
2. **`flux_offset`/`freq_offset`/`quad_term` come out `NaN`, or the node throws during the parabola fit** → `frequency.where(frequency > 0, drop=True)` silently discards any (qubit, flux) point whose fitted oscillation frequency came out negative (common at low SNR or when the idle-time span doesn't cover a full period at that flux point); a degree-2 polyfit needs at least 3 surviving points per qubit. Inspect `ds_fit.fit_results.sel(fit_vals="f")` across `flux_bias` for the affected qubit (see Parameter Tuning Heuristics #1 for the remedy once too many points are negative).
3. **`t2_star` looks off by roughly 1000× from what `06a_ramsey` reports for the same qubit** → this is very likely the units mislabeling noted in Outputs, not a real coherence change: the raw numeric `t2_star` is in ns despite the dataset attrs claiming `"uSec"`. Compare the raw number, not the attrs-implied unit.
4. **Results look fine on the parabola plot but state updates still seem to write garbage** → because `success` is always `True`, nothing stops a bad fit from being written to state. Cross-check `node.results["fit_results"]` numerically (not just the figure) before trusting an automated pipeline that chains straight into `update_state`.
5. **Re-running this node on the same qubit doesn't converge — offsets and frequencies keep drifting rather than settling** → unlike `03b`, *both* `independent_offset` and `joint_offset` are incremented here (never replaced), along with `f_01`/`RF_frequency`. A single good run should push all four increments toward zero on the next run. If they don't shrink, suspect flux crosstalk from other qubits when `multiplexed=True` (compensate per **[Dai+2021]**'s flux-crosstalk framework), a polarity mismatch between the swept `flux` sign and `qubit.z`'s configured wiring, or that the operating point has moved enough (e.g. after a `03b` re-run) that `flux_span` no longer straddles it.
6. **`qubit.z.flux_point` is set to something other than `"independent"`/`"joint"`** → `update_state` raises `RuntimeError("Unknown flux_point")` mid-loop; qubits already processed earlier in that call may already have staged state changes. Fix `flux_point` before re-running rather than assuming a clean rollback.
7. **Setting `reset_type="active"` (or any other value) doesn't change anything about SNR or run time** → expected; this node's QUA program never calls a reset at all (see Mechanism callout) — it relies entirely on the init/current-state XOR scheme instead of thermal or active reset between shots.

## Parameter Tuning Heuristics

See also: [General parameter tuning heuristics](_general_troubleshooting.md#general-parameter-tuning-heuristics) for guidance common to most nodes in this library. Below is guidance specific to this node.

1. **Too many (qubit, flux) points came out with a negative fitted frequency (see Troubleshooting #2), leaving `flux_offset`/`freq_offset`/`quad_term` as `NaN` or too few points for the degree-2 fit** → raise `num_shots` or reduce `flux_span`.
2. **Oscillation looks like pure exponential decay with no visible fringes** → per **[GRTW2021]**, a near-zero real detuning is indistinguishable from pure $T_2$ decay in a Ramsey trace. `frequency_detuning_in_mhz` (default 1.0 MHz) supplies the needed artificial detuning via the virtual-Z rotation; if fringes are still invisible, increase it — but keep it well under the Nyquist limit set by `wait_time_step_in_ns`.
3. **Fitted frequency vs. flux jumps erratically between adjacent flux points instead of tracing a smooth curve** → the same fixed idle-time sweep (`min_wait_time_in_ns`–`max_wait_time_in_ns` @ `wait_time_step_in_ns`) is reused at every flux point; if the real detuning swings past the Nyquist limit ($\approx 1/(2\times\text{wait\_time\_step\_in\_ns})$) across `flux_span`, individual points alias to the wrong frequency **[GRTW2021]**. Narrow `flux_span` first (it's already small by design); if that's not an option, run a short coarse pass (small `max_wait_time_in_ns`, coarse `wait_time_step_in_ns`) to locate the approximate detuning range before a longer fine pass.

## Next Steps

`04b_power_rabi` (as `power_rabi_error_amplification_x180`, then `_x90`) — both the bring-up graph (`80_calibration_graph_bringup_flux_tunable_transmon.py`) and the retuning graph (`81_calibration_graph_retuning_flux_tunable_transmon.py`) run this node immediately after `07_iq_blobs` and feed it directly into the power-Rabi error-amplification pair that follows.

## References

**[Koc+2007]** J. Koch, T. M. Yu, J. Gambetta, A. A. Houck, D. I. Schuster, J. Majer, A. Blais, M. H. Devoret, S. M. Girvin, and R. J. Schoelkopf, "Charge-insensitive qubit design derived from the Cooper pair box," *Phys. Rev. A*, vol. 76, no. 4, p. 042319, 2007.

**[Kra+2019]** P. Krantz, M. Kjaergaard, F. Yan, T. P. Orlando, S. Gustavsson, and W. D. Oliver, "A quantum engineer's guide to superconducting qubits," *Appl. Phys. Rev.*, vol. 6, no. 2, p. 021318, 2019.

**[Vio+2002]** D. Vion, A. Aassime, A. Cottet, P. Joyez, H. Pothier, C. Urbina, D. Esteve, and M. H. Devoret, "Manipulating the quantum state of an electrical circuit," *Science*, vol. 296, no. 5569, pp. 886–889, 2002.

**[GRTW2021]** Y. Y. Gao, M. A. Rol, S. Touzard, and C. Wang, "Practical guide for building superconducting quantum devices," *PRX Quantum*, vol. 2, p. 040202, 2021.

**[Dai+2021]** X. Dai, D. M. Tennant, R. Trappen, A. J. Martinez, D. Melanson, M. A. Yurtalan, Y. Tang, S. Novikov, J. A. Grover, S. M. Disseler, J. I. Basham, R. Das, D. K. Kim, A. J. Melville, B. M. Niedzielski, S. J. Weber, J. L. Yoder, D. A. Lidar, and A. Lupascu, "Calibration of flux crosstalk in large-scale flux-tunable superconducting quantum circuits," *PRX Quantum*, vol. 2, p. 040313, 2021.
