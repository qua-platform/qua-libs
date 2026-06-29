# **Controlled-Z (CZ) gate on flux-tunable transmons: physics & calibration**

This folder contains routines for the **flux-activated CZ gate** on flux-tunable transmons. The gate uses the **|11⟩ ↔ |20⟩** (state convention: |high_freq_qubit, low_freq_qubit⟩) avoided crossing; a baseband flux pulse on the moving qubit brings the pair into the interaction region.

Hardware falls into two workflows:

| Architecture        | Coupler                                        | Two-qubit entry                                            |
| ------------------- | ---------------------------------------------- | ---------------------------------------------------------- |
| **Fixed coupler**   | Bias fixed (no coupler flux sweep in CZ chain) | Chevron (**31**)                                           |
| **Tunable coupler** | Flux-tunable coupler + qubit Z flux            | Flux bootstrap (**30**); **31** optional (usually skipped) |

Distortion calibration (**17**, **18**) applies to **both** (moving-qubit flux line).

---

## Table of Contents

1. [Physics of the CZ gate (11–20 interaction)](#physics-of-the-cz-gate-based-on-11-02-interaction)
2. [Calibration workflows](#calibration-workflows)
   - [Shared: flux-line distortions](#shared-flux-line-distortions)
   - [Fixed-coupler workflow](#fixed-coupler-workflow)
   - [Tunable-coupler workflow](#tunable-coupler-workflow)
3. [Node reference](#node-reference)
4. [Project structure](#project-structure)
5. [Orchestrated graph (fixed coupler)](#orchestrated-graph-fixed-coupler)
6. [References](#references)

---

# Physics of the CZ gate

The CZ is a two-qubit entangling gate that applies a $\pi$ phase to
the $|11\rangle$ state and leaves the rest of the computational basis untouched:

$$U_\mathrm{CZ} = \mathrm{diag}(1, 1, 1, -1).$$

Equivalently, it imprints a Z on the target _conditioned_ on the control
being $|1\rangle$. Combined with single-qubit rotations it is universal, and on
flux-tunable transmons it is one of the native, highest-fidelity two-qubit
gates.

With the qubits coupled by an exchange interaction $J$, the Hamiltonian in
the ordered basis {$|20\rangle$, $|11\rangle$, $|02\rangle$} is

$$
H^{(2)} =
\begin{pmatrix}
2\omega_H + \alpha_H & \sqrt{2}\,J & 0 \\
\sqrt{2}\,J & \omega_H + \omega_L & \sqrt{2}\,J \\
0 & \sqrt{2}\,gJ& 2\omega_L + \alpha_L
\end{pmatrix}.
$$

The structure shows that $|11\rangle$ couples to **both** double-excitation
states, $|20\rangle$ and $|02\rangle$, each through a $\sqrt{2} J$ matrix element — the $\sqrt{2}$ coming
from the 1→2 transition of the doubly-excited transmon (the bosonic $\sqrt{2}$ of
$a^{\dagger}$). The corner entry is zero: $|20\rangle$ and $|02\rangle$ do not couple directly, only
through $|11\rangle$. Compare the single-excitation block,

$$
H^{(1)} =
\begin{pmatrix}
\omega_H & J\\
J & \omega_L
\end{pmatrix},
$$

whose off-diagonal is just $J$ — the double-excitation crossing is the
stronger one, which is why it both drives the gate and sets the dominant
leakage channel.

So in the two-excitation manifold there are two candidate partners for
$|11\rangle$ (shown in the figure below):

- $|11\rangle \leftrightarrow |20\rangle$ (high-frequency qubit doubly excited)
- $|11\rangle \leftrightarrow |02\rangle$ (low-frequency qubit doubly excited)

Either realizes a CZ in principle; the choice is set by the frequency
arrangement and which qubit is fluxed. **Here we use $|11\rangle \leftrightarrow |20\rangle$.**

The single-excitation manifold {$|10\rangle, |01\rangle$} is a separate resource:
brought to resonance ($\omega_H = \omega_L$) it exchanges excitations and realizes the
**iSWAP family**, not a CZ. It is shown only for orientation.

The resonance conditions follow from the diagonals:

- $|11\rangle \leftrightarrow |20\rangle$ at $\omega_H − \omega_L = |\alpha_H|$ (CZ operating point)
- $|11\rangle \leftrightarrow |02\rangle$ at $\omega_H - \omega_L = −|\alpha_L|$ (unused here)
- $|10\rangle \leftrightarrow |01\rangle$ at $\omega_H - \omega_L = 0$ (iSWAP)

## Mechanism

A baseband flux pulse on the moving qubit sweeps the detuning $\Delta = \omega_H - \omega_L$
toward the $|11\rangle \leftrightarrow |20\rangle$ resonance at $\Delta = |\alpha_H|$. There are two ways to spend
the resulting interaction as a conditional phase:

- **Adiabatic.** Ramp into the avoided crossing slowly enough that $|11\rangle$
  follows the lower eigenstate without ever fully populating $|20\rangle$,
  accumulating a dynamical phase along the way. Leakage-robust, but slower.
- **Diabatic.** Pulse fast to (or near) resonance and let $|11\rangle$ undergo a
  full 2π population exchange with $|20\rangle$ — out to $|20\rangle$ and back — returning
  to $|11\rangle$ with the conditional phase banked.

**This stack uses the diabatic gate.** Over the excursion, $|11\rangle$ acquires a
phase $\zeta(t)$ relative to the single-excitation states $|01\rangle$, $|10\rangle$; the gate is
calibrated so that

$$\int \zeta(t)\, dt = (2n+1)\pi, \quad n \in \mathbb{Z},$$

i.e. an _odd_ multiple of π. The residual single-qubit phases on $|01\rangle$ and
$|10\rangle$ are removed by virtual-Z compensation (node 34), yielding the ideal
$U_\mathrm{CZ}$ above.

**Key Reference:**

- **DiCarlo et al.**, _Nature_ (2009), _Demonstration of Two-Qubit Algorithms with a Superconducting Quantum Processor_

---

# Calibration workflows

## Shared: flux-line distortions

Compensate distortion on the **moving-qubit** flux line before two-qubit tuning.

| Step            | Node   | File                                                                                     |
| --------------- | ------ | ---------------------------------------------------------------------------------------- |
| Long timescale  | **17** | [`17_pi_vs_flux_long_distortions`](../1Q_calibrations/17_pi_vs_flux_long_distortions.py) |
| Short timescale | **18** | [`18_cryoscope`](../1Q_calibrations/18_cryoscope.py)                                     |

### Qubit spectroscopy vs. flux delay (17)

Detune the qubit with a flux pulse and probe frequency with a delayed microwave pulse. Reconstruct pulse amplitude vs. time and fit exponential filters for long-timescale distortion [1].

<p align="center">
   <img src="../.img/long_distortions_method.png" width="500" alt="Method diagram">
</p>

<p align="center">
   <img src="../.img/long_distortions_fit.png" width="1000" alt="Fit result">
</p>

### Cryoscope (18)

Sweep square-pulse duration inside a Ramsey sequence to reconstruct the pulse shape at ~1 ns resolution and fit short-timescale corrections [2].

<p align="center">
   <img src="../.img/cryoscope_fit.png" width="1000" alt="Cryoscope fit">
</p>

### GUI fitting (17 / 18)

Acquire with `update_state=False`, reload by `load_data_id`, tune fit parameters in the GUI, then set `update_state_from_GUI=True` and run to commit filters to QUAM.

<p align="center">
   <img src="../.img/cs_fit_operation.png" width="500" alt="GUI operation">
</p>

---

## Fixed-coupler workflow

Coupler bias is **not** swept in the CZ calibration chain. After distortion calibration, run the chevron → conditional-phase → phase-compensation sequence. Automate with graph **99**.

```text
17 → 18  →  31 → 33a → 33b
                    └→ 34
```

| Order | Node    | Summary                                                                                       |
| ----- | ------- | --------------------------------------------------------------------------------------------- |
| 1     | **31**  | Chevron: amplitude × duration → coarse CZ duration/amplitude                                  |
| 2     | **33a** | Fine amplitude scan → π/2 conditional-phase point                                             |
| 3     | **33b** | CZ pulse train → error-amplified amplitude fine tune                                          |
| 4     | **34**  | Virtual-Z phase compensation (can run after **33a**; **99** runs it in parallel with **33b**) |

---

## Tunable-coupler workflow

Node **30** finds coupler **decouple (idle)** and **interaction** flux biases plus moving-qubit detuning in one 2D map (CZ or iSWAP). That replaces the coarse amplitude/duration role of chevron (**31**), so the usual path is **30 → 32a → 32b → 33a → 33b → 34** without **31**. Pulse duration and macro amplitudes come from **30** and the gate macro already in QUAM. Use **32b** (PALEA) instead of **32a** for improved leakage isolation.

```text
17 → 18  →  30 → 32a → 32b → 33a → 33b → 34
```

| Order | Node    | Summary                                                                             |
| ----- | ------- | ----------------------------------------------------------------------------------- |
| 1     | **30**  | 2D coupler + moving-qubit flux → `decouple_offset`, `detuning`, `macros[operation]` |
| 2     | **32a** | Coupler amplitude via \|11⟩ leakage amplification (standard)                        |
| 3     | **32b** | Coupler amplitude via PALEA leakage amplification (alternative to **32a**)          |
| 4     | **33a** | Fine amplitude → π/2 conditional-phase point                                        |
| 5     | **33b** | CZ pulse train → error-amplified amplitude fine tune                                |
| 6     | **34**  | Virtual-Z phase compensation                                                        |

**31** remains available if you still want an explicit amplitude–duration Chevron after **30** (e.g. new macro shape or duration not set in state).

Run **30** manually or in a custom graph; graph **99** is for the fixed-coupler path only. Set `operation` and `cz_or_iswap` on **30** for the gate you are calibrating.

---

# Node reference

## Flux bootstrap — tunable coupler only

[(30_cz_iswap_flux_bootstrap)](./30_cz_iswap_flux_bootstrap.py)

2D sweep of coupler flux (around `coupler.decouple_offset`) and moving-qubit flux. Prep: |11⟩ (CZ) or |10⟩ (iSWAP). Finds idle plateau (decouple) and first interaction fringe; updates `coupler.decouple_offset`, `qubit_pair.detuning`, and `macros[operation]` amplitudes.

**Goal:** Coarse coupler/qubit flux operating point; for tunable couplers this typically **replaces 31**.

---

## Chevron — fixed coupler (optional for tunable)

[(31_chevron_11_20)](./31_chevron_11_20.py)

Prepare |11⟩, sweep CZ flux pulse amplitude and duration on the moving qubit. First Chevron fringe → initial duration/amplitude. **Required** on fixed-coupler pairs; **usually skipped** after **30** on tunable-coupler pairs.

<p align="center">
   <img src="../.img/chevron.png" width="500" alt="Chevron pattern">
</p>

**Goal:** Full π phase between control states (first yellow fringe).

---

## Leakage amplification — tunable coupler only

### Standard protocol

[(32a_cz_leakage_amplification)](./32a_cz_leakage_amplification.py)

Prepare \|11⟩, sweep **coupler flux pulse amplitude**, repeat CZ `n = 1…N`, measure P(11). Optimal amplitude maximizes mean P(11) over `n`. Requires GEF readout and `macros[operation].coupler_flux_pulse`.

**Goal:** Tune `coupler_flux_pulse.amplitude` to preserve \|11⟩ under repeated CZ.

### PALEA protocol

[(32b_cz_leakage_amplification_palea)](./32b_cz_leakage_amplification_palea.py)

Same coupler-amplitude objective as **32a**, with a dynamical-decoupling layer after each CZ (EF π on the high-frequency qubit, g–e π on the low-frequency qubit). Sweeps even `n = 2, 4, …`. See Marxer et al., [arXiv:2508.16437](https://arxiv.org/abs/2508.16437).

**Goal:** Same state update as **32a** with improved leakage-error amplification.

---

## Conditional phase — both workflows

[(33a_cz_conditional_phase)](./33a_cz_conditional_phase.py)

Use gate duration from **31** (fixed coupler) or from the macro in state after **30** (tunable coupler). On tunable couplers, run after leakage calibration (**32a** or **32b**). Sweep amplitude to the **π/2 conditional-phase** point. Tomography with rotating x90 on target.

<p align="center">
   <img src="../.img/conditional_phase.png" width="500" alt="Conditional phase plot">
</p>

**Goal:** Update optimal CZ amplitude in state.

### Error amplification

[(33b_cz_conditional_phase_error_amp)](./33b_cz_conditional_phase_error_amp.py)

Train of CZ pulses for finer amplitude tuning.

<p align="center">
   <img src="../.img/phase_error_amp.png" width="500" alt="Conditional phase plot">
</p>

**Goal:** Fine-tune gate amplitude.

---

## Phase compensation — both workflows

[(34_cz_phase_compensation)](./34_cz_phase_compensation.py)

|++⟩, apply CZ, reconstruct per-qubit phase; update virtual Z in state.

<p align="center">
  <img src="../.img/individual_phases.png" width="500" alt="Individual qubit phase reconstruction">
</p>

**Goal:** Compensate single-qubit phases acquired during CZ.

---

# Project structure

| Node    | File                                                                               | Fixed coupler |                    Tunable coupler                    |
| ------- | ---------------------------------------------------------------------------------- | :-----------: | :---------------------------------------------------: |
| **30**  | [`30_cz_iswap_flux_bootstrap.py`](./30_cz_iswap_flux_bootstrap.py)                 |       —       |                           ✓                           |
| **31**  | [`31_chevron_11_20.py`](./31_chevron_11_20.py)                                     |       ✓       |                       optional                        |
| **32a** | [`32a_cz_leakage_amplification.py`](./32a_cz_leakage_amplification.py)             |       —       |                           ✓                           |
| **32b** | [`32b_cz_leakage_amplification_palea.py`](./32b_cz_leakage_amplification_palea.py) |       —       |                           ✓                           |
| **33a** | [`33a_cz_conditional_phase.py`](./33a_cz_conditional_phase.py)                     |       ✓       |                           ✓                           |
| **33b** | [`33b_cz_conditional_phase_error_amp.py`](./33b_cz_conditional_phase_error_amp.py) |       ✓       |                           ✓                           |
| **34**  | [`34_cz_phase_compensation.py`](./34_cz_phase_compensation.py)                     |       ✓       |                           ✓                           |
| **99**  | [`99_CZ_calibration_graph.py`](./99_CZ_calibration_graph.py)                       | ✓ (31–33b–34) | — (use **30** → 32a–32b–33a/b–34 by hand or custom graph) |

Utilities: `cz_iswap_flux_bootstrap`, `chevron_cz`, `cz_conditional_phase`, `cz_conditional_phase_error_amp`, `cz_leakage_amp`, `cz_phase_compensation` under `../calibration_utils/`.

---

# Orchestrated graph (fixed coupler)

[`99_CZ_calibration_graph.py`](./99_CZ_calibration_graph.py) — `CZ_Calibration_Fixed_Couplers`:

- **31** → **33a** → **33b** → **34**

Leakage nodes (**32a** / **32b**) are tunable-coupler only and are not included in this graph.

---

# References

[1] Christoph Hellings et al., _arXiv_ (2025), _Calibrating Magnetic Flux Control in Superconducting Circuits by Compensating Distortions on Time Scales from Nanoseconds up to Tens of Microseconds_

[2] Rol et al., _Appl. Phys. Lett._ (2019), _Time-domain Characterization and Correction of On-chip Distortion of Control Pulses in a Quantum Processor_
