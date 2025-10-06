# Fixed-Frequency Two-Qubit Gates: **CZ Gate Physics & Calibration Overview**

This folder contains routines for implementing and calibrating the **flux-activated CZ gate** on **fixed-frequency transmons**.

The gate relies on the precise activation of the non-computational state **|11⟩ ↔ |02⟩ avoided-crossing interaction** to accumulate a conditional phase. This protocol relies on precise control of the qubit frequency through a short baseband flux pulse on the higher frequency qubit flux line.

---

## Table of Contents
1. [Physics of the CZ based on 11–02 Interaction](#physics-of-the-cz-based-on-11-02-interaction)
2. [Calibration Procedure](#calibration-procedure)
   - [Distortions: Cryoscope and Spectroscopy](#distortions-cryoscope-and-spectroscopy)
   - [Finding Initial Parameters – Amplitude and Duration Sweep](#finding-initial-parameters--amplitude-and-duration-sweep)
   - [Amplitude Sweep for 90° Phase Point](#amplitude-sweep-for-90-phase-point)
   - [Single-Qubit Phase Compensation – Virtual Z Rotations](#singlequbit-phase-compensation--virtual-z-rotations)
3. [Project Structure](#project-structure)
4. [References](#references)

---

# Physics of the CZ based on 11–02 Interaction

The **controlled-Z (CZ) gate** for fixed-frequency superconducting qubits operates via the *|11⟩ ↔ |02⟩* avoided crossing between two transmons coupled with exchange rate **J**.

- **Mechanism**

  The CZ gate is realized by pulsing the qubit frequencies to an avoided crossing at a specific operating point (Point II).

  <img src="../.img/spectrum_cz.png" width="500" alt="Alt text">

  At this point II, a useful two-qubit interaction is revealed in the two-excitation spectrum. The interaction involves a large cavity-mediated avoided crossing between the computational state |11⟩ and the non-computational higher-level transmon excitation |02⟩.

  This avoided crossing causes a frequency shift, $\zeta/2\pi$, in the transition frequency of the |11⟩ state. A CZ is specifically implemented by selecting a voltage pulse $V_R$ into Point II such that the time integral of the frequency shift satisfies $\int \zeta(t) dt = (2n+1)\pi$ (where $n$ is an integer). The $\zeta(t)$ frequency shift is directly mediated by the waveform amplitude, shape and duration.

- **Gate Condition**

  A **π phase accumulation** on the |11⟩ state realizes an ideal CZ:
  $$
  U_\mathrm{CZ} = \mathrm{diag}(1, 1, 1, -1).
  $$

- **Key References**
  - **DiCarlo et al.**, *Nature* (2009) – first demonstration of CZ via 11–02 transition.

---

# Calibration Procedure

The calibration sequence ensures accurate compensation of intrinsic distortions caused by non-ideal cabling and components, precise conditional phase calibration and single qubit virtual phase compensations.

---

## Distortions: Cryoscope and Spectroscopy

To achieve high-fidelity operation, flux lines tuning the qubit frequency must be characterized and corrected for distortion. Two methods are used to characterize long and short time scale distortions. The fitting parameters for the exponential fit time start fractions can be tuned interactively by loading the dataset via it's id and commiting the changes with the update state from GUI flag.

- **Spectroscopy vs flux delay**

  This method proposed in [1] consist of detuning the qubit via a flux pulse and probing it's frequency via a short microwave pulse. By sweeping the delay between the two pulses (t) one can reconstruct the pulse amplitude time evolution by tracking the qubit frequency.

  <img src="../.img/long_distortions_method.png" width="500" alt="Alt text">

  We can then fit a set of exponential filter corrections to compensate for the long time scale distortions.

  <img src="../.img/long_distortions_fit.png" width="500" alt="Alt text">

- **Cryoscope Calibration**

    This  method introduced in [2] consists of sweeping the duration of a square flux pulse between a fixed time Ramsey sequence. The Ramsey sequence allow to measure the detuning of the qubit for each pulse duration thus allowing the reconstruction on the pulse shape with 1ns resolution. We can then use this information to fit to a second set of short term exponentials to compensate for short time scales distortions.

    <img src="../.img/cryoscope_fit.png" width="500" alt="Alt text">

**References:**

[1] Christoph Hellings et al., *arXiv* (2025) *Calibrating Magnetic Flux Control in Superconducting Circuits by Compensating Distortions on Time Scales from Nanoseconds up to Tens of Microseconds*

[2] Rol et al., *Appl. Phys. Lett.* (2019)

---

## Finding Initial Parameters – Amplitude and Duration Sweep [(19_chevron_11-02)](./19_chevron_11-02.py)

The first calibration stage identifies coarse operating points using a **Chevron-pattern** experiment. The qubit pair is initiallised to the |11> state and then a flux pulse is applied to the high frequency qubit to activate the interaction with the |02> state. The amplitude and duration on this pulse are swept a Chevron like pattern is observed. The center of the first period of the pattern is saved as a first initial parameter for the CZ gate

<img src="../.img/chevron.png" width="500" alt="Alt text">

**Goal:** Find the amplitude–duration pair that produces a full π phase shift between control states (the first yellow fringe).

---

## Amplitude Sweep for 90° Phase Point [(20_cz_conditional_phase)](./20_cz_conditional_phase.py)

With the optimal duration fixed, perform a fine amplitude scan to locate the **90° conditional-phase point (π/2)**.

- Defines a linear regime where conditional phase grows proportionally with drive amplitude.
- Serves as reference for subsequent parameter optimizations (frequency, phase, and Z-corrections).

**Outcome:** Identifies the most stable working point minimizing leakage while preserving entangling rate.

---

## Single-Qubit Phase Compensation – Virtual Z Rotations [(21_cz_phase_compensation)](./21_cz_phase_compensation.py)

Microwave drives induce unwanted **single-qubit phase rotations (ZI, IZ)** via AC Stark shifts.
These must be compensated to achieve a pure entangling operation.

- Apply **virtual Z corrections** after the CZ pulse by adjusting the rotating frame phase of each qubit.
- Sweep the correction phase on each qubit independently while monitoring output populations.
- The optimal correction angle nulls the residual rotation, aligning both Bloch vectors.

**Result:** The final gate performs as
\[
U_\mathrm{CZ} = e^{-i(\pi/4)(ZZ - ZI - IZ)},
\]
realizing an ideal controlled-Z.

---
