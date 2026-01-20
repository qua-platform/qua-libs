# Rotating Frame Device Demos

This folder contains demonstration scripts for the `SuperconductingDeviceRot` class, which implements a rotating frame transformation and RWA Hamiltonian for transmon qubits with tunable couplers.

## Files

### `rabi_chevron.py`

Demonstrates Rabi oscillations by sweeping drive frequency and pulse duration, producing a characteristic chevron pattern.

**What it does:**
- Creates a 2-qubit device with 1 coupler using `SuperconductingDeviceRot`
- Sweeps drive frequency around the idling frequency (±50 MHz)
- Sweeps pulse duration (10-200 ns)
- Uses Gaussian pulse envelopes
- Computes z-projection (expectation value of σ_z) after each pulse
- Plots results as a 2D heatmap showing the chevron pattern

**Usage:**
```bash
python rabi_chevron.py
```

**Output:**
- Saves a 2D heatmap plot to `data/rabi_chevron.png`
- Shows z-projection as a function of frequency detuning and pulse duration
- The chevron pattern arises from Rabi oscillations: diagonal stripes appear when the drive is on resonance

**Key Features:**
- Uses rotating frame transformation (idling frequencies as reference)
- RWA coupling terms enabled by default
- Gaussian pulse envelope for smooth drive
- Schrödinger equation solver for closed-system dynamics

### `cz_phase_calibration.py`

Demonstrates CZ (conditional Z phase) calibration by measuring conditional phase accumulation as a function of flux pulse amplitude. This is a standard calibration protocol for flux-tunable coupler architectures.

**What it does:**
- Creates a 2-qubit device with 1 coupler using `SuperconductingDeviceRot`
- Implements Ramsey-style sequence on target qubit:
  - Prepare target in |+⟩ via RY(π/2)
  - Apply flux pulse of varying amplitude on target qubit
  - Apply second RY(π/2) to project phase onto measurable population
- Conditional variant: measures phase with control qubit in |0⟩ and |1⟩
- Extracts conditional phase φ_CZ = φ_Z^|1⟩ - φ_Z^|0⟩ (the CZ angle)
- Sweeps flux amplitude to map out φ_Z(A) and find flux amplitude for CZ gate (φ_CZ = π)

**Usage:**
```bash
python cz_phase_calibration.py
```

**Output:**
- Saves 4-panel plot to `data/cz_phase_calibration.png`:
  - Top left: Population vs flux amplitude (control in |0⟩ and |1⟩)
  - Top right: Z phase accumulation vs flux amplitude
  - Bottom left: Conditional phase (CZ angle) vs flux amplitude
  - Bottom right: Zoom on CZ angle with π marker
- Saves data to `data/cz_phase_calibration_data.npz`
- Prints summary with flux amplitude for CZ gate

**Key Features:**
- Uses rotating frame transformation (idling frequencies as reference)
- RWA coupling terms enabled by default
- Pauli rotation matrices for X/Y gates (instantaneous, perfect rotations)
- Time evolution only for flux pulse (phase accumulation)
- Square pulses for flux modulation
- Ramsey interferometry for phase measurement
- Conditional phase extraction for CZ gate calibration

**Physical Background:**
When a flux pulse is applied to a transmon, it modulates the qubit frequency, accumulating phase:
    φ_Z = ∫₀ᵀ Δω(t) dt
where Δω(t) = ω_q(Φ(t)) - ω_q^idle. The conditional phase difference between control in |1⟩ and |0⟩ gives the CZ gate angle, which is used to calibrate flux pulse amplitudes for two-qubit gates.

**References:**
- Yan et al., PRX 8, 041020 (2018) - Tunable Coupling Scheme
- Mundada et al., PRApplied 12, 054023 (2019) - Qubit Crosstalk Suppression
- Sung et al., PRX 11, 021058 (2021) - High-Fidelity CZ Gates
- Rol et al., PRL 123, 120502 (2019) - Fast Conditional-Phase Gate
- Negîrneac et al., PRX Quantum 2, 020319 (2021) - High-Fidelity Controlled-Z

### `cz_phase_calibration_2d.py`

2D version of the CZ phase calibration that sweeps both target qubit flux and coupler flux simultaneously, creating a 2D heatmap of the conditional phase.

**What it does:**
- Creates a 2-qubit device with 1 coupler using `SuperconductingDeviceRot`
- Implements Ramsey-style sequence on target qubit with:
  - Prepare target in |+⟩ via RY(π/2)
  - Apply flux pulses on BOTH target qubit AND coupler simultaneously
  - Apply second RY(π/2) to project phase onto population
- Conditional variant: measures phase with control qubit in |0⟩ and |1⟩
- Sweeps both target qubit flux amplitude and coupler flux amplitude in a 2D grid
- Extracts conditional phase φ_CZ = φ_Z^|1⟩ - φ_Z^|0⟩ for each flux combination
- Creates 2D heatmaps showing how CZ angle depends on both flux biases

**Usage:**
```bash
python cz_phase_calibration_2d.py
```

**Output:**
- Saves 6-panel plot to `data/cz_phase_calibration_2d.png`:
  - Top row: Populations and phases for control in |0⟩ and |1⟩
  - Bottom row: Conditional phase (CZ angle) with contour lines for π
- Saves data to `data/cz_phase_calibration_2d_data.npz`
- Prints summary with optimal flux amplitudes for CZ gate

**Key Features:**
- Uses rotating frame transformation (idling frequencies as reference)
- RWA coupling terms enabled by default
- Pauli rotation matrices for X/Y gates (instantaneous)
- Time evolution only for flux pulses
- 2D parameter sweep for comprehensive calibration
- Contour plots showing π phase lines (CZ gate operating points)

**Use Cases:**
- Finding optimal operating points for CZ gates
- Understanding crosstalk between qubit and coupler flux lines
- Calibrating multi-parameter flux pulses
- Characterizing how coupler flux affects target qubit phase

## Requirements

- `dynamiqs` for time evolution
- `jax` and `jax.numpy` for numerical computations
- `matplotlib` for plotting
- `numpy` for array operations

