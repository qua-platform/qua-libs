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

## Requirements

- `dynamiqs` for time evolution
- `jax` and `jax.numpy` for numerical computations
- `matplotlib` for plotting
- `numpy` for array operations

