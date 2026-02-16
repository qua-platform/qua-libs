"""
Time-Dependent Simulation Utilities
====================================

This module consolidates common functionality for running time-dependent quantum
simulations on multi-qubit devices.

Core API Components
-------------------
- TwoSpinDevice: Two-qubit device with configurable frequencies and coupling
- Circuit: Combines device + gates for time evolution
- sweep_circuit: Vectorized parameter sweeps using JAX
- embed_single_qubit: Creates single-qubit observables in multi-qubit space

Conventions
-----------
- Frequencies are angular frequencies in rad/time-unit
- "N GHz" means 2π·N rad/time-unit
- Times (t0, duration, Tphi) use consistent time-units
- Lab frame requires `drive_freq` for each pulse
- Rotating frame removes need for explicit carrier frequencies

Examples
--------
See the bottom of this file for complete usage examples demonstrating:
1. Single simulation: Build circuit, evolve state, measure observables
2. 1D sweep: Rabi amplitude oscillations using sweep_circuit
3. 2D sweep: Rabi chevron (detuning × duration)
4. 2D sweep: Exchange chevron (J_max × duration)
"""

from __future__ import annotations
from typing import Tuple, Any
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

try:
    import dynamiqs as dq
except ModuleNotFoundError:
    print("Failed to import dynamiqs")

from src.device import TwoSpinDevice
from src.circuit import Circuit
from src.utils import embed_single_qubit_op, SZ, sweep_circuit

# =============================================================================
# Usage Examples
# =============================================================================

if __name__ == "__main__":
    """
    Comprehensive examples demonstrating the time-dynamics API.

    These examples show the standard workflow for quantum circuit simulations:
    - Creating devices and initial states
    - Building circuits with gates
    - Running single simulations
    - Parameter sweeps using sweep_circuit
    - Measuring observables and plotting results
    """

    # Import gate definitions
    from src.circuit import X, Y, ExchangeRampGate
    from src.pulse import GaussianPulse

    print("=" * 70)
    print("Time-Dependent Simulation Examples")
    print("=" * 70)

    # =========================================================================
    # Setup: Create device and initial state
    # =========================================================================
    # Define angular frequencies (rad/time-unit)
    # "5.000 GHz" → 2π·5.000, "5.050 GHz" → 2π·5.050
    omega1 = 2.0 * jnp.pi * 5.000
    omega2 = 2.0 * jnp.pi * 5.050

    # Weak (approx. ZZ-type) coupling; equal Jxx/Jyy/Jzz for simplicity
    J = 2.0 * jnp.pi * 5.0e-5  # “~5 MHz” → 2π·5e-5 (rad/unit)

    # Simple dephasing times (same time-units as durations below)
    dev = TwoSpinDevice(
        n=2,
        frame="rot",
        omega=(omega1, omega2),
        ref_omega=(omega1, omega2),
        J0=J,
        Tphi1=500.0,
        Tphi2=500.0,
    )

    # Create initial state |00⟩ (both qubits in ground state)
    # -----------------------
    # 3) Initial state |00⟩
    # -----------------------
    psi0 = dq.basis([2, 2], [0, 0])

    # =========================================================================
    # Example 1: Single Simulation Run
    # =========================================================================
    # Demonstrates: Building a circuit, evolving state, measuring observables
    # =========================================================================
    print("\n[Example 1] Single Run: X(π/2) on qubit 0, Y(π/2) on qubit 1")
    print("-" * 70)

    # Build circuit with two gates
    # Note: Lab frame requires drive_freq for each pulse
    gates = [
        X(which=0, amp=jnp.pi / 2, t0=0.0, duration=100.0, drive_freq=omega1),
        Y(which=1, amp=jnp.pi / 2, t0=10.0, duration=100.0, drive_freq=omega2),
    ]
    circ = Circuit(device=dev, gates=gates)

    # Evolve initial state using Schrödinger equation (closed system)
    # Use solver="me" for master equation (includes dephasing)
    psi_final = circ.final_state(psi0, solver="se")

    # Measure ⟨Z⟩ on each qubit
    # embed_single_qubit creates the operator I⊗...⊗Z⊗...⊗I
    proj_z0 = embed_single_qubit_op(SZ, 0, 2)  # Z on qubit 0
    proj_z1 = embed_single_qubit_op(SZ, 1, 2)  # Z on qubit 1

    exp_z0 = circ.project(psi_final, proj_z0)
    exp_z1 = circ.project(psi_final, proj_z1)

    print(f"⟨Z₀⟩ = {float(np.asarray(exp_z0.real)):.4f}")
    print(f"⟨Z₁⟩ = {float(np.asarray(exp_z1.real)):.4f}")

    # =========================================================================
    # Example 2: 1D Parameter Sweep (Rabi Amplitude)
    # =========================================================================
    # Demonstrates: Using sweep_circuit for vectorized parameter sweeps
    # =========================================================================
    print("\n[Example 2] 1D Sweep: Rabi oscillations vs amplitude")
    print("-" * 70)

    # Define sweep parameter: pulse amplitude
    amps = jnp.linspace(3e-5, 0.025, 51)

    # Create circuit factory function
    # This function takes the sweep parameter and returns a Circuit
    def make_circuit_from_amp(amp: float) -> Circuit:
        """Build circuit with Gaussian X pulse at given amplitude."""
        gates = [
            X(
                which=0,
                amp=amp,
                t0=0.0,
                duration=100.0,
                drive_freq=omega1,
                pulse_class=GaussianPulse,
            ),
        ]
        return Circuit(dev, gates)

    # Prepare observable projector
    proj_z0 = embed_single_qubit_op(SZ, 0, 2)

    # sweep_circuit: Core API for parameter sweeps
    # - Takes circuit factory, initial state, parameter array(s), and projector
    # - Returns expectation values for each parameter
    # - Uses JAX vmap for efficient vectorization
    expectations = sweep_circuit(
        make_circuit_from_amp,  # Circuit factory
        psi0,                    # Initial state
        amps,                    # Parameter(s) to sweep
        projector=proj_z0,       # Observable to measure
        solver="me",             # Master equation (includes dephasing)
    )

    print(f"Swept {len(amps)} amplitude values")
    print(f"⟨Z₀⟩ range: [{expectations.real.min():.3f}, {expectations.real.max():.3f}]")

    # Plot results
    plt.figure()
    plt.plot(amps, expectations.real, marker="o", ms=3, lw=1)
    plt.xlabel("Drive amplitude (arb.)")
    plt.ylabel("⟨Z₀⟩")
    plt.title("Rabi Oscillations (Amplitude Sweep)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # =========================================================================
    # Example 3: 2D Parameter Sweep (Rabi Chevron)
    # =========================================================================
    # Demonstrates: 2D sweeps by passing multiple parameter arrays to sweep_circuit
    # =========================================================================
    print("\n[Example 3] 2D Sweep: Rabi chevron (detuning × duration)")
    print("-" * 70)

    # Chevron parameters
    amp_chevron = 1.0e-2       # Fixed amplitude
    delta_omega = 5e-2         # Detuning window around omega1
    n = 61                     # Grid resolution

    # Define sweep parameters
    drive_freqs = jnp.linspace(-delta_omega, delta_omega, n) + omega1
    pulse_times = jnp.linspace(1e-5, 2000.0, n)

    # Circuit factory for 2D sweep
    # Takes TWO parameters: duration and drive frequency
    def circuit_from_time_and_drive(time: float, drive_freq: float) -> Circuit:
        """Build circuit with Gaussian X pulse at given duration and frequency."""
        gates = [
            X(
                which=0,
                amp=amp_chevron,
                t0=0.0,
                duration=time,
                drive_freq=drive_freq,
                pulse_class=GaussianPulse,
            )
        ]
        return Circuit(dev, gates)

    # For 2D sweeps, create meshgrid and flatten
    # This creates all combinations of (time, freq) pairs
    FREQ_GRID, TIME_GRID = jnp.meshgrid(drive_freqs, pulse_times)
    times_flat = TIME_GRID.ravel()
    freqs_flat = FREQ_GRID.ravel()

    # sweep_circuit with multiple parameters
    # Pass multiple arrays as positional arguments after psi0
    # The circuit factory should accept the same number of parameters
    expectations_flat = sweep_circuit(
        circuit_from_time_and_drive,  # Takes (time, freq) -> Circuit
        psi0,                          # Initial state
        times_flat,                    # First parameter array
        freqs_flat,                    # Second parameter array
        projector=proj_z0,             # Observable
        solver="me",
    )

    print(f"Swept {n}×{n} = {n*n} parameter combinations")

    # Reshape flat results back to 2D grid for plotting
    expectations_2d = np.asarray(expectations_flat.real).reshape(n, n)
    print(f"⟨Z₀⟩ range: [{expectations_2d.min():.3f}, {expectations_2d.max():.3f}]")

    # Plot chevron as 2D heatmap
    detunings = np.asarray(drive_freqs - omega1)  # Convert to detuning Δ
    times_np = np.asarray(pulse_times)

    plt.figure()
    plt.imshow(
        expectations_2d,
        origin="lower",
        aspect="auto",
        extent=(detunings[0], detunings[-1], times_np[0], times_np[-1]),
    )
    plt.xlabel("Detuning Δ = ω_d − ω₁ (rad/time-unit)")
    plt.ylabel("Pulse duration (time-unit)")
    plt.title("Rabi Chevron: ⟨Z₀⟩ vs (Δ, duration)")
    plt.colorbar(label="⟨Z₀⟩")
    plt.tight_layout()
    plt.show()

    # =========================================================================
    # Example 4: 2D Exchange Sweep (J_max × duration)
    # =========================================================================
    # Demonstrates: Two-qubit gates and exchange interactions
    # =========================================================================
    print("\n[Example 4] 2D Sweep: Exchange chevron (J_max × duration)")
    print("-" * 70)

    # Start from |10⟩ to see population transfer
    psi0 = dq.basis([2, 2], [1, 0])

    # Exchange pulse parameters
    t_ramp = 16.0          # Ramp-up/down time
    n_exchange = 101       # Grid resolution

    # Define sweep ranges
    jmaxs = jnp.linspace(1e-5, 2.5, n_exchange)
    # Ensure duration >= 2*t_ramp (required by HeisenbergRampGate)
    times_exchange = jnp.linspace(0.0, 50.0, n_exchange) + 2.0 * t_ramp + 1.0

    # Circuit factory for exchange gates
    def make_exchange_circuit(jmax: float, total_time: float) -> Circuit:
        """Build circuit with Heisenberg exchange pulse."""
        gates = [
            ExchangeRampGate(
                which=(0, 1),      # Acts on qubit pair (0,1)
                Jmax=jmax,         # Maximum exchange strength
                t0=0.0,
                t_ramp=t_ramp,     # Smooth turn-on/off
                duration=total_time,
            )
        ]
        return Circuit(dev, gates)

    # Create parameter grid
    J_GRID, T_GRID = jnp.meshgrid(jmaxs, times_exchange)
    jmaxs_flat = J_GRID.ravel()
    times_flat = T_GRID.ravel()

    # Sweep exchange parameters
    expectations_exchange_flat = sweep_circuit(
        make_exchange_circuit,
        psi0,
        jmaxs_flat,
        times_flat,
        projector=proj_z0,
        solver="se",  # Schrödinger equation (coherent evolution)
    )

    print(f"Swept {n_exchange}×{n_exchange} = {n_exchange**2} parameter combinations")

    # Reshape and plot
    expectations_exchange_2d = np.asarray(expectations_exchange_flat.real).reshape(
        n_exchange, n_exchange
    )
    print(f"⟨Z₀⟩ range: [{expectations_exchange_2d.min():.3f}, {expectations_exchange_2d.max():.3f}]")

    plt.figure()
    plt.imshow(
        expectations_exchange_2d,
        origin="lower",
        aspect="auto",
        cmap="gray",
        extent=(
            jmaxs[0],
            jmaxs[-1],
            times_exchange[0],
            times_exchange[-1],
        ),
    )
    plt.xlabel(r"$J_{\max}$ (rad/time-unit)")
    plt.ylabel("Total pulse duration (time-unit)")
    plt.title(r"Exchange Chevron: $\langle Z_0 \rangle$ vs $(J_{\max}, T)$")
    plt.colorbar(label=r"$\langle Z_0 \rangle$")
    plt.tight_layout()
    plt.show()