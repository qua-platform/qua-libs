"""
CZ Phase Calibration: Conditional Z Phase vs Flux Amplitude

This demo implements a standard calibration protocol for flux-tunable coupler architectures
to characterize how much Z rotation a qubit accumulates as a function of flux pulse amplitude.
The "conditional" aspect measures this phase conditioned on the state of a neighboring qubit,
which lets you extract both single-qubit phase accumulation and any residual ZZ interaction.

Physical Picture:
-----------------
When you apply a flux pulse to a transmon, you're modulating its frequency. The accumulated
phase is:
    φ_Z = ∫₀ᵀ Δω(t) dt
where Δω(t) = ω_q(Φ(t)) - ω_q^idle is the instantaneous detuning from the idle frequency.
For a square pulse of duration T and amplitude A, this simplifies to:
    φ_Z = Δω(A) · T

The conditional phase φ_Z^|1⟩ - φ_Z^|0⟩ gives the actual CZ angle.

Calibration Protocol:
---------------------
1. Ramsey-style measurement:
   - Prepare target qubit in |+⟩ = (|0⟩ + |1⟩)/√2 via RY(π/2)
   - Apply flux pulse of varying amplitude A on target qubit
   - Apply second RY(±π/2) to project phase onto measurable population
   - Sweep amplitude to map out φ_Z(A)

2. Conditional variant (for CZ-type gates):
   - Prepare control qubit in |0⟩ or |1⟩
   - Repeat Ramsey sequence on target
   - The difference φ_Z^|1⟩ - φ_Z^|0⟩ gives the conditional phase (the actual CZ angle)

Key Literature:
---------------
- Yan et al., PRX 8, 041020 (2018) - Tunable Coupling Scheme for High-Fidelity Two-Qubit Gates
- Mundada et al., PRApplied 12, 054023 (2019) - Suppression of Qubit Crosstalk
- Sung et al., PRX 11, 021058 (2021) - High-Fidelity CZ and ZZ-Free iSWAP Gates
- Rol et al., PRL 123, 120502 (2019) - Fast Conditional-Phase Gate with Leakage Interference
- Negîrneac et al., PRX Quantum 2, 020319 (2021) - High-Fidelity Controlled-Z Gate
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Suppress sparse array warnings from dynamiqs
warnings.filterwarnings("ignore", category=UserWarning, message=".*sparse.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Sparse.*")

import jax.numpy as jnp
import dynamiqs as dq
import matplotlib.pyplot as plt
from dynamiqs.method import Tsit5
from tqdm import tqdm

# Ensure repo root is on sys.path when running as a script
_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qualibration_graphs.superconducting.validation_utils.time_dynamics.device_rot import (
    SuperconductingDeviceRot,
)
from qualibration_graphs.validation_utils.time_dynamics import SquarePulse, kron_n


def compute_z_projection(state: dq.QArray, qubit_idx: int, levels: int, n_modes: int) -> float:
    """
    Compute z-projection (expectation value of σ_z) for a qubit.

    For a transmon truncated to 3 levels, we define:
    Z = |0⟩⟨0| - |1⟩⟨1| + 0*|2⟩⟨2|
    This gives Z = +1 for |0⟩, Z = -1 for |1⟩, Z = 0 for |2⟩.

    Parameters
    ----------
    state : dq.QArray
        Quantum state (ket or density matrix)
    qubit_idx : int
        Index of the qubit (0-based)
    levels : int
        Number of levels per mode
    n_modes : int
        Total number of modes (qubits + couplers)

    Returns
    -------
    float
        Expectation value ⟨Z⟩ = P(|0⟩) - P(|1⟩)
    """
    # Create Z operator for a single mode: Z = |0⟩⟨0| - |1⟩⟨1|
    proj_0 = dq.proj(dq.basis([levels], [0]))  # |0⟩⟨0|
    proj_1 = dq.proj(dq.basis([levels], [1]))  # |1⟩⟨1|
    z_local = proj_0 - proj_1

    # Embed into full Hilbert space
    ops = [dq.eye(levels) for _ in range(n_modes)]
    ops[qubit_idx] = z_local
    z_op = kron_n(ops)

    # Convert to dense to avoid sparse array warnings
    z_op = z_op.asdense()

    # Compute expectation value
    return float(jnp.real(dq.expect(z_op, state)))


def compute_leakage(state: dq.QArray, qubit_idx: int, levels: int, n_modes: int) -> float:
    """
    Compute leakage to |2⟩ state (F state) for a qubit.

    Leakage is defined as the population in |2⟩ state: P(|2⟩).

    Parameters
    ----------
    state : dq.QArray
        Quantum state (ket or density matrix)
    qubit_idx : int
        Index of the qubit (0-based)
    levels : int
        Number of levels per mode
    n_modes : int
        Total number of modes (qubits + couplers)

    Returns
    -------
    float
        Leakage P(|2⟩) = ⟨2|ρ|2⟩
    """
    # Create projector for |2⟩ state
    proj_2 = dq.proj(dq.basis([levels], [2]))  # |2⟩⟨2|

    # Embed into full Hilbert space
    ops = [dq.eye(levels) for _ in range(n_modes)]
    ops[qubit_idx] = proj_2
    proj_op = kron_n(ops)

    # Convert to dense to avoid sparse array warnings
    proj_op = proj_op.asdense()

    # Compute expectation value (population)
    return float(jnp.real(dq.expect(proj_op, state)))


def create_pauli_x(qubit_idx: int, levels: int, n_modes: int) -> dq.QArray:
    """
    Create Pauli X operator for a qubit, embedded in full Hilbert space.

    For a 3-level system, X acts on the qubit subspace (|0⟩, |1⟩):
    X = |0⟩⟨1| + |1⟩⟨0|

    Parameters
    ----------
    qubit_idx : int
        Index of the qubit
    levels : int
        Number of levels per mode
    n_modes : int
        Total number of modes

    Returns
    -------
    dq.QArray
        Pauli X operator embedded in full Hilbert space
    """
    # Pauli X on qubit subspace: |0⟩⟨1| + |1⟩⟨0|
    x_local = dq.create(levels) + dq.destroy(levels)

    # Embed into full Hilbert space
    ops = [dq.eye(levels) for _ in range(n_modes)]
    ops[qubit_idx] = x_local
    x_op = kron_n(ops)
    
    # Convert to dense to avoid sparse array warnings
    return x_op.asdense()


def create_pauli_y(qubit_idx: int, levels: int, n_modes: int) -> dq.QArray:
    """
    Create Pauli Y operator for a qubit, embedded in full Hilbert space.

    For a 3-level system, Y acts on the qubit subspace (|0⟩, |1⟩):
    Y = -i|0⟩⟨1| + i|1⟩⟨0|

    Parameters
    ----------
    qubit_idx : int
        Index of the qubit
    levels : int
        Number of levels per mode
    n_modes : int
        Total number of modes

    Returns
    -------
    dq.QArray
        Pauli Y operator embedded in full Hilbert space
    """
    # Pauli Y on qubit subspace: -i|0⟩⟨1| + i|1⟩⟨0| = -i(a - a†)
    a = dq.destroy(levels)
    adag = dq.create(levels)
    y_local = -1j * (a - adag)

    # Embed into full Hilbert space
    ops = [dq.eye(levels) for _ in range(n_modes)]
    ops[qubit_idx] = y_local
    y_op = kron_n(ops)
    
    # Convert to dense to avoid sparse array warnings
    return y_op.asdense()


def apply_ry_gate(state: dq.QArray, qubit_idx: int, levels: int, n_modes: int, angle: float) -> dq.QArray:
    """
    Apply RY(angle) rotation gate to a qubit.

    RY(θ) = exp(-i θ Y/2)

    Parameters
    ----------
    state : dq.QArray
        Quantum state (ket)
    qubit_idx : int
        Index of the qubit
    levels : int
        Number of levels per mode
    n_modes : int
        Total number of modes
    angle : float
        Rotation angle in radians

    Returns
    -------
    dq.QArray
        Rotated state
    """
    Y = create_pauli_y(qubit_idx, levels, n_modes)
    # RY(θ) = exp(-i θ Y/2)
    # Y is already dense from create_pauli_y, but ensure result is dense
    U = dq.expm(-1j * angle / 2.0 * Y)
    # Convert result to dense to avoid sparse array warnings
    U = U.asdense()
    return U @ state


def apply_x_gate(state: dq.QArray, qubit_idx: int, levels: int, n_modes: int) -> dq.QArray:
    """
    Apply X gate (bit flip) to a qubit.

    X = |0⟩⟨1| + |1⟩⟨0|

    Parameters
    ----------
    state : dq.QArray
        Quantum state (ket)
    qubit_idx : int
        Index of the qubit
    levels : int
        Number of levels per mode
    n_modes : int
        Total number of modes

    Returns
    -------
    dq.QArray
        Rotated state
    """
    X = create_pauli_x(qubit_idx, levels, n_modes)
    return X @ state


def ramsey_sequence(
    device: SuperconductingDeviceRot,
    target_qubit: int,
    control_state: int,
    flux_amplitude: float,
    flux_duration: float,
    return_state: bool = False,
) -> float | tuple[float, dq.QArray]:
    """
    Execute Ramsey sequence to measure accumulated phase.

    Sequence:
    1. Prepare control qubit in |control_state⟩ (if control_state >= 0) using X gate
    2. RY(π/2) on target qubit (prepare |+⟩) using Pauli rotation
    3. Apply flux pulse on coupler (time evolution) - changes coupling strength
    4. RY(π/2) on target qubit (project phase onto z-axis) using Pauli rotation
    5. Measure target qubit z-projection ⟨Z⟩

    Note: Coupler flux changes the coupling strength between qubits, which creates
    a conditional phase. The coupling scales as (ω_coupler/ω_coupler_max)^0.25,
    so this is a relatively weak effect compared to direct qubit flux.

    Parameters
    ----------
    device : SuperconductingDeviceRot
        Device instance
    target_qubit : int
        Index of target qubit
    control_state : int
        Control qubit state: 0 for |0⟩, 1 for |1⟩, -1 for no control qubit
    flux_amplitude : float
        Coupler flux pulse amplitude (reduced flux phi = 2π*Φ/Φ0)
    flux_duration : float
        Flux pulse duration in ns

    Returns
    -------
    float
        Z-projection ⟨Z⟩ of target qubit (after second RY(π/2), ⟨Z⟩ = cos(φ))
    """
    # Initial state: all in ground state
    dims = [device.levels] * device.n_modes
    psi = dq.basis(dims, [0] * device.n_modes)

    # Step 1: Prepare control qubit if needed
    if control_state >= 0 and control_state < device.n_qubits:
        if control_state == 1:
            # Apply X gate to control qubit to prepare |1⟩
            psi = apply_x_gate(psi, control_state, device.levels, device.n_modes)

    # Step 2: First RY(π/2) on target qubit (prepare |+⟩)
    psi = apply_ry_gate(psi, target_qubit, device.levels, device.n_modes, jnp.pi / 2.0)

    # Step 3: Apply flux pulse on target qubit (time evolution)
    # Flux pulse: constant amplitude for flux_duration
    flux_pulse = SquarePulse(
        t0=0.0,  # Start immediately
        duration=flux_duration,
        amp=flux_amplitude,  # Reduced flux amplitude
        phase=0.0,
    )

    # Build Hamiltonian with only flux pulse (no drives)
    Ht = device.construct_h(
        drives=(),  # No drive pulses
        # qubit_flux=((target_qubit, flux_pulse),),
        coupler_flux=((0, flux_pulse),),
    )

    # Time points: save at the end
    tsave = jnp.asarray([0.0, flux_duration])

    # Solve Schrödinger equation during flux pulse
    res = dq.sesolve(
        Ht,
        psi,
        tsave=tsave,
        method=Tsit5(max_steps=100_000, rtol=1e-6, atol=1e-8),
        options=dq.Options(save_states=True, progress_meter=False),
    )

    # Get state after flux pulse
    psi = res.states[-1]

    # Step 4: Second RY(π/2) on target qubit (project phase onto z-axis)
    psi = apply_ry_gate(psi, target_qubit, device.levels, device.n_modes, jnp.pi / 2.0)

    # Step 5: Measure z-projection ⟨Z⟩
    # After the second RY(π/2), the phase φ is directly encoded in ⟨Z⟩ = cos(φ)
    z_proj = compute_z_projection(psi, target_qubit, device.levels, device.n_modes)
    
    if return_state:
        return z_proj, psi
    return z_proj


def main():
    """Run CZ phase calibration sweep."""
    # Device parameters for 2-qubit system with tunable coupler
    n_qubits = 2
    levels = 3

    # Qubit parameters (GHz)
    # Typical transmon frequencies
    max_qubit_freqs = (5.0, 5.1)  # Maximum frequencies at sweet spot
    qubit_anharm = (0.23, 0.23)  # Anharmonicities (EC)

    # Coupler parameters
    max_coupler_freqs = (6.0,)  # One coupler between two qubits
    coupler_anharm = (0.20,)

    # Coupling parameters (GHz)
    # Note: For coupler-flux-only CZ calibration, we need stronger couplings
    # to see significant conditional phase. The coupling scales as (omega_c/omega_c_max)^0.25,
    # so the effect is relatively weak. Increasing coupling helps.
    max_g_qubit_coupler = ((0.1, 0.1),)  # Qubit-coupler couplings (increased from 0.01)
    max_g_direct = (0.005,)  # Direct qubit-qubit coupling (increased from 0.001)

    # Create device
    device = SuperconductingDeviceRot(
        n_qubits=n_qubits,
        max_qubit_freqs=max_qubit_freqs,
        max_coupler_freqs=max_coupler_freqs,
        qubit_anharm=qubit_anharm,
        coupler_anharm=coupler_anharm,
        max_g_qubit_coupler=max_g_qubit_coupler,
        max_g_direct=max_g_direct,
        levels=levels,
        use_rwa=True,  # Use RWA for coupling and drive terms
    )

    print("CZ Phase Calibration: Conditional Z Phase vs Coupler Flux Amplitude")
    print("=" * 60)
    print(f"Device: {n_qubits} qubits, {device.n_couplers} coupler(s)")
    print("Target qubit: 1 (index 1)")
    print("Control qubit: 0 (index 0)")
    print("Flux applied to: Coupler (index 0)")
    print(f"Idling frequencies: {[f'{f:.4f}' for f in device.idling_qubit_freqs]} GHz")
    print(f"Max qubit-coupler couplings: {max_g_qubit_coupler[0]} GHz")
    print(f"Max direct qubit-qubit coupling: {max_g_direct[0]} GHz")

    # Sweep parameters
    n_amplitudes = 31  # Number of flux amplitude points
    # Note: For coupler-flux-only, longer pulse duration may be needed
    # because the coupling change is weak (scales as 4th root of frequency ratio)
    flux_duration = 200.0  # ns - fixed flux pulse duration (longer for coupler flux)

    # Flux amplitude sweep: reduced flux phi = 2π*Φ/Φ0
    # Typical range: -0.5 to 0.5 (corresponds to significant frequency shift)
    flux_amplitudes = jnp.linspace(jnp.pi - 0.5, jnp.pi + 0.5, n_amplitudes)

    print("\nSweep parameters:")
    print(f"  Coupler flux amplitude range: [{float(flux_amplitudes[0]):.3f}, {float(flux_amplitudes[-1]):.3f}] (reduced flux)")
    print(f"  Flux pulse duration: {flux_duration} ns")
    print(f"  Number of points: {n_amplitudes}")
    print(f"\nNote: Coupler flux changes coupling strength (scales as (ω_c/ω_c_max)^0.25)")
    print(f"      This is a weak effect, so longer pulses or stronger couplings are needed.")
    print(f"\nRunning {n_amplitudes * 2} simulations (control in |0⟩ and |1⟩)...")
    print("  Using Pauli rotation matrices for X/Y gates, time evolution only for flux pulse")

    # Measure for control in |0⟩ and |1⟩
    z_projections_control_0 = []
    z_projections_control_1 = []
    leakage_control_0 = []
    leakage_control_1 = []

    # Create progress bar
    pbar = tqdm(total=n_amplitudes * 2, desc="Running simulations", unit="sim")

    for i, flux_amp in enumerate(flux_amplitudes):
        # Control in |0⟩
        z_proj_0, state_0 = ramsey_sequence(
            device=device,
            target_qubit=1,
            control_state=0,
            flux_amplitude=float(flux_amp),
            flux_duration=flux_duration,
            return_state=True,
        )
        z_projections_control_0.append(z_proj_0)
        leakage_0 = compute_leakage(state_0, 1, device.levels, device.n_modes)
        leakage_control_0.append(leakage_0)
        pbar.update(1)

        # Control in |1⟩
        z_proj_1, state_1 = ramsey_sequence(
            device=device,
            target_qubit=1,
            control_state=1,
            flux_amplitude=float(flux_amp),
            flux_duration=flux_duration,
            return_state=True,
        )
        z_projections_control_1.append(z_proj_1)
        leakage_1 = compute_leakage(state_1, 1, device.levels, device.n_modes)
        leakage_control_1.append(leakage_1)
        pbar.update(1)

    pbar.close()

    z_projections_control_0 = jnp.array(z_projections_control_0)
    z_projections_control_1 = jnp.array(z_projections_control_1)
    leakage_control_0 = jnp.array(leakage_control_0)
    leakage_control_1 = jnp.array(leakage_control_1)

    # Extract phases directly from z-projections: ⟨Z⟩ = cos(φ), so φ = arccos(⟨Z⟩)
    # Clamp to valid range for arccos
    z_clipped_0 = jnp.clip(z_projections_control_0, -1.0, 1.0)
    z_clipped_1 = jnp.clip(z_projections_control_1, -1.0, 1.0)
    phases_control_0 = jnp.arccos(z_clipped_0)
    phases_control_1 = jnp.arccos(z_clipped_1)

    # Conditional phase (CZ angle)
    conditional_phases = phases_control_1 - phases_control_0

    print("\nSimulation complete!")

    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    flux_amps_array = jnp.array(flux_amplitudes)

    # Plot 1: Z-projections vs flux amplitude
    ax = axes[0, 0]
    ax.plot(flux_amps_array, z_projections_control_0, "o-", label="Control in |0⟩", markersize=4)
    ax.plot(flux_amps_array, z_projections_control_1, "s-", label="Control in |1⟩", markersize=4)
    ax.set_xlabel("Coupler Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=11)
    ax.set_ylabel("Target Qubit Z-Projection ⟨Z⟩", fontsize=11)
    ax.set_title("Ramsey Z-Projection vs Flux Amplitude", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1, 1])
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Plot 2: Phases vs flux amplitude
    ax = axes[0, 1]
    ax.plot(flux_amps_array, phases_control_0, "o-", label="φ_Z^|0⟩", markersize=4)
    ax.plot(flux_amps_array, phases_control_1, "s-", label="φ_Z^|1⟩", markersize=4)
    ax.set_xlabel("Coupler Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=11)
    ax.set_ylabel("Accumulated Phase φ (radians)", fontsize=11)
    ax.set_title("Z Phase Accumulation vs Flux Amplitude", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Leakage to |2⟩ state
    ax = axes[0, 2]
    ax.plot(flux_amps_array, leakage_control_0, "o-", label="Control in |0⟩", markersize=4)
    ax.plot(flux_amps_array, leakage_control_1, "s-", label="Control in |1⟩", markersize=4)
    ax.set_xlabel("Coupler Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=11)
    ax.set_ylabel("Leakage P(|2⟩)", fontsize=11)
    ax.set_title("Leakage to |2⟩ State vs Flux Amplitude", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, None])  # Leakage is always non-negative

    # Plot 4: Conditional phase (CZ angle)
    ax = axes[1, 0]
    ax.plot(flux_amps_array, conditional_phases, "o-", color="red", markersize=4)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.axhline(y=jnp.pi, color="g", linestyle="--", alpha=0.5, label="π (CZ gate)")
    ax.set_xlabel("Coupler Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=11)
    ax.set_ylabel("Conditional Phase φ_CZ (radians)", fontsize=11)
    ax.set_title("Conditional Z Phase (CZ Angle) vs Flux Amplitude", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Phase difference (zoom on conditional phase)
    ax = axes[1, 1]
    ax.plot(flux_amps_array, conditional_phases, "o-", color="red", markersize=4)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.axhline(y=jnp.pi, color="g", linestyle="--", alpha=0.5, label="π (CZ gate)")
    ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Coupler Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=11)
    ax.set_ylabel("Conditional Phase φ_CZ (radians)", fontsize=11)
    ax.set_title("CZ Angle (Zoom)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Find flux amplitude that gives π phase (CZ gate)
    # Interpolate to find where conditional_phases ≈ π
    idx_closest_pi = jnp.argmin(jnp.abs(conditional_phases - jnp.pi))
    flux_at_pi = float(flux_amps_array[idx_closest_pi])
    phase_at_pi = float(conditional_phases[idx_closest_pi])
    ax.plot([flux_at_pi], [phase_at_pi], "r*", markersize=15, label=f"φ_CZ = π at φ = {flux_at_pi:.3f}")
    ax.legend()

    # Plot 6: Maximum leakage (worst case)
    ax = axes[1, 2]
    max_leakage = jnp.maximum(leakage_control_0, leakage_control_1)
    ax.plot(flux_amps_array, max_leakage, "o-", color="purple", markersize=4)
    ax.set_xlabel("Coupler Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=11)
    ax.set_ylabel("Max Leakage P(|2⟩)", fontsize=11)
    ax.set_title("Maximum Leakage (Worst Case)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, None])

    plt.tight_layout()

    # Save plot
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "cz_phase_calibration.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")

    # Save data
    import numpy as np
    data_path = data_dir / "cz_phase_calibration_data.npz"
    np.savez(
        data_path,
        flux_amplitudes=np.array(flux_amps_array),
        z_projections_control_0=np.array(z_projections_control_0),
        z_projections_control_1=np.array(z_projections_control_1),
        leakage_control_0=np.array(leakage_control_0),
        leakage_control_1=np.array(leakage_control_1),
        phases_control_0=np.array(phases_control_0),
        phases_control_1=np.array(phases_control_1),
        conditional_phases=np.array(conditional_phases),
        flux_duration=flux_duration,
        flux_at_pi=flux_at_pi,
        phase_at_pi=phase_at_pi,
    )
    print(f"Saved data to {data_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Flux amplitude for CZ gate (φ_CZ = π): {flux_at_pi:.4f}")
    print(f"  Actual conditional phase at that point: {phase_at_pi:.4f} rad ({phase_at_pi / jnp.pi:.4f} π)")
    print(f"  Flux pulse duration: {flux_duration} ns")
    # Leakage at the CZ point
    leakage_at_pi_0 = float(leakage_control_0[idx_closest_pi])
    leakage_at_pi_1 = float(leakage_control_1[idx_closest_pi])
    max_leakage_at_pi = max(leakage_at_pi_0, leakage_at_pi_1)
    print(f"  Leakage at CZ point: {leakage_at_pi_0:.4e} (|0⟩), {leakage_at_pi_1:.4e} (|1⟩), max: {max_leakage_at_pi:.4e}")
    print("=" * 60)

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
