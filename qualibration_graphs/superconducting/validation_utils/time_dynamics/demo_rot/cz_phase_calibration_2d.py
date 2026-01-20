"""
CZ Phase Calibration 2D: Conditional Z Phase vs Target and Coupler Flux Amplitudes

This is a 2D version of the CZ phase calibration that sweeps both the target qubit flux
and the coupler flux simultaneously, creating a 2D heatmap of the conditional phase.

This is useful for understanding how the CZ gate angle depends on both flux biases,
which is important for:
- Finding optimal operating points
- Understanding crosstalk between qubit and coupler flux lines
- Calibrating multi-parameter flux pulses for CZ gates

Physical Picture:
-----------------
When you apply flux pulses to both a target qubit and its coupler, the accumulated
phase on the target qubit depends on both:
1. Direct frequency shift from target qubit flux: Δω_target(Φ_target)
2. Indirect shift from coupler flux via coupling: Δω_target(Φ_coupler)

The conditional phase φ_CZ = φ_Z^|1⟩ - φ_Z^|0⟩ gives the CZ angle as a function
of both flux amplitudes.
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


def ramsey_sequence_2d(
    device: SuperconductingDeviceRot,
    target_qubit: int,
    coupler_idx: int,
    control_state: int,
    target_flux_amplitude: float,
    coupler_flux_amplitude: float,
    flux_duration: float,
    return_state: bool = False,
) -> float | tuple[float, dq.QArray]:
    """
    Execute Ramsey sequence with both target qubit and coupler flux pulses.

    Sequence:
    1. Prepare control qubit in |control_state⟩ (if control_state >= 0) using X gate
    2. RY(π/2) on target qubit (prepare |+⟩) using Pauli rotation
    3. Apply flux pulses on BOTH target qubit AND coupler (time evolution)
    4. RY(π/2) on target qubit (project phase onto z-axis) using Pauli rotation
    5. Measure target qubit z-projection ⟨Z⟩

    Parameters
    ----------
    device : SuperconductingDeviceRot
        Device instance
    target_qubit : int
        Index of target qubit
    coupler_idx : int
        Index of coupler (0-based coupler index, not global mode index)
    control_state : int
        Control qubit state: 0 for |0⟩, 1 for |1⟩, -1 for no control qubit
    target_flux_amplitude : float
        Flux pulse amplitude on target qubit (reduced flux phi = 2π*Φ/Φ0)
    coupler_flux_amplitude : float
        Flux pulse amplitude on coupler (reduced flux phi = 2π*Φ/Φ0)
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

    # Step 3: Apply flux pulses on BOTH target qubit AND coupler (time evolution)
    # Target qubit flux pulse
    target_flux_pulse = SquarePulse(
        t0=0.0,  # Start immediately
        duration=flux_duration,
        amp=target_flux_amplitude,
        phase=0.0,
    )

    # Coupler flux pulse
    coupler_flux_pulse = SquarePulse(
        t0=0.0,  # Start immediately
        duration=flux_duration,
        amp=coupler_flux_amplitude,
        phase=0.0,
    )

    # Build Hamiltonian with both flux pulses (no drives)
    Ht = device.construct_h(
        drives=(),  # No drive pulses
        qubit_flux=((target_qubit, target_flux_pulse),),
        coupler_flux=((coupler_idx, coupler_flux_pulse),),
    )

    # Time points: save at the end
    tsave = jnp.asarray([0.0, flux_duration])

    # Solve Schrödinger equation during flux pulses
    res = dq.sesolve(
        Ht,
        psi,
        tsave=tsave,
        method=Tsit5(max_steps=100_000, rtol=1e-6, atol=1e-8),
        options=dq.Options(save_states=True, progress_meter=False),
    )

    # Get state after flux pulses
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
    """Run 2D CZ phase calibration sweep."""
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

    print("CZ Phase Calibration 2D: Conditional Z Phase vs Target and Coupler Flux")
    print("=" * 70)
    print(f"Device: {n_qubits} qubits, {device.n_couplers} coupler(s)")
    print("Target qubit: 1 (index 1)")
    print("Control qubit: 0 (index 0)")
    print("Coupler: 0 (index 0)")
    print(f"Idling frequencies: {[f'{f:.4f}' for f in device.idling_qubit_freqs]} GHz")
    print(f"Max qubit-coupler couplings: {max_g_qubit_coupler[0]} GHz")
    print(f"Max direct qubit-qubit coupling: {max_g_direct[0]} GHz")

    # Sweep parameters
    n_target = 51  # Number of target qubit flux amplitude points
    n_coupler = 51  # Number of coupler flux amplitude points (matching 1D version)
    # Note: For coupler-flux-only, longer pulse duration may be needed
    # because the coupling change is weak (scales as 4th root of frequency ratio)
    flux_duration = 200.0  # ns - fixed flux pulse duration (matching 1D version)

    # Flux amplitude sweeps: reduced flux phi = 2π*Φ/Φ0
    target_flux_amplitudes = jnp.linspace(-0.1, 0.1, n_target)  # Keep original range
    coupler_flux_amplitudes = jnp.linspace(jnp.pi - 0.25, jnp.pi + 0.25, n_coupler)  # Match 1D version

    print("\nSweep parameters:")
    print(f"  Target qubit flux range: [{float(target_flux_amplitudes[0]):.3f}, {float(target_flux_amplitudes[-1]):.3f}] (reduced flux)")
    print(f"  Coupler flux range: [{float(coupler_flux_amplitudes[0]):.3f}, {float(coupler_flux_amplitudes[-1]):.3f}] (reduced flux)")
    print(f"  Flux pulse duration: {flux_duration} ns")
    print(f"  Grid size: {n_target} × {n_coupler} = {n_target * n_coupler} points")
    print(f"\nNote: Coupler flux changes coupling strength (scales as (ω_c/ω_c_max)^0.25)")
    print(f"      This is a weak effect, so longer pulses or stronger couplings are needed.")
    print(f"\nRunning {n_target * n_coupler * 2} simulations (control in |0⟩ and |1⟩)...")
    print("  Using Pauli rotation matrices for X/Y gates, time evolution only for flux pulses")

    # Create parameter grid
    target_grid, coupler_grid = jnp.meshgrid(target_flux_amplitudes, coupler_flux_amplitudes, indexing="ij")
    target_flat = target_grid.flatten()
    coupler_flat = coupler_grid.flatten()

    # Measure for control in |0⟩ and |1⟩
    z_projections_control_0 = []
    z_projections_control_1 = []
    leakage_control_0 = []
    leakage_control_1 = []

    total_points = len(target_flat)
    
    # Create progress bar
    pbar = tqdm(total=total_points * 2, desc="Running simulations", unit="sim")
    
    for i, (target_amp, coupler_amp) in enumerate(zip(target_flat, coupler_flat)):
        # Control in |0⟩
        z_proj_0, state_0 = ramsey_sequence_2d(
            device=device,
            target_qubit=1,
            coupler_idx=0,  # First (and only) coupler
            control_state=0,
            target_flux_amplitude=float(target_amp),
            coupler_flux_amplitude=float(coupler_amp),
            flux_duration=flux_duration,
            return_state=True,
        )
        z_projections_control_0.append(z_proj_0)
        leakage_0 = compute_leakage(state_0, 1, device.levels, device.n_modes)
        leakage_control_0.append(leakage_0)
        pbar.update(1)

        # Control in |1⟩
        z_proj_1, state_1 = ramsey_sequence_2d(
            device=device,
            target_qubit=1,
            coupler_idx=0,  # First (and only) coupler
            control_state=1,
            target_flux_amplitude=float(target_amp),
            coupler_flux_amplitude=float(coupler_amp),
            flux_duration=flux_duration,
            return_state=True,
        )
        z_projections_control_1.append(z_proj_1)
        leakage_1 = compute_leakage(state_1, 1, device.levels, device.n_modes)
        leakage_control_1.append(leakage_1)
        pbar.update(1)
    
    pbar.close()

    z_projections_control_0 = jnp.array(z_projections_control_0).reshape(n_target, n_coupler)
    z_projections_control_1 = jnp.array(z_projections_control_1).reshape(n_target, n_coupler)
    leakage_control_0 = jnp.array(leakage_control_0).reshape(n_target, n_coupler)
    leakage_control_1 = jnp.array(leakage_control_1).reshape(n_target, n_coupler)

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
    fig = plt.figure(figsize=(20, 14))

    # Plot 1: Z-projection with control in |0⟩
    ax1 = plt.subplot(3, 3, 1)
    im1 = ax1.imshow(
        z_projections_control_0,
        extent=[
            float(coupler_flux_amplitudes[0]),
            float(coupler_flux_amplitudes[-1]),
            float(target_flux_amplitudes[0]),
            float(target_flux_amplitudes[-1]),
        ],
        aspect="auto",
        origin="lower",
        cmap="RdBu",
        vmin=-1.0,
        vmax=1.0,
    )
    ax1.set_xlabel("Coupler Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=10)
    ax1.set_ylabel("Target Qubit Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=10)
    ax1.set_title("Z-Projection ⟨Z⟩ - Control in |0⟩", fontsize=11)
    plt.colorbar(im1, ax=ax1)

    # Plot 2: Z-projection with control in |1⟩
    ax2 = plt.subplot(3, 3, 2)
    im2 = ax2.imshow(
        z_projections_control_1,
        extent=[
            float(coupler_flux_amplitudes[0]),
            float(coupler_flux_amplitudes[-1]),
            float(target_flux_amplitudes[0]),
            float(target_flux_amplitudes[-1]),
        ],
        aspect="auto",
        origin="lower",
        cmap="RdBu",
        vmin=-1.0,
        vmax=1.0,
    )
    ax2.set_xlabel("Coupler Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=10)
    ax2.set_ylabel("Target Qubit Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=10)
    ax2.set_title("Z-Projection ⟨Z⟩ - Control in |1⟩", fontsize=11)
    plt.colorbar(im2, ax=ax2)

    # Plot 3: Leakage with control in |0⟩
    ax3 = plt.subplot(3, 3, 3)
    im3 = ax3.imshow(
        leakage_control_0,
        extent=[
            float(coupler_flux_amplitudes[0]),
            float(coupler_flux_amplitudes[-1]),
            float(target_flux_amplitudes[0]),
            float(target_flux_amplitudes[-1]),
        ],
        aspect="auto",
        origin="lower",
        cmap="hot",
    )
    ax3.set_xlabel("Coupler Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=10)
    ax3.set_ylabel("Target Qubit Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=10)
    ax3.set_title("Leakage P(|2⟩) - Control in |0⟩", fontsize=11)
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label("P(|2⟩)", fontsize=10)

    # Plot 4: Phase with control in |0⟩
    ax4 = plt.subplot(3, 3, 4)
    im4 = ax4.imshow(
        phases_control_0,
        extent=[
            float(coupler_flux_amplitudes[0]),
            float(coupler_flux_amplitudes[-1]),
            float(target_flux_amplitudes[0]),
            float(target_flux_amplitudes[-1]),
        ],
        aspect="auto",
        origin="lower",
        cmap="plasma",
    )
    ax4.set_xlabel("Coupler Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=10)
    ax4.set_ylabel("Target Qubit Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=10)
    ax4.set_title("Z Phase φ_Z^|0⟩ (radians)", fontsize=11)
    plt.colorbar(im4, ax=ax4)

    # Plot 5: Phase with control in |1⟩
    ax5 = plt.subplot(3, 3, 5)
    im5 = ax5.imshow(
        phases_control_1,
        extent=[
            float(coupler_flux_amplitudes[0]),
            float(coupler_flux_amplitudes[-1]),
            float(target_flux_amplitudes[0]),
            float(target_flux_amplitudes[-1]),
        ],
        aspect="auto",
        origin="lower",
        cmap="plasma",
    )
    ax5.set_xlabel("Coupler Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=10)
    ax5.set_ylabel("Target Qubit Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=10)
    ax5.set_title("Z Phase φ_Z^|1⟩ (radians)", fontsize=11)
    plt.colorbar(im5, ax=ax5)

    # Plot 6: Leakage with control in |1⟩
    ax6 = plt.subplot(3, 3, 6)
    im6 = ax6.imshow(
        leakage_control_1,
        extent=[
            float(coupler_flux_amplitudes[0]),
            float(coupler_flux_amplitudes[-1]),
            float(target_flux_amplitudes[0]),
            float(target_flux_amplitudes[-1]),
        ],
        aspect="auto",
        origin="lower",
        cmap="hot",
    )
    ax6.set_xlabel("Coupler Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=10)
    ax6.set_ylabel("Target Qubit Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=10)
    ax6.set_title("Leakage P(|2⟩) - Control in |1⟩", fontsize=11)
    cbar6 = plt.colorbar(im6, ax=ax6)
    cbar6.set_label("P(|2⟩)", fontsize=10)

    # Plot 7: Conditional phase (CZ angle) - main result
    ax7 = plt.subplot(3, 3, 7)
    im7 = ax7.imshow(
        conditional_phases,
        extent=[
            float(coupler_flux_amplitudes[0]),
            float(coupler_flux_amplitudes[-1]),
            float(target_flux_amplitudes[0]),
            float(target_flux_amplitudes[-1]),
        ],
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
    )
    ax7.set_xlabel("Coupler Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=10)
    ax7.set_ylabel("Target Qubit Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=10)
    ax7.set_title("Conditional Phase φ_CZ = φ_Z^|1⟩ - φ_Z^|0⟩ (radians)", fontsize=11)
    cbar7 = plt.colorbar(im7, ax=ax7)
    cbar7.set_label("φ_CZ (radians)", fontsize=10)
    # Add contour lines for π
    contours = ax7.contour(
        coupler_flux_amplitudes,
        target_flux_amplitudes,
        conditional_phases,
        levels=[jnp.pi],
        colors=["green"],
        linestyles=["--"],
        linewidths=2,
    )
    ax7.clabel(contours, inline=True, fontsize=10, fmt="π")

    # Plot 8: Maximum leakage (worst case)
    ax8 = plt.subplot(3, 3, 8)
    max_leakage = jnp.maximum(leakage_control_0, leakage_control_1)
    im8 = ax8.imshow(
        max_leakage,
        extent=[
            float(coupler_flux_amplitudes[0]),
            float(coupler_flux_amplitudes[-1]),
            float(target_flux_amplitudes[0]),
            float(target_flux_amplitudes[-1]),
        ],
        aspect="auto",
        origin="lower",
        cmap="hot",
    )
    ax8.set_xlabel("Coupler Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=10)
    ax8.set_ylabel("Target Qubit Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=10)
    ax8.set_title("Maximum Leakage (Worst Case)", fontsize=11)
    cbar8 = plt.colorbar(im8, ax=ax8)
    cbar8.set_label("P(|2⟩)", fontsize=10)

    # Plot 9: Conditional phase with π contour highlighted
    ax9 = plt.subplot(3, 3, 9)
    im9 = ax9.imshow(
        conditional_phases,
        extent=[
            float(coupler_flux_amplitudes[0]),
            float(coupler_flux_amplitudes[-1]),
            float(target_flux_amplitudes[0]),
            float(target_flux_amplitudes[-1]),
        ],
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
    )
    ax9.set_xlabel("Coupler Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=10)
    ax9.set_ylabel("Target Qubit Flux Amplitude (reduced flux φ = 2π·Φ/Φ₀)", fontsize=10)
    ax9.set_title("CZ Angle with π Contour (Zoom)", fontsize=11)
    cbar9 = plt.colorbar(im9, ax=ax9)
    cbar9.set_label("φ_CZ (radians)", fontsize=10)
    # Add contour lines for π and other values
    contours9 = ax9.contour(
        coupler_flux_amplitudes,
        target_flux_amplitudes,
        conditional_phases,
        levels=[-jnp.pi, -jnp.pi / 2, 0, jnp.pi / 2, jnp.pi],
        colors=["red", "orange", "black", "blue", "green"],
        linestyles=["--", "--", "-", "--", "--"],
        linewidths=2,
    )
    ax9.clabel(contours9, inline=True, fontsize=9, fmt=lambda x: f"{x/jnp.pi:.2f}π")

    plt.tight_layout()

    # Save plot
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "cz_phase_calibration_2d.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")

    # Save data
    import numpy as np
    data_path = data_dir / "cz_phase_calibration_2d_data.npz"
    np.savez(
        data_path,
        target_flux_amplitudes=np.array(target_flux_amplitudes),
        coupler_flux_amplitudes=np.array(coupler_flux_amplitudes),
        z_projections_control_0=np.array(z_projections_control_0),
        z_projections_control_1=np.array(z_projections_control_1),
        leakage_control_0=np.array(leakage_control_0),
        leakage_control_1=np.array(leakage_control_1),
        phases_control_0=np.array(phases_control_0),
        phases_control_1=np.array(phases_control_1),
        conditional_phases=np.array(conditional_phases),
        flux_duration=flux_duration,
    )
    print(f"Saved data to {data_path}")

    # Find flux amplitudes that give π phase (CZ gate)
    # Find all points where conditional_phases is closest to π
    diff_from_pi = jnp.abs(conditional_phases - jnp.pi)
    min_idx = jnp.unravel_index(jnp.argmin(diff_from_pi), conditional_phases.shape)
    target_at_pi = float(target_flux_amplitudes[min_idx[0]])
    coupler_at_pi = float(coupler_flux_amplitudes[min_idx[1]])
    phase_at_pi = float(conditional_phases[min_idx])

    # Print summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("  Flux amplitudes for CZ gate (φ_CZ = π):")
    print(f"    Target qubit flux: {target_at_pi:.4f}")
    print(f"    Coupler flux: {coupler_at_pi:.4f}")
    print(f"  Actual conditional phase at that point: {phase_at_pi:.4f} rad ({phase_at_pi / jnp.pi:.4f} π)")
    print(f"  Flux pulse duration: {flux_duration} ns")
    # Leakage at the CZ point
    leakage_at_pi_0 = float(leakage_control_0[min_idx])
    leakage_at_pi_1 = float(leakage_control_1[min_idx])
    max_leakage_at_pi = max(leakage_at_pi_0, leakage_at_pi_1)
    print(f"  Leakage at CZ point: {leakage_at_pi_0:.4e} (|0⟩), {leakage_at_pi_1:.4e} (|1⟩), max: {max_leakage_at_pi:.4e}")
    print("=" * 70)

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
