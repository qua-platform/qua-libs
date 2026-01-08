"""
Rabi Chevron Demo: Sweep drive frequency and duration to observe Rabi oscillations.

This demo creates a 2D chevron pattern by:
1. Sweeping the qubit drive frequency around the idling frequency
2. Sweeping the pulse duration
3. Computing the z-projection (expectation value of σ_z) after each pulse
4. Plotting the results as a 2D heatmap showing the characteristic chevron pattern

The chevron pattern arises from Rabi oscillations: when the drive is on resonance,
the qubit oscillates between |0⟩ and |1⟩ states, creating diagonal stripes in
the frequency-duration plane.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
from jax import vmap, jit
import dynamiqs as dq
import matplotlib.pyplot as plt
from dynamiqs.method import Tsit5

# Ensure repo root is on sys.path when running as a script
_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qualibration_graphs.superconducting.validation_utils.time_dynamics.device_rot import (
    SuperconductingDeviceRot,
)
from qualibration_graphs.validation_utils.time_dynamics import GaussianPulse, kron_n


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
    # Use projectors: Z = |0⟩⟨0| - |1⟩⟨1|
    proj_0 = dq.proj(dq.basis([levels], [0]))  # |0⟩⟨0|
    proj_1 = dq.proj(dq.basis([levels], [1]))  # |1⟩⟨1|
    z_local = proj_0 - proj_1

    # Embed into full Hilbert space
    ops = [dq.eye(levels) for _ in range(n_modes)]
    ops[qubit_idx] = z_local
    z_op = kron_n(ops)

    # Compute expectation value
    return jnp.real(dq.expect(z_op, state))


def main():
    """Run Rabi chevron sweep."""
    # Device parameters (simplified for demo)
    n_qubits = 2
    levels = 3

    # Qubit parameters (GHz)
    max_qubit_freqs = (5.0, 5.1)  # Maximum frequencies at sweet spot
    qubit_anharm = (0.23, 0.23)  # Anharmonicities (EC)

    # Coupler parameters
    max_coupler_freqs = (6.0,)  # One coupler between two qubits
    coupler_anharm = (0.20,)

    # Coupling parameters (GHz)
    max_g_qubit_coupler = ((0.01, 0.01),)  # Qubit-coupler couplings
    max_g_direct = (0.001,)  # Direct qubit-qubit coupling

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

    # Initial state: all qubits and couplers in ground state |00...0⟩
    dims = [levels] * device.n_modes
    psi0 = dq.basis(dims, [0] * device.n_modes)

    # Get idling frequency for qubit 0 (reference for detuning sweep)
    omega_idle = device.idling_qubit_freqs[0]

    # Sweep parameters (reduced for faster testing - increase for higher resolution)
    # Note: JIT compilation makes larger sweeps much faster after initial compilation
    n_freqs = 41  # Number of frequency points
    n_durations = 41  # Number of duration points

    # Frequency sweep: ±50 MHz around idling frequency
    freq_span = 0.05  # GHz
    drive_freqs = jnp.linspace(omega_idle - freq_span, omega_idle + freq_span, n_freqs)

    # Duration sweep: 0 to 200 ns
    duration_max = 800.0  # ns
    durations = jnp.linspace(4.0, duration_max, n_durations)

    # Pulse amplitude (Rabi frequency in GHz)
    pulse_amp = 0.05  # GHz

    print(f"Idling frequency: {omega_idle:.4f} GHz")
    print(f"Frequency sweep: [{omega_idle - freq_span:.4f}, {omega_idle + freq_span:.4f}] GHz")
    print(f"Duration sweep: [10.0, {duration_max:.1f}] ns")
    print(f"Pulse amplitude: {pulse_amp:.4f} GHz")
    print(f"Computing {n_freqs} × {n_durations} = {n_freqs * n_durations} simulations...")

    # Create Z operator once (for qubit 0) - using the same method as compute_z_projection
    def create_z_operator():
        """Create Z operator for qubit 0."""
        proj_0 = dq.proj(dq.basis([levels], [0]))  # |0⟩⟨0|
        proj_1 = dq.proj(dq.basis([levels], [1]))  # |1⟩⟨1|
        z_local = proj_0 - proj_1
        ops = [dq.eye(levels) for _ in range(device.n_modes)]
        ops[0] = z_local  # qubit 0
        return kron_n(ops)

    z_op = create_z_operator()
    # Pre-convert to JAX array for faster expectation value computation
    z_op_jax = z_op.to_jax() if hasattr(z_op, "to_jax") else z_op

    @jit
    def single_simulation(drive_freq: float, duration: float) -> float:
        """
        Run a single simulation for given drive frequency and duration.
        JIT compiled for performance.

        Parameters
        ----------
        drive_freq : float
            Drive frequency in GHz
        duration : float
            Pulse duration in ns

        Returns
        -------
        float
            Z-projection expectation value
        """
        # Create Gaussian pulse
        pulse = GaussianPulse(
            t0=0.0,
            duration=duration,
            amp=pulse_amp,
            phase=0.0,
            drive_freq=drive_freq,
            sigma=5.0,  # Standard Gaussian width parameter
        )

        # Build Hamiltonian
        Ht = device.construct_h(drives=((0, pulse),))

        # Time points: save only at the end (minimal output for speed)
        tsave = jnp.asarray([0.0, duration])

        # Solve Schrödinger equation with optimized settings for speed
        # Relaxed tolerances: rtol=1e-6, atol=1e-8 (was 1e-8, 1e-10)
        # Reduced max_steps: 100k (was 1M) - sufficient for most cases
        res = dq.sesolve(
            Ht,
            psi0,
            tsave=tsave,
            method=Tsit5(max_steps=100_000, rtol=1e-6, atol=1e-8),
            options=dq.Options(save_states=True, progress_meter=False),
        )

        # Get final state and convert to JAX array
        final_state = res.states[-1].to_jax()

        # Compute z-projection for qubit 0 (using pre-converted operator)
        z_proj = jnp.real(dq.expect(z_op_jax, final_state))
        return z_proj  # Return JAX array, not float

    # Create parameter grid
    freq_grid, duration_grid = jnp.meshgrid(drive_freqs, durations, indexing="ij")
    # Flatten to 1D arrays for vmap
    freq_flat = freq_grid.flatten()
    duration_flat = duration_grid.flatten()

    print("Compiling JIT function (this may take a moment on first run)...")
    # Warm-up compilation: compile the function once with example inputs
    # This ensures the first real computation is fast
    _ = single_simulation(freq_flat[0], duration_flat[0]).block_until_ready()
    
    print("Running vectorized simulations with vmap (JIT compiled)...")
    print("  (Optimizations: JIT compilation, relaxed solver tolerances, reduced max_steps)")
    # Vectorize over all parameter combinations
    # vmap will use the already JIT-compiled single_simulation function
    z_projections_flat = vmap(single_simulation)(freq_flat, duration_flat)

    # Reshape back to 2D (freq x duration) and convert to numpy for plotting
    z_projections = jnp.array(z_projections_flat.reshape(n_freqs, n_durations).T)  # Transpose to match (duration, freq) shape

    print("Simulation complete!")

    # Create plot
    freq_mhz = (drive_freqs - omega_idle) * 1000  # Convert to MHz detuning
    durations_ns = durations

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create 2D heatmap
    im = ax.imshow(
        z_projections,
        extent=[
            float(freq_mhz[0]),
            float(freq_mhz[-1]),
            float(durations_ns[0]),
            float(durations_ns[-1]),
        ],
        aspect="auto",
        origin="lower",
        cmap="RdBu",
        vmin=-1.0,
        vmax=1.0,
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("⟨Z⟩", fontsize=12)

    # Labels and title
    ax.set_xlabel("Drive Frequency Detuning (MHz)", fontsize=12)
    ax.set_ylabel("Pulse Duration (ns)", fontsize=12)
    ax.set_title("Rabi Chevron: Z-Projection vs Frequency and Duration", fontsize=14)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")

    # Save plot
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "rabi_chevron.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

