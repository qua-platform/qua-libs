"""Analysis test for 09b_time_rabi_chevron_parity_diff.

Uses virtual_qpu to simulate a time-Rabi chevron for a Loss-DiVincenzo
spin qubit, builds a synthetic ``ds_raw`` xarray.Dataset, injects it
into the node, and runs the analysis pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import jax.numpy as jnp

from virtual_qpu.dynamics import simulate
from virtual_qpu.operators import expval
from virtual_qpu.pulse import SquarePulse
from virtual_qpu.schedule import Schedule
from virtual_qpu.sweep import sweep

from quantum_dots.device import LossDiVincenzoDevice
from quantum_dots.params import LossDiVincenzoParams


NODE_NAME = "09b_time_rabi_chevron_parity_diff"

# ── Simulation parameters ──────────────────────────────────────────────────
# Keep sweeps small for fast test execution.
QUBIT_FREQ_GHZ = 10.0  # Zeeman splitting (GHz)
DRIVE_AMP_GHZ = 0.02  # Drive amplitude (GHz)

# Sweep grid
MAX_DURATION_NS = 400
N_TIME_POINTS = 200  # Dense enough to resolve Rabi oscillations
FREQ_SPAN_MHZ = 100.0
FREQ_STEP_MHZ = 1.0


def _simulate_chevron(
    device: LossDiVincenzoDevice,
    pulse_durations_ns: jnp.ndarray,
    drive_freqs_ghz: jnp.ndarray,
    drive_amp: float = DRIVE_AMP_GHZ,
) -> np.ndarray:
    """Simulate a time-Rabi chevron using ``sweep()`` and return the result.

    The simulation sweeps drive frequency via ``jax.vmap`` (no Python
    loops).  A single square pulse of length ``max(pulse_durations_ns)``
    is used, and the solver samples at every requested time-point so
    that ``tsave`` doubles as the pulse-duration axis.

    Parameters
    ----------
    device : LossDiVincenzoDevice
        Configured virtual_qpu device.
    pulse_durations_ns : jnp.ndarray
        1-D JAX array of pulse durations / time-save points (ns).
    drive_freqs_ghz : jnp.ndarray
        1-D JAX array of *absolute* drive frequencies in GHz.
    drive_amp : float
        Drive amplitude in GHz.

    Returns
    -------
    pdiff : np.ndarray, shape ``(n_freqs, n_durations)``
        Probability of spin-flip P(|↓⟩) at each time-point, for each
        drive frequency.
    """
    psi0 = device.ground_state()

    # |↓⟩⟨↓| projector for qubit 0
    p1_local = jnp.diag(jnp.array([0.0, 1.0], dtype=jnp.complex64))
    P1_q0 = device.embed(p1_local, mode=0)

    max_duration = float(pulse_durations_ns[-1])

    def run_rabi(freq):
        """Simulate Rabi for a given drive frequency, return P(|↓⟩) trace."""
        pulse = SquarePulse(
            duration=max_duration,
            amplitude=drive_amp,
            frequency=freq,
        )
        sched = Schedule()
        sched.play(pulse, channel="drive_q0")
        resolved = sched.resolve()
        H_t = device.hamiltonian(resolved)
        sol = simulate(H_t, psi0, pulse_durations_ns)
        return expval(sol.states, P1_q0)  # (n_durations,)

    # Vectorised sweep over frequencies — shape (n_freqs, n_durations)
    pop1_vs_freq = sweep(run_rabi, freq=drive_freqs_ghz)

    # Convert JAX array back to numpy for xarray / downstream consumers
    return np.asarray(pop1_vs_freq)


def _build_ds_raw(
    qubit_names: list[str],
    detunings_hz: np.ndarray,
    pulse_durations_ns: np.ndarray,
    pdiff_data: dict[str, np.ndarray],
) -> xr.Dataset:
    """Build an xarray.Dataset matching the ``execute_qua_program`` output format.

    Parameters
    ----------
    qubit_names : list[str]
        Names of qubits (e.g. ``["Q1", "Q2"]``).
    detunings_hz : np.ndarray
        Frequency detunings in Hz.
    pulse_durations_ns : np.ndarray
        Pulse durations in ns.
    pdiff_data : dict
        Mapping from qubit name to a ``(n_detunings, n_durations)`` array
        of parity-difference values.
    """
    data_vars = {}
    for qname in qubit_names:
        pd = pdiff_data.get(qname, np.zeros((len(detunings_hz), len(pulse_durations_ns))))
        # p1 = pre-pulse parity (always 0 for ground state)
        data_vars[f"p1_{qname}"] = xr.DataArray(
            np.zeros_like(pd),
            dims=["detuning", "pulse_duration"],
        )
        # p2 = post-pulse parity = spin-flip probability
        data_vars[f"p2_{qname}"] = xr.DataArray(
            pd,
            dims=["detuning", "pulse_duration"],
        )
        # pdiff = |p1 - p2| (since p1=0, pdiff = p2)
        data_vars[f"pdiff_{qname}"] = xr.DataArray(
            pd,
            dims=["detuning", "pulse_duration"],
        )

    ds = xr.Dataset(
        data_vars,
        coords={
            "detuning": xr.DataArray(
                detunings_hz,
                dims="detuning",
                attrs={"long_name": "qubit frequency", "units": "Hz"},
            ),
            "pulse_duration": xr.DataArray(
                pulse_durations_ns,
                dims="pulse_duration",
                attrs={"long_name": "qubit pulse duration", "units": "ns"},
            ),
        },
        attrs={"qubit_names": qubit_names},
    )
    return ds


def _plot_chevron(
    freq_hz: np.ndarray,
    pulse_durations_ns: np.ndarray,
    pdiff: np.ndarray,
    qubit_name: str,
    qubit_freq_hz: float = QUBIT_FREQ_GHZ * 1e9,
) -> "matplotlib.figure.Figure":
    """Create an imshow chevron plot of the simulated parity difference."""
    import matplotlib.pyplot as plt

    detuning_mhz = (freq_hz - qubit_freq_hz) * 1e-6  # Hz -> MHz detuning

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        pdiff,
        aspect="auto",
        origin="lower",
        extent=[
            pulse_durations_ns[0],
            pulse_durations_ns[-1],
            detuning_mhz[0],
            detuning_mhz[-1],
        ],
        cmap="RdBu_r",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_xlabel("Pulse duration (ns)")
    ax.set_ylabel("Drive detuning (MHz)")
    ax.set_title(f"Simulated Rabi chevron (parity diff) — {qubit_name}")
    fig.colorbar(im, ax=ax, label="P(spin flip)")
    fig.tight_layout()
    return fig


# =============================================================================
# Test
# =============================================================================


@pytest.mark.analysis
def test_09b_time_rabi_chevron_analysis(analysis_runner):
    """Simulate a time-Rabi chevron with virtual_qpu and run the analysis pipeline."""

    # ── 1. Set up virtual_qpu device ────────────────────────────────────────
    params = LossDiVincenzoParams(
        n_qubits=2,
        qubit_freqs=[QUBIT_FREQ_GHZ, QUBIT_FREQ_GHZ + 0.2],
        exchange_couplings=[0.001],
        ref_freqs=None,
        frame="rot",
        use_rwa=True,
    )
    device = LossDiVincenzoDevice(params=params)

    # ── 2. Define sweep axes (JAX arrays for the simulator) ─────────────────
    # tsave must start at 0 so the solver places psi0 at t=0 and evolves
    # through the full pulse duration.  Use enough points to resolve the
    # generalised Rabi frequency at the largest detuning.
    pulse_durations_ns_jax = jnp.linspace(0, MAX_DURATION_NS, N_TIME_POINTS, dtype=jnp.float32)

    # Drive frequencies are *absolute* in GHz (simulator convention).
    # frequency=0 is a special sentinel meaning "on-resonance with qubit",
    # so we sweep absolute frequencies around the qubit Zeeman splitting.
    span_ghz = FREQ_SPAN_MHZ * 1e-3  # MHz -> GHz
    step_ghz = FREQ_STEP_MHZ * 1e-3
    drive_freqs_ghz = jnp.arange(
        QUBIT_FREQ_GHZ - span_ghz / 2,
        QUBIT_FREQ_GHZ + span_ghz / 2,
        step_ghz,
        dtype=jnp.float32,
    )

    # ── 3. Simulate Rabi chevron for one qubit (vectorised via sweep) ───────
    # We simulate only Q1; other qubits get zeros.
    pdiff_q1 = _simulate_chevron(device, pulse_durations_ns_jax, drive_freqs_ghz)

    # ── 4. Convert sweep axes to numpy for ds_raw ───────────────────────────
    pulse_durations_ns = np.asarray(pulse_durations_ns_jax)
    # ds_raw stores absolute frequencies in Hz
    detunings_hz = np.asarray(drive_freqs_ghz) * 1e9  # GHz -> Hz

    # The QuAM factory creates qubits Q1..Q4.
    qubit_names = ["Q1", "Q2", "Q3", "Q4"]
    pdiff_data = {
        qname: pdiff_q1 if qname == "Q1" else np.zeros_like(pdiff_q1)
        for qname in qubit_names
    }

    # ── 5. Build ds_raw ─────────────────────────────────────────────────────
    ds_raw = _build_ds_raw(qubit_names, detunings_hz, pulse_durations_ns, pdiff_data)

    # ── 6. Create chevron plot ──────────────────────────────────────────────
    fig = _plot_chevron(detunings_hz, pulse_durations_ns, pdiff_q1, "Q1")

    # ── 7. Run analysis pipeline via fixture ────────────────────────────────
    node = analysis_runner(
        node_name=NODE_NAME,
        ds_raw=ds_raw,
        fig=fig,
        param_overrides={
            "num_shots": 4,
            "min_wait_time_in_ns": 0,
            "max_wait_time_in_ns": MAX_DURATION_NS,
            "frequency_span_in_mhz": FREQ_SPAN_MHZ,
            "frequency_step_in_mhz": FREQ_STEP_MHZ,
        },
    )

    # ── 8. Basic assertions ─────────────────────────────────────────────────
    assert "ds_raw" in node.results
    assert "pdiff_Q1" in node.results["ds_raw"]

    # The chevron should show non-trivial dynamics (not all zeros)
    pdiff_values = node.results["ds_raw"]["pdiff_Q1"].values
    assert np.max(pdiff_values) > 0.05, "Chevron simulation produced no spin-flip signal"
