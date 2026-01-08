"""
Sweep qubit flux and locate flux-insensitive (sweet spot) points.

Uses SuperconductingDevice with paper symmetric parameters, evaluates
omega_q(phi) for each qubit, and identifies turning points where dω/dφ=0.

Units: energies in GHz, flux in reduced units (phi = 2π Φ / Φ0).
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path when running as a script
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qualibration_graphs.superconducting.validation_utils.time_dynamics import SuperconductingDevice
from qualibration_graphs.superconducting.validation_utils.time_dynamics.params_debug import (
    PAPER_PARAMS_SYMMETRIC, DEBUG_PARAMS
)


def _find_turning_points(phi_vals: jnp.ndarray, omega_vals: jnp.ndarray) -> list[float]:
    # Identify indices where the derivative changes sign.
    deriv = jnp.gradient(omega_vals, phi_vals)
    sign = jnp.sign(deriv)
    idx = jnp.where(sign[:-1] * sign[1:] <= 0)[0]
    return [float(phi_vals[int(i)]) for i in idx]


def main() -> None:
    params = PAPER_PARAMS_SYMMETRIC
    device = SuperconductingDevice(
        n_qubits=2,
        levels=3,
        frame="rot",
        g_couplings=params["g_couplings"],
        g_direct=params.get("g_direct", (0.0,)),
        qubit_freqs=params["qubit_freqs"],
        qubit_anharm=params["qubit_anharm"],
        coupler_freqs=params["coupler_freqs"],
        coupler_anharm=params["coupler_anharm"],
        ref_qubit_freqs=params["qubit_freqs"],
        ref_coupler_freqs=params["coupler_freqs"],
    )

    phi_vals = jnp.linspace(-jnp.pi, jnp.pi, 401)

    fig, ax = plt.subplots(figsize=(7, 4))
    for q in range(device.n_qubits):
        EJ_s = jnp.asarray(device.qubit_EJ_small[q])
        EJ_l = jnp.asarray(device.qubit_EJ_large[q])
        EC = jnp.asarray(device.qubit_EC[q])
        omega = device._omega_from_phi(EJ_s, EJ_l, EC, phi_vals)
        ax.plot(phi_vals, omega, label=f"Q{q} ω(φ)")

        turning_pts = _find_turning_points(phi_vals, omega)
        for phi0 in turning_pts:
            ax.axvline(phi0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)

    ax.set_xlabel("Reduced flux φ")
    ax.set_ylabel("Qubit frequency ω/2π (GHz)")
    ax.set_title("Qubit Flux Sweet Spots (Turning Points)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()

    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "qubit_flux_sweetspots.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
