"""
Demo: sweep pi-pulse duration on resonance and plot state populations vs duration.

Runs time evolution for a range of pulse durations centered on an estimated
pi time, then plots |psi_k|^2 at the end of each pulse for every basis state.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import dynamiqs as dq
import matplotlib.pyplot as plt
from dynamiqs.method import Tsit5

# Ensure repo root is on sys.path when running as a script
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qualibration_graphs.superconducting.validation_utils.time_dynamics import (
    SuperconductingDevice,
)
from qualibration_graphs.validation_utils.time_dynamics import SquarePulse
from qualibration_graphs.superconducting.validation_utils.time_dynamics.params_debug import (
    USER_PARAMS_SYMMETRIC,
)


def omega_from_phi(
    EJ_small: jnp.ndarray, EJ_large: jnp.ndarray, EC: jnp.ndarray, phi: jnp.ndarray
) -> jnp.ndarray:
    EJ = jnp.sqrt(EJ_small ** 2 + EJ_large ** 2 + 2.0 * EJ_small * EJ_large * jnp.cos(phi))
    xi = jnp.sqrt(2.0 * EC / EJ)
    return jnp.sqrt(8.0 * EJ * EC) - EC * (1.0 + xi / 4.0)


if __name__ == "__main__":
    params = USER_PARAMS_SYMMETRIC
    n_qubits = 2
    levels = 3

    device_keys = set(SuperconductingDevice.__dataclass_fields__.keys())
    device_params = {k: v for k, v in params.items() if k in device_keys}
    base_params = {
        "n_qubits": n_qubits,
        "levels": levels,
        "frame": "lab",
        "rot_frame_use_kerr": True,
    }

    qubit_EC = params["qubit_EC"]
    qubit_EJ_small = params["qubit_EJ_small"]
    qubit_EJ_large = params["qubit_EJ_large"]
    qubit_phi_ext = params["qubit_phi_ext"]

    coupler_EC = params["coupler_EC"]
    coupler_EJ_small = params["coupler_EJ_small"]
    coupler_EJ_large = params["coupler_EJ_large"]
    coupler_phi_ext = params["coupler_phi_ext"]

    qubit_refs = tuple(
        omega_from_phi(qubit_EJ_small[i], qubit_EJ_large[i], qubit_EC[i], qubit_phi_ext[i])
        for i in range(n_qubits)
    )
    coupler_refs = tuple(
        omega_from_phi(coupler_EJ_small[k], coupler_EJ_large[k], coupler_EC[k], coupler_phi_ext[k])
        for k in range(n_qubits - 1)
    )

    base_params["ref_qubit_freqs"] = qubit_refs
    base_params["ref_coupler_freqs"] = coupler_refs
    device = SuperconductingDevice(**base_params, **device_params)

    dims = [levels] * device.n_modes
    psi0 = dq.basis(dims, [0] * device.n_modes)

    w0 = jnp.asarray(qubit_refs[0])
    amp = float(params["pulse_amp"])
    t_pi = jnp.pi / (2.0 * amp)
    sweep = jnp.linspace(0.0, 1.4 * t_pi, 21)

    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    max_duration = float(sweep[-1])
    tsave = jnp.concatenate([jnp.asarray([0.0]), sweep])
    pulse = SquarePulse(
        t0=0.0,
        duration=max_duration,
        amp=amp,
        phase=0.0,
        drive_freq=w0,
    )

    Ht = device.construct_h(drives=((0, pulse),))
    res = dq.sesolve(
        Ht,
        psi0,
        tsave=tsave,
        method=Tsit5(max_steps=5_000_000, rtol=1e-8, atol=1e-10),
        options=dq.Options(save_states=True, progress_meter=True),
    )

    states = [s.to_jax() for s in res.states[1:]]
    pops = jnp.abs(jnp.stack(states, axis=0)) ** 2
    total_pop = jnp.sum(pops, axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in range(pops.shape[1]):
        ax.plot(sweep, pops[:, k], linewidth=0.9, alpha=0.85)
    ax.plot(sweep, total_pop, color="black", linewidth=1.4, label="sum |psi|^2")
    ax.set_xlabel("Pulse duration (ns)")
    ax.set_ylabel("Population |psi_k|^2")
    ax.set_title("State populations vs pulse duration")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out_path = data_dir / "x_rot_populations_vs_duration.png"
    fig.savefig(out_path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"Saved plot to {out_path}")
