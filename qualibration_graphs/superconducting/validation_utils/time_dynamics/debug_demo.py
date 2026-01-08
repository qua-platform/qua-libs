"""
Debug demo: detuning sweep Rabi oscillations on qubit 0.

Runs time evolution for a set of detunings around resonance and plots
sigma-z on qubit 0 vs time and detuning (imshow).

Units: energies in GHz, times in ns (consistent internal units).
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
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
from qualibration_graphs.validation_utils.time_dynamics import GaussianPulse, SquarePulse, kron_n
from qualibration_graphs.superconducting.validation_utils.time_dynamics.params_debug import (
    USER_PARAMS_SYMMETRIC,
)


def embed_op(op: dq.QArray, which: int, n_modes: int, levels: int) -> dq.QArray:
    ops = [dq.eye(levels) for _ in range(n_modes)]
    ops[which] = op
    return kron_n(ops)


def omega_from_phi(
    EJ_small: jnp.ndarray, EJ_large: jnp.ndarray, EC: jnp.ndarray, phi: jnp.ndarray
) -> jnp.ndarray:
    EJ = jnp.sqrt(EJ_small ** 2 + EJ_large ** 2 + 2.0 * EJ_small * EJ_large * jnp.cos(phi))
    xi = jnp.sqrt(2.0 * EC / EJ)
    return jnp.sqrt(8.0 * EJ * EC) - EC * (1.0 + xi / 4.0)


if __name__ == "__main__":
    params = USER_PARAMS_SYMMETRIC
    use_square = True
    n_detunings = int(params["n_detunings"])
    n_t = int(params["n_t"])
    t_max = float(params["t_max"])
    n_qubits = 2
    levels = 3

    device_keys = set(SuperconductingDevice.__dataclass_fields__.keys())
    device_params = {k: v for k, v in params.items() if k in device_keys}
    base_params = {"n_qubits": n_qubits, "levels": levels, "frame": "lab"}

    if "qubit_freqs" in params:
        qubit_freqs = params["qubit_freqs"]
        coupler_freqs = params["coupler_freqs"]
        base_params["ref_qubit_freqs"] = qubit_freqs
        base_params["ref_coupler_freqs"] = coupler_freqs
        device = SuperconductingDevice(**base_params, **device_params)
        w0 = jnp.asarray(qubit_freqs[0])
    else:
        qubit_EC = params["qubit_EC"]
        qubit_EJ_small = params["qubit_EJ_small"]
        qubit_EJ_large = params["qubit_EJ_large"]
        qubit_phi_ext = params["qubit_phi_ext"]

        coupler_EC = params["coupler_EC"]
        coupler_EJ_small = params["coupler_EJ_small"]
        coupler_EJ_large = params["coupler_EJ_large"]
        # Bias the coupler away from qubits; coupling remains weak via g_couplings.
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
        w0 = jnp.asarray(qubit_refs[0])
        print(f"w0: {w0}")

    dims = [levels] * device.n_modes
    psi0 = dq.basis(dims, [0] * device.n_modes)

    # Sigma-Z on the |0>,|1> subspace of qubit 0 (0-based index).
    sz01 = jnp.diag(jnp.array([1.0, -1.0, 0.0]))
    sz_op = embed_op(dq.asqarray(sz01, dims=(levels,)), which=0, n_modes=device.n_modes, levels=levels)

    detuning_span = float(params["detuning_span"])
    detunings = jnp.linspace(-detuning_span, detuning_span, n_detunings)
    tsave = jnp.linspace(0.0, t_max, n_t)
    options = dq.Options(save_states=False, progress_meter=True)
    pulse_cls = SquarePulse if use_square else GaussianPulse

    def _simulate_detuning(d: jnp.ndarray) -> jnp.ndarray:
        pulse = pulse_cls(
            t0=0.0,
            duration=float(params["pulse_duration"]),
            amp=float(params["pulse_amp"]),
            phase=0.0,
            drive_freq=w0 + d,
        )
        Ht = device.construct_h(drives=((0, pulse),))
        res = dq.sesolve(
            Ht,
            psi0,
            tsave=tsave,
            exp_ops=(sz_op,),
            method=Tsit5(max_steps=1_000_000),
            options=options,
        )
        return jnp.real(res.expects[0])

    sigmaz = jax.vmap(_simulate_detuning)(detunings)
    t = tsave

    fig, ax = plt.subplots(figsize=(7, 4))
    extent = [float(t[0]), float(t[-1]), float(detunings[0]), float(detunings[-1])]
    im = ax.imshow(
        sigmaz,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        vmin=-1.0,
        vmax=1.0,
    )
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Detuning (GHz)")
    ax.set_title("Q0 Sigma-Z vs Time and Detuning")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "rabi_debug_sigmaz_q0.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved debug plot to {out_path}")
