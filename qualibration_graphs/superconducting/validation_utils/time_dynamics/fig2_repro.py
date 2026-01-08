"""
Qualitative reproduction of Fig. 2(a,b) trends from 2402.04238v2.

Excited-state populations vs coupler flux pulse duration and amplitude,
computed via time evolution using SuperconductingDevice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import dynamiqs as dq
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path when running as a script
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qualibration_graphs.superconducting.validation_utils.time_dynamics.params_debug import (
    USER_PARAMS_SYMMETRIC,
)
from qualibration_graphs.superconducting.validation_utils.time_dynamics import (
    SuperconductingDevice,
)
from qualibration_graphs.validation_utils.time_dynamics import ErfPulse


def projector_from_basis(dims: list[int], occ: list[int]) -> dq.QArray:
    ket = dq.basis(dims, occ)
    ket_dense = ket.asdense().data
    proj = jnp.outer(ket_dense, ket_dense.conj())
    return dq.asqarray(proj, dims=tuple(dims))


def main() -> None:
    params = USER_PARAMS_SYMMETRIC

    # Qubit flux amplitude (Phi0 units) and pulse duration (ns)
    flux_amp = jnp.linspace(0.75,1.25, 51)
    durations = jnp.linspace(0.0, 100.0, 201)
    max_duration = float(durations[-1])

    n_qubits = 2
    levels = 3
    dims = [levels] * (n_qubits + 1)

    base_params = {
        "n_qubits": n_qubits,
        "levels": levels,
        "frame": "lab",
    }
    device_keys = set(SuperconductingDevice.__dataclass_fields__.keys())
    device_params = {k: v for k, v in params.items() if k in device_keys}

    device_params["coupler_phi_ext"] = (2.25,)
    device = SuperconductingDevice(
        **base_params,
        **device_params,
    )

    proj_10 = projector_from_basis(dims, [1, 0, 0])
    proj_01 = projector_from_basis(dims, [0, 1, 0])
    psi0 = dq.basis(dims, [1, 0, 0])

    tsave = durations
    options = dq.Options(save_states=False, progress_meter=True)
    method = dq.method.Euler(dt=0.005)

    def _simulate(amp):
        qubit_pulse = ErfPulse(
            t0=0.0,
            duration=max_duration,
            amp=2.0 * jnp.pi * amp,
            phase=0.0,
            t_r=4.0,
            t_wl=2.0,
            t_wr=2.0,
        )
        Ht = device.construct_h(qubit_flux=((0, qubit_pulse),))
        res = dq.sesolve(
            Ht,
            psi0,
            tsave=tsave,
            exp_ops=(proj_10, proj_01),
            method=method,
            options=options,
        )
        return jnp.real(res.expects[0]), jnp.real(res.expects[1])

    pop_10 = []
    pop_01 = []
    for amp in flux_amp:
        p10, p01 = _simulate(amp)
        pop_10.append(p10)
        pop_01.append(p01)
    pop_10 = jnp.asarray(pop_10)
    pop_01 = jnp.asarray(pop_01)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), constrained_layout=True)

    im0 = axes[0].imshow(
        pop_10,
        origin="lower",
        aspect="auto",
        extent=[float(durations[0]), float(durations[-1]), float(flux_amp[0]), float(flux_amp[-1])],
        cmap="RdBu_r",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_ylabel("q1 flux amp (Phi0)")
    axes[0].set_title("qubit 1: |10> population")
    fig.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.02, label="Excited state pop.")

    im1 = axes[1].imshow(
        pop_01,
        origin="lower",
        aspect="auto",
        extent=[float(durations[0]), float(durations[-1]), float(flux_amp[0]), float(flux_amp[-1])],
        cmap="RdBu_r",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_ylabel("q1 flux amp (Phi0)")
    axes[1].set_xlabel("Flux pulse duration (ns)")
    axes[1].set_title("qubit 2: |01> population")
    fig.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.02, label="Excited state pop.")

    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "fig2_ab_repro.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
