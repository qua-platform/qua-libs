"""
Compute effective qubit-qubit coupling vs coupler flux.

Uses the paper-aligned Hamiltonian in SuperconductingDevice and extracts
the effective coupling from the splitting between eigenstates that overlap
with |10> and |01> (coupler in ground).

Units: energies in GHz, times in ns.
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

from qualibration_graphs.superconducting.validation_utils.time_dynamics import SuperconductingDevice
from qualibration_graphs.superconducting.validation_utils.time_dynamics.params_debug import (
    PAPER_PARAMS_SYMMETRIC,
)


def _ket_vec(levels: int, n_modes: int, occ: list[int]) -> jnp.ndarray:
    ket = dq.basis([levels] * n_modes, occ)
    return ket.asdense().data.reshape(-1)


def effective_coupling_from_eigensplitting(H: dq.QArray | dq.TimeQArray, levels: int) -> float:
    H0 = H(0.0) if callable(H) else H
    H_dense = H0.asdense().data
    evals, evecs = jnp.linalg.eigh(H_dense)

    ket_10 = _ket_vec(levels, 3, [1, 0, 0])
    ket_01 = _ket_vec(levels, 3, [0, 1, 0])
    ket_sym = (ket_10 + ket_01) / jnp.sqrt(2.0)
    ket_asym = (ket_10 - ket_01) / jnp.sqrt(2.0)

    overlaps = jnp.abs(evecs.conj().T @ ket_10) ** 2 + jnp.abs(evecs.conj().T @ ket_01) ** 2
    overlaps = overlaps.reshape(-1)
    order = jnp.argsort(overlaps)
    idx_low = int(jnp.asarray(order[-2]).reshape(()))
    idx_high = int(jnp.asarray(order[-1]).reshape(()))
    e_low = evals[idx_low]
    e_high = evals[idx_high]
    v_low = evecs[:, idx_low]
    v_high = evecs[:, idx_high]
    sym_low = jnp.abs(jnp.vdot(ket_sym, v_low)) ** 2
    sym_high = jnp.abs(jnp.vdot(ket_sym, v_high)) ** 2
    asym_low = jnp.abs(jnp.vdot(ket_asym, v_low)) ** 2
    asym_high = jnp.abs(jnp.vdot(ket_asym, v_high)) ** 2
    # If the higher-energy state is more symmetric, g_eff is positive.
    sign = 1.0
    if sym_high < sym_low and asym_high > asym_low:
        sign = -1.0
    return float(sign * 0.5 * (e_high - e_low))

def _derive_ej_from_omega(omega: float, EC: float) -> float:
    return float((omega + EC) ** 2 / (8.0 * EC))


def _g_to_E(g: float, EJ_i: float, EC_i: float, EJ_j: float, EC_j: float) -> float:
    xi_i = (2.0 * EC_i / EJ_i) ** 0.5
    xi_j = (2.0 * EC_j / EJ_j) ** 0.5
    scale = (2.0 ** 0.5) * (EJ_i / EC_i * EJ_j / EC_j) ** 0.25
    corr = (1.0 - 0.125 * (xi_i + xi_j))
    return float(g / (scale * corr))


def build_device(params: dict[str, object], phi: float, g_direct, g_couplings, E_direct=None, E_couplings=None) -> SuperconductingDevice:
    qubit_freqs = params["qubit_freqs"]
    coupler_freqs = params.get("coupler_freqs_tuned", params["coupler_freqs"])
    qubit_anharm = params["qubit_anharm"]
    coupler_anharm = params["coupler_anharm"]

    return SuperconductingDevice(
        n_qubits=2,
        levels=3,
        frame="rot",
        g_couplings=g_couplings,
        g_direct=g_direct,
        E_couplings=E_couplings,
        E_direct=E_direct,
        qubit_freqs=qubit_freqs,
        qubit_anharm=qubit_anharm,
        coupler_freqs=coupler_freqs,
        coupler_anharm=coupler_anharm,
        ref_qubit_freqs=qubit_freqs,
        ref_coupler_freqs=coupler_freqs,
        coupler_phi_ext=(float(phi),),
    )


def main() -> None:
    params = PAPER_PARAMS_SYMMETRIC

    levels = 3
    n_points = 81
    phi_vals = jnp.linspace(-jnp.pi, jnp.pi, n_points)

    # Match the analytic example (Table I/II symmetric, Fig. 4a style)
    qubit_freqs = params["qubit_freqs"]
    coupler_freqs = params["coupler_freqs"]
    g1c, g2c = params["g_couplings"][0]
    g12 = float(params.get("g_direct", (0.0,))[0])

    # Derive EJ from reported frequencies (transmon approximation)
    EC_q = [float(x) for x in params["qubit_anharm"]]
    EC_c = [float(x) for x in params["coupler_anharm"]]
    EJ_q = [_derive_ej_from_omega(f, ec) for f, ec in zip(qubit_freqs, EC_q)]
    EJ_c = [_derive_ej_from_omega(f, ec) for f, ec in zip(coupler_freqs, EC_c)]

    # Convert g -> E using paper Eq. (2)/(3) at phi=0
    E1c = _g_to_E(float(g1c), EJ_q[0], EC_q[0], EJ_c[0], EC_c[0])
    E2c = _g_to_E(float(g2c), EJ_q[1], EC_q[1], EJ_c[0], EC_c[0])
    E12 = _g_to_E(g12, EJ_q[0], EC_q[0], EJ_q[1], EC_q[1])

    g_total = []
    omega_c = []

    for phi in phi_vals:
        device_total = build_device(
            {
                **params,
                "qubit_freqs": qubit_freqs,
                "coupler_freqs": coupler_freqs,
            },
            float(phi),
            g_direct=params.get("g_direct", (0.0,)),
            g_couplings=((0.0, 0.0),),
            E_direct=(E12,),
            E_couplings=((E1c, E2c),),
        )
        H_total = device_total.construct_h()
        g_total.append(effective_coupling_from_eigensplitting(H_total, levels))

        omega = device_total._omega_from_phi(
            jnp.asarray(device_total.coupler_EJ_small[0]),
            jnp.asarray(device_total.coupler_EJ_large[0]),
            jnp.asarray(device_total.coupler_EC[0]),
            jnp.asarray(device_total.coupler_phi_ext[0]),
        )
        omega_c.append(float(omega))

    g_total = jnp.asarray(g_total)
    omega_c = jnp.asarray(omega_c)

    # Keep full omega_c range to display both sides of the avoided crossing.
    finite_mask = jnp.isfinite(omega_c)
    omega_c = omega_c[finite_mask]
    g_total = g_total[finite_mask]
    sort_idx = jnp.argsort(omega_c)
    omega_c = omega_c[sort_idx]
    g_total = g_total[sort_idx]

    # Identify zero crossing of total coupling (linear interpolation)
    sign = jnp.sign(g_total)
    idx = jnp.where(sign[:-1] * sign[1:] <= 0)[0]
    zero_phi = None
    if idx.size > 0:
        i = int(idx[0])
        x0, x1 = float(phi_vals[i]), float(phi_vals[i + 1])
        y0, y1 = float(g_total[i]), float(g_total[i + 1])
        if y1 != y0:
            zero_phi = x0 - y0 * (x1 - x0) / (y1 - y0)

    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(omega_c, g_total * 1000.0, label="Effective g", linewidth=2.0)
    ax.plot(omega_c, g_total * 1000.0, "*", color="gray", label="Data (placeholder)")
    ax.set_xlabel("Coupler frequency ωc/2π (GHz)")
    ax.set_ylabel("g/2π (MHz)")
    ax.set_title("Fig. 4(a) Symmetric Coupler (Hamiltonian)")
    x_min = float(jnp.min(omega_c))
    x_max = float(jnp.max(omega_c))
    ax.set_xlim(x_min, x_max + 0.25)
    y_max = float(jnp.max(jnp.abs(g_total))) * 1000.0
    ax.set_ylim(-1.05 * y_max, 1.05 * y_max)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    out_path = data_dir / "coupling_vs_flux.png"
    fig.savefig(out_path, dpi=150)

    if zero_phi is not None:
        print(f"Zero-coupling (total) near phi = {zero_phi:.4f} rad")
    else:
        print("No zero-coupling crossing found in the scanned range.")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
