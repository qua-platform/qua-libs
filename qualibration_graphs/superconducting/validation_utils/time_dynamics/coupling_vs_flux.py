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
    USER_PARAMS_SYMMETRIC,
)


def _ket_vec(levels: int, n_modes: int, occ: list[int]) -> jnp.ndarray:
    ket = dq.basis([levels] * n_modes, occ)
    return ket.asdense().data.reshape(-1)


def effective_coupling_from_eigensplitting(H: dq.QArray | dq.TimeQArray, levels: int) -> float:
    H0 = H(0.0) if callable(H) else H
    H_dense = H0.asdense().data

    ket_10 = _ket_vec(levels, 3, [1, 0, 0])
    ket_01 = _ket_vec(levels, 3, [0, 1, 0])
    ket_001 = _ket_vec(levels, 3, [0, 0, 1])
    P = jnp.stack([ket_10, ket_01, ket_001], axis=1)
    H_proj = P.conj().T @ H_dense @ P
    evals, evecs = jnp.linalg.eigh(H_proj)

    overlaps = jnp.abs(evecs[0, :]) ** 2 + jnp.abs(evecs[1, :]) ** 2
    order = jnp.argsort(overlaps)
    idx_low = int(jnp.asarray(order[-2]).reshape(()))
    idx_high = int(jnp.asarray(order[-1]).reshape(()))
    e_low = evals[idx_low]
    e_high = evals[idx_high]
    v_low = evecs[:, idx_low]
    v_high = evecs[:, idx_high]
    ket_sym = jnp.asarray([1.0, 1.0, 0.0]) / jnp.sqrt(2.0)
    ket_asym = jnp.asarray([1.0, -1.0, 0.0]) / jnp.sqrt(2.0)
    sym_low = jnp.abs(jnp.vdot(ket_sym, v_low)) ** 2
    sym_high = jnp.abs(jnp.vdot(ket_sym, v_high)) ** 2
    asym_low = jnp.abs(jnp.vdot(ket_asym, v_low)) ** 2
    asym_high = jnp.abs(jnp.vdot(ket_asym, v_high)) ** 2
    # If the higher-energy state is more symmetric, g_eff is positive.
    sign = 1.0
    if sym_high < sym_low and asym_high > asym_low:
        sign = -1.0
    return float(sign * 0.5 * (e_high - e_low))

def _ej_from_phi(EJ_small: float, EJ_large: float, phi: float) -> float:
    return float((EJ_small ** 2 + EJ_large ** 2 + 2.0 * EJ_small * EJ_large * jnp.cos(phi)) ** 0.5)


def _omega_from_phi(EJ_small: float, EJ_large: float, EC: float, phi: float) -> float:
    EJ = _ej_from_phi(EJ_small, EJ_large, phi)
    if EJ == 0.0:
        return float("inf")
    xi = (2.0 * EC / EJ) ** 0.5
    return float((8.0 * EJ * EC) ** 0.5 - EC * (1.0 + xi / 4.0))


def _g_to_E(g: float, EJ_i: float, EC_i: float, EJ_j: float, EC_j: float) -> float:
    xi_i = (2.0 * EC_i / EJ_i) ** 0.5
    xi_j = (2.0 * EC_j / EJ_j) ** 0.5
    scale = (2.0 ** 0.5) * (EJ_i / EC_i * EJ_j / EC_j) ** 0.25
    corr = (1.0 - 0.125 * (xi_i + xi_j))
    return float(g / (scale * corr))


def build_device(
    params: dict[str, object],
    phi: float,
    E_couplings: tuple[tuple[float, float], ...],
    E_direct: tuple[float, ...],
) -> SuperconductingDevice:
    qubit_EC = params["qubit_EC"]
    qubit_EJ_small = params["qubit_EJ_small"]
    qubit_EJ_large = params["qubit_EJ_large"]
    qubit_phi_ext = params["qubit_phi_ext"]

    coupler_EC = params["coupler_EC"]
    coupler_EJ_small = params["coupler_EJ_small"]
    coupler_EJ_large = params["coupler_EJ_large"]

    return SuperconductingDevice(
        n_qubits=2,
        levels=3,
        frame="lab",
        qubit_EC=qubit_EC,
        qubit_EJ_small=qubit_EJ_small,
        qubit_EJ_large=qubit_EJ_large,
        qubit_phi_ext=qubit_phi_ext,
        coupler_EC=coupler_EC,
        coupler_EJ_small=coupler_EJ_small,
        coupler_EJ_large=coupler_EJ_large,
        coupler_phi_ext=(float(phi),),
        g_couplings=((0.0, 0.0),),
        g_direct=(0.0,),
        E_couplings=E_couplings,
        E_direct=E_direct,
    )


def main() -> None:
    params = USER_PARAMS_SYMMETRIC

    levels = 3
    n_points = 281
    phi_vals = jnp.linspace(2.0, 3.0, n_points)

    EJ_q0 = [
        _ej_from_phi(
            float(params["qubit_EJ_small"][idx_q]),
            float(params["qubit_EJ_large"][idx_q]),
            float(params["qubit_phi_ext"][idx_q]),
        )
        for idx_q in range(2)
    ]
    EJ_c0 = _ej_from_phi(
        float(params["coupler_EJ_small"][0]),
        float(params["coupler_EJ_large"][0]),
        0.0,
    )
    g1c, g2c = params["g_couplings"][0]
    g12 = float(params["g_direct"][0])
    E1c = _g_to_E(float(g1c), EJ_q0[0], params["qubit_EC"][0], EJ_c0, params["coupler_EC"][0])
    E2c = _g_to_E(float(g2c), EJ_q0[1], params["qubit_EC"][1], EJ_c0, params["coupler_EC"][0])
    E12 = _g_to_E(g12, EJ_q0[0], params["qubit_EC"][0], EJ_q0[1], params["qubit_EC"][1])
    E_couplings = ((E1c, E2c),)
    E_direct = (E12,)

    omega_q = [
        _omega_from_phi(
            float(params["qubit_EJ_small"][idx_q]),
            float(params["qubit_EJ_large"][idx_q]),
            float(params["qubit_EC"][idx_q]),
            float(params["qubit_phi_ext"][idx_q]),
        )
        for idx_q in range(2)
    ]

    g_total = []
    omega_c = []

    for phi in phi_vals:
        device_total = build_device(params, float(phi), E_couplings, E_direct)
        EJ_c = _ej_from_phi(
            float(params["coupler_EJ_small"][0]),
            float(params["coupler_EJ_large"][0]),
            float(phi),
        )
        omega = _omega_from_phi(
            float(params["coupler_EJ_small"][0]),
            float(params["coupler_EJ_large"][0]),
            float(params["coupler_EC"][0]),
            float(phi),
        )
        omega_c.append(float(omega))

        g1c = device_total._g_from_E(
            jnp.asarray(E_couplings[0][0]),
            jnp.asarray(EJ_q0[0]),
            jnp.asarray(params["qubit_EC"][0]),
            jnp.asarray(EJ_c),
            jnp.asarray(params["coupler_EC"][0]),
        )
        g2c = device_total._g_from_E(
            jnp.asarray(E_couplings[0][1]),
            jnp.asarray(EJ_q0[1]),
            jnp.asarray(params["qubit_EC"][1]),
            jnp.asarray(EJ_c),
            jnp.asarray(params["coupler_EC"][0]),
        )

        EJ_c0 = _ej_from_phi(
            float(params["coupler_EJ_small"][0]),
            float(params["coupler_EJ_large"][0]),
            0.0,
        )
        scale = (EJ_c0 / EJ_c) ** 0.25 if EJ_c != 0.0 else float("inf")
        g1c = float(g1c) * scale
        g2c = float(g2c) * scale

        denom1 = omega_q[0] - omega
        denom2 = omega_q[1] - omega
        if denom1 == 0.0 or denom2 == 0.0:
            g_eff = float("inf")
        else:
            g_eff = float(params["g_direct"][0]) + 0.5 * g1c * g2c * (
                1.0 / denom1 + 1.0 / denom2
            )
        g_total.append(g_eff)

    g_total = jnp.asarray(g_total)
    omega_c = jnp.asarray(omega_c)

    # Keep full omega_c range to display both sides of the avoided crossing.
    omega_min = 3.7
    omega_max = 4.2
    finite_mask = (
        jnp.isfinite(omega_c)
        & jnp.isfinite(g_total)
        & (omega_c > omega_min)
        & (omega_c < omega_max)
    )
    omega_c = omega_c[finite_mask]
    g_total = g_total[finite_mask]
    phi_finite = phi_vals[finite_mask]
    sort_idx = jnp.argsort(omega_c)
    omega_c = omega_c[sort_idx]
    g_total = g_total[sort_idx]
    phi_finite = phi_finite[sort_idx]

    # Identify zero crossing of total coupling (linear interpolation)
    sign = jnp.sign(g_total)
    idx = jnp.where(sign[:-1] * sign[1:] <= 0)[0]
    zero_phis = []
    zero_omegas = []
    if idx.size > 0:
        for i in idx.tolist():
            x0, x1 = float(phi_finite[i]), float(phi_finite[i + 1])
            y0, y1 = float(g_total[i]), float(g_total[i + 1])
            if y1 != y0:
                zero_phi = x0 - y0 * (x1 - x0) / (y1 - y0)
                zero_phis.append(zero_phi)
                w0, w1 = float(omega_c[i]), float(omega_c[i + 1])
                if x1 != x0:
                    zero_omegas.append(w0 + (zero_phi - x0) * (w1 - w0) / (x1 - x0))

    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax, ax_phi) = plt.subplots(figsize=(7, 7), nrows=2, sharex=True)
    ax.plot(omega_c, g_total * 1000.0, label="Effective g", linewidth=2.0)
    # ax.plot(omega_c, g_total * 1000.0, "*", color="gray", label="Data (placeholder)")
    ax.set_xlabel("Coupler frequency ωc/2π (GHz)")
    ax.set_ylabel("g/2π (MHz)")
    ax.set_title("Fig. 4(a) Symmetric Coupler (Hamiltonian)")
    if g_total.size == 0:
        raise ValueError("No finite coupling values found; cannot set plot limits.")
    ax.set_xlim(omega_min, omega_max)
    ax.set_ylim(-1000.0, 1000.0)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)
    for zero_omega in zero_omegas:
        ax.axvline(zero_omega, color="black", linestyle=":", linewidth=1.0, alpha=0.8)
    for idx_q, q_freq in enumerate(omega_q):
        ax.axvline(
            q_freq,
            color=f"C{idx_q + 1}",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=f"Q{idx_q + 1} freq",
        )
    ax.legend()

    ax_phi.plot(omega_c, phi_finite, label="Flux vs ωc", linewidth=2.0)
    ax_phi.set_xlabel("Coupler frequency ωc/2π (GHz)")
    ax_phi.set_ylabel("Flux φ (rad)")
    ax_phi.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax_phi.minorticks_on()
    ax_phi.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)
    for zero_omega in zero_omegas:
        ax_phi.axvline(zero_omega, color="black", linestyle=":", linewidth=1.0, alpha=0.8)
    for idx_q, q_freq in enumerate(omega_q):
        ax_phi.axvline(
            q_freq,
            color=f"C{idx_q + 1}",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=f"Q{idx_q + 1} freq",
        )
    ax_phi.legend()
    fig.tight_layout()
    out_path = data_dir / "coupling_vs_flux.png"
    fig.savefig(out_path, dpi=150)

    if zero_phis:
        for zero_phi in zero_phis:
            print(f"Zero-coupling (total) near phi = {zero_phi:.4f} rad")
    else:
        print("No zero-coupling crossing found in the scanned range.")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
