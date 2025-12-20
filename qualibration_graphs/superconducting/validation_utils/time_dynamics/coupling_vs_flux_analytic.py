"""
Analytic tunable coupling vs flux (paper Eq. 5).

Computes g(φ) = g12 - g_eff(φ), where
g_eff = (g1c g2c / 2) * sum_j (1/Δj + 1/Σj)
Δj = ωc(φ) - ωj, Σj = ωc(φ) + ωj

Units: energies in GHz, times in ns.
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

from qualibration_graphs.superconducting.validation_utils.time_dynamics.params_debug import (
    PAPER_PARAMS_SYMMETRIC,
)


def omega_from_phi(EJ_small: float, EJ_large: float, EC: float, phi: jnp.ndarray) -> jnp.ndarray:
    EJ = jnp.sqrt(EJ_small ** 2 + EJ_large ** 2 + 2.0 * EJ_small * EJ_large * jnp.cos(phi))
    xi = jnp.sqrt(2.0 * EC / EJ)
    return jnp.sqrt(8.0 * EJ * EC) - EC * (1.0 + xi / 4.0)


def derive_EJ_max_from_fmax(f_max: float, EC: float) -> float:
    # Invert ω ≈ sqrt(8 EJ EC) - EC
    return float((f_max + EC) ** 2 / (8.0 * EC))


def main() -> None:
    params = PAPER_PARAMS_SYMMETRIC

    # Fig. 4(a): measured symmetric device (Table I/II)
    f1, f2 = params["qubit_freqs"]
    g12 = float(params.get("g_direct", (0.0,))[0])

    # g1c and g2c magnitudes from table (assume given in params, already signed)
    g1c, g2c = params["g_couplings"][0]
    g1c = float(g1c)
    g2c = float(g2c)

    EC_c = float(params.get("coupler_EC", (0.20,))[0])
    f_c_max = float(params["coupler_freqs"][0])
    EJ_max = derive_EJ_max_from_fmax(f_c_max, EC_c)
    EJ_small = EJ_max / 2.0
    EJ_large = EJ_max / 2.0

    # Sweep flux broadly; select points that fall in the target band if provided.
    phi = jnp.linspace(-jnp.pi, jnp.pi, 2001)
    wc = omega_from_phi(EJ_small, EJ_large, EC_c, phi)

    # Focus on the Fig. 4(a) range for ωc
    wc_lo, wc_hi = 1.0, 6.0
    mask = (wc >= wc_lo) & (wc <= wc_hi)
    phi_plot = phi[mask]
    wc_plot = wc[mask]

    delta1 = wc_plot - f1
    delta2 = wc_plot - f2
    sigma1 = wc_plot + f1
    sigma2 = wc_plot + f2

    geff = 0.5 * g1c * g2c * ((1.0 / delta1) + (1.0 / sigma1) + (1.0 / delta2) + (1.0 / sigma2))
    g_total = g12 - geff

    # Zero-crossing interpolation for total coupling
    sign = jnp.sign(g_total)
    idx = jnp.where(sign[:-1] * sign[1:] <= 0)[0]
    phi_zero = None
    if idx.size > 0:
        i = int(jnp.asarray(idx[0]).reshape(()))
        x0, x1 = float(phi_plot[i]), float(phi_plot[i + 1])
        y0, y1 = float(g_total[i]), float(g_total[i + 1])
        if y1 != y0:
            phi_zero = x0 - y0 * (x1 - x0) / (y1 - y0)

    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    g_mhz = g_total * 1000.0

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(wc_plot, g_mhz, label="Analytic g", linewidth=2.0)
    ax.plot(wc_plot, g_mhz, linestyle=":", linewidth=1.0, color="gray", label="Data (placeholder)")
    ax.set_xlabel("Coupler frequency ωc/2π (GHz)")
    ax.set_ylabel("g/2π (MHz)")
    ax.set_title("Fig. 4(a) Symmetric Coupler (Analytic)")
    ax.set_ylim(-25.0, 25.0)
    ax.set_xlim(1.0, 6.0)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    if phi_zero is not None:
        # Interpolate zero crossing in ωc
        wc_zero = float(jnp.interp(phi_zero, phi_plot, wc_plot))
        ax.axvline(wc_zero, color="black", linestyle="--", linewidth=1.0)
        ax.text(wc_zero, 22.0, "g=0", rotation=90, va="top", ha="right")
    ax.legend()
    fig.tight_layout()

    out_path = data_dir / "fig4a_repro.png"
    fig.savefig(out_path, dpi=150)
    if phi_zero is not None:
        wc_zero = float(jnp.interp(phi_zero, phi_plot, wc_plot))
        print(f"Zero-coupling near ωc/2π = {wc_zero:.4f} GHz")
    else:
        print("No zero-coupling crossing found in the scanned range.")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
