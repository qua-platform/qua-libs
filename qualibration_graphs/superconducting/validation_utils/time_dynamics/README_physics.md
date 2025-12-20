# Superconducting Time‑Dynamics Model (Paper‑Aligned)

This note documents the physics implemented in
`qualibration_graphs/superconducting/validation_utils/time_dynamics/device.py`
and how each Hamiltonian term is constructed. The model is aligned to the
floating tunable‑coupler Hamiltonian in Sete et al. (2021), Appendix B.

## System overview

We model a chain of **N tunable transmon qubits** with **N−1 tunable couplers**.
Each element is a multi‑level transmon (default 3 levels). Indices:

- Qubits: `q = 0..N-1`
- Couplers: `c = 0..N-2` (coupler `c` connects qubits `c` and `c+1`)

Units are **GHz for energies** and **ns for time** (numerically stable for the
ODE solver). All frequencies and couplings in the code use these units.

## Hamiltonian structure (Appendix B)

The total Hamiltonian is built as:

```
H(t) = Σ_k H_k(t) + Σ_c [ H_qc,c(t) + H_qq,c(t) ] + H_drives(t)
```

### 1) Mode Hamiltonian (Duffing transmon)

Each mode `k ∈ {qubits, couplers}` is modeled as a Duffing oscillator derived
from the cosine expansion in Appendix B:

```
H_k = ω_k a†_k a_k - (K_k / 2) a†_k a†_k a_k a_k
```

with

```
ω_k = sqrt(8 EJ_k EC_k) - EC_k (1 + ξ_k/4)
K_k = EC_k (1 + 9 ξ_k / 16) / 2
ξ_k = sqrt(2 EC_k / EJ_k)
```

These match Eq. (B19)–(B20). In code:

- `ω_k` is computed by `_omega_from_phi(...)`
- `K_k` is computed by `_kerr_from_phi(...)`
- The Kerr term is added with a **minus sign**, i.e. `-K_k * a†a†aa`

### 2) Flux‑to‑frequency mapping

Each tunable element is a SQUID with Josephson energy:

```
EJ_k(φ) = sqrt(EJ_small^2 + EJ_large^2 + 2 EJ_small EJ_large cos φ)
```

The reduced external flux is `φ = 2π Φ / Φ0`. Flux pulses modify `φ` as:

```
φ(t) = φ_ext + Re[ pulse(t) ]
```

The code uses this directly to recompute `ω_k(t)` and `K_k(t)` when a flux
pulse is provided on that element. This matches the paper’s transmon
approximation and flux tuning (Appendix B).

### 3) Coupling terms (charge‑coupling form)

The paper Hamiltonian (Eq. B19) includes **charge‑coupling** between modes:

```
H_qc = g_jc ( a_j a_c† + a†_j a_c − a_j a_c − a†_j a†_c )
H_qq = g_12 ( a_1 a_2† + a†_1 a_2 − a_1 a_2 − a†_1 a†_2 )
```

This is the exact structure implemented in `SuperconductingDevice` for:

- **Qubit–coupler coupling** on each link: `g_couplings = (g_left, g_right)`
- **Direct qubit–qubit coupling** on each link: `g_direct`

These terms are constructed using embedded ladder operators and the exact
operator combination `(a a† + a† a − a a − a† a†)` from the paper.

### 4) Coupling strengths from coupling energies

Appendix B gives:

```
g_jc = E_jc * sqrt(2) * (EJ_j/EC_j * EJ_c/EC_c)^(1/4) * [1 − (ξ_j + ξ_c)/8]
g_12 = E_12 * sqrt(2) * (EJ_1/EC_1 * EJ_2/EC_2)^(1/4) * [1 − (ξ_1 + ξ_2)/8]
```

In the code:

- `_g_from_E(...)` implements this mapping.
- If you provide `E_couplings` / `E_direct`, the code computes `g` from these.
- If you provide `g_couplings` / `g_direct` directly, they are used as given.

### 5) Flux‑dependent coupling scaling Υ(Φec)

The paper introduces a flux‑dependent coupling prefactor:

```
Υ(Φec) = [ EJ_c(0) / EJ_c(Φec) ]^(1/4)
```

In code this is enabled by `use_flux_coupling_scaling=True` and is applied when
coupler flux pulses are present. It multiplies the `g_jc(t)` computed from
`E_jc` and the instantaneous EJ.

### 6) Drive terms

Qubit drive lines are modeled as in the existing framework:

```
H_drive(t) = Re[s(t) e^{i ω_d t}] X_q + Im[s(t) e^{i ω_d t}] Y_q
```

The drive envelope `s(t)` comes from `GaussianPulse` (or `SquarePulse`).
In the rotating frame, `ω_d` is interpreted relative to `ref_qubit_freqs`.

## How the code constructs H(t)

At runtime `construct_h(...)`:

1) Builds **time‑dependent** `ω_k(t)` and `K_k(t)` for each mode using EJ(φ).
2) Adds **qubit–coupler** and **direct qubit–qubit** coupling terms with the
   charge‑coupling operator structure from Eq. (B19).
3) Adds **drive terms** for any listed microwave pulses.

All time‑dependent components are created via `dynamiqs.modulated(...)` so the
system can be solved with JAX/dynamiqs time‑dependent solvers.

## Symmetric vs asymmetric configurations

The paper’s “symmetric” and “asymmetric” layouts are encoded by:

- The **signs** of `g1c` and `g2c`
- The sign of `g_direct`
- The operating **coupler frequency range** (below or above qubits)

In `params_debug.py`:

- `PAPER_PARAMS_SYMMETRIC` uses both `g1c` and `g2c` negative.
- `PAPER_PARAMS_ASYMMETRIC` uses opposite signs for `g1c` and `g2c`.

These sign choices allow the effective coupling to cancel `g_direct` at a
tunable coupler frequency, as described in the main text.

## Known gaps vs paper data

The paper does **not** report EJ asymmetry or coupler anharmonicity. When
only `f01` is provided, the code derives `EJ` assuming a symmetric SQUID and
uses the transmon approximation. Coupler `EC` is set to a placeholder value.

If you have measured `EC`, EJ asymmetry, or coupler anharmonicity, you should
set them explicitly to align quantitatively with the paper.

## File pointers

- `qualibration_graphs/superconducting/validation_utils/time_dynamics/device.py`
  - `construct_h`: full time‑dependent Hamiltonian construction
  - `_omega_from_phi`, `_kerr_from_phi`, `_g_from_E`: paper formulas
  - coupling operator structure for `g_jc` and `g_12`
- `qualibration_graphs/superconducting/validation_utils/time_dynamics/params_debug.py`
  - `PAPER_PARAMS_SYMMETRIC` / `PAPER_PARAMS_ASYMMETRIC` from the PDF
  - `DEBUG_PARAMS` for the current working demo values
