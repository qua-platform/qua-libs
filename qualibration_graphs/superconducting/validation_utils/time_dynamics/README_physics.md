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

## Second‑quantization primer (operators used in code)

Each transmon mode is represented in a truncated harmonic‑oscillator basis
`|0>, |1>, |2>, ...` with `levels` states. The ladder operators are:

- `a |n> = sqrt(n) |n-1>`
- `a† |n> = sqrt(n+1) |n+1>`

From these we build:

- Number operator: `n = a† a`
- Kerr (Duffing) operator: `a† a† a a`
- Quadrature operators:
  - `X = a + a†`
  - `Y = -i (a - a†)`

In the full system, each operator is embedded into the tensor product space
by taking a Kronecker product with identity operators for all other modes.
This is what `_embed_op(...)` does: it places a single‑mode operator at a
chosen index and identities elsewhere.

### Truncation note (important approximation)

Each mode is **truncated** to a finite number of levels (`levels`, default 3).
This means the Hilbert space is limited to `levels^N_modes` basis states.
The approximation is accurate when the dynamics stay in the lowest few
levels (typical for transmons under weak/medium drives). Strong drives or
large couplings can populate higher levels that are not represented.

In code, truncation happens when we construct the local basis and ladder
operators with `dq.destroy(levels)` and `dq.create(levels)`. Increasing
`levels` improves accuracy but increases the Hilbert‑space dimension and
runtime.

## How flux tunes the transmon

A SQUID replaces a single Josephson junction with two junctions in a loop.
The effective Josephson energy is flux‑dependent:

```
EJ(phi) = sqrt(EJ_small^2 + EJ_large^2 + 2 EJ_small EJ_large cos(phi))
```

The reduced flux is `phi = 2 pi Phi / Phi0`. For a **symmetric SQUID**,
`EJ_small = EJ_large`, and the frequency tuning is widest.

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

### How the coupling operator is built

For a pair of modes `i` and `j`, the code constructs:

```
O_ij = a_i a_j† + a_i† a_j - a_i a_j - a_i† a_j†
```

This expression is a compact form of the **charge‑charge coupling** in the
paper. In operator language, the charge operator is proportional to `(a - a†)`,
so a bilinear charge term produces the mix of `a a`, `a a†`, `a† a`, `a† a†`
terms. The sign pattern is important and matches Eq. (B19).

In `device.py`, these are built as:

- `op_lc` for left qubit–coupler
- `op_rc` for right qubit–coupler
- `op_lr` for direct qubit–qubit

Each is embedded into the full Hilbert space with `_embed_op(...)` and then
summed into the full Hamiltonian with the appropriate coupling rate `g`.

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

1) **Initialize** a zero Hamiltonian for the full tensor product space.
2) **Compute per‑mode operators** (`a`, `a†`, `n`, `a†a†aa`, `X`, `Y`) and embed
   them into the full Hilbert space (cached in `__post_init__`).
3) **Add mode terms**:
   - If a flux pulse is attached to a mode, define time‑dependent functions
     `omega(t)` and `Kerr(t)` and use `dynamiqs.modulated(...)`.
   - Otherwise, add static `omega * n` and `-Kerr * a†a†aa` terms.
4) **Add coupling terms**:
   - Build `op_lc`, `op_rc`, and `op_lr` for the operator structure
     `a a† + a† a − a a − a† a†`.
   - If coupling energies `E_couplings` are given, compute `g(t)` from the
     instantaneous EJ values. Otherwise use the provided `g` values.
5) **Add drive terms**:
   - For each drive pulse, build the `X` and `Y` quadrature terms and modulate
     them by the envelope and carrier frequency.

Because each piece is built as a `dynamiqs.TimeQArray`, the solver can evaluate
the Hamiltonian at any time `t` while integrating the dynamics.

## Worked two‑mode example (concrete operator matrices)

For intuition, consider two modes (i and j) truncated to 2 levels each,
so the basis is `{ |00>, |01>, |10>, |11> }`. In this basis:

```
a = [[0, 1],
     [0, 0]]
```

The embedded operators are:

```
a_i = a ⊗ I
a_j = I ⊗ a
```

The charge‑coupling operator in the paper is:

```
O_ij = a_i a_j† + a_i† a_j - a_i a_j - a_i† a_j†
```

In the 2‑level truncation this becomes:

```
O_ij =
[[ 0,  0,  0, -1],
 [ 0,  0,  1,  0],
 [ 0,  1,  0,  0],
 [-1,  0,  0,  0]]
```

You can see the physical meaning:

- The `a_i† a_j` and `a_i a_j†` terms swap `|01>` and `|10>` (exchange).
- The `a_i a_j` and `a_i† a_j†` terms couple `|00>` to `|11>` (pair creation/annihilation).

The full Hamiltonian combines this operator with a coupling rate `g_ij`.
In the weak‑coupling regime, the exchange part dominates the effective
interaction between `|01>` and `|10>`, while the pair terms shift energies.

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
