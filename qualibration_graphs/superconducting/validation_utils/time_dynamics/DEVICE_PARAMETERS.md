# SuperconductingDevice Parameters (device.py)

This note summarizes the parameters required by
`qualibration_graphs/superconducting/validation_utils/time_dynamics/device.py`,
with emphasis on the alternative ways to specify couplings.

## Core required inputs

These are required for *every* device instance:

- `n_qubits` (int): Number of qubits. Must be >= 2.
- `levels` (int): Truncation level per mode (default 3).
- `frame` ("lab" or "rot"): Whether to build the Hamiltonian in the lab frame
  or a rotating frame.

### Flux-tunable transmon parameters (required)

Each qubit and coupler is modeled as a SQUID-tunable transmon. The following
arrays must be provided and have the correct lengths:

Qubits (length `n_qubits`):
- `qubit_EJ_small`
- `qubit_EJ_large`
- `qubit_EC`
- `qubit_phi_ext` (reduced flux, `phi = 2*pi*Phi/Phi0`)

Couplers (length `n_qubits - 1`):
- `coupler_EJ_small`
- `coupler_EJ_large`
- `coupler_EC`
- `coupler_phi_ext` (reduced flux)

These values are used to compute mode frequencies and Kerr nonlinearity via:
`_omega_from_phi(...)` and `_kerr_from_phi(...)`.

## Optional frequency parameters

You can specify fixed frequencies/anharmonicities, but note the current
implementation only uses the EJ/EC/phi arrays to compute omega and Kerr.
The following are present on the dataclass but not required:

- `qubit_freqs`, `qubit_anharm`
- `coupler_freqs`, `coupler_anharm`

## USER_PARAMS_SYMMETRIC derivation (paper-backed)

The `USER_PARAMS_SYMMETRIC` set in
`qualibration_graphs/superconducting/validation_utils/time_dynamics/params_debug.py`
is derived from the paper
`qualibration_graphs/superconducting/validation_utils/time_dynamics/2103.07030v2.pdf`
using the symmetric-SQUID transmon approximation at Phi = 0:

- Reported values: f01(q1), f01(qc), f01(q2) and eta/2pi from the paper.
- Convert anharmonicity to EC: EC ≈ |eta| (GHz).
- Convert max frequency to EJ0 (symmetric SQUID: EJ_small = EJ_large = EJ0):
  EJ0 = (omega_max + EC)^2 / (16 EC), with omega_max = 2*pi*f01 if using
  angular-frequency conventions, or f01 if using GHz as angular units.
- Set phi_ext = 0.0 for all modes (Phi = 0 gives max frequency).
- Use reported couplings g: sqrt(g1c g2c)/2pi and g12/2pi from the paper,
  converted to the same GHz units as the Hamiltonian, and applied symmetrically
  (g1c = g2c).

The coupler anharmonicity is not reported in the paper, so `coupler_EC` is set
to a typical transmon value (0.20 GHz) for this parameter set.

## Coupling parameterization options

There are **two supported ways** to specify couplings between modes.

### Option A: Direct coupling rates (g)

Provide coupling rates directly:

- `g_couplings`: sequence of `(g_left, g_right)` for each coupler link
  (length `n_qubits - 1`). Each entry couples:
  - left qubit <-> coupler
  - right qubit <-> coupler
- `g_direct`: sequence of direct qubit-qubit couplings (length `n_qubits - 1`)

If `g_couplings` is provided as a list of scalars, it is expanded to
`(g, g)` for each link.

Optional modifier:
- `use_flux_coupling_scaling` (bool, default True):
  if coupler flux pulses are provided, the coupling is scaled by
  `(EJ_c0 / EJ_c(t))**0.25` even when using direct g's.

### Option B: Coupling energies (E)

Provide coupling **energies** and let the code convert them into `g(t)` using
Appendix B expressions from the paper:

- `E_couplings`: sequence of `(E_left, E_right)` for each coupler link
- `E_direct`: sequence of direct qubit-qubit coupling energies

When `E_couplings` is provided:
- `g_couplings` is allowed to be empty or set to zeros, because the code
  derives `g(t)` from `E_couplings`.
- The derived `g(t)` is *flux-dependent* if a qubit or coupler has a flux
  pulse applied.
- The same `use_flux_coupling_scaling` factor is applied when enabled.

## Rotating-frame references

If `frame="rot"`, the code subtracts reference frequencies from the
time-dependent omega terms.

Optional arrays:
- `ref_qubit_freqs`: length `n_qubits`
- `ref_coupler_freqs`: length `n_qubits - 1`

If not provided, these are set to zeros.

## Time-dependent controls

These are not stored on the device but passed to `construct_h(...)`:

- `drives`: list of `(qubit_index, GaussianPulse)` entries
- `qubit_flux`: list of `(qubit_index, Pulse)` entries
- `coupler_flux`: list of `(coupler_index, Pulse)` entries

Flux pulses modify the local reduced flux `phi = phi_ext + Re[pulse(t)]`,
which in turn changes `omega(t)`, `Kerr(t)`, and (if enabled) `g(t)`.

## Summary of required vs optional

Required (always):
- `n_qubits`, `levels`, `frame`
- All EJ/EC/phi arrays for qubits and couplers
- A coupling specification (Option A or B)

Optional:
- `ref_qubit_freqs`, `ref_coupler_freqs` (used only in rotating frame)
- `use_flux_coupling_scaling`
- The "freq/anharm" arrays (currently unused for Hamiltonian construction)
