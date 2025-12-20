"""
Superconducting qubit device for time dynamics simulations.

This module provides a framework for simulating superconducting qubit systems
with flux-tunable transmons and couplers. The Hamiltonian matches the floating
tunable-coupler model from Sete et al. (2021), using charge-coupling terms and
transmon cosine expansion to fourth order.

Classes
-------
SuperconductingDevice
    Device class for multi-level transmon qubits and tunable couplers with
    charge-mediated coupling, supporting per-line time-dependent drives and
    flux bias.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence, Tuple, Optional

import dynamiqs as dq
import jax.numpy as jnp

from qualibration_graphs.validation_utils.time_dynamics import Pulse, GaussianPulse, kron_n


@dataclass(frozen=True)
class SuperconductingDevice:
    """
    Multi-level transmon chain with tunable couplers (paper-aligned).

    The model follows the floating tunable-coupler Hamiltonian from the paper,
    with charge-charge coupling terms and Duffing-like transmon nonlinearities.
    Qubits are indexed 0..N-1, couplers are indexed 0..N-2 and placed between
    adjacent qubits (k connects qubits k and k+1).

    Mode Hamiltonian (paper, Eq. B19/B20):
        H_k = ω_k a†_k a_k - (K_k/2) a†_k a†_k a_k a_k
    with ω_k and K_k computed from EJ_k(φ) and EC_k.

    Couplings (paper, Eq. B19):
        H_int = g_jc (a_j a_c† + a†_j a_c - a_j a_c - a†_j a†_c)
             + g_12 (a_1 a_2† + a†_1 a_2 - a_1 a_2 - a†_1 a†_2)

    Time-dependent controls:
        - Qubit drive lines: complex envelopes s(t) on each qubit with optional
          carrier frequency drive_freq.
        - Qubit flux lines: reduced-flux pulses phi_e(t) mapped to EJ(φ) and
          omega(φ), including flux-dependent coupling if enabled.
        - Coupler flux lines: reduced-flux pulses phi_e(t) mapped to EJ(φ) and
          omega(φ), including flux-dependent coupling if enabled.

    Notes
    -----
    The symmetric/asymmetric operating regimes are captured via the signs of
    g_jc and g_12 and the tunable-coupler frequency placement.
    """
    n_qubits: int
    levels: int = 3
    frame: Literal["lab", "rot"] = "lab"

    # Transmon parameters (length n_qubits / n_qubits-1)
    qubit_freqs: Optional[Sequence[float]] = None
    qubit_anharm: Optional[Sequence[float]] = None
    coupler_freqs: Optional[Sequence[float]] = None
    coupler_anharm: Optional[Sequence[float]] = None
    qubit_EJ_small: Optional[Sequence[float]] = None
    qubit_EJ_large: Optional[Sequence[float]] = None
    qubit_EC: Optional[Sequence[float]] = None
    qubit_phi_ext: Optional[Sequence[float]] = None  # reduced flux phi_e = 2*pi*Phi/Phi0
    coupler_EJ_small: Optional[Sequence[float]] = None
    coupler_EJ_large: Optional[Sequence[float]] = None
    coupler_EC: Optional[Sequence[float]] = None
    coupler_phi_ext: Optional[Sequence[float]] = None  # reduced flux phi_e

    # Coupling strengths for each coupler: (g_left, g_right), and direct g between qubits
    g_couplings: Sequence[Tuple[float, float]] = ()
    g_direct: Optional[Sequence[float]] = None

    # Optional coupling energies (E_jc, E_12) to compute g via paper formulas
    E_couplings: Optional[Sequence[Tuple[float, float]]] = None
    E_direct: Optional[Sequence[float]] = None

    # Optional rotating-frame references
    ref_qubit_freqs: Optional[Sequence[float]] = None
    ref_coupler_freqs: Optional[Sequence[float]] = None

    # Flux-dependent coupling scaling (Υ(Φec) factor)
    use_flux_coupling_scaling: bool = True

    def __post_init__(self) -> None:
        if self.n_qubits < 2:
            raise ValueError("n_qubits must be >= 2")

        n_couplers = self.n_qubits - 1

        self._normalize_transmon_params()
        self._validate_flux_params()

        g_couplings = list(self.g_couplings)
        if len(g_couplings) != n_couplers:
            raise ValueError("g_couplings length must be n_qubits - 1")
        if g_couplings and not isinstance(g_couplings[0], (tuple, list)):
            g_couplings = [(float(g), float(g)) for g in g_couplings]
        object.__setattr__(self, "g_couplings", tuple((float(g0), float(g1)) for g0, g1 in g_couplings))

        if self.ref_qubit_freqs is None:
            object.__setattr__(self, "ref_qubit_freqs", (0.0,) * self.n_qubits)
        if self.ref_coupler_freqs is None:
            object.__setattr__(self, "ref_coupler_freqs", (0.0,) * n_couplers)

        if self.g_direct is None:
            object.__setattr__(self, "g_direct", (0.0,) * n_couplers)

        # Precompute embedded operators for each mode
        a = dq.destroy(self.levels)
        adag = dq.create(self.levels)
        n_local = adag @ a
        kerr_local = adag @ adag @ a @ a
        x_local = a + adag
        y_local = -1j * (a - adag)

        a_ops = []
        adag_ops = []
        n_ops = []
        kerr_ops = []
        x_ops = []
        y_ops = []

        for which in range(self.n_modes):
            a_ops.append(self._embed_op(a, which))
            adag_ops.append(self._embed_op(adag, which))
            n_ops.append(self._embed_op(n_local, which))
            kerr_ops.append(self._embed_op(kerr_local, which))
            x_ops.append(self._embed_op(x_local, which))
            y_ops.append(self._embed_op(y_local, which))

        object.__setattr__(self, "_a_ops", tuple(a_ops))
        object.__setattr__(self, "_adag_ops", tuple(adag_ops))
        object.__setattr__(self, "_n_ops", tuple(n_ops))
        object.__setattr__(self, "_kerr_ops", tuple(kerr_ops))
        object.__setattr__(self, "_x_ops", tuple(x_ops))
        object.__setattr__(self, "_y_ops", tuple(y_ops))

    @property
    def n_couplers(self) -> int:
        return self.n_qubits - 1

    @property
    def n_modes(self) -> int:
        return self.n_qubits + self.n_couplers

    def _coupler_index(self, k: int) -> int:
        return self.n_qubits + k

    def _embed_op(self, op: dq.QArray, which: int) -> dq.QArray:
        ops = [dq.eye(self.levels) for _ in range(self.n_modes)]
        ops[which] = op
        return kron_n(ops)

    def _pulse_callable(self, pulse: Pulse | Callable[[jnp.ndarray], jnp.ndarray]):
        return pulse.timecallable() if hasattr(pulse, "timecallable") else pulse

    def _validate_flux_params(self) -> None:
        if self.qubit_EJ_small is None or self.qubit_EJ_large is None or self.qubit_EC is None or self.qubit_phi_ext is None:
            raise ValueError("qubit_EJ_small/large, qubit_EC, and qubit_phi_ext are required")
        if self.coupler_EJ_small is None or self.coupler_EJ_large is None or self.coupler_EC is None or self.coupler_phi_ext is None:
            raise ValueError("coupler_EJ_small/large, coupler_EC, and coupler_phi_ext are required")

        if len(self.qubit_EJ_small) != self.n_qubits or len(self.qubit_EJ_large) != self.n_qubits:
            raise ValueError("qubit_EJ_small/large length must match n_qubits")
        if len(self.qubit_EC) != self.n_qubits or len(self.qubit_phi_ext) != self.n_qubits:
            raise ValueError("qubit_EC and qubit_phi_ext length must match n_qubits")

        if len(self.coupler_EJ_small) != self.n_couplers or len(self.coupler_EJ_large) != self.n_couplers:
            raise ValueError("coupler_EJ_small/large length must be n_qubits - 1")
        if len(self.coupler_EC) != self.n_couplers or len(self.coupler_phi_ext) != self.n_couplers:
            raise ValueError("coupler_EC and coupler_phi_ext length must be n_qubits - 1")

        if len(self.qubit_EJ_small) != self.n_qubits or len(self.qubit_EJ_large) != self.n_qubits:
            raise ValueError("qubit_EJ_small/large length must match n_qubits")
        if len(self.qubit_EC) != self.n_qubits or len(self.qubit_phi_ext) != self.n_qubits:
            raise ValueError("qubit_EC and qubit_phi_ext length must match n_qubits")

        if len(self.coupler_EJ_small) != self.n_couplers or len(self.coupler_EJ_large) != self.n_couplers:
            raise ValueError("coupler_EJ_small/large length must be n_qubits - 1")
        if len(self.coupler_EC) != self.n_couplers or len(self.coupler_phi_ext) != self.n_couplers:
            raise ValueError("coupler_EC and coupler_phi_ext length must be n_qubits - 1")

        if self.E_couplings is not None and len(self.E_couplings) != self.n_couplers:
            raise ValueError("E_couplings length must be n_qubits - 1 when provided")
        if self.E_direct is not None and len(self.E_direct) != self.n_couplers:
            raise ValueError("E_direct length must be n_qubits - 1 when provided")

    def _ej_from_phi(self, EJ_small: jnp.ndarray, EJ_large: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(EJ_small ** 2 + EJ_large ** 2 + 2.0 * EJ_small * EJ_large * jnp.cos(phi))

    def _derive_ej_from_omega(self, omega: float, EC: float) -> float:
        # Invert ω ≈ sqrt(8 EJ EC) - EC (transmon approximation).
        return float((omega + EC) ** 2 / (8.0 * EC))

    def _normalize_transmon_params(self) -> None:
        if self.qubit_EC is None and self.qubit_anharm is not None:
            object.__setattr__(self, "qubit_EC", tuple(float(a) for a in self.qubit_anharm))
        if self.coupler_EC is None and self.coupler_anharm is not None:
            object.__setattr__(self, "coupler_EC", tuple(float(a) for a in self.coupler_anharm))

        if self.qubit_EJ_small is None and self.qubit_EJ_large is None and self.qubit_freqs is not None and self.qubit_EC is not None:
            EJ_max = tuple(self._derive_ej_from_omega(w, ec) for w, ec in zip(self.qubit_freqs, self.qubit_EC))
            object.__setattr__(self, "qubit_EJ_small", tuple(0.5 * ej for ej in EJ_max))
            object.__setattr__(self, "qubit_EJ_large", tuple(0.5 * ej for ej in EJ_max))
            if self.qubit_phi_ext is None:
                object.__setattr__(self, "qubit_phi_ext", (0.0,) * len(EJ_max))

        if self.coupler_EJ_small is None and self.coupler_EJ_large is None and self.coupler_freqs is not None and self.coupler_EC is not None:
            EJ_max = tuple(self._derive_ej_from_omega(w, ec) for w, ec in zip(self.coupler_freqs, self.coupler_EC))
            object.__setattr__(self, "coupler_EJ_small", tuple(0.5 * ej for ej in EJ_max))
            object.__setattr__(self, "coupler_EJ_large", tuple(0.5 * ej for ej in EJ_max))
            if self.coupler_phi_ext is None:
                object.__setattr__(self, "coupler_phi_ext", (0.0,) * len(EJ_max))
    def _xi_from_ej_ec(self, EJ: jnp.ndarray, EC: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(2.0 * EC / EJ)

    def _omega_from_phi(self, EJ_small: jnp.ndarray, EJ_large: jnp.ndarray, EC: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
        EJ = self._ej_from_phi(EJ_small, EJ_large, phi)
        xi = self._xi_from_ej_ec(EJ, EC)
        return jnp.sqrt(8.0 * EJ * EC) - EC * (1.0 + xi / 4.0)

    def _kerr_from_phi(self, EJ_small: jnp.ndarray, EJ_large: jnp.ndarray, EC: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
        EJ = self._ej_from_phi(EJ_small, EJ_large, phi)
        xi = self._xi_from_ej_ec(EJ, EC)
        return 0.5 * EC * (1.0 + 9.0 * xi / 16.0)

    def _g_from_E(self, Eij: jnp.ndarray, EJ_i: jnp.ndarray, EC_i: jnp.ndarray, EJ_j: jnp.ndarray, EC_j: jnp.ndarray) -> jnp.ndarray:
        xi_i = self._xi_from_ej_ec(EJ_i, EC_i)
        xi_j = self._xi_from_ej_ec(EJ_j, EC_j)
        scale = jnp.sqrt(2.0) * (EJ_i / EC_i * EJ_j / EC_j) ** 0.25
        return Eij * scale * (1.0 - 0.125 * (xi_i + xi_j))

    def construct_h(
        self,
        drives: Sequence[tuple[int, GaussianPulse]] = (),
        qubit_flux: Sequence[tuple[int, Pulse]] = (),
        coupler_flux: Sequence[tuple[int, Pulse]] = (),
    ) -> dq.TimeQArray:
        """
        Build H(t) = H_static + H_flux_qubits(t) + H_flux_couplers(t) + H_drives(t).
        """
        Ht = dq.constant(0.0 * kron_n([dq.eye(self.levels)] * self.n_modes))

        qubit_flux_map = {i: self._pulse_callable(p) for i, p in qubit_flux}
        coupler_flux_map = {i: self._pulse_callable(p) for i, p in coupler_flux}

        # Mode terms (omega and Kerr), possibly flux-modulated
        for i in range(self.n_qubits):
            EJ_s = jnp.asarray(self.qubit_EJ_small[i])
            EJ_l = jnp.asarray(self.qubit_EJ_large[i])
            EC = jnp.asarray(self.qubit_EC[i])
            phi0 = jnp.asarray(self.qubit_phi_ext[i])
            n_op = self._n_ops[i]
            kerr_op = self._kerr_ops[i]

            if i in qubit_flux_map:
                f = qubit_flux_map[i]

                def omega_t(t, EJ_s=EJ_s, EJ_l=EJ_l, EC=EC, phi0=phi0, f=f):
                    omega = self._omega_from_phi(EJ_s, EJ_l, EC, phi0 + jnp.real(f(t)))
                    if self.frame == "rot":
                        omega = omega - jnp.asarray(self.ref_qubit_freqs[i])
                    return omega

                def kerr_t(t, EJ_s=EJ_s, EJ_l=EJ_l, EC=EC, phi0=phi0, f=f):
                    return -self._kerr_from_phi(EJ_s, EJ_l, EC, phi0 + jnp.real(f(t)))

                Ht = Ht + dq.modulated(omega_t, n_op)
                Ht = Ht + dq.modulated(kerr_t, kerr_op)
            else:
                omega0 = self._omega_from_phi(EJ_s, EJ_l, EC, phi0)
                kerr0 = -self._kerr_from_phi(EJ_s, EJ_l, EC, phi0)
                if self.frame == "rot":
                    omega0 = omega0 - jnp.asarray(self.ref_qubit_freqs[i])
                Ht = Ht + omega0 * n_op + kerr0 * kerr_op

        for k in range(self.n_couplers):
            EJ_s = jnp.asarray(self.coupler_EJ_small[k])
            EJ_l = jnp.asarray(self.coupler_EJ_large[k])
            EC = jnp.asarray(self.coupler_EC[k])
            phi0 = jnp.asarray(self.coupler_phi_ext[k])
            idx = self._coupler_index(k)
            n_op = self._n_ops[idx]
            kerr_op = self._kerr_ops[idx]

            if k in coupler_flux_map:
                f = coupler_flux_map[k]

                def omega_t(t, EJ_s=EJ_s, EJ_l=EJ_l, EC=EC, phi0=phi0, f=f):
                    omega = self._omega_from_phi(EJ_s, EJ_l, EC, phi0 + jnp.real(f(t)))
                    if self.frame == "rot":
                        omega = omega - jnp.asarray(self.ref_coupler_freqs[k])
                    return omega

                def kerr_t(t, EJ_s=EJ_s, EJ_l=EJ_l, EC=EC, phi0=phi0, f=f):
                    return -self._kerr_from_phi(EJ_s, EJ_l, EC, phi0 + jnp.real(f(t)))

                Ht = Ht + dq.modulated(omega_t, n_op)
                Ht = Ht + dq.modulated(kerr_t, kerr_op)
            else:
                omega0 = self._omega_from_phi(EJ_s, EJ_l, EC, phi0)
                kerr0 = -self._kerr_from_phi(EJ_s, EJ_l, EC, phi0)
                if self.frame == "rot":
                    omega0 = omega0 - jnp.asarray(self.ref_coupler_freqs[k])
                Ht = Ht + omega0 * n_op + kerr0 * kerr_op

        # Couplings (direct and via couplers)
        for k in range(self.n_couplers):
            q_left = k
            q_right = k + 1
            c_idx = self._coupler_index(k)

            a_l = self._a_ops[q_left]
            ad_l = self._adag_ops[q_left]
            a_r = self._a_ops[q_right]
            ad_r = self._adag_ops[q_right]
            a_c = self._a_ops[c_idx]
            ad_c = self._adag_ops[c_idx]

            op_lc = (a_l @ ad_c + ad_l @ a_c - a_l @ a_c - ad_l @ ad_c).asdense()
            op_rc = (a_r @ ad_c + ad_r @ a_c - a_r @ a_c - ad_r @ ad_c).asdense()
            op_lr = (a_l @ ad_r + ad_l @ a_r - a_l @ a_r - ad_l @ ad_r).asdense()

            g12 = jnp.asarray(self.g_direct[k]).reshape(())
            Ht = Ht + g12 * op_lr

            g_left, g_right = self.g_couplings[k]
            g_left = jnp.asarray(g_left).reshape(())
            g_right = jnp.asarray(g_right).reshape(())

            if self.E_couplings is not None:
                EJ_l0 = self._ej_from_phi(
                    jnp.asarray(self.qubit_EJ_small[q_left]),
                    jnp.asarray(self.qubit_EJ_large[q_left]),
                    jnp.asarray(self.qubit_phi_ext[q_left]),
                )
                EJ_r0 = self._ej_from_phi(
                    jnp.asarray(self.qubit_EJ_small[q_right]),
                    jnp.asarray(self.qubit_EJ_large[q_right]),
                    jnp.asarray(self.qubit_phi_ext[q_right]),
                )
                EJ_c0 = self._ej_from_phi(
                    jnp.asarray(self.coupler_EJ_small[k]),
                    jnp.asarray(self.coupler_EJ_large[k]),
                    jnp.asarray(self.coupler_phi_ext[k]),
                )

                E_left, E_right = self.E_couplings[k]
                E_left = jnp.asarray(E_left)
                E_right = jnp.asarray(E_right)

                f_c = coupler_flux_map.get(k)
                f_l = qubit_flux_map.get(q_left)
                f_r = qubit_flux_map.get(q_right)

                def EJ_l_t(t, EJ_s=jnp.asarray(self.qubit_EJ_small[q_left]), EJ_l=jnp.asarray(self.qubit_EJ_large[q_left]), phi0=jnp.asarray(self.qubit_phi_ext[q_left]), f_l=f_l):
                    if f_l is None:
                        return EJ_l0
                    return self._ej_from_phi(EJ_s, EJ_l, phi0 + jnp.real(f_l(t)))

                def EJ_r_t(t, EJ_s=jnp.asarray(self.qubit_EJ_small[q_right]), EJ_l=jnp.asarray(self.qubit_EJ_large[q_right]), phi0=jnp.asarray(self.qubit_phi_ext[q_right]), f_r=f_r):
                    if f_r is None:
                        return EJ_r0
                    return self._ej_from_phi(EJ_s, EJ_l, phi0 + jnp.real(f_r(t)))

                def EJ_c_t(t, EJ_s=jnp.asarray(self.coupler_EJ_small[k]), EJ_l=jnp.asarray(self.coupler_EJ_large[k]), phi0=jnp.asarray(self.coupler_phi_ext[k]), f_c=f_c):
                    if f_c is None:
                        return EJ_c0
                    return self._ej_from_phi(EJ_s, EJ_l, phi0 + jnp.real(f_c(t)))

                def g_left_t(t, E_left=E_left, EC_l=jnp.asarray(self.qubit_EC[q_left]), EC_c=jnp.asarray(self.coupler_EC[k])):
                    EJ_l = EJ_l_t(t)
                    EJ_c = EJ_c_t(t)
                    g = self._g_from_E(E_left, EJ_l, EC_l, EJ_c, EC_c)
                    if self.use_flux_coupling_scaling:
                        g = g * (EJ_c0 / EJ_c) ** 0.25
                    return g

                def g_right_t(t, E_right=E_right, EC_r=jnp.asarray(self.qubit_EC[q_right]), EC_c=jnp.asarray(self.coupler_EC[k])):
                    EJ_r = EJ_r_t(t)
                    EJ_c = EJ_c_t(t)
                    g = self._g_from_E(E_right, EJ_r, EC_r, EJ_c, EC_c)
                    if self.use_flux_coupling_scaling:
                        g = g * (EJ_c0 / EJ_c) ** 0.25
                    return g

                if f_c is not None or f_l is not None or f_r is not None:
                    Ht = Ht + dq.modulated(g_left_t, op_lc)
                    Ht = Ht + dq.modulated(g_right_t, op_rc)
                else:
                    g_l = jnp.asarray(self._g_from_E(E_left, EJ_l0, jnp.asarray(self.qubit_EC[q_left]), EJ_c0, jnp.asarray(self.coupler_EC[k]))).reshape(())
                    g_r = jnp.asarray(self._g_from_E(E_right, EJ_r0, jnp.asarray(self.qubit_EC[q_right]), EJ_c0, jnp.asarray(self.coupler_EC[k]))).reshape(())
                    Ht = Ht + g_l * op_lc + g_r * op_rc
            else:
                if self.use_flux_coupling_scaling:
                    f_c = coupler_flux_map.get(k)

                    def scale_t(t, EJ_s=jnp.asarray(self.coupler_EJ_small[k]), EJ_l=jnp.asarray(self.coupler_EJ_large[k]), phi0=jnp.asarray(self.coupler_phi_ext[k]), f_c=f_c, EJ_c0=self._ej_from_phi(jnp.asarray(self.coupler_EJ_small[k]), jnp.asarray(self.coupler_EJ_large[k]), jnp.asarray(self.coupler_phi_ext[k]))):
                        if f_c is None:
                            return 1.0
                        EJ_c = self._ej_from_phi(EJ_s, EJ_l, phi0 + jnp.real(f_c(t)))
                        return (EJ_c0 / EJ_c) ** 0.25

                    if f_c is None:
                        Ht = Ht + g_left * op_lc + g_right * op_rc
                    else:
                        Ht = Ht + dq.modulated(lambda t, g=g_left: g * scale_t(t), op_lc)
                        Ht = Ht + dq.modulated(lambda t, g=g_right: g * scale_t(t), op_rc)
                else:
                    Ht = Ht + g_left * op_lc + g_right * op_rc

        # Qubit drive lines
        for which, pulse in drives:
            for f, op in self._drive_terms(which, pulse):
                Ht = Ht + dq.modulated(f, op)

        return Ht

    def _static_h(self) -> dq.QArray:
        return 0.0 * kron_n([dq.eye(self.levels)] * self.n_modes)

    def _drive_terms(
        self, which: int, pulse: GaussianPulse
    ) -> Sequence[tuple[Callable[[jnp.ndarray], jnp.ndarray], dq.QArray]]:
        s_base = self._pulse_callable(pulse)
        drive_freq = getattr(pulse, "drive_freq", None)

        if self.frame == "rot":
            wref = jnp.asarray(self.ref_qubit_freqs[which])
            omega_eff = jnp.asarray(0.0) if drive_freq is None else drive_freq - wref
        else:
            omega_eff = jnp.asarray(0.0) if drive_freq is None else drive_freq

        def wobble(t):
            return jnp.exp(1j * (omega_eff * t))

        Xj = self._x_ops[which]
        Yj = self._y_ops[which]

        def s_re(t):
            return jnp.real(s_base(t) * wobble(t))

        def s_im(t):
            return jnp.imag(s_base(t) * wobble(t))

        return ((s_re, Xj), (s_im, Yj))
