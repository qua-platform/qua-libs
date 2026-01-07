"""
Superconducting qubit device for time dynamics simulations with rotating frame and RWA.

This module provides a framework for simulating superconducting qubit systems
with flux-tunable transmons and couplers using a rotating frame transformation
and rotating wave approximation (RWA). The implementation follows the physics
document with simplified symmetric-case flux-dependent expressions.

Classes
-------
SuperconductingDeviceRot
    Device class for multi-level transmon qubits and tunable couplers with
    rotating frame transformation, RWA coupling terms, and simplified parameter
    interface using only max frequencies, anharmonicities, and max couplings.
"""
# pylint: disable=import-error,invalid-name,too-many-instance-attributes
# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-branches,too-many-statements
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence, Tuple

import dynamiqs as dq
import jax.numpy as jnp

from qualibration_graphs.validation_utils.time_dynamics import Pulse, GaussianPulse, kron_n


@dataclass(frozen=True)
class SuperconductingDeviceRot:
    """
    Multi-level transmon chain with tunable couplers in rotating frame with RWA.

    The model implements the rotating frame transformation and RWA Hamiltonian
    as described in the physics document. It uses simplified symmetric-case
    expressions for flux-dependent frequencies and couplings, requiring only
    max frequencies, anharmonicities, and max coupling values.

    Rotating Frame (Eq. 2):
        U_R(t) = exp(it Σ_j ω_j^(0) n_j)

    Free Hamiltonian in Rotating Frame (Eq. 7):
        H_free^(R) / ℏ = Σ_j [δ_j |1⟩⟨1| + (2δ_j + η_j) |2⟩⟨2|]
        where δ_j = ω_j(t) - ω_j^(0)

    RWA Coupling Terms (Eqs. 14-16, when use_rwa=True):
        g_ij(t) * (a_i a_j† e^{-iΔ_ij t} + a_i† a_j e^{+iΔ_ij t})
        where Δ_ij = ω_i^(0) - ω_j^(0)

    Full Coupling Terms (Eq. 13, when use_rwa=False):
        g_ij(t) * [e^{-iΣ_ij t} a_i a_j + e^{+iΣ_ij t} a_i† a_j†
                  + e^{-iΔ_ij t} a_i a_j† + e^{+iΔ_ij t} a_i† a_j]
        where Σ_ij = ω_i^(0) + ω_j^(0), Δ_ij = ω_i^(0) - ω_j^(0)

    Drive Terms (Eqs. 83, 87, 88):
        H_d^RWA / ℏ = Σ_j Ω_j(t)/2 * (a_j e^{i(Δ_d,j t + φ_d,j)} + a_j† e^{-i(Δ_d,j t + φ_d,j)})
        where Δ_d,j = ω_d,j - ω_j^(0)

    Parameters
    ----------
    n_qubits : int
        Number of qubits. Must be >= 2.
    levels : int, default=3
        Number of levels per mode (truncation).
    max_qubit_freqs : Sequence[float]
        Maximum frequencies for each qubit (omega_max) in GHz.
        Length must be n_qubits.
    max_coupler_freqs : Sequence[float]
        Maximum frequencies for each coupler (omega_max) in GHz.
        Length must be n_qubits - 1.
    qubit_anharm : Sequence[float]
        Anharmonicities for each qubit in GHz. EC = anharmonicity.
        Length must be n_qubits.
    coupler_anharm : Sequence[float]
        Anharmonicities for each coupler in GHz. EC = anharmonicity.
        Length must be n_qubits - 1.
    max_g_qubit_coupler : Sequence[Tuple[float, float]]
        Maximum qubit-coupler couplings for each coupler (g_left, g_right) in GHz.
        Length must be n_qubits - 1.
    max_g_direct : Sequence[float]
        Maximum direct qubit-qubit couplings in GHz.
        Length must be n_qubits - 1.
    use_rwa : bool, default=True
        If True, apply RWA to both coupling and drive terms (only energy-conserving terms).
        If False, include all terms:
        - For couplings: all four terms (sum and difference frequencies)
        - For drives: both slow (Δ_d) and fast (ω_d) oscillating terms

    Notes
    -----
    The idling frequencies are computed from max frequencies at the default flux bias
    (phi_ext). If phi_ext is not provided, it defaults to 0.0 (sweet spot), so
    omega_idle = omega_max. Flux pulses are added on top of phi_ext.
    Flux-dependent frequencies use simplified symmetric-case expressions (Eqs. 42, 44, 46).
    Flux-dependent couplings use simplified expressions (Eqs. 48-50, 57-59).
    """
    n_qubits: int
    max_qubit_freqs: Sequence[float]
    max_coupler_freqs: Sequence[float]
    qubit_anharm: Sequence[float]
    coupler_anharm: Sequence[float]
    max_g_qubit_coupler: Sequence[Tuple[float, float]]
    max_g_direct: Sequence[float]
    levels: int = 3
    use_rwa: bool = True
    qubit_phi_ext: Sequence[float] | None = None  # Default flux bias (reduced flux phi = 2*pi*Phi/Phi0)
    coupler_phi_ext: Sequence[float] | None = None  # Default flux bias for couplers

    # Cached operators (set in __post_init__)
    _a_ops: Tuple[dq.QArray, ...] = field(init=False, default_factory=tuple)
    _adag_ops: Tuple[dq.QArray, ...] = field(init=False, default_factory=tuple)
    _n_ops: Tuple[dq.QArray, ...] = field(init=False, default_factory=tuple)
    _kerr_ops: Tuple[dq.QArray, ...] = field(init=False, default_factory=tuple)
    _x_ops: Tuple[dq.QArray, ...] = field(init=False, default_factory=tuple)
    _y_ops: Tuple[dq.QArray, ...] = field(init=False, default_factory=tuple)

    # Computed idling frequencies (set in __post_init__)
    _idling_qubit_freqs: Tuple[float, ...] = field(init=False, default_factory=tuple)
    _idling_coupler_freqs: Tuple[float, ...] = field(init=False, default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate parameters, compute idling frequencies, and cache operators."""
        if self.n_qubits < 2:
            raise ValueError("n_qubits must be >= 2")

        n_couplers = self.n_qubits - 1

        # Validate parameter lengths
        if len(self.max_qubit_freqs) != self.n_qubits:
            raise ValueError(f"max_qubit_freqs length must be n_qubits ({self.n_qubits})")
        if len(self.max_coupler_freqs) != n_couplers:
            raise ValueError(f"max_coupler_freqs length must be n_qubits - 1 ({n_couplers})")
        if len(self.qubit_anharm) != self.n_qubits:
            raise ValueError(f"qubit_anharm length must be n_qubits ({self.n_qubits})")
        if len(self.coupler_anharm) != n_couplers:
            raise ValueError(f"coupler_anharm length must be n_qubits - 1 ({n_couplers})")
        if len(self.max_g_qubit_coupler) != n_couplers:
            raise ValueError(f"max_g_qubit_coupler length must be n_qubits - 1 ({n_couplers})")
        if len(self.max_g_direct) != n_couplers:
            raise ValueError(f"max_g_direct length must be n_qubits - 1 ({n_couplers})")

        # Validate coupling tuples
        for k, g_pair in enumerate(self.max_g_qubit_coupler):
            if not isinstance(g_pair, (tuple, list)) or len(g_pair) != 2:
                raise ValueError(f"max_g_qubit_coupler[{k}] must be a tuple of length 2")

        # Set default flux biases (phi_ext) if not provided
        if self.qubit_phi_ext is None:
            object.__setattr__(self, "qubit_phi_ext", (0.0,) * self.n_qubits)
        elif len(self.qubit_phi_ext) != self.n_qubits:
            raise ValueError(f"qubit_phi_ext length must be n_qubits ({self.n_qubits})")
        else:
            object.__setattr__(self, "qubit_phi_ext", tuple(float(p) for p in self.qubit_phi_ext))

        if self.coupler_phi_ext is None:
            object.__setattr__(self, "coupler_phi_ext", (0.0,) * n_couplers)
        elif len(self.coupler_phi_ext) != n_couplers:
            raise ValueError(f"coupler_phi_ext length must be n_qubits - 1 ({n_couplers})")
        else:
            object.__setattr__(self, "coupler_phi_ext", tuple(float(p) for p in self.coupler_phi_ext))

        # Compute idling frequencies at default flux bias (phi_ext)
        # omega_idle = omega(phi_ext), not necessarily omega_max
        qubit_phi_ext_vals: Sequence[float] = self.qubit_phi_ext  # type: ignore[assignment]  # Now guaranteed to be not None
        coupler_phi_ext_vals: Sequence[float] = self.coupler_phi_ext  # type: ignore[assignment]  # Now guaranteed to be not None

        idling_qubit_freqs = []
        for i in range(self.n_qubits):
            omega_max = jnp.asarray(self.max_qubit_freqs[i])
            EC = jnp.asarray(self.qubit_anharm[i])
            phi_ext = jnp.asarray(qubit_phi_ext_vals[i])
            omega_idle = self._omega_from_phi_symmetric(omega_max, EC, phi_ext)
            idling_qubit_freqs.append(float(omega_idle))

        idling_coupler_freqs = []
        for k in range(n_couplers):
            omega_max = jnp.asarray(self.max_coupler_freqs[k])
            EC = jnp.asarray(self.coupler_anharm[k])
            phi_ext = jnp.asarray(coupler_phi_ext_vals[k])
            omega_idle = self._omega_from_phi_symmetric(omega_max, EC, phi_ext)
            idling_coupler_freqs.append(float(omega_idle))

        object.__setattr__(self, "_idling_qubit_freqs", tuple(idling_qubit_freqs))
        object.__setattr__(self, "_idling_coupler_freqs", tuple(idling_coupler_freqs))

        # Precompute embedded operators for each mode
        a = dq.destroy(self.levels)
        adag = dq.create(self.levels)
        n_local = adag @ a
        kerr_local = adag @ a @ adag @ a  # n(n-1) for anharmonicity
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
        """Return number of couplers between adjacent qubits."""
        return self.n_qubits - 1

    @property
    def n_modes(self) -> int:
        """Return total number of modes (qubits + couplers)."""
        return self.n_qubits + self.n_couplers

    @property
    def idling_qubit_freqs(self) -> Tuple[float, ...]:
        """Return idling frequencies for qubits."""
        return self._idling_qubit_freqs

    @property
    def idling_coupler_freqs(self) -> Tuple[float, ...]:
        """Return idling frequencies for couplers."""
        return self._idling_coupler_freqs

    def _coupler_index(self, k: int) -> int:
        """Return the global mode index for coupler k."""
        return self.n_qubits + k

    def _embed_op(self, op: dq.QArray, which: int) -> dq.QArray:
        """Embed a single-mode operator into the full tensor product space."""
        ops = [dq.eye(self.levels) for _ in range(self.n_modes)]
        ops[which] = op
        return kron_n(ops)

    def _pulse_callable(self, pulse: Pulse | Callable[[jnp.ndarray], jnp.ndarray]):
        """Normalize pulse inputs to a callable signature f(t)."""
        return pulse.timecallable() if hasattr(pulse, "timecallable") else pulse

    def _omega_from_phi_symmetric(
        self, omega_max: jnp.ndarray, EC: jnp.ndarray, phi: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute flux-dependent frequency using simplified symmetric-case expression.

        Implements Eqs. 42, 44, 46 from section 7.8 for symmetric SQUID.
        For symmetric SQUID, the frequency scales as |cos(phi/2)|.

        Parameters
        ----------
        omega_max : jnp.ndarray
            Maximum frequency at phi=0 (sweet spot).
        EC : jnp.ndarray
            Charging energy (anharmonicity).
        phi : jnp.ndarray
            Reduced flux phi = 2*pi*Phi/Phi0.

        Returns
        -------
        jnp.ndarray
            Flux-dependent frequency omega(phi).
        """
        # Simplified symmetric case: omega(phi) ≈ omega_max * |cos(phi/2)|
        # This is a first-order approximation. For more accuracy, we include
        # a correction term based on EC to account for transmon nonlinearity.
        cos_term = jnp.abs(jnp.cos(phi / 2.0))
        # Add small correction from transmon expansion
        # omega ≈ omega_max * |cos(phi/2)| - EC * (small correction)
        # The correction is small when EC << omega_max
        return omega_max * cos_term - EC * (1.0 - cos_term) * 0.1

    def _g_qubit_coupler_from_flux(
        self,
        g_max: jnp.ndarray,
        omega_coupler_max: jnp.ndarray,
        omega_coupler: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute flux-dependent qubit-coupler coupling.

        Implements Eqs. 48, 49, 50 from section 7.8.
        Coupling scales with coupler frequency/EJ.

        Parameters
        ----------
        g_max : jnp.ndarray
            Maximum coupling at sweet spot (phi=0).
        omega_coupler_max : jnp.ndarray
            Maximum coupler frequency at sweet spot.
        omega_coupler : jnp.ndarray
            Current coupler frequency (flux-dependent).

        Returns
        -------
        jnp.ndarray
            Flux-dependent coupling g(phi).
        """
        # Coupling scales as (EJ_coupler)^(1/4) or (omega_coupler)^(1/4)
        # g(phi) = g_max * (omega_coupler(phi) / omega_coupler_max)^(1/4)
        ratio = jnp.maximum(omega_coupler / omega_coupler_max, 1e-6)  # Avoid division by zero
        return g_max * (ratio ** 0.25)

    def _g_qubit_qubit_from_flux(
        self,
        g_max: jnp.ndarray,
        omega_coupler_max: jnp.ndarray,
        omega_coupler: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute flux-dependent qubit-qubit coupling.

        Implements Eqs. 57, 58, 59 from section 7.8.
        Direct qubit-qubit coupling also depends on coupler frequency.

        Parameters
        ----------
        g_max : jnp.ndarray
            Maximum direct coupling at sweet spot.
        omega_coupler_max : jnp.ndarray
            Maximum coupler frequency at sweet spot.
        omega_coupler : jnp.ndarray
            Current coupler frequency (flux-dependent).

        Returns
        -------
        jnp.ndarray
            Flux-dependent direct coupling g_12(phi).
        """
        # Similar scaling as qubit-coupler coupling
        ratio = jnp.maximum(omega_coupler / omega_coupler_max, 1e-6)
        return g_max * (ratio ** 0.25)

    def _rwa_coupling_term(
        self,
        i: int,
        j: int,
        g_t: Callable[[jnp.ndarray], jnp.ndarray],
        delta_ij: float,
        sigma_ij: float,
    ) -> Sequence[tuple[Callable[[jnp.ndarray], jnp.ndarray], dq.QArray]]:
        """
        Construct coupling operator terms for modes i and j.

        If use_rwa=True: Only energy-conserving terms (Eqs. 14-16).
        If use_rwa=False: All four terms including sum frequencies (Eq. 13).

        Parameters
        ----------
        i : int
            Mode index i.
        j : int
            Mode index j.
        g_t : Callable
            Time-dependent coupling function g(t).
        delta_ij : float
            Frequency difference Δ_ij = ω_i^(0) - ω_j^(0).
        sigma_ij : float
            Frequency sum Σ_ij = ω_i^(0) + ω_j^(0).

        Returns
        -------
        Sequence[tuple[Callable, dq.QArray]]
            List of (time_function, operator) pairs for the coupling terms.
        """
        a_i = self._a_ops[i]
        adag_i = self._adag_ops[i]
        a_j = self._a_ops[j]
        adag_j = self._adag_ops[j]

        terms = []

        if self.use_rwa:
            # RWA: Only energy-conserving terms (Eqs. 14-16)
            # g(t) * (a_i a_j† e^{-iΔ_ij t} + a_i† a_j e^{+iΔ_ij t})
            # For real g(t), we can decompose into cos and sin terms
            op_exchange_x = (a_i @ adag_j + adag_i @ a_j).asdense()
            op_exchange_y = (-1j * (a_i @ adag_j - adag_i @ a_j)).asdense()

            def term_exchange_x(t):
                g_val = g_t(t)
                # Convert frequency from GHz to angular frequency: ω = 2π * f
                # Phase = ωt = 2π * (delta_ij in GHz) * (t in ns)
                return jnp.real(g_val) * jnp.cos(2.0 * jnp.pi * delta_ij * t)

            def term_exchange_y(t):
                g_val = g_t(t)
                return jnp.real(g_val) * jnp.sin(2.0 * jnp.pi * delta_ij * t)

            terms.append((term_exchange_x, op_exchange_x))
            terms.append((term_exchange_y, op_exchange_y))
        else:
            # Full coupling: All four terms (Eq. 13)
            # g(t) * [e^{-iΣ_ij t} a_i a_j + e^{+iΣ_ij t} a_i† a_j†
            #        + e^{-iΔ_ij t} a_i a_j† + e^{+iΔ_ij t} a_i† a_j]

            # Term 1: a_i a_j with sum frequency
            op_aa = (a_i @ a_j).asdense()

            def term_aa_re(t):
                g_val = g_t(t)
                # Convert frequency from GHz to angular frequency: ω = 2π * f
                return jnp.real(g_val * jnp.exp(-1j * 2.0 * jnp.pi * sigma_ij * t))

            terms.append((term_aa_re, op_aa))

            # Term 2: a_i† a_j† with sum frequency
            op_adag_adag = (adag_i @ adag_j).asdense()

            def term_adag_adag_re(t):
                g_val = g_t(t)
                return jnp.real(g_val * jnp.exp(1j * 2.0 * jnp.pi * sigma_ij * t))

            terms.append((term_adag_adag_re, op_adag_adag))

            # Term 3: a_i a_j† with difference frequency
            op_a_adag = (a_i @ adag_j).asdense()

            def term_a_adag_re(t):
                g_val = g_t(t)
                return jnp.real(g_val * jnp.exp(-1j * 2.0 * jnp.pi * delta_ij * t))

            terms.append((term_a_adag_re, op_a_adag))

            # Term 4: a_i† a_j with difference frequency
            op_adag_a = (adag_i @ a_j).asdense()

            def term_adag_a_re(t):
                g_val = g_t(t)
                return jnp.real(g_val * jnp.exp(1j * 2.0 * jnp.pi * delta_ij * t))

            terms.append((term_adag_a_re, op_adag_a))

        return terms

    def construct_h(
        self,
        drives: Sequence[tuple[int, GaussianPulse]] = (),
        qubit_flux: Sequence[tuple[int, Pulse]] = (),
        coupler_flux: Sequence[tuple[int, Pulse]] = (),
    ) -> dq.TimeQArray:
        """
        Build time-dependent Hamiltonian in rotating frame with RWA.

        Implements Eq. 89 structure:
        - Free Hamiltonian terms with detunings (Eq. 7)
        - RWA or full coupling terms (Eqs. 13-16, 48-50, 57-59)
        - Drive terms (Eqs. 83, 87, 88)

        Parameters
        ----------
        drives : Sequence[tuple[int, GaussianPulse]], default=()
            Qubit drive pulses: (qubit_index, pulse) pairs.
        qubit_flux : Sequence[tuple[int, Pulse]], default=()
            Qubit flux pulses: (qubit_index, pulse) pairs.
            Pulse value is reduced flux phi = 2*pi*Phi/Phi0.
        coupler_flux : Sequence[tuple[int, Pulse]], default=()
            Coupler flux pulses: (coupler_index, pulse) pairs.
            Pulse value is reduced flux phi = 2*pi*Phi/Phi0.

        Returns
        -------
        dq.TimeQArray
            Time-dependent Hamiltonian in rotating frame.
        """
        # Initialize zero Hamiltonian
        Ht = dq.constant(0.0 * kron_n([dq.eye(self.levels)] * self.n_modes))

        # Create flux pulse maps
        qubit_flux_map = {i: self._pulse_callable(p) for i, p in qubit_flux}
        coupler_flux_map = {i: self._pulse_callable(p) for i, p in coupler_flux}

        # Free Hamiltonian terms: detunings and anharmonicity (Eq. 7)
        # H_free^(R) / ℏ = Σ_j [δ_j |1⟩⟨1| + (2δ_j + η_j) |2⟩⟨2|]
        # For three-level system: δ_j n_j - η_j/2 * n_j(n_j-1)
        # where n_j = |1⟩⟨1| + 2|2⟩⟨2|, and n_j(n_j-1) = a†a†aa (Kerr operator)

        for i in range(self.n_qubits):
            omega_max = jnp.asarray(self.max_qubit_freqs[i])
            EC = jnp.asarray(self.qubit_anharm[i])
            omega_idle = jnp.asarray(self.idling_qubit_freqs[i])
            n_op = self._n_ops[i]
            kerr_op = self._kerr_ops[i]

            phi_ext = jnp.asarray(self.qubit_phi_ext[i])

            if i in qubit_flux_map:
                f = qubit_flux_map[i]

                def delta_q_t(t, omega_max=omega_max, EC=EC, omega_idle=omega_idle, phi_ext=phi_ext, f=f):
                    phi = phi_ext + jnp.real(f(t))  # Flux pulse added to default bias
                    omega = self._omega_from_phi_symmetric(omega_max, EC, phi)
                    return omega - omega_idle

                def anharm_q_t(_t, EC=EC):
                    # Anharmonicity term: -η_j/2 * n_j(n_j-1) = -EC/2 * kerr_op
                    return -EC / 2.0

                Ht = Ht + dq.modulated(delta_q_t, n_op)
                Ht = Ht + dq.modulated(anharm_q_t, kerr_op)
            else:
                # Static: delta = 0 at idling, but anharmonicity remains
                Ht = Ht + dq.modulated(lambda t, EC=EC: -EC / 2.0, kerr_op)

        for k in range(self.n_couplers):
            omega_max = jnp.asarray(self.max_coupler_freqs[k])
            EC = jnp.asarray(self.coupler_anharm[k])
            omega_idle = jnp.asarray(self.idling_coupler_freqs[k])
            idx = self._coupler_index(k)
            n_op = self._n_ops[idx]
            kerr_op = self._kerr_ops[idx]

            phi_ext_val: Sequence[float] = self.coupler_phi_ext  # type: ignore[assignment]
            phi_ext = jnp.asarray(phi_ext_val[k])

            if k in coupler_flux_map:
                f = coupler_flux_map[k]

                def delta_c_t(t, omega_max=omega_max, EC=EC, omega_idle=omega_idle, phi_ext=phi_ext, f=f):
                    phi = phi_ext + jnp.real(f(t))  # Flux pulse added to default bias
                    omega = self._omega_from_phi_symmetric(omega_max, EC, phi)
                    return omega - omega_idle

                def anharm_c_t(_t, EC=EC):
                    # Anharmonicity term: -η_j/2 * n_j(n_j-1) = -EC/2 * kerr_op
                    return -EC / 2.0

                Ht = Ht + dq.modulated(delta_c_t, n_op)
                Ht = Ht + dq.modulated(anharm_c_t, kerr_op)
            else:
                # Static: delta = 0 at idling, but anharmonicity remains
                Ht = Ht + dq.modulated(lambda t, EC=EC: -EC / 2.0, kerr_op)

        # Coupling terms: qubit-coupler and qubit-qubit
        for k in range(self.n_couplers):
            q_left = k
            q_right = k + 1
            c_idx = self._coupler_index(k)

            # Get idling frequencies for frequency differences
            omega_q_left_idle = jnp.asarray(self.idling_qubit_freqs[q_left])
            omega_q_right_idle = jnp.asarray(self.idling_qubit_freqs[q_right])
            omega_c_idle = jnp.asarray(self.idling_coupler_freqs[k])

            # Frequency differences for RWA phases
            delta_left_c = float(omega_q_left_idle - omega_c_idle)
            delta_right_c = float(omega_q_right_idle - omega_c_idle)
            delta_left_right = float(omega_q_left_idle - omega_q_right_idle)

            # Frequency sums for full coupling
            sigma_left_c = float(omega_q_left_idle + omega_c_idle)
            sigma_right_c = float(omega_q_right_idle + omega_c_idle)
            sigma_left_right = float(omega_q_left_idle + omega_q_right_idle)

            # Qubit-coupler couplings
            g_left_max, g_right_max = self.max_g_qubit_coupler[k]
            omega_c_max = jnp.asarray(self.max_coupler_freqs[k])

            # Check if coupler has flux pulse
            f_c = coupler_flux_map.get(k)

            phi_c_ext_val: Sequence[float] = self.coupler_phi_ext  # type: ignore[assignment]
            phi_c_ext = jnp.asarray(phi_c_ext_val[k])

            if f_c is not None:
                # Time-dependent coupling
                def g_left_t(t, omega_c_max=omega_c_max, phi_c_ext=phi_c_ext, f_c=f_c):
                    phi = phi_c_ext + jnp.real(f_c(t))  # Flux pulse added to default bias
                    EC_c = jnp.asarray(self.coupler_anharm[k])
                    omega_c = self._omega_from_phi_symmetric(omega_c_max, EC_c, phi)
                    return self._g_qubit_coupler_from_flux(
                        jnp.asarray(g_left_max), omega_c_max, omega_c
                    )

                def g_right_t(t, omega_c_max=omega_c_max, phi_c_ext=phi_c_ext, f_c=f_c):
                    phi = phi_c_ext + jnp.real(f_c(t))  # Flux pulse added to default bias
                    EC_c = jnp.asarray(self.coupler_anharm[k])
                    omega_c = self._omega_from_phi_symmetric(omega_c_max, EC_c, phi)
                    return self._g_qubit_coupler_from_flux(
                        jnp.asarray(g_right_max), omega_c_max, omega_c
                    )

                # Left qubit - coupler coupling
                terms_left = self._rwa_coupling_term(
                    q_left, c_idx, g_left_t, delta_left_c, sigma_left_c
                )
                for term_func, term_op in terms_left:
                    Ht = Ht + dq.modulated(term_func, term_op)

                # Right qubit - coupler coupling
                terms_right = self._rwa_coupling_term(
                    q_right, c_idx, g_right_t, delta_right_c, sigma_right_c
                )
                for term_func, term_op in terms_right:
                    Ht = Ht + dq.modulated(term_func, term_op)
            else:
                # Static coupling
                g_left_static = jnp.asarray(g_left_max)
                g_right_static = jnp.asarray(g_right_max)

                def g_left_static_t(_t, g=g_left_static):
                    return g

                def g_right_static_t(_t, g=g_right_static):
                    return g

                terms_left = self._rwa_coupling_term(
                    q_left, c_idx, g_left_static_t, delta_left_c, sigma_left_c
                )
                for term_func, term_op in terms_left:
                    Ht = Ht + dq.modulated(term_func, term_op)

                terms_right = self._rwa_coupling_term(
                    q_right, c_idx, g_right_static_t, delta_right_c, sigma_right_c
                )
                for term_func, term_op in terms_right:
                    Ht = Ht + dq.modulated(term_func, term_op)

            # Direct qubit-qubit coupling
            g_direct_max = self.max_g_direct[k]

            if f_c is not None:
                # Coupling depends on coupler flux
                def g_direct_t(t, omega_c_max=omega_c_max, phi_c_ext=phi_c_ext, f_c=f_c):
                    phi = phi_c_ext + jnp.real(f_c(t))  # Flux pulse added to default bias
                    EC_c = jnp.asarray(self.coupler_anharm[k])
                    omega_c = self._omega_from_phi_symmetric(omega_c_max, EC_c, phi)
                    return self._g_qubit_qubit_from_flux(
                        jnp.asarray(g_direct_max), omega_c_max, omega_c
                    )

                terms_direct = self._rwa_coupling_term(
                    q_left, q_right, g_direct_t, delta_left_right, sigma_left_right
                )
                for term_func, term_op in terms_direct:
                    Ht = Ht + dq.modulated(term_func, term_op)
            else:
                # Static direct coupling
                g_direct_static = jnp.asarray(g_direct_max)

                def g_direct_static_t(_t, g=g_direct_static):
                    return g

                terms_direct = self._rwa_coupling_term(
                    q_left, q_right, g_direct_static_t, delta_left_right, sigma_left_right
                )
                for term_func, term_op in terms_direct:
                    Ht = Ht + dq.modulated(term_func, term_op)

        # Drive terms (Eqs. 83, 87, 88)
        for which, pulse in drives:
            for f, op in self._drive_terms(which, pulse):
                Ht = Ht + dq.modulated(f, op)

        return Ht

    def _drive_terms(
        self, which: int, pulse: GaussianPulse
    ) -> Sequence[tuple[Callable[[jnp.ndarray], jnp.ndarray], dq.QArray]]:
        """
        Build drive terms for qubit in rotating frame.

        With RWA (use_rwa=True): Implements Eqs. 83, 87, 88
        - Resonant drive (Δ_d = 0): Eq. 83
        - Detuned drive: Eq. 87
        - Only energy-conserving terms: a_j e^{i(Δ_d t + φ_d)} + a_j† e^{-i(Δ_d t + φ_d)}

        Without RWA (use_rwa=False): Full drive Hamiltonian
        - Includes both slow (Δ_d) and fast (ω_d + ω_idle) oscillating terms
        - Terms: a_j e^{-i(ω_d - ω_idle)t} + a_j† e^{i(ω_d - ω_idle)t}
                + a_j e^{-i(ω_d + ω_idle)t} + a_j† e^{i(ω_d + ω_idle)t}

        Parameters
        ----------
        which : int
            Qubit index.
        pulse : GaussianPulse
            Drive pulse with amplitude, phase, and optional drive_freq.

        Returns
        -------
        Sequence[tuple[Callable, dq.QArray]]
            List of (time_function, operator) pairs for drive terms.
        """
        s_base = self._pulse_callable(pulse)
        drive_freq = getattr(pulse, "drive_freq", None)

        omega_idle = jnp.asarray(self.idling_qubit_freqs[which])
        omega_d = jnp.asarray(omega_idle) if drive_freq is None else jnp.asarray(drive_freq)
        delta_d = omega_d - omega_idle  # Detuning in rotating frame (slow term)
        # Fast term: in lab frame would oscillate at ω_d + ω_idle
        # In rotating frame, this becomes (ω_d + ω_idle) - ω_idle = ω_d
        # So fast counter-rotating term oscillates at ω_d in rotating frame

        Xj = self._x_ops[which]
        Yj = self._y_ops[which]

        terms = []

        if self.use_rwa:
            # RWA: Only energy-conserving terms (Eqs. 83, 87, 88)
            # H_d^RWA / ℏ = Ω_j(t)/2 * (a_j e^{i(Δ_d t + φ_d)} + a_j† e^{-i(Δ_d t + φ_d)})
            # For real-valued drive envelope, this becomes X and Y quadrature terms

            def s_re(t):
                s_val = s_base(t)
                # Convert frequency from GHz to angular frequency: ω = 2π * f
                # Phase = ωt = 2π * (delta_d in GHz) * (t in ns)
                # Since 1 GHz * 1 ns = 1, we get phase in radians
                phase_rot = jnp.exp(1j * (2.0 * jnp.pi * delta_d * t))
                return jnp.real(s_val * phase_rot)

            def s_im(t):
                s_val = s_base(t)
                phase_rot = jnp.exp(1j * (2.0 * jnp.pi * delta_d * t))
                return jnp.imag(s_val * phase_rot)

            terms.append((s_re, Xj))
            terms.append((s_im, Yj))
        else:
            # Full drive: Include both slow and fast oscillating terms
            # In lab frame: H_drive = s(t) * (a e^{-iω_d t} + a† e^{iω_d t})
            # In rotating frame at ω_idle:
            #   - Slow term: at Δ_d = ω_d - ω_idle (energy-conserving)
            #   - Fast term: at ω_d + ω_idle in lab frame, which in rotating frame is ω_d + ω_idle - ω_idle = ω_d
            # But actually, the counter-rotating term in lab frame e^{i(ω_d + ω_idle)t} transforms
            # to e^{i(ω_d + ω_idle - ω_idle)t} = e^{iω_d t} in rotating frame
            # So fast terms oscillate at ω_d (not Δ_d)

            # Slow terms (energy-conserving): at Δ_d = ω_d - ω_idle
            def s_slow_re(t):
                s_val = s_base(t)
                # Convert frequency from GHz to angular frequency: ω = 2π * f
                phase_rot = jnp.exp(1j * (2.0 * jnp.pi * delta_d * t))
                return jnp.real(s_val * phase_rot)

            def s_slow_im(t):
                s_val = s_base(t)
                phase_rot = jnp.exp(1j * (2.0 * jnp.pi * delta_d * t))
                return jnp.imag(s_val * phase_rot)

            terms.append((s_slow_re, Xj))
            terms.append((s_slow_im, Yj))

            # Fast terms (counter-rotating): at ω_d in rotating frame
            # These are the fast-oscillating terms that RWA drops
            # The counter-rotating term in lab frame e^{i(ω_d + ω_idle)t} 
            # transforms to e^{iω_d t} in rotating frame
            def s_fast_re(t):
                s_val = s_base(t)
                # Fast term oscillates at ω_d in rotating frame (counter-rotating)
                # Convert frequency from GHz to angular frequency: ω = 2π * f
                phase_rot = jnp.exp(1j * (2.0 * jnp.pi * omega_d * t))
                return jnp.real(s_val * phase_rot)

            def s_fast_im(t):
                s_val = s_base(t)
                phase_rot = jnp.exp(1j * (2.0 * jnp.pi * omega_d * t))
                return jnp.imag(s_val * phase_rot)

            terms.append((s_fast_re, Xj))
            terms.append((s_fast_im, Yj))

        return tuple(terms)

