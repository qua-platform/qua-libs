"""Circuit construction for Context Aware Fidelity Estimation (CAFE) of a CZ gate.

Reference: D. M. Debroy et al., "Context Aware Fidelity Estimation", arXiv:2303.17565.

Every CAFE circuit has three parts:

1. Prepare one of the 16 states of a two-qubit SIC (a 2-design) with one CZ:
   ``L1 · CZ · L0 |00>`` (Appendix A of the paper).
2. Apply the cycle ``n`` times. The cycle is the CZ, followed by X on both qubits for
   the DECAF variant.
3. Undo the state that the *reference* cycle would have produced, ``C_ref^n |psi>``,
   with the inverse of its own one-CZ preparation circuit: ``L0'^† · CZ · L1'^†``.

Averaging P(|00>) over the 16 states gives the average gate fidelity between the
implemented cycle repeated ``n`` times and ``C_ref^n``.

Single-qubit layers are compiled to ``Rz(a) · Ry(π/2) · Rz(b) · Ry(π/2) · Rz(c)``,
played as three virtual-Z frame rotations and two ``y90`` pulses. Every gate in a
CAFE circuit (Ry(π/2), CZ, X, Rz up to its sign) then has a real matrix or maps to
its own complex conjugate, so P(|00>) is unchanged if the hardware realises Rz(-θ)
for ``frame_rotation(θ)``, or Ry(-π/2) for ``y90``. The protocol is therefore
insensitive to these two sign conventions.

Qubit ordering: index 0 is ``qubit_control`` and index 1 is ``qubit_target``, with
``|q0 q1>`` ordered as ``kron(q0, q1)``.
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

# Pauli matrices and fixed gates
_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
CZ = np.diag([1, 1, 1, -1]).astype(complex)
XX = np.kron(_X, _X)

# Fiducial of a Weyl-Heisenberg-covariant SIC in dimension 4 (|<psi|D_ab|psi>|^2 = 1/5 for all
# 15 non-trivial displacement operators D_ab = X^a Z^b, with X|j> = |j+1 mod 4>, Z|j> = i^j |j>).
# Found numerically; the 2-design property is checked in the unit tests.
_SIC_FIDUCIAL = np.array(
    [
        0.4008483913243409 + 0.0j,
        -0.15440391488017483 - 0.12898169793524528j,
        -0.557833840897345 + 0.50174572332246425j,
        -0.311389364453179 + 0.37276402538721909j,
    ]
)

NUM_STATES = 16
NUM_ANGLES_PER_QUBIT_LAYER = 3
# Per circuit part (preparation or undo): 2 layers x 2 qubits x 3 angles
NUM_ANGLES_PER_PART = 2 * 2 * NUM_ANGLES_PER_QUBIT_LAYER

VARIANTS = ("cafe", "decaf")


def rz(angle: float) -> np.ndarray:
    """Z rotation exp(-i angle Z / 2)."""
    return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])


def ry(angle: float) -> np.ndarray:
    """Y rotation exp(-i angle Y / 2)."""
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def sic_states() -> np.ndarray:
    """Return the 16 states of the two-qubit SIC, shape (16, 4)."""
    d = 4
    shift = np.roll(np.eye(d), 1, axis=0)
    clock = np.diag([1j**k for k in range(d)])
    return np.array(
        [
            np.linalg.matrix_power(shift, a) @ np.linalg.matrix_power(clock, b) @ _SIC_FIDUCIAL
            for a in range(d)
            for b in range(d)
        ]
    )


def fsim_unitary(delta_theta: float = 0.0, delta_gamma: float = 0.0, delta_phi: float = 0.0) -> np.ndarray:
    """Excitation-preserving two-qubit unitary close to a CZ (Eq. 2 of the paper).

    ``delta_theta`` is the swap angle, ``delta_gamma`` the single-qubit phase and ``delta_phi``
    the conditional-phase error. All zero gives the ideal CZ.
    """
    phase = np.exp(-1j * delta_gamma)
    c, s = np.cos(delta_theta), np.sin(delta_theta)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, phase * c, -1j * phase * s, 0],
            [0, -1j * phase * s, phase * c, 0],
            [0, 0, 0, -np.exp(-1j * (delta_phi + 2 * delta_gamma))],
        ],
        dtype=complex,
    )


def cycle_unitary(gate: np.ndarray, variant: str) -> np.ndarray:
    """Unitary of one cycle: the two-qubit gate, followed by X on both qubits for DECAF."""
    if variant == "cafe":
        return gate
    if variant == "decaf":
        return XX @ gate
    raise ValueError(f"Unknown CAFE variant {variant!r}; expected one of {VARIANTS}.")


def zyzyz_angles(u: np.ndarray) -> Tuple[float, float, float]:
    """Return (a, b, c) with ``u ∝ Rz(a) · Ry(π/2) · Rz(b) · Ry(π/2) · Rz(c)``.

    Uses the ZYZ Euler decomposition ``u ∝ Rz(φ) Ry(θ) Rz(λ)`` and the identity
    ``Ry(π/2) Rz(θ + π) Ry(π/2) ∝ Rz(-π/2) Ry(θ) Rz(-π/2)``.
    """
    u = u / np.sqrt(np.linalg.det(u))
    theta = 2 * np.arctan2(abs(u[1, 0]), abs(u[0, 0]))
    sum_angle = -2 * np.angle(u[0, 0]) if abs(u[0, 0]) > 1e-12 else 0.0  # φ + λ
    diff_angle = 2 * np.angle(u[1, 0]) if abs(u[1, 0]) > 1e-12 else 0.0  # φ - λ
    phi = 0.5 * (sum_angle + diff_angle)
    lam = 0.5 * (sum_angle - diff_angle)
    return phi + np.pi / 2, theta + np.pi, lam + np.pi / 2


def zyzyz_unitary(a: float, b: float, c: float) -> np.ndarray:
    """Matrix of the compiled layer ``Rz(a) · Ry(π/2) · Rz(b) · Ry(π/2) · Rz(c)``."""
    return rz(a) @ ry(np.pi / 2) @ rz(b) @ ry(np.pi / 2) @ rz(c)


@dataclass
class OneCzPreparation:
    """One-CZ circuit ``(l1_q0 ⊗ l1_q1) · CZ · (l0_q0 ⊗ l0_q1)`` mapping |00> to a target state."""

    l0: Tuple[np.ndarray, np.ndarray]
    l1: Tuple[np.ndarray, np.ndarray]

    def unitary(self) -> np.ndarray:
        """Full 4x4 unitary of the preparation circuit."""
        return np.kron(*self.l1) @ CZ @ np.kron(*self.l0)


def one_cz_preparation(state: np.ndarray) -> OneCzPreparation:
    """Build the one-CZ circuit preparing ``state`` from |00> (Appendix A of the paper).

    The amplitude matrix ``M[i, j] = <i j|state>`` has singular values (s0, s1). The state
    ``CZ (|+> ⊗ Ry(α)|0>)`` with ``cos(α/2) = s0`` has the same singular values, so the two
    are related by local unitaries: ``(A ⊗ B)`` acts on M as ``A M B^T``.
    """
    m_target = np.asarray(state, dtype=complex).reshape(2, 2)
    u_t, s_t, vh_t = np.linalg.svd(m_target)
    alpha = 2 * np.arccos(np.clip(s_t[0], -1.0, 1.0))
    l0 = (ry(np.pi / 2), ry(alpha))
    m_cz = (CZ @ np.kron(*l0) @ np.array([1, 0, 0, 0], dtype=complex)).reshape(2, 2)
    u_c, _, vh_c = np.linalg.svd(m_cz)
    # m_target = (u_t u_c^†) m_cz (vh_c^† vh_t), and B^T = vh_c^† vh_t
    a = u_t @ u_c.conj().T
    b = (vh_c.conj().T @ vh_t).T
    return OneCzPreparation(l0=l0, l1=(a, b))


def _layer_angles(layer: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """ZYZYZ angles of a two-qubit layer, shape (2 qubits, 3)."""
    return np.array([zyzyz_angles(u) for u in layer])


def preparation_angles(prep: OneCzPreparation) -> np.ndarray:
    """Angles played to prepare the state, in time order: shape (2 layers, 2 qubits, 3)."""
    return np.array([_layer_angles(prep.l0), _layer_angles(prep.l1)])


def undo_angles(prep: OneCzPreparation) -> np.ndarray:
    """Angles played to undo the state (inverse circuit), in time order: shape (2 layers, 2 qubits, 3)."""
    l1_dag = tuple(u.conj().T for u in prep.l1)
    l0_dag = tuple(u.conj().T for u in prep.l0)
    return np.array([_layer_angles(l1_dag), _layer_angles(l0_dag)])


@dataclass
class CafeCircuitAngles:
    """Single-qubit layer angles for all CAFE circuits of one qubit pair.

    Angles are the (a, b, c) of each ZYZYZ layer, in radians. Within a layer, the pulse
    order in time is ``Rz(c)``, ``y90``, ``Rz(b)``, ``y90``, ``Rz(a)``.
    """

    preparation: np.ndarray
    """Shape (16 states, 2 layers, 2 qubits, 3)."""
    undo: np.ndarray
    """Shape (n_variants, n_depths, 16 states, 2 layers, 2 qubits, 3)."""

    def flat_preparation_2pi(self) -> list:
        """Preparation angles in units of 2π, wrapped to [-0.5, 0.5), flattened (C order)."""
        return _wrap_2pi(self.preparation).ravel().tolist()

    def flat_undo_2pi(self) -> list:
        """Undo angles in units of 2π, wrapped to [-0.5, 0.5), flattened (C order)."""
        return _wrap_2pi(self.undo).ravel().tolist()


def _wrap_2pi(angles: np.ndarray) -> np.ndarray:
    turns = np.asarray(angles) / (2 * np.pi)
    return (turns + 0.5) % 1.0 - 0.5


def build_circuit_angles(
    variants: Sequence[str],
    depths: Sequence[int],
    reference_gate: np.ndarray = CZ,
) -> CafeCircuitAngles:
    """Compute preparation and undo angles for every (variant, depth, state).

    ``reference_gate`` is the two-qubit unitary the undo step assumes for the CZ
    (ideal CZ, or a characterized ``fsim_unitary``).
    """
    states = sic_states()
    preps = [one_cz_preparation(psi) for psi in states]
    preparation = np.array([preparation_angles(p) for p in preps])

    undo = np.zeros((len(variants), len(depths), NUM_STATES, 2, 2, NUM_ANGLES_PER_QUBIT_LAYER))
    for iv, variant in enumerate(variants):
        cycle = cycle_unitary(reference_gate, variant)
        for idepth, depth in enumerate(depths):
            cycle_n = np.linalg.matrix_power(cycle, int(depth))
            for istate, psi in enumerate(states):
                undo[iv, idepth, istate] = undo_angles(one_cz_preparation(cycle_n @ psi))
    return CafeCircuitAngles(preparation=preparation, undo=undo)


def compiled_layer(angles: np.ndarray) -> np.ndarray:
    """4x4 unitary of a compiled two-qubit layer from its angles, shape (2 qubits, 3)."""
    return np.kron(zyzyz_unitary(*angles[0]), zyzyz_unitary(*angles[1]))


def simulate_return_probability(
    prep_angles: np.ndarray,
    undo_angles_: np.ndarray,
    implemented_cycle: np.ndarray,
    depth: int,
) -> float:
    """Ideal-gate probability of returning to |00> for one compiled CAFE circuit.

    Used in tests to check that the compiled circuits are correct.
    """
    psi = np.array([1, 0, 0, 0], dtype=complex)
    psi = compiled_layer(prep_angles[1]) @ CZ @ compiled_layer(prep_angles[0]) @ psi
    psi = np.linalg.matrix_power(implemented_cycle, depth) @ psi
    psi = compiled_layer(undo_angles_[1]) @ CZ @ compiled_layer(undo_angles_[0]) @ psi
    return float(abs(psi[0]) ** 2)


def reference_gates_per_pair(
    pair_names: Sequence[str],
    reference_unitary: str,
    characterized_angles: Dict[str, Sequence[float]],
) -> Dict[str, np.ndarray]:
    """Return the reference two-qubit gate for each pair.

    With ``reference_unitary="characterized"``, pairs listed in ``characterized_angles`` use
    ``fsim_unitary(Δθ, Δγ, Δφ)``; other pairs fall back to the ideal CZ.
    """
    gates = {}
    for name in pair_names:
        if reference_unitary == "characterized" and name in characterized_angles:
            gates[name] = fsim_unitary(*characterized_angles[name])
        else:
            gates[name] = CZ
    return gates
