"""Default Loss–DiVincenzo device parameters for virtual_qpu analysis tests.

Single source for the exchange model and 2-qubit Zeeman settings used by
``tests/analysis/.../loss_divincenzo/conftest.py`` (``DEFAULT_LD_PARAMS`` / ``ld_device``).

The ``exchange_0_1`` channel carries barrier voltage *V* (volts); the device uses
``DEFAULT_LD_EXCHANGE_MODEL`` so :math:`J(V) = J_0 \\exp((V - V_{\\rm ref})/\\lambda)`.
"""

from __future__ import annotations

from quantum_dots.params import ExchangeModel, LossDiVincenzoParams, MU_B_OVER_H

# Zeeman target for qubit 0 (g=2.0) — same convention as analysis conftest.
TARGET_Q0_ZEEMAN_GHZ = 5.0

# Parking scale J_0 (GHz); small so CROT lines meet at low barrier voltage.
EXCHANGE_PARKING_J0_GHZ = 5e-5

DEFAULT_LD_EXCHANGE_MODEL = ExchangeModel(
    J_0=EXCHANGE_PARKING_J0_GHZ,
    V_ref=0.0,
    lever_arm=0.050,
)


# Decoherence times (ns) — physically realistic for spin qubits.
# T1 = 100 µs, T2 = 10 µs (T2 ≤ 2·T1 satisfied).
DEFAULT_T1_NS: list[float] = [10_000.0, 10_000.0]
DEFAULT_T2_NS: list[float] = [1_500.0, 1_500.0]


def default_virtual_ld_params(
    *,
    target_q0_zeeman_ghz: float = TARGET_Q0_ZEEMAN_GHZ,
    g0: float = 2.0,
    g1: float = 2.03,
    exchange_model: ExchangeModel | None = None,
    t1: list[float] | None = None,
    t2: list[float] | None = None,
) -> LossDiVincenzoParams:
    """Two-qubit rotating-frame params used in loss_divincenzo analysis tests."""
    b_field = target_q0_zeeman_ghz / (g0 * MU_B_OVER_H)
    em = exchange_model if exchange_model is not None else DEFAULT_LD_EXCHANGE_MODEL
    return LossDiVincenzoParams(
        n_qubits=2,
        g_factors=[g0, g1],
        magnetic_field=b_field,
        exchange_models=[em],
        ref_freqs=None,
        frame="rot",
        use_rwa=True,
        t1=t1 if t1 is not None else DEFAULT_T1_NS,
        t2=t2 if t2 is not None else DEFAULT_T2_NS,
    )
