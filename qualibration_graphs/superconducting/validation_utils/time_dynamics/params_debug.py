"""
Parameter bundles for superconducting time-dynamics demos.

Values are in GHz (energies) and ns (times).
"""
from __future__ import annotations

USER_PARAMS_SYMMETRIC = {
    # User-provided f01 at Phi=0 and eta/2pi; symmetric SQUIDs (d = 0).
    "qubit_EC": (0.230, 0.233),
    "qubit_EJ_small": (4.550126086956523, 4.90914270386266),
    "qubit_EJ_large": (4.550126086956523, 4.90914270386266),
    "qubit_phi_ext": (0.0, 0.0),
    # Assumed typical coupler anharmonicity (GHz).
    "coupler_EC": (0.20,),
    "coupler_EJ_small": (12.1719003125,),
    "coupler_EJ_large": (12.1719003125,),
    "coupler_phi_ext": (0.0,),
    # Reported couplings g/2pi in MHz -> GHz (same units as omega in this demo).
    "g_couplings": ((0.111, 0.111),),
    "g_direct": (-0.0091,),
    "detuning_span": 0.2,
    "pulse_amp": 0.05,
    "pulse_duration": 200.0,
    "n_detunings": 101,
    "n_t": 201,
    "t_max": 200.0,
}
