"""
Parameter bundles for superconducting time-dynamics demos.

Values are in GHz (energies) and ns (times).
"""
from __future__ import annotations

import jax.numpy as jnp

DEBUG_PARAMS = {
    "qubit_EC": (0.25, 0.25),
    "qubit_EJ_small": (12.0, 11.0),
    "qubit_EJ_large": (20.0, 19.0),
    "qubit_phi_ext": (0.0, 0.0),
    "coupler_EC": (0.20,),
    "coupler_EJ_small": (14.0,),
    "coupler_EJ_large": (14.0,),
    "coupler_phi_ext": (0.7 * jnp.pi,),
    "g_couplings": ((0.001, 0.001),),
    "g_direct": (-0.0,),
    "detuning_span": 0.2,
    "pulse_amp": 0.05,
    "pulse_duration": 200.0,
    "n_detunings": 11,
    "n_t": 201,
    "t_max": 200.0,
}

PAPER_PARAMS_SYMMETRIC = {
    # Table I (symmetric): f01 for q1, qc, q2 in GHz
    "qubit_freqs": (3.862, 4.045),
    "coupler_freqs": (6.041,),
    # Tuned max coupler frequency for plotting the avoided crossing near ~4 GHz.
    "coupler_freqs_tuned": (4.2,),
    # Table I: anharmonicities eta/2pi in MHz -> GHz
    "qubit_anharm": (0.230, 0.233),
    # Coupler anharmonicity not provided; set to a small transmon-like value.
    "coupler_anharm": (0.20,),
    "coupler_EC": (0.20,),
    # Table II: sqrt(g1c g2c)/2pi = 111 MHz -> 0.111 GHz (g1c,g2c < 0 in symmetric config)
    "g_couplings": ((-0.111, -0.111),),
    # Table II: g12/2pi = -5.7 MHz -> -0.0057 GHz
    "g_direct": (-0.0057,),
    "detuning_span": 0.2,
    "pulse_amp": 0.05,
    "pulse_duration": 200.0,
    "n_detunings": 101,
    "n_t": 201,
    "t_max": 200.0,
    # Text (Fig. 6 discussion): symmetric coupler frequency range (GHz)
    "coupler_freq_range": (2.787, 3.663),
    # Text (Fig. 6 discussion): qubit frequencies used in simulation (GHz)
    "qubit_freqs_sim": (4.18, 4.54),
}

PAPER_PARAMS_ASYMMETRIC = {
    # Table I (asymmetric): f01 for q1, qc, q2 in GHz
    "qubit_freqs": (3.449, 3.63),
    "coupler_freqs": (6.526,),
    # Table I: anharmonicities eta/2pi in MHz -> GHz
    "qubit_anharm": (0.219, 0.215),
    "coupler_anharm": (0.20,),
    "coupler_EC": (0.20,),
    # Table II: sqrt(g1c g2c)/2pi = 150 MHz -> 0.150 GHz (g1c and g2c opposite signs)
    "g_couplings": ((-0.150, 0.150),),
    # Table II: g12/2pi = -9.4 MHz -> -0.0094 GHz
    "g_direct": (-0.0094,),
    "detuning_span": 0.2,
    "pulse_amp": 0.05,
    "pulse_duration": 200.0,
    "n_detunings": 11,
    "n_t": 201,
    "t_max": 200.0,
    # Text (Fig. 6 discussion): asymmetric coupler frequency range (GHz)
    "coupler_freq_range": (4.38, 5.71),
}
