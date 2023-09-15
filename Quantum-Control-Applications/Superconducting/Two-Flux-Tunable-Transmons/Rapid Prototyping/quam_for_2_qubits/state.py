import quam_sdk.constructor

# The system state is a high level abstraction of the experiment written in the language of physicists
# The structure is almost completely free
state = {
    "network": {"qop_ip": "127.0.0.1", "cluster_name": "my_cluster", "save_dir": ""},
    "local_oscillators": {
        "qubits": [
            {"freq": 4.5e9, "power": 18},
            {"freq": 6e9, "power": 18},
        ],
        "readout": [
            {"freq": 7.35e9, "power": 15},
        ],
    },
    "qubits": [
        {
            "name": "q0",
            "xy": {
                "LO_index": 0,
                "f_01": 4.5e9,
                "anharmonicity": 250e6,
                "drag_coefficient": 0.0,
                "ac_stark_detuning": 0.0,
                "pi_length": 40,
                "pi_amp": 0.124,
                "wiring": {
                    "controller": "con1",
                    "I": 3,
                    "Q": 4,
                    "mixer_correction": {"offset_I": 0.0, "offset_Q": -0.0, "gain": 0.0, "phase": -0.0},
                },
            },
            "z": {
                "wiring": {
                    "controller": "con1",
                    "port": 7,
                    "filter": {"iir_taps": [], "fir_taps": []},
                },
                "flux_pulse_length": 16,
                "flux_pulse_amp": 0.175,
                "max_frequency_point": 0.0,
                "min_frequency_point": 0.0,
                "iswap": {
                    "length": 16,
                    "level": 0.075,
                },
                "cz": {
                    "length": 16,
                    "level": 0.075,
                },
            },
            "ge_threshold": 0.0,
            "T1": 5000,
            "T2": 1000,
            "T2echo": 2000,
        },
        {
            "name": "q1",
            "xy": {
                "LO_index": 1,
                "f_01": 6.25e9,
                "anharmonicity": 250e6,
                "drag_coefficient": 0.0,
                "ac_stark_detuning": 0.0,
                "pi_length": 40,
                "pi_amp": 0.124,
                "wiring": {
                    "controller": "con1",
                    "I": 5,
                    "Q": 6,
                    "mixer_correction": {"offset_I": 0.0, "offset_Q": -0.0, "gain": 0.0, "phase": -0.0},
                },
            },
            "z": {
                "wiring": {
                    "controller": "con1",
                    "port": 8,
                    "filter": {"iir_taps": [], "fir_taps": []},
                },
                "flux_pulse_length": 16,
                "flux_pulse_amp": 0.175,
                "max_frequency_point": 0.0,
                "min_frequency_point": 0.0,
                "iswap": {
                    "length": 16,
                    "level": 0.075,
                },
                "cz": {
                    "length": 16,
                    "level": 0.075,
                },
            },
            "ge_threshold": 0.0,
            "T1": 5000,
            "T2": 1000,
            "T2echo": 2000,
        },
    ],
    "resonators": [
        {
            "name": "rr0",
            "LO_index": 0,
            "f_res": 7.245e9,
            "f_opt": 7.245e9,
            "depletion_time": 1_000,
            "readout_pulse_length": 4_000,
            "readout_pulse_amp": 0.05,
            "rotation_angle": 0.0,
            "readout_fidelity": 0.0,
            "wiring": {
                "controller": "con1",
                "I": 1,
                "Q": 2,
                "mixer_correction": {"offset_I": 0.0, "offset_Q": -0.0, "gain": 0.0, "phase": -0.0},
            },
            "opt_weights": {
                "weights_real": [1.0] * 4_000,
                "weights_minus_imag": [0.0] * 4_000,
                "weights_imag": [0.0] * 4_000,
                "weights_minus_real": [-1.0] * 4_000,
            },
        },
        {
            "name": "rr1",
            "LO_index": 0,
            "f_res": 7.31e9,
            "f_opt": 7.31e9,
            "depletion_time": 1_000,
            "readout_pulse_length": 4_000,
            "readout_pulse_amp": 0.07,
            "rotation_angle": 0.0,
            "readout_fidelity": 0.0,
            "wiring": {
                "controller": "con1",
                "I": 1,
                "Q": 2,
                "mixer_correction": {"offset_I": 0.0, "offset_Q": -0.0, "gain": 0.0, "phase": -0.0},
            },
            "opt_weights": {
                "weights_real": [1.0] * 4_000,
                "weights_minus_imag": [0.0] * 4_000,
                "weights_imag": [0.0] * 4_000,
                "weights_minus_real": [-1.0] * 4_000,
            },
        },
    ],
    "crosstalk": {
        "z": [
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        "xy": [
            [0.0, 0.0],
            [0.0, 0.0],
        ],
    },
    "global_parameters": {
        "time_of_flight": 24,
        "downconversion_offset_I": 0.0,
        "downconversion_offset_Q": 0.0,
    },
}

# Now we use QuAM SDK to construct the Python class out of the state
quam_sdk.constructor.quamConstructor(state)
