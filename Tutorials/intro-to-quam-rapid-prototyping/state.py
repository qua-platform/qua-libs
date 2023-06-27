import quam_sdk.constructor

# The system state is a high level abstraction of the experiment written in the language of physicists
# The structure is almost completely free
state = {
    "network": {"qop_ip": "127.0.0.1", "qop_port": 80, "save_dir": ""},
    "local_oscillators": {
        "qubits": [
            {"freq": 3.3e9, "power": 18},
        ],
        "readout": [
            {"freq": 6.5e9, "power": 15},
        ],
    },
    "qubits": [
        {
            "xy": {
                "f_01": 3.52e9,
                "anharmonicity": 250e6,
                "drag_coefficient": 0.0,
                "ac_stark_detuning": 0.0,
                "pi_length": 40,
                "pi_amp": 0.124,
                "wiring": {
                    "I": 1,
                    "Q": 2,
                    "mixer_correction": {"offset_I": 0.01, "offset_Q": -0.041, "gain": 0.015, "phase": -0.0236},
                },
            },
            "z": {
                "wiring": {
                    "port": 7,
                    "filter": {"iir_taps": [], "fir_taps": []},
                },
                "flux_pulse_length": 16,
                "flux_pulse_amp": 0.175,
                "max_frequency_point": 0.0,
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
            "T1": 1230,
            "T2": 123,
        },
        {
            "xy": {
                "f_01": 3.25e9,
                "anharmonicity": 250e6,
                "drag_coefficient": 0.0,
                "ac_stark_detuning": 0.0,
                "pi_length": 40,
                "pi_amp": 0.124,
                "wiring": {
                    "I": 3,
                    "Q": 4,
                    "mixer_correction": {"offset_I": 0.01, "offset_Q": -0.041, "gain": 0.015, "phase": -0.0236},
                },
            },
            "z": {
                "wiring": {
                    "port": 8,
                    "filter": {"iir_taps": [], "fir_taps": []},
                },
                "flux_pulse_length": 16,
                "flux_pulse_amp": 0.175,
                "max_frequency_point": 0.0,
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
            "T1": 1232,
            "T2": 122,
        },
    ],
    "resonators": [
        {
            "f_res": 6.3e9,
            "f_opt": 6.3e9,
            "depletion_time": 10_000,
            "readout_pulse_length": 1_000,
            "readout_pulse_amp": 0.05,
            "rotation_angle": 0.0,
            "wiring": {
                "I": 5,
                "Q": 6,
                "mixer_correction": {"offset_I": 0.01, "offset_Q": -0.041, "gain": 0.015, "phase": -0.0236},
            },
        },
        {
            "f_res": 6.75e9,
            "f_opt": 6.75e9,
            "depletion_time": 10_000,
            "readout_pulse_length": 1_000,
            "readout_pulse_amp": 0.07,
            "rotation_angle": 0.0,
            "wiring": {
                "I": 5,
                "Q": 6,
                "mixer_correction": {"offset_I": 0.01, "offset_Q": -0.041, "gain": 0.019, "phase": -0.0214},
            },
        },
    ],
    "crosstalk": {
        "flux": {"dc": [[0.0, 0.0], [0.0, 0.0]], "fast_flux": [[0.0, 0.0], [0.0, 0.0]]},
        "rf": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    },
    "global_parameters": {
        "time_of_flight": 24,
        "downconversion_offset_I": 0.0,
        "downconversion_offset_Q": 0.0,
    },
}

# Now we use QuAM SDK to construct the Python class out of the state
quam_sdk.constructor.quamConstructor(state, flat_data=False)
