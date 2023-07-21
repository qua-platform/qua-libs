import quam_sdk.constructor

# The system state is a high level abstraction of the experiment written in the language of physicists
# The structure is almost completely free
state = {
    "network": {"qop_ip": "172.16.33.100", "octave1_ip": "192.168.88.150", "octave2_ip": "192.168.88.151", "qop_port": 80, "cluster_name": 'Cluster_81', "save_dir": ""},
    "qubits": [
        {
            "name": "q0",
            "f_01": 3.1817e9,
            "lo": 3.2317e9,
            "rf_gain": 0,
            "rf_switch_mode": "on",
            "mixer_name": "octave_octave1_2",
            "anharmonicity": 250e6,
            "drag_coefficient": 0.0,
            "ac_stark_detuning": 0.0,
            "pi_length": 40,
            "pi_amp": 0.124,
            "wiring": {
                "controller": "con1",
                "I": 3,
                "Q": 4,
            },
            "T1": 1230,
            "T2": 123,
        },
    ],
    "resonators": [
        {
            "name": "rr0",
            "f_readout": 7.0840e9,
            "lo": 7.05e9,
            "rf_gain": 0,
            "rf_switch_mode": "on",
            "depletion_time": 10_000,
            "readout_pulse_length": 1_000,
            "optimal_pulse_length": 2_000,
            "readout_pulse_amp": 0.05,
            "rotation_angle": 0.0,
            "ge_threshold": 0.0,
            "wiring": {
                "controller": "con1",
                "I": 1,
                "Q": 2,
            },
        },
    ],
    "global_parameters": {
        "time_of_flight": 24,
        "saturation_amp": 0.1,
        "saturation_len": 14000,
        "con1_downconversion_offset_I": 0.0,
        "con1_downconversion_offset_Q": 0.0,
        "con1_downconversion_gain": 0,
        "con2_downconversion_offset_I": 0.0,
        "con2_downconversion_offset_Q": 0.0,
        "con2_downconversion_gain": 0,
    },
}

# Now we use QuAM SDK to construct the Python class out of the state
quam_sdk.constructor.quamConstructor(state, flat_data=False)
