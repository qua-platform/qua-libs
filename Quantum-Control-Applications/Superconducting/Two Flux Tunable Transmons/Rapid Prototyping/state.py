import quam_sdk.constructor

# system state is high level abstraction of the experiment
# written in the language of physicists
# structure is almost completely free

state = {
    "network": {"qop_ip": "172.16.33.100", "qop_port": 83},
    "local_oscillators": {
        "qubits": {
            "freq": 3.3,
            "power": 3.3
        },
        "readout": [
            {
                "freq": 6.5,
                "power": 3.3
            },
        ]
    },
    "crosstalk": {
        "flux": {
            "dc": [[0.0, 0.0], [0.0, 0.0]],
            "fast_flux": [[0.0, 0.0], [0.0, 0.0]]
        },
        "rf": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    },
    "qubits": [  
        {
            "f01": 3.52,
            "T1": 123.2,
            "wiring": {
                "I": 1,
                "Q": 2,
            },
        },
        {
            "f01": 3.21,
            "T1": 99.2,
            "wiring": {
                "I": 3,
                "Q": 4,
            },
        },
    ],
    "qubits_docs": "list of all qubits in the experiment",
    "resonators": [
        {
            "f_res": 6.3,
            "depletion_time": 10_000,
            "wiring": {
                "I": 5,
                "Q": 6,
            },
            "time_of_flight": 24,
        },
        {
            "f_res": 6.75,
            "depletion_time": 10_000,
            "wiring": {
                "I": 5,
                "Q": 6,
            },
        },
    ],
}

# Now we use QuAM SDK

quam_sdk.constructor.quamConstructor(state)
