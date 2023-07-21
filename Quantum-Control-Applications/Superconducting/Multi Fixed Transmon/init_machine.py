from quam import QuAM

machine = QuAM("quam_bootstrap_state.json", flat_data=False)

qubits_name = ["q" + str(i) for i in range(1, 9)]
qubits_connectivity = [(5, 6, "con1"), (7, 8, "con1"), (9, 10, "con1"), (1, 2, "con2"), (3, 4, "con2"), (5, 6, "con2"), (7, 8, "con2"), (9, 10, "con2")]
qubits_mixers_names = ["octave_octave1_3", "octave_octave1_4", "octave_octave1_5", "octave_octave2_1", "octave_octave2_2", "octave_octave2_3", "octave_octave2_4", "octave_octave2_5"]
qubits_frequencies = [(3.1218e9, 3.2317e9), (3.2113e9, 3.3744e9), (3.3244e9, 3.3744e9), (3.5078e9, 3.680e9), (3.6301e9, 3.680e9), (3.67e9, 3.680e9), (4.1e9, 4e9), (4.1e9, 4e9)]
resonators_name = ["rr" + str(i) for i in range(1, 9)]
resonators_connectivity = [(1, 2, "con1")]
resonators_frequencies = [(7.2236e9, 7.05e9), (7.2068e9, 7.05e9), (7.2780e9, 7.05e9), (7.0776e9, 7.05e9), (7.3406e9, 7.05e9), (7.15e9, 7.05e9), (7.15e9, 7.05e9), (7.15e9, 7.05e9)]

for i in range(8):
    machine.qubits.append(
        {
            "name": qubits_name[i],
            "f_01": qubits_frequencies[i][0],
            "lo": qubits_frequencies[i][1],
            "rf_gain": 0,
            "rf_switch_mode": "on",
            "mixer_name": qubits_mixers_names[i],
            "anharmonicity": 250e6,
            "drag_coefficient": 0.0,
            "ac_stark_detuning": 0.0,
            "pi_length": 40,
            "pi_amp": 0.124,
            "wiring": {
                "controller": qubits_connectivity[i][2],
                "I": qubits_connectivity[i][0],
                "Q": qubits_connectivity[i][1],
            },
            "T1": 1230,
            "T2": 123,
        }
    )
    machine.resonators.append(
        {
            "name": resonators_name[i],
            "f_readout": resonators_frequencies[i][0],
            "lo": resonators_frequencies[i][1],
            "rf_gain": 0,
            "rf_switch_mode": "on",
            "depletion_time": 10_000,
            "readout_pulse_length": 1_000,
            "optimal_pulse_length": 2_000,
            "readout_pulse_amp": 0.05,
            "rotation_angle": 0.0,
            "ge_threshold": 0.0,
            "wiring": {
                "controller": resonators_connectivity[0][2],
                "I": resonators_connectivity[0][0],
                "Q": resonators_connectivity[0][1],
            },
        }
    )

machine._save("quam_state.json", flat_data=False)