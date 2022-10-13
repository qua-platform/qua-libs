import quam_sdk.constructor
from scipy.signal.windows import gaussian
import numpy as np


# IQ imbalance matrix
def IQ_imbalance(g, phi):
    """
    Creates the correction matrix for the mixer imbalance caused by the gain and phase imbalances, more information can
    be seen here:
    https://docs.qualang.io/libs/examples/mixer-calibration/#non-ideal-mixer
    :param g: relative gain imbalance between the I & Q ports (unit-less). Set to 0 for no gain imbalance.
    :param phi: relative phase imbalance between the I & Q ports (radians). Set to 0 for no phase imbalance.
    """
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [
        float(N * x)
        for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]
    ]


# Gaussian pulse parameters
gauss_amp = 0.4  # The gaussian is used when calibrating pi and pi_half pulses
gauss_len = 80  # The gaussian is used when calibrating pi and pi_half pulses
gauss_wf = (gauss_amp * (gaussian(gauss_len, gauss_len / 5) -
                         gaussian(gauss_len, gauss_len / 5)[-1])).tolist()  # waveform / subtracted gaussian

qubit_control = {

    # individual qubit RF control para #
    'LO': [4.75e9]*4,
    'IF_freqs': [-150e6, -150e6, -150e6, -150e6],
    'pi_amp' : [0.4398157,  0.38254119, 0.38919345, 0.41357646],
    'gaussian_amp': [gauss_amp]*4,
    'gaussian_len': [gauss_len]*4,  # ns
    'gaussian_sigma': [gauss_len/5]*4,  # ns
    # Drive wiring and mixer cal #
    'drive_line': [ {"I": ["con1", 1], "Q": ["con1", 2]},
                    {"I": ["con1", 1], "Q": ["con1", 2]},
                    {"I": ["con1", 1], "Q": ["con1", 2]},
                    {"I": ["con1", 1], "Q": ["con1", 2]},
                    ],
    'correction_matrix': IQ_imbalance(0.003, 0.045),
    # flux line #
    'flux_line':
        {"f0": ["con2", 5],
         "f1": ["con2", 6],
         "f2": ["con2", 7],
         "f3": ["con2", 8],
         },
    "crosstalk_matrix": {
        "static": [  # index 0, 1 -> correspond to qubit0 talking to qubit1
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        "fast": [  # in unit of GHz/V, column vector is the normalized fast flux vector
            [-0.6161186, -0.05305659, -0.09892138, -0.03762675],
            [-0.11224173,  0.32837944, -0.09124057, -0.04864316],
            [-0.14225446,  0.06075035,  0.39291193, -0.10490801],
            [-0.27958306,  0.13788062,  0.16568033,  0.65200215]
        ],
    },
}
readout_control = {
    'LO':  [6.37e9]*4,
    'IF_freqs': [-52.9e6, -94.5e6, -137.35e6, -197.00e6],
    'read_amp': [0.3]*4,
    'readout_len': [2000]*4,   # ns
    'tof': [272]*4,            # ns

    # Mixer Cal #
    'readout_line': [{"I": ["con1", 3], "Q": ["con1", 4]},
                    {"I": ["con1", 3], "Q": ["con1", 4]},
                    {"I": ["con1", 3], "Q": ["con1", 4]},
                    {"I": ["con1", 3], "Q": ["con1", 4]},
                    ],
    'correction_matrix': IQ_imbalance(0.006, 0.017),
}
sideband_control = {
    'LO' : [11e9, 1.8e9],
    'IF_freqs': [-183e6, -300e6],
    'sideband_amp': [0.3, 0.3],
    'sideband_line': [
        {"I": ["con1", 7], "Q": ["con1", 8]},
        {"I": ["con1", 9], "Q": ["con1", 10]},
                    ],
    'correction_matrix': [
        IQ_imbalance(0.0, -0.02),
        IQ_imbalance(0.03, 0.101),
    ],
}


state = {
    "_func": ["config.build_config"],
    "analog_outputs": [
        {
            "controller": "con1",
            "output": 1,
            "offset": 0.0,
        },
        {
            "controller": "con1",
            "output": 2,
            "offset": 0.0,
        },
        {
            "controller": "con1",
            "output": 3,
            "offset": 0.0,
        },
        {
            "controller": "con1",
            "output": 4,
            "offset": 0.0,
        },
        {"controller": "con1", "output": 5, "offset": 0.0},
        {"controller": "con1", "output": 6, "offset": 0.0},
        {"controller": "con1", "output": 7, "offset": 0.0},
        {"controller": "con1", "output": 8, "offset": 0.0},
        {"controller": "con1", "output": 9, "offset": 0.0},
        {"controller": "con1", "output": 10, "offset": 0.0},
        {"controller": "con2", "output": 1, "offset": 0.0},
        {"controller": "con2", "output": 2, "offset": 0.0},
        {"controller": "con2", "output": 3, "offset": 0.0},
        {"controller": "con2", "output": 4, "offset": 0.0},
        {"controller": "con2", "output": 5, "offset": 0.0},
    ],
    "analog_inputs": [
        {"controller": "con1", "input": 1, "offset": 0.0, "gain_db": 0},
        {"controller": "con1", "input": 2, "offset": 0.0, "gain_db": 0},
        {"controller": "con2", "input": 1, "offset": 0.0, "gain_db": 0},
    ],
    "analog_waveforms": [
        {"name": "const_wf", "type": "constant", "samples": [0.2]},
        {"name": "saturation_drive_wf", "type": "constant", "samples": [0.2]},
        {"name": "zero_wf", "type": "constant", "samples": [0.0]},
        {"name": "gaussian_wf", "type": "arbitrary", "samples": gauss_wf},
    ],
    "digital_waveforms": [{"name": "ON", "samples": [[1, 0]]}],
    "pulses": [
        {
            "name": "const_pulse",
            "operation": "control",
            "length": 100,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
        },
        {
            "name": "saturation_pulse",
            "operation": "control",
            "length": 100000,
            "waveforms": {"I": "saturation_drive_wf", "Q": "zero_wf"},
        },
    ],
    "pulses_single": [
        {
            "name": "const_flux_pulse",
            "operation": "control",
            "length": 100,
            "waveforms": {
                "single": "const_wf",
            },
        }
    ],
    "readout_lines": [
        {
            "length": 0.8e-6,
            "length_docs": "readout time (seconds) on this drive line",
            "lo_freq": 6.57e9,
            "lo_freq_docs": "LO frequency for readout line"
        }
    ],
    "readout_resonators": [
        {
            "f_res": 6.45218e9,
            "f_res_docs": "resonator frequency (Hz)",
            "q_factor": 1.0,
            "readout_regime": "low_power",
            "readout_amplitude": 0.2,
            "opt_readout_frequency": 4.52503e9,
            "rotation_angle": 41.3,  # degrees
            "readout_fidelity": 0.84,
            "chi": 1e6,
            "wiring": {
                "readout_line_index": 0,
                "time_of_flight": 260,
                "I": ["con1", 9],
                "Q": ["con1", 10],
                "correction_matrix": IQ_imbalance(0, 0),
            },
        },
        {
            "f_res": 6.53269e9,
            "f_res_docs": "resonator frequency (Hz)",
            "q_factor": 1.0,
            "readout_regime": "low_power",
            "readout_amplitude": 0.2,
            "opt_readout_frequency": 4.52503e9,
            "rotation_angle": 41.3,  # degrees
            "readout_fidelity": 0.84,
            "chi": 1e6,
            "wiring": {
                "readout_line_index": 0,
                "time_of_flight": 260,
                "I": ["con1", 9],
                "Q": ["con1", 10],
                "correction_matrix": IQ_imbalance(0, 0),
            },
        },
        {
            "f_res": 6.35218e9,
            "f_res_docs": "resonator frequency (Hz)",
            "q_factor": 1.0,
            "readout_regime": "low_power",
            "readout_amplitude": 0.2,
            "opt_readout_frequency": 4.52503e9,
            "rotation_angle": 41.3,  # degrees
            "readout_fidelity": 0.84,
            "chi": 1e6,
            "wiring": {
                "readout_line_index": 0,
                "time_of_flight": 260,
                "I": ["con1", 9],
                "Q": ["con1", 10],
                "correction_matrix": IQ_imbalance(0, 0),
            },
        },
        {
            "f_res": 6.63269e9,
            "f_res_docs": "resonator frequency (Hz)",
            "q_factor": 1.0,
            "readout_regime": "low_power",
            "readout_amplitude": 0.2,
            "opt_readout_frequency": 4.52503e9,
            "rotation_angle": 41.3,  # degrees
            "readout_fidelity": 0.84,
            "chi": 1e6,
            "wiring": {
                "readout_line_index": 0,
                "time_of_flight": 260,
                "I": ["con1", 9],
                "Q": ["con1", 10],
                "correction_matrix": IQ_imbalance(0, 0),
            },
        },
        {
            "f_res": 6.63269e9,
            "f_res_docs": "resonator frequency (Hz)",
            "q_factor": 1.0,
            "readout_regime": "low_power",
            "readout_amplitude": 0.2,
            "opt_readout_frequency": 4.52503e9,
            "rotation_angle": 41.3,  # degrees
            "readout_fidelity": 0.84,
            "chi": 1e6,
            "wiring": {
                "readout_line_index": 0,
                "time_of_flight": 260,
                "I": ["con1", 9],
                "Q": ["con1", 10],
                "correction_matrix": IQ_imbalance(0, 0),
            },
        },
    ],
    "crosstalk_matrix": {
        "static": [   # index 0, 1 -> correspond to qubit0 talking to qubit1
            [1.0, 0.0],
            [0.0, 1.0]
        ],
        "static_docs": "crosstalk matrix for slow flux lines",
        "fast": [
            [1.0, 0.0],
            [-0.2, 1.0]
        ],
        "fast_docs": "crosstalk matrix for fast flux lines",
    },
    "drive_line": [
        {
            "qubits": [0, 1],
            "qubits_docs": "qubits associated with this drive line",
            "freq":4.6e9,
            "freq_docs": "LO frequency",
            "power": 15,
            "power_docs": "LO power to mixer",
        },
        {
            "qubits": [2, 3, 4],
            "qubits_docs": "qubits associated with this drive line",
            "freq":5.1e9,
            "freq_docs": "LO frequency",
            "power": 15,
            "power_docs": "LO power to mixer",
        },
    ],
    "qubits": [
        {
            "f_01": 4.52503e9,
            "f_01_docs": "01 transition frequency (Hz)",
            "anharmonicity": 0.0,
            "rabi_freq": 1.0e6,
            "t1": 18e-6,
            "t2": 200e-6,
            "t2star": 5e-6,
            "driving": {
                "gate_len": 60e-9,
                "gate_len_docs": "(seconds)",
                "gate_sigma": 20e-9,
                "gate_shape": "gaussian",
                "angle2volt": {"deg90": 0.1, "deg180": 0.2},
            },
            "wiring": {
                "I": ["con1", 1],
                "Q": ["con1", 2],
                "correction_matrix": IQ_imbalance(0, 0),
                "flux_line": ["con2", 1],
                "flux_line_docs": "controller port associated with fast flux line",
                "flux_filter_coef": {
                    "feedforward": [0.932282, -0.92300557],
                    "feedback": [0.99072356]
                },
                "flux_filter_coef_docs": "filter taps IIR and FIR to fast flux line",
            },
            "sequence_states":[
                {"name": "dissipative_stabilization",
                "amplitude": 0.2,
                "length": 200
                },
                {"name": "Excitation",
                "amplitude": 0.3,
                "length": 80
                },
                {"name": "Free_evolution",
                "amplitude": 0.2,
                "length": 200
                },
                {"name": "Jump",
                "amplitude": 0.4,
                "length": 16
                },
                {"name": "Readout",
                "amplitude": 0.35,
                "length": 1000
                },
                {"name": "flux_balancing",
                "amplitude": -0.35,
                "length": 400
                },
            ]
        },
        {
            "f_01": 4.63097e9,
            "f_01_docs": "01 transition frequency (Hz)",
            "anharmonicity": 0.0,
            "rabi_freq": 1.0e6,
            "t1": 18e-6,
            "t2": 200e-6,
            "t2star": 5e-6,
            "driving": {
                "gate_len": 60e-9,
                "gate_len_docs": "(seconds)",
                "gate_sigma": 20e-9,
                "gate_shape": "gaussian",
                "angle2volt": {"deg90": 0.1, "deg180": 0.2},
            },
            "wiring": {
                "I": ["con1", 3],
                "Q": ["con1", 4],
                "correction_matrix": IQ_imbalance(0, 0),
                "flux_line": ["con2", 2],
                "flux_line_docs": "controller port associated with fast flux line",
                "flux_filter_coef": {
                    "feedforward": [],
                    "feedback": []
                },
                "flux_filter_coef_docs": "filter taps IIR and FIR to fast flux line",
            },
            "sequence_states":[
                {"name": "dissipative_stabilization",
                "amplitude": 0.2,
                "length": 200
                },
                {"name": "Excitation",
                "amplitude": 0.3,
                "length": 80
                },
                {"name": "Free_evolution",
                "amplitude": 0.2,
                "length": 200
                },
                {"name": "Jump",
                "amplitude": 0.4,
                "length": 16
                },
                {"name": "Readout",
                "amplitude": 0.35,
                "length": 1000
                },
                {"name": "flux_balancing",
                "amplitude": -0.35,
                "length": 400
                },
            ]
        },
        {
            "f_01": 4.95097e9,
            "f_01_docs": "01 transition frequency (Hz)",
            "anharmonicity": 0.0,
            "rabi_freq": 1.0e6,
            "t1": 18e-6,
            "t2": 200e-6,
            "t2star": 5e-6,
            "driving": {
                "gate_len": 60e-9,
                "gate_len_docs": "(seconds)",
                "gate_sigma": 20e-9,
                "gate_shape": "gaussian",
                "angle2volt": {"deg90": 0.1, "deg180": 0.2},
            },
            "wiring": {
                "I": ["con1", 5],
                "Q": ["con1", 6],
                "correction_matrix": IQ_imbalance(0, 0),
                "flux_line": ["con2", 3],
                "flux_line_docs": "controller port associated with fast flux line",
                "flux_filter_coef": {
                    "feedforward": [],
                    "feedback": []
                },
                "flux_filter_coef_docs": "filter taps IIR and FIR to fast flux line",
            },
            "sequence_states":[
                {"name": "dissipative_stabilization",
                "amplitude": 0.2,
                "length": 200
                },
                {"name": "Excitation",
                "amplitude": 0.3,
                "length": 80
                },
                {"name": "Free_evolution",
                "amplitude": 0.2,
                "length": 200
                },
                {"name": "Jump",
                "amplitude": 0.4,
                "length": 16
                },
                {"name": "Readout",
                "amplitude": 0.35,
                "length": 1000
                },
                {"name": "flux_balancing",
                "amplitude": -0.35,
                "length": 400
                },
            ]
        },
        {
            "f_01": 5.05097e9,
            "f_01_docs": "01 transition frequency (Hz)",
            "anharmonicity": 0.0,
            "rabi_freq": 1.0e6,
            "t1": 18e-6,
            "t2": 200e-6,
            "t2star": 5e-6,
            "driving": {
                "gate_len": 60e-9,
                "gate_len_docs": "(seconds)",
                "gate_sigma": 20e-9,
                "gate_shape": "gaussian",
                "angle2volt": {"deg90": 0.1, "deg180": 0.2},
            },
            "wiring": {
                "I": ["con1", 7],
                "Q": ["con1", 8],
                "correction_matrix": IQ_imbalance(0, 0),
                "flux_line": ["con2", 4],
                "flux_line_docs": "controller port associated with fast flux line",
                "flux_filter_coef": {
                    "feedforward": [],
                    "feedback": []
                },
                "flux_filter_coef_docs": "filter taps IIR and FIR to fast flux line",
            },
            "sequence_states":[
                {"name": "dissipative_stabilization",
                "amplitude": 0.2,
                "length": 200
                },
                {"name": "Excitation",
                "amplitude": 0.3,
                "length": 80
                },
                {"name": "Free_evolution",
                "amplitude": 0.2,
                "length": 200
                },
                {"name": "Jump",
                "amplitude": 0.4,
                "length": 16
                },
                {"name": "Readout",
                "amplitude": 0.35,
                "length": 1000
                },
                {"name": "flux_balancing",
                "amplitude": -0.35,
                "length": 400
                },
            ]
        },
        {
            "f_01": 5.15097e9,  # Hz
            "anharmonicity": 0.0,
            "rabi_freq": 1.0e6,
            "t1": 18e-6,
            "t2": 200e-6,
            "t2star": 5e-6,
            "driving": {
                "gate_len": 60e-9,
                "gate_len_docs": "(seconds)",
                "gate_sigma": 20e-9,
                "gate_shape": "gaussian",
                "angle2volt": {"deg90": 0.1, "deg180": 0.2},
            },
            "wiring": {
                "I": ["con1", 7],
                "Q": ["con1", 8],
                "correction_matrix": IQ_imbalance(0, 0),
                "flux_line": ["con2", 5],
                "flux_line_docs": "controller port associated with fast flux line",
                "flux_filter_coef": {
                    "feedforward": [],
                    "feedback": []
                },
                "flux_filter_coef_docs": "filter taps IIR and FIR to fast flux line",
            },
            "sequence_states":[
                {"name": "dissipative_stabilization",
                "amplitude": 0.2,
                "length": 200
                },
                {"name": "Excitation",
                "amplitude": 0.3,
                "length": 80
                },
                {"name": "Free_evolution",
                "amplitude": 0.2,
                "length": 200
                },
                {"name": "Jump",
                "amplitude": 0.4,
                "length": 16
                },
                {"name": "Readout",
                "amplitude": 0.35,
                "length": 1000
                },
                {"name": "flux_balancing",
                "amplitude": -0.35,
                "length": 400
                },
            ]
        },
    ],
    "single_qubit_operations": [
        {"direction": "x", "angle": 180},
        {"direction": "x", "angle": -180},
        {"direction": "x", "angle": 90},
        {"direction": "x", "angle": -90},
        {"direction": "y", "angle": 180},
        {"direction": "y", "angle": -180},
        {"direction": "y", "angle": 90},
        {"direction": "y", "angle": -90},
    ],
    "running_strategy": {"running": True, "start": [], "end": []},
    "sideband": [
        {
            "f_res": sideband_control['IF_freqs'][0] + sideband_control['LO'][0],  # Hz
            "LO": sideband_control['LO'][0],
            'IF': sideband_control['IF_freqs'][0],
            "sideband_amplitude": sideband_control['sideband_amp'][0],
            "wiring": {
                "I": sideband_control['sideband_line'][0]['I'],
                "Q": sideband_control['sideband_line'][0]['Q'],
                "correction_matrix": sideband_control['correction_matrix'][0],
            },
        },

        {
            "f_res": sideband_control['IF_freqs'][1] + sideband_control['LO'][1],  # Hz
            "LO": sideband_control['LO'][1],
            'IF': sideband_control['IF_freqs'][1],
            "sideband_amplitude": sideband_control['sideband_amp'][1],
            "wiring": {
                "I": sideband_control['sideband_line'][1]['I'],
                "Q": sideband_control['sideband_line'][1]['Q'],
                "correction_matrix": sideband_control['correction_matrix'][1],
            },
        },
    ],

}

# Now we use QuAM SDK

quam_sdk.constructor.quamConstructor(state)