qop_ip = "127.0.0.1"
cluster_name = "Cluster_1"

# Detection / state discrimination
detection_threshold = 15  # photon counts; |1⟩ is dark (counts <= threshold)

# Cooling beam
cooling_amp = 0.25
cooling_if = 200_000_000
cooling_len = 500_000

# Repump beam
repump_amp = 0.2
repump_if = 100_000_000
repump_len = 10_000

# Detection beam
detection_amp = 0.3
detection_if = 220_000_000
detection_len = 500_000

# Shelving beam (optical or auxiliary drive to metastable state)
shelving_amp = 0.2
shelving_if = 180_000_000
shelving_len = 50_000

# Microwave qubit drive (carrier gates)
qubit_lo = 3_000_000_000
qubit_if = 100_000_000
qubit_band = 1
qubit_power_dbm = 1
pi_amp = 0.25
pi_len = 1000  # clock cycles (4 ns)

# Raman beam pair (always played together; beat note = effective transition)
raman_a_amp = 0.25
raman_a_if = 80_000_000
raman_b_amp = 0.25
raman_b_if = 80_000_000
raman_pi_amp = 0.25
raman_pi_len = 2000  # clock cycles (4 ns)
raman_sideband_if = 2_000_000  # detuning from motional frequency

# Tickle drive (electrode-based motional spectroscopy)
tickle_amp = 0.1
tickle_if = 2_000_000
tickle_len = 10_000

# Paul-trap RF. Set this to the ACTUAL trap-drive frequency in Hz (e.g. 30_970_560).
# Photon-RF correlation folds OPX timestamps against this period, so the RF
# generator must share a 10 MHz reference with the OPX.
trap_rf_freq = 20_000_000.0
# Analog time-tag unit (ns). 0.5 on OPX1000 / QOP >= 3.5.0, else 1.0.
tag_to_ns = 0.5

# PMT time tagging
pmt_max_tags = 1000  # max photons stored per tagging window
pmt_readout_len = 1_000_000  # ns; "readout" window used by detection/counting scripts
# RF-correlation tagging window (ns). This is the measure duration, so two PMT
# elements at 50% duty staggered by one window tile the timeline exactly.
rf_window_len = 10_000

# Radial compensation electrodes (DC). Amplified LF outputs: ±2.5 V.
comp_x_offset = 0.0
comp_y_offset = 0.0


config = {
    "controllers": {
        "con1": {
            "type": "opx1000",
            "fems": {
                1: {
                    "type": "LF",
                    "analog_outputs": {
                        1: {"offset": 0.0, "output_mode": "direct", "upsampling_mode": "mw"},
                        2: {"offset": 0.0, "output_mode": "direct", "upsampling_mode": "mw"},
                        3: {"offset": 0.0, "output_mode": "direct", "upsampling_mode": "mw"},
                        4: {"offset": 0.0, "output_mode": "direct", "upsampling_mode": "mw"},
                    },
                    "analog_inputs": {
                        1: {"offset": 0, "gain_db": 0},
                        2: {"offset": 0, "gain_db": 0},
                    },
                },
                2: {
                    "type": "LF",
                    "analog_outputs": {
                        1: {"offset": 0.0, "output_mode": "direct", "upsampling_mode": "mw"},
                        2: {"offset": 0.0, "output_mode": "direct", "upsampling_mode": "mw"},
                        3: {"offset": 0.0, "output_mode": "direct", "upsampling_mode": "mw"},
                        4: {"offset": comp_x_offset, "output_mode": "amplified", "upsampling_mode": "pulse"},
                        5: {"offset": comp_y_offset, "output_mode": "amplified", "upsampling_mode": "pulse"},
                    },
                },
                8: {
                    "type": "MW",
                    "analog_outputs": {
                        1: {
                            "band": qubit_band,
                            "full_scale_power_dbm": qubit_power_dbm,
                            "upconverters": {1: {"frequency": qubit_lo}},
                        },
                    },
                },
            },
        },
    },
    "elements": {
        "cooling": {
            "singleInput": {"port": ("con1", 1, 1)},
            "intermediate_frequency": cooling_if,
            "operations": {"constant": "cooling_pulse"},
        },
        "repump": {
            "singleInput": {"port": ("con1", 1, 2)},
            "intermediate_frequency": repump_if,
            "operations": {"constant": "repump_pulse"},
        },
        "detection": {
            "singleInput": {"port": ("con1", 1, 3)},
            "intermediate_frequency": detection_if,
            "operations": {"constant": "detection_pulse"},
        },
        "shelving": {
            "singleInput": {"port": ("con1", 1, 4)},
            "intermediate_frequency": shelving_if,
            "operations": {"constant": "shelving_pulse"},
        },
        "qubit": {
            "MWInput": {"port": ("con1", 8, 1), "upconverter": 1},
            "intermediate_frequency": qubit_if,
            "operations": {"x180": "x180_pulse", "x90": "x90_pulse", "constant": "constant_pulse"},
        },
        "raman_a": {
            "singleInput": {"port": ("con1", 2, 1)},
            "intermediate_frequency": raman_a_if,
            "operations": {
                "constant": "raman_a_constant_pulse",
                "x180": "raman_a_x180_pulse",
                "x90": "raman_a_x90_pulse",
                "red_sideband": "raman_a_red_sideband_pulse",
                "blue_sideband": "raman_a_blue_sideband_pulse",
            },
        },
        "raman_b": {
            "singleInput": {"port": ("con1", 2, 2)},
            "intermediate_frequency": raman_b_if,
            "operations": {
                "constant": "raman_b_constant_pulse",
                "x180": "raman_b_x180_pulse",
                "x90": "raman_b_x90_pulse",
                "red_sideband": "raman_b_red_sideband_pulse",
                "blue_sideband": "raman_b_blue_sideband_pulse",
            },
        },
        "tickle": {
            "singleInput": {"port": ("con1", 2, 3)},
            "intermediate_frequency": tickle_if,
            "operations": {"constant": "tickle_pulse"},
        },
        "comp_x": {
            "singleInput": {"port": ("con1", 2, 4)},
            "operations": {"const": "dc_pulse"},
        },
        "comp_y": {
            "singleInput": {"port": ("con1", 2, 5)},
            "operations": {"const": "dc_pulse"},
        },
        "pmt": {
            "outputs": {"out1": ("con1", 1, 1)},
            "intermediate_frequency": 0,
            "timeTaggingParameters": {
                "signalThreshold": -2000,
                "signalPolarity": "Below",
                "derivativeThreshold": -2000,
                "derivativePolarity": "Above",
            },
            "time_of_flight": 28,
            "smearing": 0,
            "operations": {"readout": "readout_pulse", "tag": "rf_tag_pulse"},
        },
        # Same physical PMT on a second ADC, used to tile tagging windows with no gaps.
        "pmt_2": {
            "outputs": {"out1": ("con1", 1, 2)},
            "intermediate_frequency": 0,
            "timeTaggingParameters": {
                "signalThreshold": -2000,
                "signalPolarity": "Below",
                "derivativeThreshold": -2000,
                "derivativePolarity": "Above",
            },
            "time_of_flight": 28,
            "smearing": 0,
            "operations": {"readout": "readout_pulse", "tag": "rf_tag_pulse"},
        },
    },
    "pulses": {
        "constant_pulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {"single": "constant_wf"},
        },
        "cooling_pulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {"single": "cooling_wf"},
        },
        "repump_pulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {"single": "repump_wf"},
        },
        "detection_pulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {"single": "detection_wf"},
        },
        "shelving_pulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {"single": "shelving_wf"},
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": pmt_readout_len,
            "waveforms": {"single": "zero_wf"},
        },
        "rf_tag_pulse": {
            "operation": "measurement",
            "length": rf_window_len,
            "waveforms": {"single": "zero_wf"},
        },
        "x180_pulse": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {"I": "pi_wf", "Q": "zero_iq_wf"},
        },
        "x90_pulse": {
            "operation": "control",
            "length": pi_len // 2,
            "waveforms": {"I": "pi_wf", "Q": "zero_iq_wf"},
        },
        "raman_a_constant_pulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {"single": "raman_a_wf"},
        },
        "raman_b_constant_pulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {"single": "raman_b_wf"},
        },
        "raman_a_x180_pulse": {
            "operation": "control",
            "length": raman_pi_len,
            "waveforms": {"single": "raman_pi_wf"},
        },
        "raman_b_x180_pulse": {
            "operation": "control",
            "length": raman_pi_len,
            "waveforms": {"single": "raman_pi_wf"},
        },
        "raman_a_x90_pulse": {
            "operation": "control",
            "length": raman_pi_len // 2,
            "waveforms": {"single": "raman_pi_wf"},
        },
        "raman_b_x90_pulse": {
            "operation": "control",
            "length": raman_pi_len // 2,
            "waveforms": {"single": "raman_pi_wf"},
        },
        "raman_a_red_sideband_pulse": {
            "operation": "control",
            "length": raman_pi_len,
            "waveforms": {"single": "raman_pi_wf"},
        },
        "raman_b_red_sideband_pulse": {
            "operation": "control",
            "length": raman_pi_len,
            "waveforms": {"single": "raman_pi_wf"},
        },
        "raman_a_blue_sideband_pulse": {
            "operation": "control",
            "length": raman_pi_len,
            "waveforms": {"single": "raman_pi_wf"},
        },
        "raman_b_blue_sideband_pulse": {
            "operation": "control",
            "length": raman_pi_len,
            "waveforms": {"single": "raman_pi_wf"},
        },
        "tickle_pulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {"single": "tickle_wf"},
        },
        "dc_pulse": {
            "operation": "control",
            "length": 16,
            "waveforms": {"single": "zero_wf"},
        },
    },
    "waveforms": {
        "constant_wf": {"type": "constant", "sample": 0.25},
        "cooling_wf": {"type": "constant", "sample": cooling_amp},
        "repump_wf": {"type": "constant", "sample": repump_amp},
        "detection_wf": {"type": "constant", "sample": detection_amp},
        "shelving_wf": {"type": "constant", "sample": shelving_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "pi_wf": {"type": "constant", "sample": pi_amp},
        "zero_iq_wf": {"type": "constant", "sample": 0.0},
        "raman_a_wf": {"type": "constant", "sample": raman_a_amp},
        "raman_b_wf": {"type": "constant", "sample": raman_b_amp},
        "raman_pi_wf": {"type": "constant", "sample": raman_pi_amp},
        "tickle_wf": {"type": "constant", "sample": tickle_amp},
    },
}
