def strip_octaves_from_config(config):
    octaves_config = config.pop("octaves")
    for key, channel in config["elements"].items():
        if "RF_outputs" in channel:  # OPX input IQ channels
            RF_outputs_entry = channel.pop("RF_outputs")
            octave, idx = RF_outputs_entry["port"]
            if idx != 1:
                raise ValueError("Only RF output 1 is supported for Octave")

            IF_outputs = octaves_config[octave]["IF_outputs"]
            channel["outputs"] = {
                "out1": IF_outputs["IF_out1"]["port"],
                "out2": IF_outputs["IF_out2"]["port"],
            }
        if "RF_inputs" in channel:  # OPX output IQ channesl
            RF_inputs_entry = channel.pop("RF_inputs")
            octave, idx = RF_inputs_entry["port"]
            RF_input = octaves_config[octave]["RF_outputs"][idx]

            channel["mixInputs"] = {
                "I": RF_input["I_connection"],
                "Q": RF_input["Q_connection"],
                "mixer": f"{key}.mixer",
                "lo_frequency": RF_input["LO_frequency"],
            }

            try:
                q_name, ch_name = key.split(".")
                qubit = machine.qubits[q_name]
                ch = getattr(qubit, ch_name)
                mixer_calibration = ch.mixer_calibration or [1, 0, 0, 1]
            except Exception:
                mixer_calibration = [1, 0, 0, 1]

            config["mixers"] = {
                f"{key}.mixer": [
                    {
                        "intermediate_frequency": channel["intermediate_frequency"],
                        "lo_frequency": RF_input["LO_frequency"],
                        "correction": mixer_calibration,
                    }
                ]
            }

    return config