import numpy as np
from qm.qua import *
from copy import deepcopy


class DCoffsetCalibrator:
    @staticmethod
    def _update_config(config, qe, freq):
        config["pulses"]["dc_offset_readout_pulse"] = {
            "operation": "measurement",
            "length": int(1e6),
            "waveforms": {
                "I": "dc_offset_zero_wf",
                "Q": "dc_offset_zero_wf",
            },
            "digital_marker": "dc_offset_ON",
        }
        config["digital_waveforms"]["dc_offset_ON"] = {"samples": [(1, 0)]}
        config["waveforms"]["dc_offset_zero_wf"] = {"type": "constant", "sample": 0.0}
        config["elements"][qe]["operations"]["dc_offset_readout"] = "dc_offset_readout_pulse"
        config["elements"][qe]["intermediate_frequency"] = freq
        config["mixers"][config["elements"][qe]["mixInputs"]["mixer"]][0]["intermediate_frequency"] = freq
        print(
            f"ATTENTION: Using the mixer at the 0'th position of {config['elements'][qe]['mixInputs']['mixer']} to "
            f"calibrate DC offset."
        )
        con_name = config["elements"][qe]["outputs"]["out1"][0]
        config["controllers"][con_name]["analog_inputs"][1]["offset"] = 0.0
        config["controllers"][con_name]["analog_inputs"][2]["offset"] = 0.0
        return config

    @staticmethod
    def calibrate(qmm, config, qe, **execute_args):
        """
        Returns the offset that should be applied for the analog inputs of each controller.
        Assumes that when nothing is played there should be zero incoming signal.
        :param qmm: the QuantumMachineManager to execute the program on
        :return:
        """
        config = deepcopy(config)
        with program() as cal_dc:
            reset_phase(qe)
            measure("dc_offset_readout", qe, "adc")
        freq = 1.33e4
        offsets = {}
        qm = qmm.open_qm(DCoffsetCalibrator._update_config(config, qe, freq))
        job = qm.execute(cal_dc, **execute_args)
        job.result_handles.wait_for_all_values()
        adc1 = np.mean(job.result_handles.adc_input1.fetch_all()["value"])
        adc2 = np.mean(job.result_handles.adc_input2.fetch_all()["value"])
        con_name = config["elements"][qe]["outputs"]["out1"][0]
        print("DC offsets to apply:")
        print(f"input 1 on {con_name}: ", -adc1 * (2**-12))
        print(f"input 2 on {con_name}: ", -adc2 * (2**-12))
        offsets[con_name] = (-adc1 * (2**-12), -adc2 * (2**-12))
        return offsets
