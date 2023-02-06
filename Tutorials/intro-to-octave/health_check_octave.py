from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.octave import *
from qm.qua import *
import os
import time
import matplotlib.pyplot as plt
from qualang_tools.units import unit
"""
octave_health_check.py: checks the octave's clock, synthesizers, up-converters, triggers, down-converters and calibration
"""

opx_ip = '172.0.0.1'
opx_port = 80
octave_ip = '172.0.0.1'
octave_port = 50
IF = 50e6
LO = 6e9
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": 0.0},
                2: {"offset": 0.0},
                3: {"offset": 0.0},
                4: {"offset": 0.0},
                5: {"offset": 0.0},
                6: {"offset": 0.0},
                7: {"offset": 0.0},
                8: {"offset": 0.0},
                9: {"offset": 0.0},
                10: {"offset": 0.0},
            },
            "digital_outputs": {
                1: {},
                2: {},
                3: {},
                4: {},
                5: {},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
            },
        }
    },
    "elements": {
        "qe1": {
            "mixInputs": {
                    "I": ("con1", 1),
                    "Q": ("con1", 2),
                    "lo_frequency": LO,
                    "mixer": "octave_octave1_1", # a fixed name, do not change.
                },
            "intermediate_frequency": IF,
            "operations": {
                "cw": "const",
                "readout": "readout_pulse",
            },
            "digitalInputs": {
                "switch": {
                    "port": ("con1", 1),
                    "delay": 136,
                    "buffer": 0,
                },
            },
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
            "time_of_flight": 24,
            "smearing": 0,
        },
        "qe2": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": LO,
                "mixer": "octave_octave1_2",  # a fixed name, do not change.
            },
            "intermediate_frequency": IF,
            "operations": {
                "cw": "const",
                "readout": "readout_pulse",
            },
            "digitalInputs": {
                "switch": {
                    "port": ("con1", 2),
                    "delay": 136,
                    "buffer": 0,
                },
            },
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
            "time_of_flight": 24,
            "smearing": 0,
        },
        "qe3": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": LO,
                "mixer": "octave_octave1_3",  # a fixed name, do not change.
            },
            "intermediate_frequency": IF,
            "operations": {
                "cw": "const",
            },
            "digitalInputs": {
                "switch": {
                    "port": ("con1", 3),
                    "delay": 136,
                    "buffer": 0,
                },
            },
        },
        "qe4": {
            "mixInputs": {
                "I": ("con1", 7),
                "Q": ("con1", 8),
                "lo_frequency": LO,
                "mixer": "octave_octave1_4",  # a fixed name, do not change.
            },
            "intermediate_frequency": IF,
            "operations": {
                "cw": "const",
            },
            "digitalInputs": {
                "switch": {
                    "port": ("con1", 4),
                    "delay": 136,
                    "buffer": 0,
                },
            },
        },
        "qe5": {
            "mixInputs": {
                "I": ("con1", 9),
                "Q": ("con1", 10),
                "lo_frequency": LO,
                "mixer": "octave_octave1_5",  # a fixed name, do not change.
            },
            "intermediate_frequency": IF,
            "operations": {
                "cw": "const",
            },
            "digitalInputs": {
                "switch": {
                    "port": ("con1", 5),
                    "delay": 136,
                    "buffer": 0,
                },
            },
        },


    },
    "pulses": {
            "const": {
                "operation": "control",
                "length": 1000,
                "waveforms": {
                    "I": "const_wf",
                    "Q": "zero_wf",
                },
                "digital_marker": "ON",
            },
            "readout_pulse": {
                "operation": "measurement",
                "length": 1000,
                "waveforms": {
                    "I": "readout_wf",
                    "Q": "zero_wf",
                },
                "integration_weights": {
                    "cos": "cosine_weights",
                    "sin": "sine_weights",
                    "minus_sin": "minus_sine_weights",
                },
                "digital_marker": "ON",
            },
        },
    "waveforms": {
        "zero_wf": {
            "type": "constant",
            "sample": 0.0,
        },
        "const_wf": {
            "type": "constant",
            "sample": 0.125,
        },
        "readout_wf": {
            "type": "constant",
            "sample": 0.125,
        },
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
        "OFF": {"samples": [(0, 0)]},
    },
    "integration_weights": {
        "cosine_weights": {
            "cosine": [(1.0, 1000)],
            "sine": [(0.0, 1000)],
        },
        "sine_weights": {
            "cosine": [(0.0, 1000)],
            "sine": [(1.0, 1000)],
        },
        "minus_sine_weights": {
            "cosine": [(0.0, 1000)],
            "sine": [(-1.0, 1000)],
        },
        },
    "mixers": {
            "octave_octave1_1": [
                {
                    "intermediate_frequency": IF,
                    "lo_frequency": LO,
                    "correction": (1, 0, 0, 1),
                },
            ],
            "octave_octave1_2": [
            {
                "intermediate_frequency": IF,
                "lo_frequency": LO,
                "correction": (1, 0, 0, 1),
            },
        ],
            "octave_octave1_3": [
                {
                    "intermediate_frequency": IF,
                    "lo_frequency": LO,
                    "correction": (1, 0, 0, 1),
                },
            ],
            "octave_octave1_4": [
                {
                    "intermediate_frequency": IF,
                    "lo_frequency": LO,
                    "correction": (1, 0, 0, 1),
                },
            ],
            "octave_octave1_5": [
                {
                    "intermediate_frequency": IF,
                    "lo_frequency": LO,
                    "correction": (1, 0, 0, 1),
                },
            ],
        },
}

octave_config = QmOctaveConfig()
octave_config.set_calibration_db(os.getcwd())

octave_config.add_device_info('octave1', octave_ip, octave_port)
octave_config.set_opx_octave_mapping(
    [('con1', 'octave1')])  # set default mapping between analog outputs of OPX and the octave

qmm = QuantumMachinesManager(host=opx_ip, port=opx_port, octave=octave_config)
qm = qmm.open_qm(config)

with program() as hello_octave:
    with infinite_loop_():
        play('cw', 'qe1')
        play('cw', 'qe2')
        play('cw', 'qe3')
        play('cw', 'qe4')
        play('cw', 'qe5')

set_clock = True
set_LO_and_RF_gain_mode = True
check_synthesizers = False
check_up_converters = False
check_triggers = False
check_down_converters = False
calibration = False

###########################
# Step 1 : clock settings #
###########################
if set_clock:
    external_clock = False
    if external_clock == '10MHz':
        qmm.octave_manager.set_clock("octave1", ClockType.External, ClockFrequency.MHZ_10)
    elif external_clock == '100MHz':
        qmm.octave_manager.set_clock("octave1", ClockType.External, ClockFrequency.MHZ_100)
    elif external_clock == '1000MHz' or external_clock == '1GHz':
        qmm.octave_manager.set_clock("octave1", ClockType.External, ClockFrequency.MHZ_1000)
    else:
        qmm.octave_manager.set_clock("octave1", ClockType.Internal, ClockFrequency.MHZ_10)

    # connect clock out from rear panel to oscilloscope and see 1GHz
############################################
# Step 2 : set LO, RF gain, and RF mode #
############################################
if set_LO_and_RF_gain_mode:
    ############################################
    # Step 2 : set LO, RF gain, and RF mode #
    ############################################
    elements = ['qe1', 'qe2', 'qe3', 'qe4', 'qe5']

    for i in range(len(elements)):
        qm.octave.set_lo_source(elements[i], OctaveLOSource.Internal) # Internal by default
        qm.octave.set_lo_frequency(elements[i], LO)  # assign the LO inside the octave to element
        qm.octave.set_rf_output_gain(elements[i], 0)  # can set gain from -10dB to 20dB
        qm.octave.set_rf_output_mode(elements[i], RFOutputMode.on)  # set the behaviour of the RF switch to be on. Can change it to : off, trig_normal, trig_inverse
######################################
# Step 3 : checking the synthesizers #
######################################
if check_synthesizers:
    # connect Synth1, Synth2, Synth3 (in the rear panel) to spectrum analyzer and check whether you get 6GHz signal
    job = qm.execute(hello_octave)
    job.halt()
#######################################
# Step 4 : checking the up-converters #
#######################################
# Connect RF1, RF2, RF3, RF4, RF5 to spectrum analyzer and check if you get 3 peaks: 1. LO-IF at 5.95GHz,  2. LO at 6GHz,  3. LO+IF at 6.05GHz
if check_up_converters:
    job = qm.execute(hello_octave)
    time.sleep(180) #The program will run for 3 minutes
    job.halt()
##################################
# Step 4 : checking the triggers #
##################################
if check_triggers:
    # Connect RF1, RF2, RF3, RF4, RF5 to spectrum analyzer and check if you get 3 peaks: 1. LO-IF at 5.95GHz,  2. LO at 6GHz,  3. LO+IF at 6.05GHz
    for i in range(len(elements)):
        qm.octave.set_rf_output_mode(elements[i], RFOutputMode.trig_normal)  # set the behaviour of the RF switch to be on only when triggered
    with program() as hello_octave_trigger:
        play('cw', 'qe1', duration=1e6)
        play('cw', 'qe2', duration=1e6)
        play('cw', 'qe3', duration=1e6)
        play('cw', 'qe4', duration=1e6)
        play('cw', 'qe5', duration=1e6)
    job = qm.execute(hello_octave_trigger)
    time.sleep(180) #The program will run for 3 minutes
    job.halt()
#########################################
# Step 5 : checking the down-converters #
#########################################
if check_down_converters:
    # Connect RF1 ->RF1In, RF2 ->RF2In
    # Connect IFOUT1-> AI1 , IFOUT2-> AI2
    check_down_converter_1 = True
    check_down_converter_2 = False
    u = unit()
    if check_down_converter_1:
        qm.octave.set_rf_output_gain(elements[0], -10)
        qm.octave.set_qua_element_octave_rf_in_port(elements[0], "octave1", 1)
        qm.octave.set_downconversion(elements[0], lo_source=RFInputLOSource.Internal)

        with program() as hello_octave_readout_1:
            raw_ADC_1 = declare_stream(adc_trace=True)
            measure('readout', 'qe1', raw_ADC_1)
            with stream_processing():
                raw_ADC_1.input1().save('adc_1')
                raw_ADC_1.input2().save('adc_2')

        job = qm.execute(hello_octave_readout_1)
        res = job.result_handles
        res.wait_for_all_values()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Inputs from down conversion 1')
        adc_1 = u.raw2volts(res.get("adc_1").fetch_all())
        ax1.plot(adc_1, label="Input 1")
        ax1.set_title("amp VS time Input 1")
        ax1.set_xlabel("Time [ns]")
        ax1.set_ylabel("Signal amplitude [V]")

        adc_2 = u.raw2volts(res.get("adc_2").fetch_all())
        ax2.plot(adc_2, label="Input 2")
        ax2.set_title("amp VS time Input 2")
        ax2.set_xlabel("Time [ns]")
        ax2.set_ylabel("Signal amplitude [V]")

    if check_down_converter_2:
        qm.octave.set_rf_output_gain(elements[1], -10)
        qm.octave.set_qua_element_octave_rf_in_port(elements[1], "octave1", 2)
        qm.octave.set_downconversion(elements[1], lo_source=RFInputLOSource.Dmd2LO)
        with program() as hello_octave_readout_2:
            raw_ADC_2 = declare_stream(adc_trace=True)
            measure('readout', 'qe2', raw_ADC_2)
            with stream_processing():
                raw_ADC_2.input1().save('adc_1')
                raw_ADC_2.input2().save('adc_2')

        job = qm.execute(hello_octave_readout_2)
        res = job.result_handles
        res.wait_for_all_values()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Inputs from down conversion 2')
        adc_1 = u.raw2volts(res.get("adc_1").fetch_all())
        ax1.plot(adc_1, label="Input 1")
        ax1.set_title("amp VS time Input 1")
        ax1.set_xlabel("Time [ns]")
        ax1.set_ylabel("Signal amplitude [V]")

        adc_2 = u.raw2volts(res.get("adc_2").fetch_all())
        ax2.plot(adc_2, label="Input 2")
        ax2.set_title("amp VS time Input 2")
        ax2.set_xlabel("Time [ns]")
        ax2.set_ylabel("Signal amplitude [V]")
#################################
# Step 5 : checking calibration #
#################################
if calibration:
    # Step 5.1: Connect RF1 and run these lines in order to see the uncalibrated signal first
    job = qm.execute(hello_octave)
    time.sleep(30) #The program will run for 30 seconds
    job.halt()
    # Step 5.2: Run this in order to calibrate
    for i in range(len(elements)):
        qm.octave.calibrate_element(elements[i], [(LO, IF)])  # can provide many pairs of LO & IFs.
        qm = qmm.open_qm(config)
    #Step 5.3: Run these and look at the spectrum analyzer and check if you get 1 peak at LO+IF (i.e. 6.05GHz)
    job = qm.execute(hello_octave)
    time.sleep(30) #The program will run for 30 seconds
    job.halt()



