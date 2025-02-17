"""
octave_introduction.py: shows the basic commands to control the octave's clock, synthesizers, up-converters, triggers,
down-converters and calibration
"""

from qm import QuantumMachinesManager
from qm.octave import *
from qm.octave.octave_manager import ClockMode
from qm.qua import *
import os
import time
import matplotlib.pyplot as plt
from qualang_tools.units import unit

# Flags to switch between different modes defined below
check_up_converters = False
check_triggers = False
check_down_converters = False
calibration = False

#################################
# Step 0 : Octave configuration #
#################################
qop_ip = "172.0.0.1"
cluster_name = "Cluster_1"
opx_port = None

octave_port = 11250  # Must be 11xxx, where xxx are the last three digits of the Octave IP address
con = "con1"
octave = "octave1"
# The elements used to test the ports of the Octave
elements = ["qe1", "qe2", "qe3", "qe4", "qe5"]
IF = 50e6  # The IF frequency
LO = 6e9  # The LO frequency
# The configuration used here
config = {
    "version": 1,
    "controllers": {
        con: {
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
                3: {},
                5: {},
                7: {},
                9: {},
            },
            "analog_inputs": {
                1: {"offset": 0.0},
                2: {"offset": 0.0},
            },
        }
    },
    "elements": {
        "qe1": {
            "RF_inputs": {"port": (octave, 1)},
            "RF_outputs": {"port": (octave, 1)},
            "intermediate_frequency": IF,
            "operations": {
                "cw": "const",
                "cw_wo_trig": "const_wo_trig",
                "readout": "readout_pulse",
            },
            "digitalInputs": {
                "switch": {
                    "port": (con, 1),
                    "delay": 136,
                    "buffer": 0,
                },
            },
            "time_of_flight": 24,
            "smearing": 0,
        },
        "qe2": {
            "RF_inputs": {"port": (octave, 2)},
            "RF_outputs": {"port": (octave, 2)},
            "intermediate_frequency": IF,
            "operations": {
                "cw": "const",
                "cw_wo_trig": "const_wo_trig",
                "readout": "readout_pulse",
            },
            "digitalInputs": {
                "switch": {
                    "port": (con, 3),
                    "delay": 136,
                    "buffer": 0,
                },
            },
            "time_of_flight": 24,
            "smearing": 0,
        },
        "qe3": {
            "RF_inputs": {"port": (octave, 3)},
            "intermediate_frequency": IF,
            "operations": {
                "cw": "const",
                "cw_wo_trig": "const_wo_trig",
                "readout": "readout_pulse",
            },
            "digitalInputs": {
                "switch": {
                    "port": (con, 5),
                    "delay": 136,
                    "buffer": 0,
                },
            },
        },
        "qe4": {
            "RF_inputs": {"port": (octave, 4)},
            "intermediate_frequency": IF,
            "operations": {
                "cw": "const",
                "cw_wo_trig": "const_wo_trig",
                "readout": "readout_pulse",
            },
            "digitalInputs": {
                "switch": {
                    "port": (con, 7),
                    "delay": 136,
                    "buffer": 0,
                },
            },
        },
        "qe5": {
            "RF_inputs": {"port": (octave, 5)},
            "intermediate_frequency": IF,
            "operations": {
                "cw": "const",
                "cw_wo_trig": "const_wo_trig",
                "readout": "readout_pulse",
            },
            "digitalInputs": {
                "switch": {
                    "port": (con, 9),
                    "delay": 136,
                    "buffer": 0,
                },
            },
        },
    },
    "octaves": {
        octave: {
            "RF_outputs": {
                1: {
                    "LO_frequency": LO,
                    "LO_source": "internal",  # can be external or internal. internal is the default
                    "output_mode": "always_on",  # can be: "always_on" / "always_off"/ "triggered" / "triggered_reversed". "always_off" is the default
                    "gain": 0,  # can be in the range [-20 : 0.5 : 20]dB
                },
                2: {
                    "LO_frequency": LO,
                    "LO_source": "internal",
                    "output_mode": "always_on",
                    "gain": 0,
                },
                3: {
                    "LO_frequency": LO,
                    "LO_source": "internal",
                    "output_mode": "always_on",
                    "gain": 0,
                },
                4: {
                    "LO_frequency": LO,
                    "LO_source": "internal",
                    "output_mode": "always_on",
                    "gain": 0,
                },
                5: {
                    "LO_frequency": LO,
                    "LO_source": "internal",
                    "output_mode": "always_on",
                    "gain": 0,
                },
            },
            "RF_inputs": {
                1: {
                    "LO_frequency": LO,
                    "LO_source": "internal",  # internal is the default
                    "IF_mode_I": "direct",  # can be: "direct" / "mixer" / "envelope" / "off". direct is default
                    "IF_mode_Q": "direct",
                },
                2: {
                    "LO_frequency": LO,
                    "LO_source": "external",  # external is the default
                    "IF_mode_I": "direct",
                    "IF_mode_Q": "direct",
                },
            },
            "connectivity": con,
        }
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
        "const_wo_trig": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
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
}

# Create the octave config object
octave_config = QmOctaveConfig()
# Specify where to store the outcome of the calibration (correction matrix, offsets...)
octave_config.set_calibration_db(os.getcwd())
# Add an Octave called 'octave1' with the specified IP and port
octave_config.add_device_info(octave, qop_ip, octave_port)

qmm = QuantumMachinesManager(host=qop_ip, port=opx_port, cluster_name=cluster_name, octave=octave_config)

qm = qmm.open_qm(config)

# Simple test program that plays a continuous wave through all ports
with program() as hello_octave:
    with infinite_loop_():
        for el in elements:
            play("cw", el)

###########################
# Step 1 : clock settings #
###########################
external_clock = False
if external_clock:
    # Change to the relevant external frequency
    qm.octave.set_clock(octave, clock_mode=ClockMode.External_10MHz)
else:
    qm.octave.set_clock(octave, clock_mode=ClockMode.Internal)
# You can connect clock out from rear panel to a spectrum analyzer  to see the 1GHz signal

#######################################
# Step 2 : checking the up-converters #
#######################################
if check_up_converters:
    print("-" * 37 + " Checking up-converters")
    job = qm.execute(hello_octave)
    time.sleep(60)  # The program will run for 1 minute
    job.halt()
    # You can connect RF1, RF2, RF3, RF4, RF5 to a spectrum analyzer and check the 3 peaks before calibration:
    # 1. LO-IF, 2. LO, 3. LO+IF

##################################
# Step 3 : checking the triggers #
##################################
if check_triggers:
    print("-" * 37 + " Checking triggers")
    # Connect RF1, RF2, RF3, RF4, RF5 to a spectrum analyzer and check that you get a signal for 4sec then don't get a signal fo 4 sec and so on.
    for el in elements:
        # set the behaviour of the RF switch to be on only when triggered
        qm.octave.set_rf_output_mode(el, RFOutputMode.trig_normal)

    with program() as hello_octave_trigger:
        with infinite_loop_():
            for el in elements:
                play("cw", el, duration=1e9)
                play("cw_wo_trig", el, duration=1e9)
    job = qm.execute(hello_octave_trigger)
    time.sleep(60)  # The program will run for 1 minute
    job.halt()

#########################################
# Step 4 : checking the down-converters #
#########################################
if check_down_converters:
    print("-" * 37 + " Checking down-converters")
    # Connect RF1 -> RF1In, RF2 -> RF2In
    # Connect IFOUT1 -> AI1 , IFOUT2 -> AI2
    check_down_converter_1 = True
    check_down_converter_2 = False
    u = unit()
    if check_down_converter_1:
        # Reduce the Octave gain to avoid saturating the OPX ADC or add a 20dB attenuator
        qm.octave.set_rf_output_gain(elements[0], -10)
        with program() as hello_octave_readout_1:
            raw_ADC_1 = declare_stream(adc_trace=True)
            measure("readout", elements[0], raw_ADC_1)
            with stream_processing():
                raw_ADC_1.input1().save("adc_1")
                raw_ADC_1.input2().save("adc_2")
        # Execute the program
        job = qm.execute(hello_octave_readout_1)
        res = job.result_handles
        res.wait_for_all_values()
        # Plot the results
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Inputs from down conversion 1")
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
        plt.tight_layout()

    if check_down_converter_2:
        # Reduce the Octave gain to avoid saturating the OPX ADC or add a 20dB attenuator
        qm.octave.set_rf_output_gain(elements[1], -10)

        with program() as hello_octave_readout_2:
            raw_ADC_2 = declare_stream(adc_trace=True)
            measure("readout", elements[1], raw_ADC_2)
            with stream_processing():
                raw_ADC_2.input1().save("adc_1")
                raw_ADC_2.input2().save("adc_2")
        # Execute the program
        job = qm.execute(hello_octave_readout_2)
        res = job.result_handles
        res.wait_for_all_values()
        # Plot the results
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Inputs from down conversion 2")
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
        plt.tight_layout()

#################################
# Step 5 : checking calibration #
#################################
if calibration:
    print("-" * 37 + " Play before calibration")
    # Step 5.1: Connect RF1 and run these lines in order to see the uncalibrated signal first
    job = qm.execute(hello_octave)
    time.sleep(10)  # The program will run for 10 seconds
    job.halt()
    # Step 5.2: Run this in order to calibrate
    for element in elements:
        print("-" * 37 + f" Calibrates {element}")
        qm.calibrate_element(element, {LO: (IF,)})  # can provide many IFs for specific LO
    # Step 5.3: Run these and look at the spectrum analyzer and check if you get 1 peak at LO+IF (i.e. 6.05GHz)
    print("-" * 37 + " Play after calibration")
    job = qm.execute(hello_octave)
    time.sleep(30)  # The program will run for 30 seconds
    job.halt()
